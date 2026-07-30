from dataclasses import dataclass
from datetime import date
import logging
from typing import Any, Literal, Mapping

import asyncpg
from aiogram import Bot
from aiogram.enums import ChatType

from app.db.publications import (
    create_publication_attempt,
    mark_publication_failed,
    mark_publication_published,
    mark_publication_unknown,
)

logger = logging.getLogger(__name__)

PublicationStatus = Literal["published", "unknown"]


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """Результат отправки публикации в Telegram."""

    batch_id: int
    generated_post_id: int
    publication_attempt_id: int
    publication_date: date
    edition: int
    telegram_message_id: int
    database_status: PublicationStatus
    requires_review: bool


class PublicationStateUncertainError(RuntimeError):
    """
    Telegram подтвердил публикацию, но её состояние не удалось записать.

    Повторная отправка в таком случае запрещена до ручной проверки.
    """

    def __init__(
        self,
        *,
        publication_attempt_id: int,
        telegram_message_id: int,
    ) -> None:
        self.publication_attempt_id = publication_attempt_id
        self.telegram_message_id = telegram_message_id

        super().__init__(
            "Telegram подтвердил публикацию, но состояние попытки "
            "не удалось надёжно записать в PostgreSQL. "
            "Повторная отправка запрещена до ручной проверки: "
            f"publication_attempt_id={publication_attempt_id}, "
            f"telegram_message_id={telegram_message_id}"
        )


def _enum_value(value: Any) -> str:
    """Возвращает строковое значение Enum или обычной строки."""

    return str(getattr(value, "value", value))


def _error_text(error: BaseException) -> str:
    """Формирует единый текст ошибки для аудита."""

    return f"{type(error).__name__}: {error}"


async def _close_bot_session(bot: Bot) -> None:
    """
    Закрывает HTTP-сессию, не меняя результат Telegram-публикации.

    Ошибка закрытия сессии не означает, что сообщение не было отправлено.
    """

    try:
        await bot.session.close()
    except Exception:
        logger.exception(
            "Не удалось корректно закрыть сессию Telegram Bot API"
        )


async def publish_text_to_channel(
    pool: asyncpg.Pool,
    *,
    bot_token: str,
    telegram_chat_id: int,
    publication_date: date,
    post_text: str,
    metadata: Mapping[str, Any],
    disable_notification: bool = True,
    simulate_finalization_failure: bool = False,
) -> PublicationResult:
    """
    Публикует текст в Telegram и сохраняет жизненный цикл в PostgreSQL.

    До получения telegram_message_id ошибка считается failed.

    После получения telegram_message_id повторная отправка запрещена.
    Если финализация в PostgreSQL не удалась, попытка получает unknown.
    """

    request_payload = {
        "chat_id": telegram_chat_id,
        "text": post_text,
        "text_format": "plain_text",
        "disable_notification": disable_notification,
    }

    publication = await create_publication_attempt(
        pool,
        publication_date=publication_date,
        telegram_chat_id=telegram_chat_id,
        post_text=post_text,
        request_payload=request_payload,
        metadata=metadata,
    )

    bot = Bot(token=bot_token)

    try:
        try:
            chat = await bot.get_chat(telegram_chat_id)

            if chat.type != ChatType.CHANNEL:
                raise RuntimeError(
                    "TELEGRAM_CHANNEL_ID указывает не на канал: "
                    f"chat_type={chat.type}"
                )

            message = await bot.send_message(
                chat_id=telegram_chat_id,
                text=post_text,
                disable_notification=disable_notification,
            )

        except Exception as error:
            error_message = _error_text(error)

            try:
                await mark_publication_failed(
                    pool,
                    publication,
                    error_message=error_message,
                )
            except Exception as database_error:
                raise RuntimeError(
                    "Публикация в Telegram завершилась ошибкой, "
                    "и статус failed не удалось записать в PostgreSQL: "
                    f"publication_attempt_id="
                    f"{publication.publication_attempt_id}"
                ) from database_error

            raise

        response_payload = {
            "message_id": message.message_id,
            "chat_id": message.chat.id,
            "chat_type": _enum_value(message.chat.type),
            "chat_title": message.chat.title,
            "message_date": message.date.isoformat(),
        }

        try:
            if simulate_finalization_failure:
                raise RuntimeError(
                    "Simulated database finalization failure"
                )

            await mark_publication_published(
                pool,
                publication,
                telegram_message_id=message.message_id,
                response_payload=response_payload,
            )

        except Exception as finalization_error:
            error_message = _error_text(finalization_error)

            try:
                await mark_publication_unknown(
                    pool,
                    publication,
                    telegram_message_id=message.message_id,
                    response_payload=response_payload,
                    error_message=error_message,
                )
            except Exception as unknown_status_error:
                raise PublicationStateUncertainError(
                    publication_attempt_id=(
                        publication.publication_attempt_id
                    ),
                    telegram_message_id=message.message_id,
                ) from unknown_status_error

            return PublicationResult(
                batch_id=publication.batch_id,
                generated_post_id=publication.generated_post_id,
                publication_attempt_id=(
                    publication.publication_attempt_id
                ),
                publication_date=publication.publication_date,
                edition=publication.edition,
                telegram_message_id=message.message_id,
                database_status="unknown",
                requires_review=True,
            )

        return PublicationResult(
            batch_id=publication.batch_id,
            generated_post_id=publication.generated_post_id,
            publication_attempt_id=(
                publication.publication_attempt_id
            ),
            publication_date=publication.publication_date,
            edition=publication.edition,
            telegram_message_id=message.message_id,
            database_status="published",
            requires_review=False,
        )

    finally:
        await _close_bot_session(bot)