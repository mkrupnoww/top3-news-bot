import logging
from typing import Any

import asyncpg
from aiogram import Bot
from aiogram.enums import ChatType, ParseMode

from app.db.approved_publications import (
    prepare_approved_publication,
)
from app.db.publications import (
    mark_publication_failed,
    mark_publication_published,
    mark_publication_unknown,
)
from app.publication.service import (
    PublicationResult,
    PublicationStateUncertainError,
)


logger = logging.getLogger(__name__)


def _enum_value(value: Any) -> str:
    """Возвращает строковое значение Enum или обычной строки."""

    return str(getattr(value, "value", value))


def _error_text(error: BaseException) -> str:
    """Формирует текст ошибки для аудита."""

    return f"{type(error).__name__}: {error}"


def _resolve_parse_mode(
    text_format: str,
) -> ParseMode | None:
    """Преобразует формат из PostgreSQL в parse_mode Telegram."""

    parse_modes = {
        "plain_text": None,
        "markdown": ParseMode.MARKDOWN,
        "markdown_v2": ParseMode.MARKDOWN_V2,
        "html": ParseMode.HTML,
    }

    if text_format not in parse_modes:
        raise ValueError(
            "Неподдерживаемый формат текста: "
            f"text_format={text_format}"
        )

    return parse_modes[text_format]


async def _close_bot_session(bot: Bot) -> None:
    """Закрывает HTTP-сессию Telegram без изменения статуса поста."""

    try:
        await bot.session.close()
    except Exception:
        logger.exception(
            "Не удалось корректно закрыть "
            "сессию Telegram Bot API"
        )


async def publish_approved_post(
    pool: asyncpg.Pool,
    *,
    bot_token: str,
    generated_post_id: int,
    disable_notification: bool = True,
) -> PublicationResult:
    """
    Публикует существующий одобренный generated_post.

    Новый batch и новый generated_post не создаются.
    После получения telegram_message_id повторная отправка запрещена.
    """

    # Сначала читаем формат, не создавая попытку.
    async with pool.acquire() as connection:
        text_format = await connection.fetchval(
            """
            SELECT text_format
            FROM generated_posts
            WHERE generated_post_id = $1
            """,
            generated_post_id,
        )

    if text_format is None:
        raise LookupError(
            "Пост не найден: "
            f"generated_post_id={generated_post_id}"
        )

    parse_mode = _resolve_parse_mode(text_format)

    prepared = await prepare_approved_publication(
        pool,
        generated_post_id=generated_post_id,
        disable_notification=disable_notification,
        parse_mode=(
            parse_mode.value
            if parse_mode is not None
            else None
        ),
    )

    publication = prepared.publication
    bot = Bot(token=bot_token)

    try:
        try:
            chat = await bot.get_chat(
                prepared.telegram_chat_id
            )

            if chat.type != ChatType.CHANNEL:
                raise RuntimeError(
                    "Целевой Telegram chat не является каналом: "
                    f"chat_type={chat.type}"
                )

            message = await bot.send_message(
                chat_id=prepared.telegram_chat_id,
                text=prepared.post_text,
                parse_mode=parse_mode,
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
                    "Telegram-публикация завершилась ошибкой, "
                    "и статус failed не удалось записать: "
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
            "existing_approved_post": True,
        }

        try:
            await mark_publication_published(
                pool,
                publication,
                telegram_message_id=message.message_id,
                response_payload=response_payload,
            )

        except Exception as finalization_error:
            error_message = _error_text(
                finalization_error
            )

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
                    telegram_message_id=(
                        message.message_id
                    ),
                ) from unknown_status_error

            return PublicationResult(
                batch_id=publication.batch_id,
                generated_post_id=(
                    publication.generated_post_id
                ),
                publication_attempt_id=(
                    publication.publication_attempt_id
                ),
                publication_date=(
                    publication.publication_date
                ),
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