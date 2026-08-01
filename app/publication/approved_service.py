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
from app.publication.telegram_text import (
    prepare_telegram_text,
)


logger = logging.getLogger(__name__)


def _enum_value(
    value: Any,
) -> str:
    """Возвращает значение Enum или обычной строки."""

    return str(
        getattr(
            value,
            "value",
            value,
        )
    )


def _error_text(
    error: BaseException,
) -> str:
    """Формирует текст ошибки для аудита."""

    return (
        f"{type(error).__name__}: "
        f"{error}"
    )


def _resolve_parse_mode(
    value: str | None,
) -> ParseMode | None:
    """Преобразует строку parse_mode в Enum aiogram."""

    if value is None:
        return None

    try:
        return ParseMode(value)
    except ValueError as error:
        raise ValueError(
            "Неподдерживаемый Telegram "
            "parse_mode: "
            f"{value}"
        ) from error


async def _close_bot_session(
    bot: Bot,
) -> None:
    """Закрывает HTTP-сессию Telegram."""

    try:
        await bot.session.close()
    except Exception:
        logger.exception(
            "Не удалось корректно закрыть "
            "сессию Telegram Bot API"
        )


async def _load_source_post(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
) -> tuple[str, str]:
    """Читает исходный текст и формат generated_post."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                post_text,
                text_format
            FROM generated_posts
            WHERE generated_post_id = $1
            """,
            generated_post_id,
        )

    if record is None:
        raise LookupError(
            "Пост не найден: "
            f"generated_post_id="
            f"{generated_post_id}"
        )

    post_text = record["post_text"]
    text_format = record["text_format"]

    if not isinstance(post_text, str):
        raise ValueError(
            "generated_posts.post_text "
            "должен быть строкой."
        )

    if not isinstance(text_format, str):
        raise ValueError(
            "generated_posts.text_format "
            "должен быть строкой."
        )

    return (
        post_text,
        text_format,
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

    Внутренний Markdown проекта преобразуется
    в безопасный Telegram HTML до создания
    publication_attempt.

    В request_payload сохраняется именно тот
    текст, который передаётся Telegram Bot API.

    Новый batch и generated_post не создаются.
    После получения telegram_message_id
    повторная отправка запрещена.
    """

    (
        source_post_text,
        source_text_format,
    ) = await _load_source_post(
        pool,
        generated_post_id=generated_post_id,
    )

    prepared_text = prepare_telegram_text(
        source_post_text,
        text_format=source_text_format,
    )

    prepared = await prepare_approved_publication(
        pool,
        generated_post_id=generated_post_id,
        disable_notification=(
            disable_notification
        ),
        telegram_text=prepared_text.text,
        telegram_text_format=(
            prepared_text.text_format
        ),
        parse_mode=prepared_text.parse_mode,
        source_post_text=source_post_text,
        source_text_format=(
            source_text_format
        ),
    )

    publication = prepared.publication

    parse_mode = _resolve_parse_mode(
        prepared.parse_mode
    )

    bot = Bot(
        token=bot_token
    )

    try:
        try:
            chat = await bot.get_chat(
                prepared.telegram_chat_id
            )

            if chat.type != ChatType.CHANNEL:
                raise RuntimeError(
                    "Целевой Telegram chat "
                    "не является каналом: "
                    f"chat_type={chat.type}"
                )

            message = await bot.send_message(
                chat_id=(
                    prepared.telegram_chat_id
                ),
                text=prepared.post_text,
                parse_mode=parse_mode,
                disable_notification=(
                    disable_notification
                ),
            )

        except Exception as error:
            error_message = _error_text(
                error
            )

            try:
                await mark_publication_failed(
                    pool,
                    publication,
                    error_message=(
                        error_message
                    ),
                )
            except Exception as database_error:
                raise RuntimeError(
                    "Telegram-публикация "
                    "завершилась ошибкой, "
                    "и статус failed не удалось "
                    "записать: "
                    "publication_attempt_id="
                    f"{publication.publication_attempt_id}"
                ) from database_error

            raise

        response_payload = {
            "message_id": (
                message.message_id
            ),
            "chat_id": message.chat.id,
            "chat_type": _enum_value(
                message.chat.type
            ),
            "chat_title": (
                message.chat.title
            ),
            "message_date": (
                message.date.isoformat()
            ),
            "existing_approved_post": True,
            "sent_text_format": (
                prepared.text_format
            ),
            "source_text_format": (
                prepared.source_text_format
            ),
            "parse_mode": (
                prepared.parse_mode
            ),
        }

        try:
            await mark_publication_published(
                pool,
                publication,
                telegram_message_id=(
                    message.message_id
                ),
                response_payload=(
                    response_payload
                ),
            )

        except Exception as finalization_error:
            error_message = _error_text(
                finalization_error
            )

            try:
                await mark_publication_unknown(
                    pool,
                    publication,
                    telegram_message_id=(
                        message.message_id
                    ),
                    response_payload=(
                        response_payload
                    ),
                    error_message=(
                        error_message
                    ),
                )
            except Exception as unknown_status_error:
                raise PublicationStateUncertainError(
                    publication_attempt_id=(
                        publication
                        .publication_attempt_id
                    ),
                    telegram_message_id=(
                        message.message_id
                    ),
                ) from unknown_status_error

            return PublicationResult(
                batch_id=(
                    publication.batch_id
                ),
                generated_post_id=(
                    publication
                    .generated_post_id
                ),
                publication_attempt_id=(
                    publication
                    .publication_attempt_id
                ),
                publication_date=(
                    publication
                    .publication_date
                ),
                edition=(
                    publication.edition
                ),
                telegram_message_id=(
                    message.message_id
                ),
                database_status="unknown",
                requires_review=True,
            )

        return PublicationResult(
            batch_id=(
                publication.batch_id
            ),
            generated_post_id=(
                publication.generated_post_id
            ),
            publication_attempt_id=(
                publication
                .publication_attempt_id
            ),
            publication_date=(
                publication.publication_date
            ),
            edition=publication.edition,
            telegram_message_id=(
                message.message_id
            ),
            database_status="published",
            requires_review=False,
        )

    finally:
        await _close_bot_session(
            bot
        )