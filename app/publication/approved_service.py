from hashlib import sha256
import logging
from pathlib import Path
from typing import Any

import asyncpg
from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.types import FSInputFile

from app.db.approved_publications import (
    prepare_approved_publication,
)
from app.db.publications import (
    mark_publication_failed,
    mark_publication_published,
    mark_publication_unknown,
)
from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
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
) -> tuple[
    str,
    str,
    str,
    str,
]:
    """Читает текст, формат и изображение generated_post."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                post_text,
                text_format,
                image_path,
                image_sha256
            FROM generated_posts
            WHERE generated_post_id = $1
            """,
            generated_post_id,
        )

    if record is None:
        raise LookupError(
            "Пост не найден: "
            f"generated_post_id={generated_post_id}"
        )

    post_text = record["post_text"]
    text_format = record["text_format"]
    image_path = record["image_path"]
    image_sha256 = record["image_sha256"]

    if not isinstance(post_text, str):
        raise ValueError(
            "generated_posts.post_text должен быть строкой."
        )

    if not isinstance(text_format, str):
        raise ValueError(
            "generated_posts.text_format должен быть строкой."
        )

    if (
        not isinstance(image_path, str)
        or not image_path.strip()
    ):
        raise ValueError(
            "generated_posts.image_path не заполнен."
        )

    if (
        not isinstance(image_sha256, str)
        or not image_sha256.strip()
    ):
        raise ValueError(
            "generated_posts.image_sha256 не заполнен."
        )

    return (
        post_text,
        text_format,
        image_path,
        image_sha256,
    )


def _resolve_image_file(
    *,
    image_path: str,
    expected_sha256: str,
) -> Path:
    """Проверяет PNG и его SHA-256 перед publication_attempt."""

    resolved_path = Path(
        image_path
    ).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = (
            Path.cwd()
            / resolved_path
        )

    resolved_path = resolved_path.resolve()

    if not resolved_path.exists():
        raise ValueError(
            "PNG публикации не найден: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "image_path не указывает на файл: "
            f"{resolved_path}"
        )

    if resolved_path.suffix.lower() != ".png":
        raise ValueError(
            "Для публикации ожидается PNG: "
            f"{resolved_path}"
        )

    if resolved_path.stat().st_size <= 0:
        raise ValueError(
            "PNG публикации пуст."
        )

    digest = sha256()

    with resolved_path.open("rb") as image_file:
        while True:
            chunk = image_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    actual_sha256 = digest.hexdigest()

    if actual_sha256 != expected_sha256:
        raise ValueError(
            "SHA-256 PNG не совпадает "
            "с generated_posts.image_sha256: "
            f"expected={expected_sha256}, "
            f"actual={actual_sha256}"
        )

    return resolved_path


async def publish_approved_post(
    pool: asyncpg.Pool,
    *,
    bot_token: str,
    generated_post_id: int,
    disable_notification: bool = True,
) -> PublicationResult:
    """
    Публикует существующий одобренный generated_post.

    Пост отправляется одним native Telegram photo-message:
    PNG + полный форматированный caption.

    До создания publication_attempt проверяются длина текста,
    наличие PNG и фактический SHA-256.

    Новый batch и generated_post не создаются.
    После получения telegram_message_id повторная отправка запрещена.
    """

    (
        source_post_text,
        source_text_format,
        source_image_path,
        source_image_sha256,
    ) = await _load_source_post(
        pool,
        generated_post_id=generated_post_id,
    )

    if len(source_post_text) > MAXIMUM_POST_LENGTH:
        raise ValueError(
            "Текст публикации превышает "
            "допустимую длину выпуска: "
            f"{MAXIMUM_POST_LENGTH} символов."
        )

    resolved_image_file = _resolve_image_file(
        image_path=source_image_path,
        expected_sha256=source_image_sha256,
    )

    prepared_text = prepare_telegram_text(
        source_post_text,
        text_format=source_text_format,
    )

    prepared = await prepare_approved_publication(
        pool,
        generated_post_id=generated_post_id,
        disable_notification=disable_notification,
        caption=prepared_text.text,
        caption_text_format=prepared_text.text_format,
        parse_mode=prepared_text.parse_mode,
        image_path=source_image_path,
        image_sha256=source_image_sha256,
        source_post_text=source_post_text,
        source_text_format=source_text_format,
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
                    "Целевой Telegram chat "
                    "не является каналом: "
                    f"chat_type={chat.type}"
                )

            message = await bot.send_photo(
                chat_id=prepared.telegram_chat_id,
                photo=FSInputFile(
                    resolved_image_file
                ),
                caption=prepared.caption,
                parse_mode=prepared.parse_mode,
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
                    "publication_attempt_id="
                    f"{publication.publication_attempt_id}"
                ) from database_error

            raise

        response_payload = {
            "message_id": message.message_id,
            "chat_id": message.chat.id,
            "chat_type": _enum_value(
                message.chat.type
            ),
            "chat_title": message.chat.title,
            "message_date": (
                message.date.isoformat()
            ),
            "transport": "native_photo",
            "caption_characters": len(
                prepared.caption
            ),
            "caption_text_format": (
                prepared.caption_text_format
            ),
            "parse_mode": prepared.parse_mode,
            "image_path": prepared.image_path,
            "image_sha256": (
                prepared.image_sha256
            ),
            "source_text_format": (
                prepared.source_text_format
            ),
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