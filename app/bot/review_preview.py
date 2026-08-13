from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aiogram import Bot
from aiogram.types import (
    FSInputFile,
    InlineKeyboardMarkup,
    Message,
)

from app.db.review_queue import ReviewDraftPreview
from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
)
from app.publication.telegram_text import (
    prepare_telegram_text,
)


class ReviewImageUnavailableError(RuntimeError):
    """Изображение черновика отсутствует или недоступно."""


@dataclass(frozen=True, slots=True)
class PreparedReviewPhoto:
    """Подготовленный native photo-preview для review."""

    image_path: Path
    caption: str
    parse_mode: str | None
    source_text_format: str


class ReviewPhotoSender(Protocol):
    """Минимальный интерфейс Telegram sender для review."""

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: FSInputFile,
        caption: str,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup,
    ) -> Message:
        """Отправляет native Telegram photo-message."""


def _resolve_review_image_path(
    draft: ReviewDraftPreview,
) -> Path:
    """Проверяет изображение, связанное с generated_post."""

    if draft.image_path is None:
        raise ReviewImageUnavailableError(
            "generated_posts.image_path не заполнен."
        )

    if draft.image_sha256 is None:
        raise ReviewImageUnavailableError(
            "generated_posts.image_sha256 не заполнен."
        )

    image_path = Path(
        draft.image_path
    ).expanduser().resolve()

    if not image_path.exists():
        raise ReviewImageUnavailableError(
            "PNG-файл черновика не найден: "
            f"{image_path}"
        )

    if not image_path.is_file():
        raise ReviewImageUnavailableError(
            "image_path не указывает на файл: "
            f"{image_path}"
        )

    if image_path.suffix.lower() != ".png":
        raise ReviewImageUnavailableError(
            "Для review-preview ожидается PNG: "
            f"{image_path}"
        )

    if image_path.stat().st_size <= 0:
        raise ReviewImageUnavailableError(
            "PNG-файл черновика пуст."
        )

    return image_path


def prepare_review_photo(
    draft: ReviewDraftPreview,
) -> PreparedReviewPhoto:
    """
    Подготавливает native Telegram photo-message:

    PNG + caption с форматированием проекта.
    """

    if len(draft.post_text) > MAXIMUM_POST_LENGTH:
        raise ValueError(
            "Текст review-preview превышает "
            "допустимую длину выпуска: "
            f"{MAXIMUM_POST_LENGTH} символов."
        )

    image_path = _resolve_review_image_path(
        draft
    )

    prepared_text = prepare_telegram_text(
        draft.post_text,
        text_format=draft.text_format,
    )

    return PreparedReviewPhoto(
        image_path=image_path,
        caption=prepared_text.text,
        parse_mode=prepared_text.parse_mode,
        source_text_format=(
            prepared_text.source_text_format
        ),
    )


async def send_review_draft_photo(
    bot: Bot | ReviewPhotoSender,
    *,
    chat_id: int,
    draft: ReviewDraftPreview,
    reply_markup: InlineKeyboardMarkup,
) -> Message:
    """
    Отправляет конкретный review draft в указанный Telegram chat.

    Используется как ручным /review, так и будущей
    автоматической review delivery.
    """

    if chat_id <= 0:
        raise ValueError(
            "chat_id должен быть больше нуля "
            "для личной review delivery."
        )

    prepared = prepare_review_photo(
        draft
    )

    return await bot.send_photo(
        chat_id=chat_id,
        photo=FSInputFile(
            prepared.image_path
        ),
        caption=prepared.caption,
        parse_mode=prepared.parse_mode,
        reply_markup=reply_markup,
    )


async def send_review_draft_preview(
    message: Message,
    *,
    draft: ReviewDraftPreview,
    reply_markup: InlineKeyboardMarkup,
) -> Message:
    """
    Совместимый wrapper для существующего /review.

    Отправляет native photo-message в тот же чат,
    из которого пользователь вызвал команду.
    """

    return await send_review_draft_photo(
        message.bot,
        chat_id=message.chat.id,
        draft=draft,
        reply_markup=reply_markup,
    )
