from pathlib import Path

from aiogram.types import (
    FSInputFile,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputRichMessage,
    InputRichMessageMedia,
    Message,
)

from app.db.review_queue import ReviewDraftPreview
from app.publication.telegram_rich_message import (
    prepare_telegram_rich_message,
)


class ReviewImageUnavailableError(RuntimeError):
    """Изображение черновика отсутствует или недоступно."""


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


def build_review_rich_message(
    draft: ReviewDraftPreview,
) -> InputRichMessage:
    """
    Формирует Telegram Rich Message
    для ручного review.
    """

    image_path = _resolve_review_image_path(
        draft
    )

    prepared = prepare_telegram_rich_message(
        draft.post_text,
        text_format=draft.text_format,
    )

    media = InputRichMessageMedia(
        id=prepared.media_id,
        media=InputMediaPhoto(
            media=FSInputFile(
                image_path
            )
        ),
    )

    return InputRichMessage(
        html=prepared.html,
        media=[media],
    )


async def send_review_draft_preview(
    message: Message,
    *,
    draft: ReviewDraftPreview,
    reply_markup: InlineKeyboardMarkup,
) -> Message:
    """
    Отправляет пользователю одно review-сообщение:

    PNG + полный форматированный текст +
    существующая inline-клавиатура.
    """

    rich_message = build_review_rich_message(
        draft
    )

    return await message.bot.send_rich_message(
        chat_id=message.chat.id,
        rich_message=rich_message,
        reply_markup=reply_markup,
    )