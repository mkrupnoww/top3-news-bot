import asyncio
from datetime import date
from pathlib import Path
import tempfile
from typing import Any

from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from app.bot.review_preview import (
    ReviewImageUnavailableError,
    send_review_draft_photo,
)
from app.db.review_queue import ReviewDraftPreview


class FakeTelegramBot:
    """Fake Telegram sender без сетевых запросов."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_call: dict[str, Any] | None = None

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: FSInputFile,
        caption: str,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup,
    ) -> object:
        self.call_count += 1

        self.last_call = {
            "chat_id": chat_id,
            "photo": photo,
            "caption": caption,
            "parse_mode": parse_mode,
            "reply_markup": reply_markup,
        }

        return object()


def build_keyboard(
    generated_post_id: int,
) -> InlineKeyboardMarkup:
    """Создаёт минимальную review-клавиатуру для теста."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Одобрить",
                    callback_data=(
                        "review:approve:"
                        f"{generated_post_id}"
                    ),
                )
            ]
        ]
    )


def build_draft(
    *,
    image_path: str | None,
    image_sha256: str | None,
) -> ReviewDraftPreview:
    """Создаёт синтетический review draft."""

    return ReviewDraftPreview(
        batch_id=1001,
        generated_post_id=2001,
        publication_date=date(
            2026,
            8,
            13,
        ),
        edition=1,
        version_number=1,
        post_text=(
            "**TOP-3 НОВОСТЕЙ КИНО**\n\n"
            "1️⃣ **Тестовая новость**\n\n"
            "Тестовый текст."
        ),
        text_format="markdown",
        image_path=image_path,
        image_sha256=image_sha256,
    )


async def main() -> int:
    """Проверяет native review sender без Telegram."""

    print("Review photo sender test")
    print("Telegram requests=not_performed")
    print("OpenAI requests=not_performed")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = (
            Path(temp_dir)
            / "review_test.png"
        )

        image_path.write_bytes(
            b"\x89PNG\r\n\x1a\nsynthetic-test-data"
        )

        draft = build_draft(
            image_path=str(image_path),
            image_sha256="synthetic-sha256",
        )

        bot = FakeTelegramBot()

        keyboard = build_keyboard(
            draft.generated_post_id
        )

        await send_review_draft_photo(
            bot,
            chat_id=200214441,
            draft=draft,
            reply_markup=keyboard,
        )

        assert bot.call_count == 1
        assert bot.last_call is not None

        assert (
            bot.last_call["chat_id"]
            == 200214441
        )

        assert isinstance(
            bot.last_call["photo"],
            FSInputFile,
        )

        assert (
            bot.last_call["caption"]
            == (
                "<b>TOP-3 НОВОСТЕЙ КИНО</b>"
                "\n\n"
                "1️⃣ <b>Тестовая новость</b>"
                "\n\n"
                "Тестовый текст."
            )
        )

        assert (
            bot.last_call["parse_mode"]
            == "HTML"
        )

        assert (
            bot.last_call["reply_markup"]
            is keyboard
        )

        print("Native send_photo call: OK")
        print("Chat ID forwarding: OK")
        print("PNG forwarding: OK")
        print("Caption formatting: OK")
        print("Review keyboard forwarding: OK")

        missing_image_draft = build_draft(
            image_path=None,
            image_sha256=None,
        )

        try:
            await send_review_draft_photo(
                bot,
                chat_id=200214441,
                draft=missing_image_draft,
                reply_markup=keyboard,
            )
        except ReviewImageUnavailableError:
            pass
        else:
            raise AssertionError(
                "Отсутствующее изображение "
                "не было заблокировано."
            )

        assert bot.call_count == 1

        print("Missing image blocking: OK")

        try:
            await send_review_draft_photo(
                bot,
                chat_id=0,
                draft=draft,
                reply_markup=keyboard,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Некорректный chat_id "
                "не был заблокирован."
            )

        assert bot.call_count == 1

        print("Invalid chat ID blocking: OK")

    print()
    print("Telegram requests=not_performed")
    print("OpenAI requests=not_performed")
    print("Review photo sender test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
