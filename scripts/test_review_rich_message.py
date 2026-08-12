from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from app.bot.review_preview import (
    ReviewImageUnavailableError,
    build_review_rich_message,
)
from app.db.review_queue import (
    ReviewDraftPreview,
)


TEST_SHA256 = (
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
)


def build_test_draft(
    *,
    image_path: str | None,
    image_sha256: str | None,
) -> ReviewDraftPreview:
    """Создаёт тестовый ReviewDraftPreview."""

    return ReviewDraftPreview(
        batch_id=1,
        generated_post_id=1,
        publication_date=date(
            2026,
            8,
            12,
        ),
        edition=1,
        version_number=1,
        post_text=(
            "**TOP-3 НОВОСТЕЙ КИНО "
            "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n\n"
            "---\n\n"
            "1️⃣ **Первая новость**\n\n"
            "Описание первой новости.\n\n"
            "2️⃣ **Вторая новость**\n\n"
            "Описание второй новости.\n\n"
            "3️⃣ **Третья новость**\n\n"
            "Описание третьей новости."
        ),
        text_format="markdown",
        image_path=image_path,
        image_sha256=image_sha256,
    )


def main() -> int:
    """
    Проверяет построение review Rich Message.

    Telegram и PostgreSQL не используются.
    """

    with TemporaryDirectory() as directory:
        image_path = (
            Path(directory)
            / "review_test.png"
        )

        image = Image.new(
            "RGB",
            (12, 18),
        )

        image.save(
            image_path,
            format="PNG",
        )

        draft = build_test_draft(
            image_path=str(image_path),
            image_sha256=TEST_SHA256,
        )

        rich_message = (
            build_review_rich_message(
                draft
            )
        )

        if rich_message.html is None:
            raise RuntimeError(
                "Rich Message не содержит HTML."
            )

        if (
            '<img src="tg://photo?id=top3_image"/>'
            not in rich_message.html
        ):
            raise RuntimeError(
                "Rich Message не содержит "
                "ссылку на media."
            )

        if (
            "<p><b>TOP-3 НОВОСТЕЙ КИНО "
            "ЗА ПОСЛЕДНИЕ 24 ЧАСА</b></p>"
            not in rich_message.html
        ):
            raise RuntimeError(
                "Заголовок Rich Message "
                "сформирован неверно."
            )

        if "<hr/>" not in rich_message.html:
            raise RuntimeError(
                "Разделитель Rich Message "
                "не сформирован."
            )

        if not rich_message.media:
            raise RuntimeError(
                "Rich Message не содержит media."
            )

        if len(rich_message.media) != 1:
            raise RuntimeError(
                "Review должен содержать "
                "ровно один media-элемент."
            )

        media_item = rich_message.media[0]

        if media_item.id != "top3_image":
            raise RuntimeError(
                "Некорректный media id: "
                f"{media_item.id!r}"
            )

        print(
            "Review Rich Message build test: OK"
        )
        print(
            "media_count="
            f"{len(rich_message.media)}"
        )
        print(
            "rich_html_characters="
            f"{len(rich_message.html)}"
        )

    missing_image_blocked = False

    try:
        build_review_rich_message(
            build_test_draft(
                image_path=None,
                image_sha256=None,
            )
        )
    except ReviewImageUnavailableError:
        missing_image_blocked = True

    if not missing_image_blocked:
        raise RuntimeError(
            "Review без изображения "
            "должен блокироваться."
        )

    print(
        "missing_image_blocked=true"
    )
    print(
        "Network requests: not performed"
    )
    print(
        "Database changes: not performed"
    )
    print(
        "Telegram requests: not performed"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())