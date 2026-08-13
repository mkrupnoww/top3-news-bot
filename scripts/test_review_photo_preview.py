from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from app.bot.review_preview import (
    ReviewImageUnavailableError,
    prepare_review_photo,
)
from app.db.review_queue import (
    ReviewDraftPreview,
)
from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
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
    post_text: str | None = None,
) -> ReviewDraftPreview:
    """Создаёт тестовый ReviewDraftPreview."""

    return ReviewDraftPreview(
        batch_id=1,
        generated_post_id=1,
        publication_date=date(
            2026,
            8,
            13,
        ),
        edition=1,
        version_number=1,
        post_text=(
            post_text
            if post_text is not None
            else (
                "**TOP-3 НОВОСТЕЙ КИНО "
                "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n\n"
                "_______________\n\n"
                "1️⃣ **Первая новость**\n\n"
                "Описание первой новости.\n\n"
                "2️⃣ **Вторая новость**\n\n"
                "Описание второй новости.\n\n"
                "3️⃣ **Третья новость**\n\n"
                "Описание третьей новости."
            )
        ),
        text_format="markdown",
        image_path=image_path,
        image_sha256=image_sha256,
    )


def main() -> int:
    """
    Проверяет подготовку native photo-preview.

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

        prepared = prepare_review_photo(
            draft
        )

        if prepared.image_path != image_path.resolve():
            raise RuntimeError(
                "Некорректный путь native photo."
            )

        if prepared.parse_mode != "HTML":
            raise RuntimeError(
                "Markdown review должен "
                "преобразовываться в HTML."
            )

        if prepared.source_text_format != "markdown":
            raise RuntimeError(
                "Исходный формат review потерян."
            )

        if (
            "<b>TOP-3 НОВОСТЕЙ КИНО "
            "ЗА ПОСЛЕДНИЕ 24 ЧАСА</b>"
            not in prepared.caption
        ):
            raise RuntimeError(
                "Заголовок caption "
                "сформирован неверно."
            )

        if (
            "1️⃣ <b>Первая новость</b>"
            not in prepared.caption
        ):
            raise RuntimeError(
                "Жирный заголовок новости "
                "сформирован неверно."
            )

        if "_______________" not in prepared.caption:
            raise RuntimeError(
                "Разделитель выпуска потерян."
            )

        print(
            "Review native photo preparation: OK"
        )
        print(
            "parse_mode="
            f"{prepared.parse_mode}"
        )
        print(
            "caption_characters="
            f"{len(prepared.caption)}"
        )
        print(
            "source_text_characters="
            f"{len(draft.post_text)}"
        )

    missing_image_blocked = False

    try:
        prepare_review_photo(
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

    oversized_blocked = False

    try:
        prepare_review_photo(
            build_test_draft(
                image_path="/tmp/not-used.png",
                image_sha256=TEST_SHA256,
                post_text=(
                    "X"
                    * (
                        MAXIMUM_POST_LENGTH
                        + 1
                    )
                ),
            )
        )
    except ValueError as error:
        if (
            str(MAXIMUM_POST_LENGTH)
            in str(error)
        ):
            oversized_blocked = True

    if not oversized_blocked:
        raise RuntimeError(
            "Review длиннее лимита "
            "должен блокироваться "
            "до проверки PNG."
        )

    print("missing_image_blocked=true")
    print("oversized_caption_blocked=true")
    print(
        "maximum_post_length="
        f"{MAXIMUM_POST_LENGTH}"
    )
    print("Network requests: not performed")
    print("Database changes: not performed")
    print("Telegram requests: not performed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())