from app.publication.telegram_rich_message import (
    RICH_MESSAGE_MEDIA_ID,
    prepare_telegram_rich_message,
)


def main() -> int:
    """Проверяет Rich Message formatter без сети."""

    post_text = """**TOP-3 НОВОСТЕЙ КИНО ЗА ПОСЛЕДНИЕ 24 ЧАСА**

---

1️⃣ **Первая новость**

Первый абзац первой новости.

Второй абзац первой новости.

2️⃣ **Вторая новость**

Описание второй новости.

3️⃣ **Третья новость**

Описание третьей новости.

……………
Подписаться на VIP канал - @kkm_vip_bot"""

    result = prepare_telegram_rich_message(
        post_text,
        text_format="markdown",
    )

    expected_image = (
        '<img src="tg://photo?id='
        f'{RICH_MESSAGE_MEDIA_ID}"/>'
    )

    assert result.html.startswith(
        expected_image
    )

    assert (
        "<p><b>TOP-3 НОВОСТЕЙ КИНО "
        "ЗА ПОСЛЕДНИЕ 24 ЧАСА</b></p>"
        in result.html
    )

    assert "<hr/>" in result.html

    assert (
        "<p>1️⃣ <b>Первая новость</b></p>"
        in result.html
    )

    assert (
        "<p>Первый абзац первой новости.</p>"
        in result.html
    )

    assert (
        "<p>Второй абзац первой новости.</p>"
        in result.html
    )

    assert (
        "<p>2️⃣ <b>Вторая новость</b></p>"
        in result.html
    )

    assert (
        "<p>3️⃣ <b>Третья новость</b></p>"
        in result.html
    )

    assert result.media_id == "top3_image"

    assert (
        result.source_text_format
        == "markdown"
    )

    print("Telegram Rich Message formatter: OK")
    print(
        f"rich_html_characters="
        f"{len(result.html)}"
    )
    print(
        f"media_id={result.media_id}"
    )
    print()
    print(result.html)
    print()
    print("Network requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())