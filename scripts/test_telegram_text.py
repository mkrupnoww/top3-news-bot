from app.publication.telegram_text import (
    convert_project_markdown_to_html,
    prepare_telegram_text,
)


def test_markdown_conversion() -> None:
    """Проверяет основной внутренний Markdown."""

    source_text = (
        "**TOP-3 киноновости дня**\n\n"
        "**1. Sony & Crunchyroll**\n"
        "Выручка <снизилась>, но направление выросло.\n\n"
        "__Какую новость обсудим?__"
    )

    expected_text = (
        "<b>TOP-3 киноновости дня</b>\n\n"
        "<b>1. Sony &amp; Crunchyroll</b>\n"
        "Выручка &lt;снизилась&gt;, "
        "но направление выросло.\n\n"
        "<i>Какую новость обсудим?</i>"
    )

    converted_text = (
        convert_project_markdown_to_html(
            source_text
        )
    )

    assert converted_text == expected_text
    assert "**" not in converted_text
    assert "__" not in converted_text

    prepared = prepare_telegram_text(
        source_text,
        text_format="markdown",
    )

    assert prepared.text == expected_text
    assert prepared.text_format == "html"
    assert prepared.parse_mode == "HTML"
    assert (
        prepared.source_text_format
        == "markdown"
    )

    print("Project Markdown conversion: OK")


def test_plain_text() -> None:
    """Проверяет обычный текст без parse_mode."""

    source_text = (
        "Обычный текст: 2 < 3 & 5 > 4"
    )

    prepared = prepare_telegram_text(
        source_text,
        text_format="plain_text",
    )

    assert prepared.text == source_text
    assert prepared.text_format == "plain_text"
    assert prepared.parse_mode is None
    assert (
        prepared.source_text_format
        == "plain_text"
    )

    print("Plain text preparation: OK")


def test_existing_html() -> None:
    """Проверяет уже подготовленный HTML."""

    source_text = (
        "<b>Готовый HTML</b>"
    )

    prepared = prepare_telegram_text(
        source_text,
        text_format="html",
    )

    assert prepared.text == source_text
    assert prepared.text_format == "html"
    assert prepared.parse_mode == "HTML"
    assert (
        prepared.source_text_format
        == "html"
    )

    print("Existing HTML preparation: OK")


def test_markdown_v2() -> None:
    """Проверяет явный MarkdownV2."""

    source_text = (
        r"\*Тест MarkdownV2\*"
    )

    prepared = prepare_telegram_text(
        source_text,
        text_format="markdown_v2",
    )

    assert prepared.text == source_text
    assert (
        prepared.text_format
        == "markdown_v2"
    )
    assert (
        prepared.parse_mode
        == "MarkdownV2"
    )
    assert (
        prepared.source_text_format
        == "markdown_v2"
    )

    print("MarkdownV2 preparation: OK")


def test_unpaired_bold_marker() -> None:
    """Блокирует непарный жирный маркер."""

    try:
        convert_project_markdown_to_html(
            "**Незакрытый жирный текст"
        )
    except ValueError as error:
        assert (
            "непарный маркер жирного"
            in str(error)
        )
    else:
        raise AssertionError(
            "Непарный маркер ** "
            "не был заблокирован."
        )

    print("Unpaired bold marker: blocked")


def test_unpaired_italic_marker() -> None:
    """Блокирует непарный курсивный маркер."""

    try:
        convert_project_markdown_to_html(
            "__Незакрытый курсив"
        )
    except ValueError as error:
        assert (
            "непарный маркер курсива"
            in str(error)
        )
    else:
        raise AssertionError(
            "Непарный маркер __ "
            "не был заблокирован."
        )

    print("Unpaired italic marker: blocked")


def test_unsupported_format() -> None:
    """Блокирует неизвестный формат."""

    try:
        prepare_telegram_text(
            "Тест",
            text_format="unknown",
        )
    except ValueError as error:
        assert (
            "Неподдерживаемый формат"
            in str(error)
        )
    else:
        raise AssertionError(
            "Неизвестный формат "
            "не был заблокирован."
        )

    print("Unsupported format: blocked")


def test_empty_text() -> None:
    """Блокирует пустую публикацию."""

    try:
        prepare_telegram_text(
            "   ",
            text_format="markdown",
        )
    except ValueError as error:
        assert (
            "не может быть пустым"
            in str(error)
        )
    else:
        raise AssertionError(
            "Пустой текст не был заблокирован."
        )

    print("Empty text: blocked")


def main() -> int:
    """Запускает изолированный тест."""

    test_markdown_conversion()
    test_plain_text()
    test_existing_html()
    test_markdown_v2()
    test_unpaired_bold_marker()
    test_unpaired_italic_marker()
    test_unsupported_format()
    test_empty_text()

    print()
    print("PostgreSQL connections: not performed")
    print("Telegram requests: not performed")
    print("Telegram text preparation test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )