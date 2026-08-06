from dataclasses import dataclass
from html import escape
import re


PROJECT_BOLD_PATTERN = re.compile(
    r"\*\*(?=\S)(.+?)(?<=\S)\*\*"
)

PROJECT_ITALIC_PATTERN = re.compile(
    r"__(?=\S)(.+?)(?<=\S)__"
)

PROJECT_SEPARATOR_LINE_PATTERN = re.compile(
    r"^_{3,}$"
)


@dataclass(frozen=True, slots=True)
class PreparedTelegramText:
    """Текст, подготовленный для Telegram Bot API."""

    text: str
    text_format: str
    parse_mode: str | None
    source_text_format: str


def _validate_post_text(
    post_text: str,
) -> str:
    """Проверяет исходный текст публикации."""

    if not isinstance(post_text, str):
        raise TypeError(
            "post_text должен быть строкой."
        )

    if not post_text.strip():
        raise ValueError(
            "post_text не может быть пустым."
        )

    return post_text


def _is_separator_line(
    line: str,
) -> bool:
    """
    Проверяет строку-разделитель из подчёркиваний.

    Такая строка является обычным текстом проекта,
    а не Markdown-разметкой курсива.
    """

    return bool(
        PROJECT_SEPARATOR_LINE_PATTERN.fullmatch(
            line.strip()
        )
    )


def _markdown_without_separator_lines(
    post_text: str,
) -> str:
    """
    Исключает строки-разделители при проверке
    парности Markdown-маркеров.
    """

    return "\n".join(
        ""
        if _is_separator_line(line)
        else line
        for line in post_text.splitlines()
    )


def _validate_project_markdown(
    post_text: str,
) -> None:
    """Проверяет парность разрешённых маркеров."""

    markdown_text = (
        _markdown_without_separator_lines(
            post_text
        )
    )

    bold_marker_count = markdown_text.count(
        "**"
    )

    italic_marker_count = markdown_text.count(
        "__"
    )

    if bold_marker_count % 2 != 0:
        raise ValueError(
            "В тексте обнаружен непарный "
            "маркер жирного текста **."
        )

    if italic_marker_count % 2 != 0:
        raise ValueError(
            "В тексте обнаружен непарный "
            "маркер курсива __."
        )


def _split_line_ending(
    value: str,
) -> tuple[str, str]:
    """Отделяет содержимое строки от перевода строки."""

    if value.endswith("\r\n"):
        return value[:-2], "\r\n"

    if value.endswith("\n"):
        return value[:-1], "\n"

    if value.endswith("\r"):
        return value[:-1], "\r"

    return value, ""


def _convert_markdown_line_to_html(
    line: str,
) -> str:
    """
    Преобразует одну обычную строку проекта
    из Markdown в безопасный HTML.
    """

    escaped_line = escape(
        line,
        quote=False,
    )

    converted_line = (
        PROJECT_BOLD_PATTERN.sub(
            r"<b>\1</b>",
            escaped_line,
        )
    )

    converted_line = (
        PROJECT_ITALIC_PATTERN.sub(
            r"<i>\1</i>",
            converted_line,
        )
    )

    if "**" in converted_line:
        raise ValueError(
            "Не удалось однозначно преобразовать "
            "маркер жирного текста **."
        )

    if "__" in converted_line:
        raise ValueError(
            "Не удалось однозначно преобразовать "
            "маркер курсива __."
        )

    return converted_line


def convert_project_markdown_to_html(
    post_text: str,
) -> str:
    """
    Преобразует внутренний Markdown проекта в HTML.

    Поддерживаются только:

    **жирный текст**
    __курсив__

    Строки, состоящие минимум из трёх символов
    подчёркивания, считаются текстовыми
    разделителями и сохраняются без изменений.

    Весь исходный HTML предварительно экранируется.
    """

    validated_text = _validate_post_text(
        post_text
    )

    _validate_project_markdown(
        validated_text
    )

    converted_lines: list[str] = []

    for raw_line in validated_text.splitlines(
        keepends=True
    ):
        line, line_ending = _split_line_ending(
            raw_line
        )

        if _is_separator_line(line):
            converted_line = escape(
                line,
                quote=False,
            )
        else:
            converted_line = (
                _convert_markdown_line_to_html(
                    line
                )
            )

        converted_lines.append(
            converted_line + line_ending
        )

    return "".join(converted_lines)


def prepare_telegram_text(
    post_text: str,
    *,
    text_format: str,
) -> PreparedTelegramText:
    """Подготавливает текст и parse_mode для Telegram."""

    validated_text = _validate_post_text(
        post_text
    )

    normalized_format = (
        text_format.strip().lower()
    )

    if normalized_format == "plain_text":
        return PreparedTelegramText(
            text=validated_text,
            text_format="plain_text",
            parse_mode=None,
            source_text_format=(
                normalized_format
            ),
        )

    if normalized_format == "markdown":
        return PreparedTelegramText(
            text=(
                convert_project_markdown_to_html(
                    validated_text
                )
            ),
            text_format="html",
            parse_mode="HTML",
            source_text_format=(
                normalized_format
            ),
        )

    if normalized_format == "markdown_v2":
        return PreparedTelegramText(
            text=validated_text,
            text_format="markdown_v2",
            parse_mode="MarkdownV2",
            source_text_format=(
                normalized_format
            ),
        )

    if normalized_format == "html":
        return PreparedTelegramText(
            text=validated_text,
            text_format="html",
            parse_mode="HTML",
            source_text_format=(
                normalized_format
            ),
        )

    raise ValueError(
        "Неподдерживаемый формат текста: "
        f"text_format={text_format}"
    )