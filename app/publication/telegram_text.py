from dataclasses import dataclass
from html import escape
import re


PROJECT_BOLD_PATTERN = re.compile(
    r"\*\*(?=\S)(.+?)(?<=\S)\*\*"
)

PROJECT_ITALIC_PATTERN = re.compile(
    r"__(?=\S)(.+?)(?<=\S)__"
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


def _validate_project_markdown(
    post_text: str,
) -> None:
    """Проверяет парность разрешённых маркеров."""

    bold_marker_count = post_text.count(
        "**"
    )

    italic_marker_count = post_text.count(
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


def convert_project_markdown_to_html(
    post_text: str,
) -> str:
    """
    Преобразует внутренний Markdown проекта в HTML.

    Поддерживаются только:

    **жирный текст**
    __курсив__

    Весь исходный HTML предварительно экранируется.
    """

    validated_text = _validate_post_text(
        post_text
    )

    _validate_project_markdown(
        validated_text
    )

    escaped_text = escape(
        validated_text,
        quote=False,
    )

    converted_text = (
        PROJECT_BOLD_PATTERN.sub(
            r"<b>\1</b>",
            escaped_text,
        )
    )

    converted_text = (
        PROJECT_ITALIC_PATTERN.sub(
            r"<i>\1</i>",
            converted_text,
        )
    )

    if "**" in converted_text:
        raise ValueError(
            "Не удалось однозначно преобразовать "
            "маркер жирного текста **."
        )

    if "__" in converted_text:
        raise ValueError(
            "Не удалось однозначно преобразовать "
            "маркер курсива __."
        )

    return converted_text


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