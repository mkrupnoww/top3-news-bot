from dataclasses import dataclass
from html import escape
import re

from app.publication.telegram_text import (
    convert_project_markdown_to_html,
)


RICH_MESSAGE_MEDIA_ID = "top3_image"
RICH_MESSAGE_MAX_CHARACTERS = 32768

_SEPARATOR_PATTERN = re.compile(
    r"^(?:-{3,}|_{3,})$"
)


@dataclass(frozen=True, slots=True)
class PreparedTelegramRichMessage:
    """Подготовленное содержимое Telegram Rich Message."""

    html: str
    media_id: str
    source_text_format: str


def _normalize_text_format(
    text_format: str,
) -> str:
    """Нормализует формат исходного текста."""

    if not isinstance(text_format, str):
        raise TypeError(
            "text_format должен быть строкой."
        )

    normalized_format = (
        text_format.strip().lower()
    )

    if not normalized_format:
        raise ValueError(
            "text_format не может быть пустым."
        )

    return normalized_format


def _prepare_inline_html(
    post_text: str,
    *,
    text_format: str,
) -> str:
    """
    Преобразует исходный текст в безопасный HTML.

    На текущем production-пути generated_posts
    используют внутренний Markdown проекта.
    """

    if not isinstance(post_text, str):
        raise TypeError(
            "post_text должен быть строкой."
        )

    if not post_text.strip():
        raise ValueError(
            "post_text не может быть пустым."
        )

    normalized_format = _normalize_text_format(
        text_format
    )

    if normalized_format == "markdown":
        return convert_project_markdown_to_html(
            post_text
        )

    if normalized_format == "plain_text":
        return escape(
            post_text,
            quote=False,
        )

    if normalized_format == "html":
        return post_text

    raise ValueError(
        "Telegram Rich Message пока поддерживает "
        "только markdown, plain_text и html: "
        f"text_format={text_format}"
    )


def _is_separator(
    value: str,
) -> bool:
    """Проверяет строку-разделитель выпуска."""

    return bool(
        _SEPARATOR_PATTERN.fullmatch(
            value.strip()
        )
    )


def _flush_paragraph(
    lines: list[str],
    blocks: list[str],
) -> None:
    """Добавляет накопленный текст как HTML paragraph."""

    if not lines:
        return

    paragraph_html = "<br/>".join(
        lines
    )

    blocks.append(
        f"<p>{paragraph_html}</p>"
    )

    lines.clear()


def _build_structured_html(
    inline_html: str,
) -> str:
    """
    Превращает обычные переводы строк проекта
    в структурированные Rich Message blocks.
    """

    blocks: list[str] = [
        (
            '<img src="tg://photo?id='
            f'{RICH_MESSAGE_MEDIA_ID}"/>'
        )
    ]

    paragraph_lines: list[str] = []

    for raw_line in inline_html.splitlines():
        line = raw_line.strip()

        if not line:
            _flush_paragraph(
                paragraph_lines,
                blocks,
            )
            continue

        if _is_separator(line):
            _flush_paragraph(
                paragraph_lines,
                blocks,
            )

            blocks.append("<hr/>")
            continue

        paragraph_lines.append(line)

    _flush_paragraph(
        paragraph_lines,
        blocks,
    )

    return "\n".join(blocks)


def prepare_telegram_rich_message(
    post_text: str,
    *,
    text_format: str,
) -> PreparedTelegramRichMessage:
    """Готовит текст поста для Telegram Rich Message."""

    normalized_format = _normalize_text_format(
        text_format
    )

    inline_html = _prepare_inline_html(
        post_text,
        text_format=normalized_format,
    )

    rich_html = _build_structured_html(
        inline_html
    )

    if len(rich_html) > RICH_MESSAGE_MAX_CHARACTERS:
        raise ValueError(
            "Telegram Rich Message превышает "
            f"лимит {RICH_MESSAGE_MAX_CHARACTERS}: "
            f"characters={len(rich_html)}"
        )

    return PreparedTelegramRichMessage(
        html=rich_html,
        media_id=RICH_MESSAGE_MEDIA_ID,
        source_text_format=normalized_format,
    )