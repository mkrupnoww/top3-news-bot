from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import html
import re
from typing import Literal
from urllib.parse import urlsplit
import xml.etree.ElementTree as ElementTree

from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException


FeedType = Literal["rss", "atom"]


_WHITESPACE_PATTERN = re.compile(r"\s+")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class ParsedFeedEntry:
    """Одна новость, извлечённая из RSS или Atom."""

    external_id: str | None
    title: str
    source_url: str
    summary: str | None
    author_name: str | None
    source_published_at: datetime | None
    primary_image_url: str | None
    categories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FeedParseResult:
    """Результат разбора одного XML-документа ленты."""

    feed_type: FeedType
    feed_title: str | None
    entries: tuple[ParsedFeedEntry, ...]
    skipped_count: int


def _local_name(tag: str) -> str:
    """Удаляет XML namespace из имени элемента."""

    return tag.rsplit("}", maxsplit=1)[-1].lower()


def _element_text(
    element: ElementTree.Element | None,
) -> str | None:
    """Извлекает и нормализует весь текст элемента."""

    if element is None:
        return None

    value = "".join(element.itertext()).strip()

    if not value:
        return None

    return _WHITESPACE_PATTERN.sub(
        " ",
        value,
    ).strip()


def _find_direct_child(
    element: ElementTree.Element,
    *local_names: str,
) -> ElementTree.Element | None:
    """Ищет первый непосредственный дочерний элемент."""

    expected_names = {
        name.lower()
        for name in local_names
    }

    for child in element:
        if _local_name(child.tag) in expected_names:
            return child

    return None


def _find_direct_text(
    element: ElementTree.Element,
    *local_names: str,
) -> str | None:
    """Возвращает текст первого подходящего элемента."""

    return _element_text(
        _find_direct_child(
            element,
            *local_names,
        )
    )


def _clean_summary(
    value: str | None,
) -> str | None:
    """Удаляет HTML-разметку и лишние пробелы."""

    if value is None:
        return None

    without_tags = _HTML_TAG_PATTERN.sub(
        " ",
        value,
    )

    normalized_value = _WHITESPACE_PATTERN.sub(
        " ",
        html.unescape(without_tags),
    ).strip()

    return normalized_value or None


def _normalize_category(
    value: str | None,
) -> str | None:
    """Нормализует одну категорию RSS или Atom."""

    if value is None:
        return None

    normalized_value = _WHITESPACE_PATTERN.sub(
        " ",
        html.unescape(value),
    ).strip()

    return normalized_value or None


def _extract_categories(
    entry: ElementTree.Element,
) -> tuple[str, ...]:
    """Извлекает категории RSS и Atom без дублей."""

    categories: list[str] = []
    seen_normalized: set[str] = set()

    for child in entry:
        if _local_name(child.tag) != "category":
            continue

        category = _normalize_category(
            child.attrib.get("term")
            or _element_text(child)
        )

        if category is None:
            continue

        duplicate_key = category.casefold()

        if duplicate_key in seen_normalized:
            continue

        seen_normalized.add(duplicate_key)
        categories.append(category)

    return tuple(categories)


def _is_http_url(value: str | None) -> bool:
    """Проверяет абсолютный HTTP или HTTPS URL."""

    if value is None:
        return False

    parsed = urlsplit(value.strip())

    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
    )


def _normalize_url(value: str) -> str:
    """Удаляет внешние пробелы из URL."""

    return value.strip()


def _parse_datetime(
    value: str | None,
) -> datetime | None:
    """Разбирает RFC 2822 или ISO 8601 дату."""

    if value is None:
        return None

    normalized_value = value.strip()

    if not normalized_value:
        return None

    parsed_value: datetime | None = None

    try:
        parsed_value = parsedate_to_datetime(
            normalized_value
        )
    except (TypeError, ValueError, OverflowError):
        parsed_value = None

    if parsed_value is None:
        iso_value = normalized_value

        if iso_value.endswith("Z"):
            iso_value = (
                iso_value[:-1]
                + "+00:00"
            )

        try:
            parsed_value = datetime.fromisoformat(
                iso_value
            )
        except ValueError:
            return None

    if (
        parsed_value.tzinfo is None
        or parsed_value.utcoffset() is None
    ):
        parsed_value = parsed_value.replace(
            tzinfo=timezone.utc
        )

    return parsed_value.astimezone(
        timezone.utc
    )


def _extract_rss_link(
    item: ElementTree.Element,
) -> str | None:
    """Извлекает ссылку из RSS item."""

    link = _find_direct_text(
        item,
        "link",
    )

    if _is_http_url(link):
        return _normalize_url(link)

    guid = _find_direct_text(
        item,
        "guid",
    )

    if _is_http_url(guid):
        return _normalize_url(guid)

    return None


def _extract_atom_link(
    entry: ElementTree.Element,
) -> str | None:
    """Извлекает основную ссылку из Atom entry."""

    fallback_url: str | None = None

    for child in entry:
        if _local_name(child.tag) != "link":
            continue

        href = child.attrib.get("href")

        if not _is_http_url(href):
            continue

        normalized_href = _normalize_url(
            href
        )

        relation = child.attrib.get(
            "rel",
            "alternate",
        ).lower()

        if relation == "alternate":
            return normalized_href

        if fallback_url is None:
            fallback_url = normalized_href

    return fallback_url


def _extract_author(
    entry: ElementTree.Element,
) -> str | None:
    """Извлекает автора RSS или Atom записи."""

    for child in entry:
        child_name = _local_name(child.tag)

        if child_name in {"creator", "author"}:
            nested_name = _find_direct_text(
                child,
                "name",
            )

            return (
                nested_name
                or _element_text(child)
            )

    return None


def _extract_image_url(
    entry: ElementTree.Element,
) -> str | None:
    """Извлекает основное изображение записи."""

    for child in entry:
        child_name = _local_name(child.tag)

        if child_name in {
            "content",
            "thumbnail",
            "enclosure",
        }:
            candidate_url = (
                child.attrib.get("url")
                or child.attrib.get("href")
            )

            if not _is_http_url(
                candidate_url
            ):
                continue

            media_type = child.attrib.get(
                "type",
                "",
            ).lower()

            medium = child.attrib.get(
                "medium",
                "",
            ).lower()

            relation = child.attrib.get(
                "rel",
                "",
            ).lower()

            if (
                child_name == "thumbnail"
                or medium == "image"
                or media_type.startswith("image/")
                or relation == "enclosure"
            ):
                return _normalize_url(
                    candidate_url
                )

    return None


def _extract_rss_entry(
    item: ElementTree.Element,
) -> ParsedFeedEntry | None:
    """Преобразует RSS item в унифицированную запись."""

    title = _find_direct_text(
        item,
        "title",
    )

    source_url = _extract_rss_link(
        item
    )

    if not title or source_url is None:
        return None

    summary = _clean_summary(
        _find_direct_text(
            item,
            "description",
            "encoded",
        )
    )

    published_at = _parse_datetime(
        _find_direct_text(
            item,
            "pubdate",
            "published",
            "date",
            "updated",
        )
    )

    return ParsedFeedEntry(
        external_id=_find_direct_text(
            item,
            "guid",
            "id",
        ),
        title=title,
        source_url=source_url,
        summary=summary,
        author_name=_extract_author(item),
        source_published_at=published_at,
        primary_image_url=(
            _extract_image_url(item)
        ),
        categories=_extract_categories(item),
    )


def _extract_atom_entry(
    entry: ElementTree.Element,
) -> ParsedFeedEntry | None:
    """Преобразует Atom entry в унифицированную запись."""

    title = _find_direct_text(
        entry,
        "title",
    )

    source_url = _extract_atom_link(
        entry
    )

    if not title or source_url is None:
        return None

    summary_element = (
        _find_direct_child(
            entry,
            "summary",
        )
        or _find_direct_child(
            entry,
            "content",
        )
    )

    summary = _clean_summary(
        _element_text(summary_element)
    )

    published_at = _parse_datetime(
        _find_direct_text(
            entry,
            "published",
            "updated",
        )
    )

    return ParsedFeedEntry(
        external_id=_find_direct_text(
            entry,
            "id",
        ),
        title=title,
        source_url=source_url,
        summary=summary,
        author_name=_extract_author(entry),
        source_published_at=published_at,
        primary_image_url=(
            _extract_image_url(entry)
        ),
        categories=_extract_categories(entry),
    )


def parse_feed_document(
    xml_content: bytes,
    *,
    max_entries: int = 100,
    max_document_bytes: int = 2_000_000,
) -> FeedParseResult:
    """
    Безопасно разбирает RSS 2.0 или Atom.

    Сеть, PostgreSQL и Telegram не используются.
    """

    if max_entries <= 0:
        raise ValueError(
            "max_entries должен быть больше нуля."
        )

    if not xml_content:
        raise ValueError(
            "XML-документ ленты пуст."
        )

    if len(xml_content) > max_document_bytes:
        raise ValueError(
            "XML-документ превышает допустимый размер: "
            f"size={len(xml_content)}, "
            f"limit={max_document_bytes}"
        )

    try:
        root = SafeElementTree.fromstring(
            xml_content
        )
    except (
        ElementTree.ParseError,
        DefusedXmlException,
    ) as error:
        raise ValueError(
            f"Некорректный или небезопасный XML: {error}"
        ) from error

    root_name = _local_name(root.tag)

    if root_name == "feed":
        feed_type: FeedType = "atom"
        container = root
        entry_elements = [
            element
            for element in root
            if _local_name(element.tag) == "entry"
        ]
        parser = _extract_atom_entry

    elif root_name in {"rss", "rdf"}:
        feed_type = "rss"

        channel = next(
            (
                element
                for element in root.iter()
                if _local_name(element.tag)
                == "channel"
            ),
            root,
        )

        container = channel
        entry_elements = [
            element
            for element in root.iter()
            if _local_name(element.tag) == "item"
        ]
        parser = _extract_rss_entry

    else:
        raise ValueError(
            "Неподдерживаемый формат XML-ленты: "
            f"root={root_name}"
        )

    feed_title = _find_direct_text(
        container,
        "title",
    )

    parsed_entries: list[ParsedFeedEntry] = []
    seen_urls: set[str] = set()
    skipped_count = 0

    for entry_element in entry_elements:
        if len(parsed_entries) >= max_entries:
            break

        parsed_entry = parser(
            entry_element
        )

        if parsed_entry is None:
            skipped_count += 1
            continue

        duplicate_key = (
            parsed_entry.source_url.rstrip("/")
        )

        if duplicate_key in seen_urls:
            skipped_count += 1
            continue

        seen_urls.add(duplicate_key)
        parsed_entries.append(parsed_entry)

    return FeedParseResult(
        feed_type=feed_type,
        feed_title=feed_title,
        entries=tuple(parsed_entries),
        skipped_count=skipped_count,
    )