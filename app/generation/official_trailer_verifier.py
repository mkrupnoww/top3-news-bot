import re
from dataclasses import dataclass

from app.generation.youtube_oembed import (
    YouTubeOEmbedMetadata,
)


_WHITESPACE_PATTERN = re.compile(r"\s+")

_TRAILER_MARKERS = (
    "trailer",
    "teaser",
)

_OFFICIAL_MARKERS = (
    "official",
)


@dataclass(frozen=True, slots=True)
class OfficialTrailerVerification:
    """Результат консервативной проверки трейлера."""

    verified: bool
    official_trailer_url: str | None
    reason: str


def _normalize_text(value: str) -> str:
    """Нормализует текст для сравнений."""

    if not isinstance(value, str):
        raise TypeError(
            "value должен быть строкой."
        )

    return _WHITESPACE_PATTERN.sub(
        " ",
        value.casefold(),
    ).strip()


def _contains_any(
    text: str,
    markers: tuple[str, ...],
) -> bool:
    """Проверяет наличие хотя бы одного маркера."""

    return any(
        marker in text
        for marker in markers
    )


def verify_official_trailer(
    metadata: YouTubeOEmbedMetadata,
    *,
    source_title: str,
    source_summary: str,
) -> OfficialTrailerVerification:
    """
    Проверяет уже найденный YouTube-ролик.

    Проверка намеренно консервативна:
    - исходная новость должна быть о trailer/teaser;
    - YouTube title должен быть о trailer/teaser;
    - хотя бы источник или YouTube title должны
      явно содержать marker official;
    - имя YouTube-канала должно присутствовать
      в title/summary исходной новости.

    Функция ничего не ищет в интернете и не выполняет
    сетевых запросов.
    """

    normalized_source_title = (
        _normalize_text(source_title)
    )

    normalized_source_summary = (
        _normalize_text(source_summary)
    )

    normalized_source_text = (
        normalized_source_title
        + " "
        + normalized_source_summary
    )

    normalized_video_title = (
        _normalize_text(metadata.title)
    )

    normalized_author_name = (
        _normalize_text(metadata.author_name)
    )

    if not _contains_any(
        normalized_source_text,
        _TRAILER_MARKERS,
    ):
        return OfficialTrailerVerification(
            verified=False,
            official_trailer_url=None,
            reason="source_is_not_trailer_news",
        )

    if not _contains_any(
        normalized_video_title,
        _TRAILER_MARKERS,
    ):
        return OfficialTrailerVerification(
            verified=False,
            official_trailer_url=None,
            reason="youtube_title_is_not_trailer",
        )

    if not (
        _contains_any(
            normalized_source_text,
            _OFFICIAL_MARKERS,
        )
        or _contains_any(
            normalized_video_title,
            _OFFICIAL_MARKERS,
        )
    ):
        return OfficialTrailerVerification(
            verified=False,
            official_trailer_url=None,
            reason="official_marker_not_confirmed",
        )

    if (
        not normalized_author_name
        or normalized_author_name
        not in normalized_source_text
    ):
        return OfficialTrailerVerification(
            verified=False,
            official_trailer_url=None,
            reason=(
                "youtube_author_not_confirmed_"
                "by_source"
            ),
        )

    return OfficialTrailerVerification(
        verified=True,
        official_trailer_url=(
            metadata.canonical_url
        ),
        reason="verified_official_trailer",
    )