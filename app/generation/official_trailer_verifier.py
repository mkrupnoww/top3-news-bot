import re
from dataclasses import dataclass

from app.generation.youtube_oembed import (
    YouTubeOEmbedMetadata,
)


_WHITESPACE_PATTERN = re.compile(r"\s+")
_WORD_PATTERN = re.compile(
    r"[a-zа-яё0-9]+",
    flags=re.IGNORECASE,
)

_TRAILER_MARKERS = (
    "trailer",
    "teaser",
    "трейлер",
    "тизер",
)

_OFFICIAL_MARKERS = (
    "official",
    "официаль",
)

_GENERIC_AUTHOR_TOKENS = {
    "official",
    "channel",
    "pictures",
    "picture",
    "films",
    "film",
    "movies",
    "movie",
    "studios",
    "studio",
    "entertainment",
    "productions",
    "production",
}


@dataclass(frozen=True, slots=True)
class OfficialTrailerVerification:
    """Результат консервативной проверки трейлера."""

    verified: bool
    official_trailer_url: str | None
    reason: str
    official_trailer_channel_name: str | None = None


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


def _author_is_confirmed_by_source(
    *,
    author_name: str,
    source_text: str,
) -> bool:
    """
    Проверяет связь YouTube-канала с исходной новостью.

    Полное совпадение остаётся предпочтительным, но
    ``Amazon MGM Studios`` также может подтверждаться
    упоминанием ``Amazon MGM`` в статье.
    """

    normalized_author = _normalize_text(
        author_name
    )

    if not normalized_author:
        return False

    if normalized_author in source_text:
        return True

    source_tokens = set(
        _WORD_PATTERN.findall(source_text)
    )

    author_tokens = [
        token
        for token in _WORD_PATTERN.findall(
            normalized_author
        )
        if (
            token not in _GENERIC_AUTHOR_TOKENS
            and (
                len(token) >= 4
                or any(
                    character.isdigit()
                    for character in token
                )
            )
        )
    ]

    if not author_tokens:
        return False

    return any(
        token in source_tokens
        for token in author_tokens
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
    - название YouTube-канала должно быть связано
      с исходной новостью хотя бы одним значимым
      токеном либо полным совпадением.

    Функция ничего не ищет в интернете и не выполняет
    сетевых запросов.
    """

    normalized_source_title = _normalize_text(
        source_title
    )
    normalized_source_summary = _normalize_text(
        source_summary
    )
    normalized_source_text = (
        normalized_source_title
        + " "
        + normalized_source_summary
    )

    normalized_video_title = _normalize_text(
        metadata.title
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

    if not _author_is_confirmed_by_source(
        author_name=metadata.author_name,
        source_text=normalized_source_text,
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
        official_trailer_channel_name=(
            metadata.author_name.strip()
        ),
    )
