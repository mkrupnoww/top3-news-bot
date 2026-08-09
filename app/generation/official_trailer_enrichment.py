from dataclasses import dataclass

import httpx

from app.collectors.article_http import (
    AddressResolver,
    ArticleDownloadError,
    download_article_document,
)
from app.collectors.feed_http import (
    resolve_host_addresses,
)
from app.generation.official_trailer_verifier import (
    verify_official_trailer,
)
from app.generation.trailer_extractor import (
    extract_youtube_iframe_urls,
)
from app.generation.youtube_oembed import (
    YouTubeOEmbedError,
    fetch_youtube_oembed_metadata,
)


_TRAILER_MARKERS = (
    "trailer",
    "teaser",
)


@dataclass(frozen=True, slots=True)
class OfficialTrailerEnrichmentResult:
    """Результат enrichment официального трейлера."""

    attempted: bool
    verified: bool
    official_trailer_url: str | None
    reason: str
    article_final_url: str | None
    youtube_candidate_urls: tuple[str, ...]
    checked_video_urls: tuple[str, ...]
    verification_reasons: tuple[str, ...]
    oembed_error_count: int
    error_type: str | None


def _required_string(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательную непустую строку."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть строкой."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _source_mentions_trailer(
    source_title: str,
    source_summary: str,
) -> bool:
    """
    Выполняет дешёвую предварительную проверку.

    Если источник вообще не говорит о trailer/teaser,
    HTML статьи не загружается.
    """

    source_text = (
        source_title
        + " "
        + source_summary
    ).casefold()

    return any(
        marker in source_text
        for marker in _TRAILER_MARKERS
    )


def _decode_html_document(
    content: bytes,
) -> str:
    """Декодирует HTML-документ для stdlib HTMLParser."""

    return content.decode(
        "utf-8",
        errors="replace",
    )


async def enrich_official_trailer(
    *,
    source_url: str,
    source_title: str,
    source_summary: str,
    article_timeout_seconds: float = 15.0,
    article_max_response_bytes: int = 2_000_000,
    article_max_redirects: int = 3,
    youtube_timeout_seconds: float = 10.0,
    resolver: AddressResolver = resolve_host_addresses,
    article_transport: (
        httpx.AsyncBaseTransport | None
    ) = None,
    youtube_transport: (
        httpx.AsyncBaseTransport | None
    ) = None,
) -> OfficialTrailerEnrichmentResult:
    """
    Ищет официальный трейлер только внутри статьи.

    Поток:

    source title/summary
    -> предварительная trailer/teaser проверка
    -> безопасная загрузка source_url
    -> извлечение YouTube iframe URL
    -> YouTube oEmbed для конкретных найденных видео
    -> консервативная official trailer verification
    -> official_trailer_url или None

    Общий web search не выполняется.
    Видео не скачивается.

    Ошибки внешних HTTP-источников считаются
    best-effort enrichment failure и не должны сами
    по себе останавливать основной generation pipeline.

    PostgreSQL, OpenAI и Telegram не используются.
    """

    normalized_source_url = _required_string(
        source_url,
        field_name="source_url",
    )

    normalized_source_title = _required_string(
        source_title,
        field_name="source_title",
    )

    if not isinstance(source_summary, str):
        raise TypeError(
            "source_summary должен быть строкой."
        )

    normalized_source_summary = (
        source_summary.strip()
    )

    if not _source_mentions_trailer(
        normalized_source_title,
        normalized_source_summary,
    ):
        return OfficialTrailerEnrichmentResult(
            attempted=False,
            verified=False,
            official_trailer_url=None,
            reason="source_is_not_trailer_news",
            article_final_url=None,
            youtube_candidate_urls=(),
            checked_video_urls=(),
            verification_reasons=(),
            oembed_error_count=0,
            error_type=None,
        )

    try:
        article = await download_article_document(
            normalized_source_url,
            timeout_seconds=(
                article_timeout_seconds
            ),
            max_response_bytes=(
                article_max_response_bytes
            ),
            max_redirects=(
                article_max_redirects
            ),
            resolver=resolver,
            transport=article_transport,
        )
    except ArticleDownloadError as error:
        return OfficialTrailerEnrichmentResult(
            attempted=True,
            verified=False,
            official_trailer_url=None,
            reason="article_download_failed",
            article_final_url=None,
            youtube_candidate_urls=(),
            checked_video_urls=(),
            verification_reasons=(),
            oembed_error_count=0,
            error_type=type(error).__name__,
        )

    html_content = _decode_html_document(
        article.content
    )

    youtube_candidate_urls = (
        extract_youtube_iframe_urls(
            html_content
        )
    )

    if not youtube_candidate_urls:
        return OfficialTrailerEnrichmentResult(
            attempted=True,
            verified=False,
            official_trailer_url=None,
            reason="youtube_iframe_not_found",
            article_final_url=article.final_url,
            youtube_candidate_urls=(),
            checked_video_urls=(),
            verification_reasons=(),
            oembed_error_count=0,
            error_type=None,
        )

    checked_video_urls: list[str] = []
    verification_reasons: list[str] = []
    oembed_error_count = 0

    for video_url in youtube_candidate_urls:
        try:
            metadata = (
                await fetch_youtube_oembed_metadata(
                    video_url,
                    timeout_seconds=(
                        youtube_timeout_seconds
                    ),
                    transport=youtube_transport,
                )
            )
        except YouTubeOEmbedError:
            oembed_error_count += 1
            continue

        checked_video_urls.append(
            metadata.canonical_url
        )

        verification = (
            verify_official_trailer(
                metadata,
                source_title=(
                    normalized_source_title
                ),
                source_summary=(
                    normalized_source_summary
                ),
            )
        )

        verification_reasons.append(
            verification.reason
        )

        if (
            verification.verified
            and verification.official_trailer_url
            is not None
        ):
            return OfficialTrailerEnrichmentResult(
                attempted=True,
                verified=True,
                official_trailer_url=(
                    verification
                    .official_trailer_url
                ),
                reason=(
                    "verified_official_trailer"
                ),
                article_final_url=(
                    article.final_url
                ),
                youtube_candidate_urls=(
                    youtube_candidate_urls
                ),
                checked_video_urls=tuple(
                    checked_video_urls
                ),
                verification_reasons=tuple(
                    verification_reasons
                ),
                oembed_error_count=(
                    oembed_error_count
                ),
                error_type=None,
            )

    if (
        oembed_error_count
        == len(youtube_candidate_urls)
    ):
        reason = "youtube_oembed_unavailable"
    else:
        reason = "official_trailer_not_verified"

    return OfficialTrailerEnrichmentResult(
        attempted=True,
        verified=False,
        official_trailer_url=None,
        reason=reason,
        article_final_url=article.final_url,
        youtube_candidate_urls=(
            youtube_candidate_urls
        ),
        checked_video_urls=tuple(
            checked_video_urls
        ),
        verification_reasons=tuple(
            verification_reasons
        ),
        oembed_error_count=oembed_error_count,
        error_type=None,
    )