import asyncio

import httpx

from app.generation.official_trailer_enrichment import (
    enrich_official_trailer,
)


ARTICLE_URL = (
    "https://news.example.com/"
    "primetime-trailer"
)

VIDEO_ID = "5fHXyqQOKL8"

WATCH_URL = (
    "https://www.youtube.com/watch?v="
    f"{VIDEO_ID}"
)


async def fake_public_resolver(
    hostname: str,
    port: int,
) -> tuple[str, ...]:
    """Возвращает тестовый публичный IP."""

    assert hostname == "news.example.com"
    assert port == 443

    return ("93.184.216.34",)


def build_article_transport(
    html_content: str,
) -> httpx.MockTransport:
    """Создаёт transport для HTML статьи."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert str(request.url) == ARTICLE_URL

        return httpx.Response(
            status_code=200,
            headers={
                "Content-Type": (
                    "text/html; charset=utf-8"
                ),
            },
            content=html_content.encode(
                "utf-8"
            ),
        )

    return httpx.MockTransport(
        handler
    )


def build_verified_oembed_transport(
) -> httpx.MockTransport:
    """Возвращает metadata официального A24 трейлера."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.url.host == (
            "www.youtube.com"
        )
        assert request.url.path == "/oembed"

        assert (
            request.url.params.get("url")
            == WATCH_URL
        )

        assert (
            request.url.params.get("format")
            == "json"
        )

        return httpx.Response(
            status_code=200,
            json={
                "title": (
                    "Primetime | "
                    "Official Trailer HD | A24"
                ),
                "author_name": "A24",
                "author_url": (
                    "https://www.youtube.com/@A24"
                ),
                "provider_name": "YouTube",
            },
        )

    return httpx.MockTransport(
        handler
    )


async def test_verified_trailer() -> None:
    """Проверяет полный успешный enrichment."""

    html_content = f"""
    <!doctype html>
    <html>
      <body>
        <iframe
          src="https://www.youtube.com/embed/{VIDEO_ID}?start=10">
        </iframe>
      </body>
    </html>
    """

    result = await enrich_official_trailer(
        source_url=ARTICLE_URL,
        source_title=(
            "Jeff Zucker Disses Robert Pattinson "
            "in Riveting Primetime Trailer"
        ),
        source_summary=(
            "The official trailer for A24's "
            "dramatic take on the series."
        ),
        resolver=fake_public_resolver,
        article_transport=(
            build_article_transport(
                html_content
            )
        ),
        youtube_transport=(
            build_verified_oembed_transport()
        ),
    )

    assert result.attempted is True
    assert result.verified is True
    assert (
        result.official_trailer_url
        == WATCH_URL
    )
    assert (
        result.reason
        == "verified_official_trailer"
    )
    assert result.article_final_url == (
        ARTICLE_URL
    )
    assert result.youtube_candidate_urls == (
        WATCH_URL,
    )
    assert result.checked_video_urls == (
        WATCH_URL,
    )
    assert result.verification_reasons == (
        "verified_official_trailer",
    )
    assert result.oembed_error_count == 0
    assert result.error_type is None

    print("Verified trailer enrichment: OK")
    print(
        "official_trailer_url="
        f"{result.official_trailer_url}"
    )


async def test_non_trailer_short_circuit() -> None:
    """Не загружает статью для обычной новости."""

    async def forbidden_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "HTTP-запрос не должен "
            "выполняться."
        )

    forbidden_transport = (
        httpx.MockTransport(
            forbidden_handler
        )
    )

    result = await enrich_official_trailer(
        source_url=ARTICLE_URL,
        source_title=(
            "Martin McDonagh to Receive "
            "Zurich Film Festival Award"
        ),
        source_summary=(
            "The filmmaker will attend "
            "the festival gala."
        ),
        resolver=fake_public_resolver,
        article_transport=(
            forbidden_transport
        ),
        youtube_transport=(
            forbidden_transport
        ),
    )

    assert result.attempted is False
    assert result.verified is False
    assert result.official_trailer_url is None
    assert (
        result.reason
        == "source_is_not_trailer_news"
    )
    assert result.article_final_url is None
    assert result.youtube_candidate_urls == ()
    assert result.checked_video_urls == ()
    assert result.verification_reasons == ()
    assert result.oembed_error_count == 0
    assert result.error_type is None

    print()
    print("Non-trailer short circuit: OK")


async def test_article_download_failure() -> None:
    """HTTP-сбой статьи остаётся best-effort."""

    async def article_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            content=b"Unavailable",
        )

    async def forbidden_youtube_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "YouTube oEmbed не должен "
            "вызываться."
        )

    result = await enrich_official_trailer(
        source_url=ARTICLE_URL,
        source_title=(
            "A24 Releases Primetime Trailer"
        ),
        source_summary=(
            "The official trailer for A24 "
            "has arrived."
        ),
        resolver=fake_public_resolver,
        article_transport=httpx.MockTransport(
            article_handler
        ),
        youtube_transport=httpx.MockTransport(
            forbidden_youtube_handler
        ),
    )

    assert result.attempted is True
    assert result.verified is False
    assert result.official_trailer_url is None
    assert (
        result.reason
        == "article_download_failed"
    )
    assert result.error_type == (
        "ArticleDownloadError"
    )

    print()
    print("Article failure best-effort: OK")
    print(
        "error_type="
        f"{result.error_type}"
    )


async def test_missing_youtube_iframe() -> None:
    """Проверяет статью без YouTube iframe."""

    html_content = """
    <!doctype html>
    <html>
      <body>
        <p>The official trailer is discussed here.</p>
      </body>
    </html>
    """

    async def forbidden_youtube_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "YouTube oEmbed не должен "
            "вызываться."
        )

    result = await enrich_official_trailer(
        source_url=ARTICLE_URL,
        source_title=(
            "A24 Releases Primetime Trailer"
        ),
        source_summary=(
            "The official trailer for A24 "
            "has arrived."
        ),
        resolver=fake_public_resolver,
        article_transport=(
            build_article_transport(
                html_content
            )
        ),
        youtube_transport=httpx.MockTransport(
            forbidden_youtube_handler
        ),
    )

    assert result.attempted is True
    assert result.verified is False
    assert result.official_trailer_url is None
    assert (
        result.reason
        == "youtube_iframe_not_found"
    )
    assert result.youtube_candidate_urls == ()
    assert result.checked_video_urls == ()

    print()
    print("Missing YouTube iframe: OK")


async def test_multiple_candidates() -> None:
    """
    Проверяет продолжение после неподходящего ролика.

    Первый iframe существует, но это не трейлер.
    Второй iframe является официальным трейлером.
    """

    unrelated_video_id = "abc123XYZ"
    unrelated_watch_url = (
        "https://www.youtube.com/watch?v="
        f"{unrelated_video_id}"
    )

    html_content = f"""
    <!doctype html>
    <html>
      <body>
        <iframe
          src="https://www.youtube.com/embed/{unrelated_video_id}">
        </iframe>
        <iframe
          src="https://www.youtube.com/embed/{VIDEO_ID}">
        </iframe>
      </body>
    </html>
    """

    requested_video_urls: list[str] = []

    async def youtube_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        video_url = request.url.params.get(
            "url"
        )

        assert video_url is not None

        requested_video_urls.append(
            video_url
        )

        if video_url == unrelated_watch_url:
            return httpx.Response(
                status_code=200,
                json={
                    "title": (
                        "Primetime Behind "
                        "the Scenes | A24"
                    ),
                    "author_name": "A24",
                    "author_url": (
                        "https://www.youtube.com/@A24"
                    ),
                    "provider_name": "YouTube",
                },
            )

        if video_url == WATCH_URL:
            return httpx.Response(
                status_code=200,
                json={
                    "title": (
                        "Primetime | "
                        "Official Trailer HD | A24"
                    ),
                    "author_name": "A24",
                    "author_url": (
                        "https://www.youtube.com/@A24"
                    ),
                    "provider_name": "YouTube",
                },
            )

        return httpx.Response(
            status_code=404
        )

    result = await enrich_official_trailer(
        source_url=ARTICLE_URL,
        source_title=(
            "A24 Releases Primetime Trailer"
        ),
        source_summary=(
            "The official trailer for A24 "
            "has arrived."
        ),
        resolver=fake_public_resolver,
        article_transport=(
            build_article_transport(
                html_content
            )
        ),
        youtube_transport=httpx.MockTransport(
            youtube_handler
        ),
    )

    assert result.verified is True
    assert (
        result.official_trailer_url
        == WATCH_URL
    )
    assert result.youtube_candidate_urls == (
        unrelated_watch_url,
        WATCH_URL,
    )
    assert result.checked_video_urls == (
        unrelated_watch_url,
        WATCH_URL,
    )
    assert result.verification_reasons == (
        "youtube_title_is_not_trailer",
        "verified_official_trailer",
    )
    assert requested_video_urls == [
        unrelated_watch_url,
        WATCH_URL,
    ]

    print()
    print("Multiple YouTube candidates: OK")
    print(
        "checked_video_count="
        f"{len(result.checked_video_urls)}"
    )


async def test_oembed_failure_best_effort() -> None:
    """Недоступный oEmbed не ломает enrichment."""

    html_content = f"""
    <!doctype html>
    <html>
      <body>
        <iframe
          src="https://www.youtube.com/embed/{VIDEO_ID}">
        </iframe>
      </body>
    </html>
    """

    async def youtube_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            json={
                "error": "Unavailable",
            },
        )

    result = await enrich_official_trailer(
        source_url=ARTICLE_URL,
        source_title=(
            "A24 Releases Primetime Trailer"
        ),
        source_summary=(
            "The official trailer for A24 "
            "has arrived."
        ),
        resolver=fake_public_resolver,
        article_transport=(
            build_article_transport(
                html_content
            )
        ),
        youtube_transport=httpx.MockTransport(
            youtube_handler
        ),
    )

    assert result.attempted is True
    assert result.verified is False
    assert result.official_trailer_url is None
    assert (
        result.reason
        == "youtube_oembed_unavailable"
    )
    assert result.oembed_error_count == 1
    assert result.checked_video_urls == ()
    assert result.verification_reasons == ()

    print()
    print("oEmbed failure best-effort: OK")


async def main() -> int:
    """Запускает enrichment-тесты без реальной сети."""

    await test_verified_trailer()
    await test_non_trailer_short_circuit()
    await test_article_download_failure()
    await test_missing_youtube_iframe()
    await test_multiple_candidates()
    await test_oembed_failure_best_effort()

    print()
    print("Real network requests: not performed")
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Official trailer enrichment test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )