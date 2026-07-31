import asyncio
from pathlib import Path

import httpx

from app.collectors.feed_http import (
    FeedTooLargeError,
    UnsafeFeedUrlError,
    UnsupportedFeedContentTypeError,
    download_feed_document,
)
from app.collectors.feed_parser import (
    parse_feed_document,
)


RSS_FIXTURE = Path(
    "tests/fixtures/sample_movie_rss.xml"
)


async def fake_public_resolver(
    hostname: str,
    port: int,
) -> tuple[str, ...]:
    """Возвращает тестовый публичный IP."""

    assert hostname in {
        "news.example.com",
        "feeds.example.com",
    }

    assert port == 443

    return ("93.184.216.34",)


async def test_successful_download() -> None:
    """Проверяет redirect, загрузку и парсинг."""

    rss_content = RSS_FIXTURE.read_bytes()
    requested_urls: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        requested_urls.append(
            str(request.url)
        )

        if (
            str(request.url)
            == "https://news.example.com/feed"
        ):
            return httpx.Response(
                status_code=302,
                headers={
                    "Location": (
                        "https://feeds.example.com/"
                        "movie-rss.xml"
                    )
                },
            )

        if (
            str(request.url)
            == (
                "https://feeds.example.com/"
                "movie-rss.xml"
            )
        ):
            return httpx.Response(
                status_code=200,
                headers={
                    "Content-Type": (
                        "application/rss+xml; "
                        "charset=utf-8"
                    ),
                    "Content-Length": str(
                        len(rss_content)
                    ),
                },
                content=rss_content,
            )

        return httpx.Response(
            status_code=404
        )

    result = await download_feed_document(
        "https://news.example.com/feed",
        resolver=fake_public_resolver,
        transport=httpx.MockTransport(
            handler
        ),
    )

    assert result.status_code == 200
    assert result.redirect_count == 1
    assert result.bytes_downloaded == len(
        rss_content
    )
    assert (
        result.content_type
        == "application/rss+xml"
    )
    assert result.final_url == (
        "https://feeds.example.com/"
        "movie-rss.xml"
    )

    assert requested_urls == [
        "https://news.example.com/feed",
        (
            "https://feeds.example.com/"
            "movie-rss.xml"
        ),
    ]

    parsed_feed = parse_feed_document(
        result.content
    )

    assert parsed_feed.feed_type == "rss"
    assert len(parsed_feed.entries) == 2

    print("Successful HTTP download: OK")
    print(
        f"final_url={result.final_url}"
    )
    print(
        f"redirect_count="
        f"{result.redirect_count}"
    )
    print(
        f"bytes_downloaded="
        f"{result.bytes_downloaded}"
    )
    print(
        f"parsed_entries="
        f"{len(parsed_feed.entries)}"
    )


async def test_unsafe_urls() -> None:
    """Проверяет блокировку внутренних адресов."""

    unsafe_urls = [
        "http://127.0.0.1/feed",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/feed",
        "http://localhost/feed",
        "http://[::1]/feed",
    ]

    async def forbidden_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "HTTP-запрос к опасному адресу "
            "не должен выполняться."
        )

    for unsafe_url in unsafe_urls:
        try:
            await download_feed_document(
                unsafe_url,
                transport=httpx.MockTransport(
                    forbidden_handler
                ),
            )
        except UnsafeFeedUrlError:
            continue

        raise AssertionError(
            "Опасный URL не был заблокирован: "
            f"url={unsafe_url}"
        )

    print()
    print("Unsafe URL blocking: OK")
    print(
        f"blocked_count={len(unsafe_urls)}"
    )


async def test_response_size_limit() -> None:
    """Проверяет раннюю блокировку большого ответа."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={
                "Content-Type": (
                    "application/rss+xml"
                ),
                "Content-Length": "10000",
            },
            content=b"<rss></rss>",
        )

    try:
        await download_feed_document(
            "https://news.example.com/feed",
            max_response_bytes=1000,
            resolver=fake_public_resolver,
            transport=httpx.MockTransport(
                handler
            ),
        )
    except FeedTooLargeError:
        print()
        print("Response size limit: OK")
        return

    raise AssertionError(
        "Большой HTTP-ответ не был заблокирован."
    )


async def test_content_type_blocking() -> None:
    """Проверяет запрет HTML вместо XML."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={
                "Content-Type": (
                    "text/html; charset=utf-8"
                )
            },
            content=b"<html>Not a feed</html>",
        )

    try:
        await download_feed_document(
            "https://news.example.com/feed",
            resolver=fake_public_resolver,
            transport=httpx.MockTransport(
                handler
            ),
        )
    except UnsupportedFeedContentTypeError:
        print()
        print("Content-Type blocking: OK")
        return

    raise AssertionError(
        "HTML-ответ не был заблокирован."
    )


async def main() -> int:
    """Запускает HTTP-тесты без реальной сети."""

    await test_successful_download()
    await test_unsafe_urls()
    await test_response_size_limit()
    await test_content_type_blocking()

    print()
    print("Real network requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Feed HTTP client test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )