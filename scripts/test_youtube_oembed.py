import asyncio

import httpx

from app.generation.youtube_oembed import (
    YouTubeOEmbedError,
    fetch_youtube_oembed_metadata,
)


VIDEO_ID = "5fHXyqQOKL8"
WATCH_URL = (
    "https://www.youtube.com/watch?v="
    f"{VIDEO_ID}"
)


async def test_successful_oembed() -> None:
    """Проверяет метаданные известного ролика."""

    requested_urls: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        requested_urls.append(
            str(request.url)
        )

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
            headers={
                "Content-Type": (
                    "application/json"
                )
            },
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
                "provider_url": (
                    "https://www.youtube.com/"
                ),
                "type": "video",
            },
        )

    metadata = (
        await fetch_youtube_oembed_metadata(
            (
                "https://www.youtube.com/embed/"
                f"{VIDEO_ID}?start=128"
            ),
            transport=httpx.MockTransport(
                handler
            ),
        )
    )

    assert metadata.video_id == VIDEO_ID
    assert metadata.canonical_url == WATCH_URL

    assert metadata.title == (
        "Primetime | "
        "Official Trailer HD | A24"
    )

    assert metadata.author_name == "A24"

    assert metadata.author_url == (
        "https://www.youtube.com/@A24"
    )

    assert len(requested_urls) == 1

    print("Successful YouTube oEmbed: OK")
    print(f"video_id={metadata.video_id}")
    print(f"title={metadata.title}")
    print(
        "author_name="
        f"{metadata.author_name}"
    )
    print(
        "author_url="
        f"{metadata.author_url}"
    )


async def test_invalid_video_url() -> None:
    """Проверяет запрет чужого URL."""

    async def forbidden_handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            "HTTP-запрос не должен "
            "выполняться."
        )

    try:
        await fetch_youtube_oembed_metadata(
            "https://example.com/video/123",
            transport=httpx.MockTransport(
                forbidden_handler
            ),
        )
    except ValueError:
        print()
        print("Invalid YouTube URL blocking: OK")
        return

    raise AssertionError(
        "Не-YouTube URL не был заблокирован."
    )


async def test_http_failure() -> None:
    """Проверяет обработку отсутствующего ролика."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={
                "error": "Not found",
            },
        )

    try:
        await fetch_youtube_oembed_metadata(
            WATCH_URL,
            transport=httpx.MockTransport(
                handler
            ),
        )
    except YouTubeOEmbedError:
        print()
        print("YouTube oEmbed HTTP failure: OK")
        return

    raise AssertionError(
        "HTTP-ошибка oEmbed не была обработана."
    )


async def test_invalid_payload() -> None:
    """Проверяет обязательные поля ответа."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "title": "Trailer",
                "author_name": "A24",
                "provider_name": "YouTube",
            },
        )

    try:
        await fetch_youtube_oembed_metadata(
            WATCH_URL,
            transport=httpx.MockTransport(
                handler
            ),
        )
    except YouTubeOEmbedError:
        print()
        print(
            "Invalid YouTube oEmbed payload: OK"
        )
        return

    raise AssertionError(
        "Неполный oEmbed payload "
        "не был заблокирован."
    )


async def main() -> int:
    """Запускает oEmbed-тесты без реальной сети."""

    await test_successful_oembed()
    await test_invalid_video_url()
    await test_http_failure()
    await test_invalid_payload()

    print()
    print("Real network requests: not performed")
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("YouTube oEmbed client test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )