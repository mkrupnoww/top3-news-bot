import asyncio

import httpx

from app.collectors.article_http import (
    ArticleTooLargeError,
    UnsafeArticleUrlError,
    UnsupportedArticleContentTypeError,
    download_article_document,
)


async def fake_public_resolver(
    hostname: str,
    port: int,
) -> tuple[str, ...]:
    """Возвращает тестовый публичный IP."""

    assert hostname in {
        "news.example.com",
        "www.example.com",
    }
    assert port == 443
    return ("93.184.216.34",)


async def test_successful_download() -> None:
    """Проверяет redirect и HTML-загрузку."""

    article_content = (
        b"<!doctype html>"
        b"<html><body>"
        b"<h1>Movie trailer</h1>"
        b"</body></html>"
    )

    requested_urls: list[str] = []

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        requested_urls.append(
            str(request.url)
        )

        if str(request.url) == (
            "https://news.example.com/article"
        ):
            return httpx.Response(
                status_code=301,
                headers={
                    "Location": (
                        "https://www.example.com/"
                        "article/"
                    )
                },
            )

        if str(request.url) == (
            "https://www.example.com/article/"
        ):
            return httpx.Response(
                status_code=200,
                headers={
                    "Content-Type": (
                        "text/html; charset=utf-8"
                    ),
                    "Content-Length": str(
                        len(article_content)
                    ),
                },
                content=article_content,
            )

        return httpx.Response(
            status_code=404
        )

    result = await download_article_document(
        "https://news.example.com/article",
        resolver=fake_public_resolver,
        transport=httpx.MockTransport(
            handler
        ),
    )

    assert result.status_code == 200
    assert result.redirect_count == 1
    assert (
        result.bytes_downloaded
        == len(article_content)
    )
    assert result.content_type == "text/html"
    assert result.final_url == (
        "https://www.example.com/article/"
    )
    assert result.content == article_content

    assert requested_urls == [
        "https://news.example.com/article",
        "https://www.example.com/article/",
    ]

    print("Successful article download: OK")
    print(f"final_url={result.final_url}")
    print(
        "redirect_count="
        f"{result.redirect_count}"
    )
    print(
        "bytes_downloaded="
        f"{result.bytes_downloaded}"
    )


async def test_unsafe_urls() -> None:
    """Проверяет блокировку внутренних адресов."""

    unsafe_urls = [
        "http://127.0.0.1/article",
        (
            "http://169.254.169.254/"
            "latest/meta-data"
        ),
        "http://10.0.0.1/article",
        "http://localhost/article",
        "http://[::1]/article",
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
            await download_article_document(
                unsafe_url,
                transport=httpx.MockTransport(
                    forbidden_handler
                ),
            )
        except UnsafeArticleUrlError:
            continue

        raise AssertionError(
            "Опасный URL статьи не был "
            "заблокирован: "
            f"url={unsafe_url}"
        )

    print()
    print("Unsafe article URL blocking: OK")
    print(
        f"blocked_count={len(unsafe_urls)}"
    )


async def test_response_size_limit() -> None:
    """Проверяет раннюю блокировку большого HTML."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={
                "Content-Type": "text/html",
                "Content-Length": "10000",
            },
            content=b"<html></html>",
        )

    try:
        await download_article_document(
            "https://news.example.com/article",
            max_response_bytes=1000,
            resolver=fake_public_resolver,
            transport=httpx.MockTransport(
                handler
            ),
        )
    except ArticleTooLargeError:
        print()
        print("Article response size limit: OK")
        return

    raise AssertionError(
        "Большой HTML-ответ не был заблокирован."
    )


async def test_content_type_blocking() -> None:
    """Проверяет запрет не-HTML ответа."""

    async def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={
                "Content-Type": (
                    "application/json"
                )
            },
            content=b'{"not":"article"}',
        )

    try:
        await download_article_document(
            "https://news.example.com/article",
            resolver=fake_public_resolver,
            transport=httpx.MockTransport(
                handler
            ),
        )
    except UnsupportedArticleContentTypeError:
        print()
        print(
            "Article Content-Type blocking: OK"
        )
        return

    raise AssertionError(
        "Не-HTML ответ не был заблокирован."
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
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Article HTTP client test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )