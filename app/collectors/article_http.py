import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.collectors.feed_http import (
    UnsafeFeedUrlError,
    resolve_host_addresses,
    validate_public_feed_url,
)


AddressResolver = Callable[
    [str, int],
    Awaitable[tuple[str, ...]],
]

_ALLOWED_CONTENT_TYPES = {
    "application/xhtml+xml",
    "text/html",
}

_REDIRECT_STATUS_CODES = {
    301,
    302,
    303,
    307,
    308,
}

_DEFAULT_USER_AGENT = (
    "top3-news-bot/0.1 "
    "(article enrichment)"
)


class ArticleDownloadError(RuntimeError):
    """Базовая ошибка загрузки HTML-статьи."""


class UnsafeArticleUrlError(
    ArticleDownloadError
):
    """Адрес статьи может вести во внутреннюю сеть."""


class ArticleTooLargeError(
    ArticleDownloadError
):
    """Ответ превышает разрешённый размер."""


class UnsupportedArticleContentTypeError(
    ArticleDownloadError
):
    """Сервер вернул неподдерживаемый тип содержимого."""


@dataclass(frozen=True, slots=True)
class ArticleDownloadResult:
    """Результат безопасной HTTP-загрузки статьи."""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str | None
    content: bytes
    bytes_downloaded: int
    redirect_count: int


def _normalize_content_type(
    raw_content_type: str | None,
) -> str | None:
    """Извлекает MIME-тип без charset."""

    if raw_content_type is None:
        return None

    normalized_value = (
        raw_content_type
        .split(";", maxsplit=1)[0]
        .strip()
        .lower()
    )

    return normalized_value or None


def _validate_article_content_type(
    content_type: str | None,
) -> None:
    """Разрешает HTML-подобные ответы."""

    if content_type is None:
        return

    if content_type in _ALLOWED_CONTENT_TYPES:
        return

    raise UnsupportedArticleContentTypeError(
        "Сервер вернул неподдерживаемый "
        "Content-Type для статьи: "
        f"content_type={content_type}"
    )


def _validate_content_length(
    raw_content_length: str | None,
    *,
    max_response_bytes: int,
) -> None:
    """Проверяет Content-Length до чтения тела."""

    if raw_content_length is None:
        return

    try:
        content_length = int(
            raw_content_length
        )
    except ValueError:
        return

    if content_length < 0:
        return

    if content_length > max_response_bytes:
        raise ArticleTooLargeError(
            "Content-Length статьи превышает "
            "лимит: "
            f"size={content_length}, "
            f"limit={max_response_bytes}"
        )


async def _read_limited_response(
    response: httpx.Response,
    *,
    max_response_bytes: int,
) -> bytes:
    """Читает тело ответа с жёстким лимитом."""

    chunks: list[bytes] = []
    total_size = 0

    async for chunk in response.aiter_bytes():
        if not chunk:
            continue

        total_size += len(chunk)

        if total_size > max_response_bytes:
            raise ArticleTooLargeError(
                "Тело HTML-ответа превышает "
                "лимит: "
                f"received={total_size}, "
                f"limit={max_response_bytes}"
            )

        chunks.append(chunk)

    if total_size == 0:
        raise ArticleDownloadError(
            "Сервер вернул пустую HTML-страницу."
        )

    return b"".join(chunks)


async def _validate_public_article_url(
    url: str,
    *,
    resolver: AddressResolver,
) -> str:
    """Переиспользует SSRF-защиту RSS HTTP-слоя."""

    try:
        return await validate_public_feed_url(
            url,
            resolver=resolver,
        )
    except UnsafeFeedUrlError as error:
        raise UnsafeArticleUrlError(
            str(error)
        ) from error


async def download_article_document(
    url: str,
    *,
    timeout_seconds: float = 15.0,
    max_response_bytes: int = 2_000_000,
    max_redirects: int = 3,
    resolver: AddressResolver = (
        resolve_host_addresses
    ),
    transport: httpx.AsyncBaseTransport | None = None,
) -> ArticleDownloadResult:
    """
    Безопасно загружает HTML страницы статьи.

    Каждый исходный URL и каждый redirect
    проверяются перед выполнением запроса.

    PostgreSQL, OpenAI и Telegram не используются.
    """

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds должен быть "
            "больше нуля."
        )

    if max_response_bytes <= 0:
        raise ValueError(
            "max_response_bytes должен быть "
            "больше нуля."
        )

    if max_redirects < 0:
        raise ValueError(
            "max_redirects не может быть "
            "отрицательным."
        )

    requested_url = (
        await _validate_public_article_url(
            url,
            resolver=resolver,
        )
    )

    current_url = requested_url
    redirect_count = 0

    timeout = httpx.Timeout(
        timeout_seconds
    )

    headers = {
        "Accept": (
            "text/html,"
            "application/xhtml+xml;q=0.9,"
            "*/*;q=0.1"
        ),
        "User-Agent": _DEFAULT_USER_AGENT,
    }

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
            transport=transport,
        ) as client:
            while True:
                async with client.stream(
                    "GET",
                    current_url,
                ) as response:
                    if (
                        response.status_code
                        in _REDIRECT_STATUS_CODES
                    ):
                        location = (
                            response.headers.get(
                                "location"
                            )
                        )

                        if location is None:
                            raise ArticleDownloadError(
                                "Redirect статьи не "
                                "содержит Location: "
                                f"status="
                                f"{response.status_code}"
                            )

                        if (
                            redirect_count
                            >= max_redirects
                        ):
                            raise ArticleDownloadError(
                                "Превышен лимит "
                                "HTTP-перенаправлений "
                                "статьи: "
                                f"limit={max_redirects}"
                            )

                        redirect_url = urljoin(
                            current_url,
                            location,
                        )

                        current_url = (
                            await _validate_public_article_url(
                                redirect_url,
                                resolver=resolver,
                            )
                        )

                        redirect_count += 1
                        continue

                    response.raise_for_status()

                    content_type = (
                        _normalize_content_type(
                            response.headers.get(
                                "content-type"
                            )
                        )
                    )

                    _validate_article_content_type(
                        content_type
                    )

                    _validate_content_length(
                        response.headers.get(
                            "content-length"
                        ),
                        max_response_bytes=(
                            max_response_bytes
                        ),
                    )

                    content = (
                        await _read_limited_response(
                            response,
                            max_response_bytes=(
                                max_response_bytes
                            ),
                        )
                    )

                    return ArticleDownloadResult(
                        requested_url=requested_url,
                        final_url=current_url,
                        status_code=(
                            response.status_code
                        ),
                        content_type=content_type,
                        content=content,
                        bytes_downloaded=len(
                            content
                        ),
                        redirect_count=redirect_count,
                    )

    except ArticleDownloadError:
        raise

    except httpx.TimeoutException as error:
        raise ArticleDownloadError(
            "Истёк таймаут загрузки статьи: "
            f"url={current_url}, error={error}"
        ) from error

    except httpx.HTTPStatusError as error:
        raise ArticleDownloadError(
            "Сервер статьи вернул HTTP-ошибку: "
            f"status={error.response.status_code}, "
            f"url={current_url}"
        ) from error

    except httpx.HTTPError as error:
        raise ArticleDownloadError(
            "Ошибка HTTP при загрузке статьи: "
            f"url={current_url}, error={error}"
        ) from error