import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx


AddressResolver = Callable[
    [str, int],
    Awaitable[tuple[str, ...]],
]


_ALLOWED_CONTENT_TYPES = {
    "application/atom+xml",
    "application/rdf+xml",
    "application/rss+xml",
    "application/xml",
    "text/plain",
    "text/xml",
}

_REDIRECT_STATUS_CODES = {
    301,
    302,
    303,
    307,
    308,
}

_BLOCKED_HOST_SUFFIXES = (
    ".home",
    ".internal",
    ".lan",
    ".local",
    ".localhost",
)

_DEFAULT_USER_AGENT = (
    "top3-news-bot/0.1 "
    "(RSS and Atom collector)"
)


class FeedDownloadError(RuntimeError):
    """Базовая ошибка загрузки RSS или Atom."""


class UnsafeFeedUrlError(FeedDownloadError):
    """Адрес ленты может вести во внутреннюю сеть."""


class FeedTooLargeError(FeedDownloadError):
    """Ответ превышает разрешённый размер."""


class UnsupportedFeedContentTypeError(
    FeedDownloadError
):
    """Сервер вернул неподдерживаемый тип содержимого."""


@dataclass(frozen=True, slots=True)
class FeedDownloadResult:
    """Результат безопасной HTTP-загрузки ленты."""

    requested_url: str
    final_url: str
    status_code: int
    content_type: str | None
    content: bytes
    bytes_downloaded: int
    redirect_count: int


def _parse_port(
    scheme: str,
    explicit_port: int | None,
) -> int:
    """Определяет порт HTTP-запроса."""

    if explicit_port is not None:
        return explicit_port

    return 443 if scheme == "https" else 80


def _normalize_ip_text(
    value: str,
) -> str:
    """Удаляет zone identifier у IPv6-адреса."""

    return value.split("%", maxsplit=1)[0]


def _is_public_ip_address(
    value: str,
) -> bool:
    """Проверяет глобальную маршрутизируемость IP."""

    try:
        address = ipaddress.ip_address(
            _normalize_ip_text(value)
        )
    except ValueError:
        return False

    return address.is_global


async def resolve_host_addresses(
    hostname: str,
    port: int,
) -> tuple[str, ...]:
    """Асинхронно получает IP-адреса хоста."""

    event_loop = asyncio.get_running_loop()

    try:
        address_info = await event_loop.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as error:
        raise FeedDownloadError(
            "Не удалось разрешить имя хоста: "
            f"hostname={hostname}, error={error}"
        ) from error

    addresses = sorted(
        {
            _normalize_ip_text(
                item[4][0]
            )
            for item in address_info
        }
    )

    if not addresses:
        raise FeedDownloadError(
            "DNS не вернул IP-адреса: "
            f"hostname={hostname}"
        )

    return tuple(addresses)


async def validate_public_feed_url(
    url: str,
    *,
    resolver: AddressResolver = (
        resolve_host_addresses
    ),
) -> str:
    """
    Проверяет адрес перед сетевым запросом.

    Разрешаются только:

    - HTTP и HTTPS;
    - порты 80 и 443;
    - публичные глобально маршрутизируемые IP;
    - адреса без username/password.
    """

    normalized_input = url.strip()

    if not normalized_input:
        raise UnsafeFeedUrlError(
            "URL ленты не может быть пустым."
        )

    try:
        parsed = urlsplit(
            normalized_input
        )
        explicit_port = parsed.port
    except ValueError as error:
        raise UnsafeFeedUrlError(
            f"Некорректный URL ленты: {error}"
        ) from error

    scheme = parsed.scheme.lower()

    if scheme not in {"http", "https"}:
        raise UnsafeFeedUrlError(
            "Разрешены только схемы http и https: "
            f"scheme={parsed.scheme or 'missing'}"
        )

    if parsed.username is not None:
        raise UnsafeFeedUrlError(
            "Username в URL ленты запрещён."
        )

    if parsed.password is not None:
        raise UnsafeFeedUrlError(
            "Password в URL ленты запрещён."
        )

    hostname = parsed.hostname

    if hostname is None:
        raise UnsafeFeedUrlError(
            "В URL ленты отсутствует hostname."
        )

    normalized_hostname = (
        hostname.rstrip(".").lower()
    )

    if (
        normalized_hostname == "localhost"
        or normalized_hostname.endswith(
            _BLOCKED_HOST_SUFFIXES
        )
    ):
        raise UnsafeFeedUrlError(
            "Локальное имя хоста запрещено: "
            f"hostname={hostname}"
        )

    port = _parse_port(
        scheme,
        explicit_port,
    )

    if port not in {80, 443}:
        raise UnsafeFeedUrlError(
            "Для RSS/Atom разрешены только "
            f"порты 80 и 443: port={port}"
        )

    try:
        literal_address = ipaddress.ip_address(
            normalized_hostname
        )
    except ValueError:
        literal_address = None

    if literal_address is not None:
        addresses = (
            str(literal_address),
        )
    else:
        addresses = await resolver(
            normalized_hostname,
            port,
        )

    blocked_addresses = [
        address
        for address in addresses
        if not _is_public_ip_address(
            address
        )
    ]

    if blocked_addresses:
        raise UnsafeFeedUrlError(
            "Хост ленты разрешается во внутренний "
            "или служебный IP-адрес: "
            f"hostname={hostname}, "
            f"addresses={','.join(blocked_addresses)}"
        )

    normalized_path = parsed.path or "/"

    return urlunsplit(
        (
            scheme,
            parsed.netloc,
            normalized_path,
            parsed.query,
            "",
        )
    )


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


def _validate_content_type(
    content_type: str | None,
) -> None:
    """Разрешает только XML-подобные ответы."""

    if content_type is None:
        # Некоторые RSS-серверы не передают
        # Content-Type. XML проверит feed_parser.
        return

    if (
        content_type in _ALLOWED_CONTENT_TYPES
        or content_type.endswith("+xml")
    ):
        return

    raise UnsupportedFeedContentTypeError(
        "Сервер вернул неподдерживаемый "
        "Content-Type: "
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
        raise FeedTooLargeError(
            "Content-Length превышает лимит: "
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
            raise FeedTooLargeError(
                "Тело HTTP-ответа превышает лимит: "
                f"received={total_size}, "
                f"limit={max_response_bytes}"
            )

        chunks.append(chunk)

    if total_size == 0:
        raise FeedDownloadError(
            "Сервер вернул пустой документ ленты."
        )

    return b"".join(chunks)


async def download_feed_document(
    url: str,
    *,
    timeout_seconds: float = 15.0,
    max_response_bytes: int = 2_000_000,
    max_redirects: int = 3,
    resolver: AddressResolver = (
        resolve_host_addresses
    ),
    transport: httpx.AsyncBaseTransport | None = None,
) -> FeedDownloadResult:
    """
    Загружает RSS/Atom с базовой защитой от SSRF.

    Каждый исходный адрес и каждый redirect
    проверяются перед выполнением запроса.

    PostgreSQL и Telegram не используются.
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

    requested_url = await validate_public_feed_url(
        url,
        resolver=resolver,
    )

    current_url = requested_url
    redirect_count = 0

    timeout = httpx.Timeout(
        timeout_seconds
    )

    headers = {
        "Accept": (
            "application/atom+xml,"
            "application/rss+xml,"
            "application/xml,"
            "text/xml;q=0.9,"
            "text/plain;q=0.5,"
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
                        location = response.headers.get(
                            "location"
                        )

                        if location is None:
                            raise FeedDownloadError(
                                "Redirect не содержит "
                                "заголовок Location: "
                                f"status="
                                f"{response.status_code}"
                            )

                        if (
                            redirect_count
                            >= max_redirects
                        ):
                            raise FeedDownloadError(
                                "Превышен лимит "
                                "HTTP-перенаправлений: "
                                f"limit={max_redirects}"
                            )

                        redirect_url = urljoin(
                            current_url,
                            location,
                        )

                        current_url = (
                            await validate_public_feed_url(
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

                    _validate_content_type(
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

                    return FeedDownloadResult(
                        requested_url=requested_url,
                        final_url=current_url,
                        status_code=(
                            response.status_code
                        ),
                        content_type=content_type,
                        content=content,
                        bytes_downloaded=len(content),
                        redirect_count=redirect_count,
                    )

    except FeedDownloadError:
        raise

    except httpx.TimeoutException as error:
        raise FeedDownloadError(
            "Истёк таймаут загрузки ленты: "
            f"url={current_url}, error={error}"
        ) from error

    except httpx.HTTPStatusError as error:
        raise FeedDownloadError(
            "Сервер ленты вернул HTTP-ошибку: "
            f"status={error.response.status_code}, "
            f"url={error.request.url}"
        ) from error

    except httpx.RequestError as error:
        raise FeedDownloadError(
            "Ошибка HTTP-запроса к ленте: "
            f"url={current_url}, error={error}"
        ) from error