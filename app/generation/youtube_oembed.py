from dataclasses import dataclass

import httpx

from app.generation.trailer_extractor import (
    build_youtube_watch_url,
    extract_youtube_video_id,
)


_OEMBED_URL = "https://www.youtube.com/oembed"
_DEFAULT_USER_AGENT = (
    "top3-news-bot/0.1 "
    "(YouTube trailer verification)"
)


class YouTubeOEmbedError(RuntimeError):
    """Ошибка получения или разбора YouTube oEmbed."""


@dataclass(frozen=True, slots=True)
class YouTubeOEmbedMetadata:
    """Проверяемые метаданные конкретного YouTube-видео."""

    video_id: str
    canonical_url: str
    title: str
    author_name: str
    author_url: str


def _required_text(
    payload: dict[str, object],
    field_name: str,
) -> str:
    """Извлекает обязательную непустую строку."""

    value = payload.get(field_name)

    if not isinstance(value, str):
        raise YouTubeOEmbedError(
            "YouTube oEmbed не содержит "
            f"строковое поле {field_name}."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise YouTubeOEmbedError(
            "YouTube oEmbed содержит пустое "
            f"поле {field_name}."
        )

    return normalized_value


async def fetch_youtube_oembed_metadata(
    video_url: str,
    *,
    timeout_seconds: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> YouTubeOEmbedMetadata:
    """
    Получает oEmbed-метаданные уже найденного YouTube-видео.

    Функция не выполняет поиск роликов. Она проверяет
    только конкретный URL, обнаруженный в исходной статье.

    PostgreSQL, OpenAI и Telegram не используются.
    """

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds должен быть "
            "больше нуля."
        )

    video_id = extract_youtube_video_id(
        video_url
    )

    if video_id is None:
        raise ValueError(
            "video_url должен быть корректным "
            "YouTube URL."
        )

    canonical_url = build_youtube_watch_url(
        video_id
    )

    timeout = httpx.Timeout(
        timeout_seconds
    )

    headers = {
        "Accept": "application/json",
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
            response = await client.get(
                _OEMBED_URL,
                params={
                    "url": canonical_url,
                    "format": "json",
                },
            )

            response.raise_for_status()

    except httpx.TimeoutException as error:
        raise YouTubeOEmbedError(
            "Истёк таймаут YouTube oEmbed: "
            f"video_id={video_id}"
        ) from error

    except httpx.HTTPStatusError as error:
        raise YouTubeOEmbedError(
            "YouTube oEmbed вернул HTTP-ошибку: "
            f"status={error.response.status_code}, "
            f"video_id={video_id}"
        ) from error

    except httpx.HTTPError as error:
        raise YouTubeOEmbedError(
            "Ошибка HTTP YouTube oEmbed: "
            f"video_id={video_id}, error={error}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise YouTubeOEmbedError(
            "YouTube oEmbed вернул некорректный JSON: "
            f"video_id={video_id}"
        ) from error

    if not isinstance(payload, dict):
        raise YouTubeOEmbedError(
            "YouTube oEmbed должен вернуть "
            "JSON-объект."
        )

    provider_name = payload.get(
        "provider_name"
    )

    if (
        isinstance(provider_name, str)
        and provider_name.strip()
        and provider_name.strip().casefold()
        != "youtube"
    ):
        raise YouTubeOEmbedError(
            "Неожиданный provider_name "
            "в YouTube oEmbed: "
            f"{provider_name}"
        )

    return YouTubeOEmbedMetadata(
        video_id=video_id,
        canonical_url=canonical_url,
        title=_required_text(
            payload,
            "title",
        ),
        author_name=_required_text(
            payload,
            "author_name",
        ),
        author_url=_required_text(
            payload,
            "author_url",
        ),
    )