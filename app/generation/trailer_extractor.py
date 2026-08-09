from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit


_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
}


class _YouTubeIframeParser(HTMLParser):
    """Извлекает URL YouTube iframe из HTML."""

    def __init__(self) -> None:
        super().__init__(
            convert_charrefs=True,
        )
        self.urls: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "iframe":
            return

        attributes = {
            name.casefold(): value
            for name, value in attrs
        }

        source_url = attributes.get("src")

        if source_url is None:
            return

        normalized_url = source_url.strip()

        if normalized_url:
            self.urls.append(
                normalized_url
            )


def _normalize_hostname(
    hostname: str | None,
) -> str | None:
    """Нормализует hostname URL."""

    if hostname is None:
        return None

    normalized_hostname = (
        hostname.rstrip(".").casefold()
    )

    return normalized_hostname or None


def extract_youtube_video_id(
    url: str,
) -> str | None:
    """
    Извлекает YouTube video_id из URL.

    Поддерживаются canonical watch URL,
    youtu.be и embed URL.
    """

    if not isinstance(url, str):
        raise TypeError(
            "url должен быть строкой."
        )

    normalized_url = url.strip()

    if not normalized_url:
        return None

    try:
        parsed = urlsplit(
            normalized_url
        )
    except ValueError:
        return None

    scheme = parsed.scheme.casefold()

    if scheme not in {"http", "https"}:
        return None

    hostname = _normalize_hostname(
        parsed.hostname
    )

    if hostname not in _YOUTUBE_HOSTS:
        return None

    path_parts = [
        part
        for part in parsed.path.split("/")
        if part
    ]

    video_id: str | None = None

    if hostname == "youtu.be":
        if path_parts:
            video_id = path_parts[0]

    elif (
        len(path_parts) >= 2
        and path_parts[0].casefold()
        in {"embed", "shorts", "live"}
    ):
        video_id = path_parts[1]

    elif (
        not path_parts
        or path_parts[0].casefold()
        == "watch"
    ):
        query = parse_qs(
            parsed.query
        )

        values = query.get("v")

        if values:
            video_id = values[0]

    if video_id is None:
        return None

    normalized_video_id = (
        video_id.strip()
    )

    if not normalized_video_id:
        return None

    if any(
        character.isspace()
        for character in normalized_video_id
    ):
        return None

    if len(normalized_video_id) > 64:
        return None

    return normalized_video_id


def build_youtube_watch_url(
    video_id: str,
) -> str:
    """Строит canonical YouTube watch URL."""

    if not isinstance(video_id, str):
        raise TypeError(
            "video_id должен быть строкой."
        )

    normalized_video_id = (
        video_id.strip()
    )

    if not normalized_video_id:
        raise ValueError(
            "video_id не может быть пустым."
        )

    if any(
        character.isspace()
        for character in normalized_video_id
    ):
        raise ValueError(
            "video_id не должен содержать "
            "пробельные символы."
        )

    if len(normalized_video_id) > 64:
        raise ValueError(
            "video_id имеет недопустимую длину."
        )

    return (
        "https://www.youtube.com/watch?v="
        f"{normalized_video_id}"
    )


def extract_youtube_iframe_urls(
    html_content: str,
) -> tuple[str, ...]:
    """
    Извлекает canonical YouTube URL из iframe.

    Возвращает уникальные URL в порядке
    появления в HTML.
    """

    if not isinstance(
        html_content,
        str,
    ):
        raise TypeError(
            "html_content должен быть строкой."
        )

    parser = _YouTubeIframeParser()
    parser.feed(html_content)
    parser.close()

    results: list[str] = []
    seen_video_ids: set[str] = set()

    for source_url in parser.urls:
        video_id = extract_youtube_video_id(
            source_url
        )

        if video_id is None:
            continue

        if video_id in seen_video_ids:
            continue

        seen_video_ids.add(video_id)

        results.append(
            build_youtube_watch_url(
                video_id
            )
        )

    return tuple(results)