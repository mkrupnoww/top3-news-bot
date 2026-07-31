import argparse
import asyncio
from datetime import datetime

from app.collectors.feed_http import (
    FeedDownloadError,
    download_feed_document,
)
from app.collectors.feed_parser import (
    ParsedFeedEntry,
    parse_feed_document,
)


def _truncate(
    value: str | None,
    *,
    max_length: int,
) -> str:
    """Сокращает длинный текст для вывода в терминал."""

    if value is None:
        return "not specified"

    normalized_value = value.strip()

    if not normalized_value:
        return "not specified"

    if len(normalized_value) <= max_length:
        return normalized_value

    return (
        normalized_value[: max_length - 1].rstrip()
        + "…"
    )


def _format_datetime(
    value: datetime | None,
) -> str:
    """Форматирует дату публикации записи."""

    if value is None:
        return "unknown"

    return value.isoformat()


def _print_entry(
    position: int,
    entry: ParsedFeedEntry,
) -> None:
    """Выводит одну запись RSS/Atom."""

    print()
    print(
        f"{position}. "
        f"{_truncate(entry.title, max_length=300)}"
    )
    print(
        f"   published_at="
        f"{_format_datetime(entry.source_published_at)}"
    )
    print(
        f"   author="
        f"{_truncate(entry.author_name, max_length=200)}"
    )
    print(
        f"   source_url={entry.source_url}"
    )
    print(
        "   primary_image_url="
        f"{_truncate(entry.primary_image_url, max_length=500)}"
    )
    print(
        f"   external_id="
        f"{_truncate(entry.external_id, max_length=300)}"
    )
    print(
        f"   summary="
        f"{_truncate(entry.summary, max_length=500)}"
    )


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры загрузки ленты."""

    parser = argparse.ArgumentParser(
        description=(
            "Загружает и разбирает публичную RSS/Atom-ленту. "
            "PostgreSQL и Telegram не используются."
        ),
    )

    parser.add_argument(
        "feed_url",
        help="Публичный HTTP или HTTPS URL RSS/Atom-ленты.",
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=20,
        help=(
            "Максимальное число выводимых записей. "
            "По умолчанию: 20."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help=(
            "Таймаут HTTP-запроса в секундах. "
            "По умолчанию: 15."
        ),
    )

    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=2_000_000,
        help=(
            "Максимальный размер XML-документа. "
            "По умолчанию: 2000000 байт."
        ),
    )

    return parser.parse_args()


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Загружает, разбирает и выводит содержимое ленты."""

    if arguments.max_entries <= 0:
        print("Feed inspection failed")
        print("--max-entries должен быть больше нуля.")
        return 2

    try:
        download_result = await download_feed_document(
            arguments.feed_url,
            timeout_seconds=(
                arguments.timeout_seconds
            ),
            max_response_bytes=(
                arguments.max_response_bytes
            ),
        )

        parse_result = parse_feed_document(
            download_result.content,
            max_entries=arguments.max_entries,
            max_document_bytes=(
                arguments.max_response_bytes
            ),
        )

    except (
        FeedDownloadError,
        ValueError,
    ) as error:
        print("Feed inspection failed")
        print(error)
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 1

    print("Feed inspection completed")
    print(
        f"requested_url="
        f"{download_result.requested_url}"
    )
    print(
        f"final_url="
        f"{download_result.final_url}"
    )
    print(
        f"status_code="
        f"{download_result.status_code}"
    )
    print(
        f"content_type="
        f"{download_result.content_type or 'not specified'}"
    )
    print(
        f"bytes_downloaded="
        f"{download_result.bytes_downloaded}"
    )
    print(
        f"redirect_count="
        f"{download_result.redirect_count}"
    )
    print(
        f"feed_type="
        f"{parse_result.feed_type}"
    )
    print(
        f"feed_title="
        f"{parse_result.feed_title or 'not specified'}"
    )
    print(
        f"entry_count="
        f"{len(parse_result.entries)}"
    )
    print(
        f"skipped_count="
        f"{parse_result.skipped_count}"
    )

    for position, entry in enumerate(
        parse_result.entries,
        start=1,
    ):
        _print_entry(
            position,
            entry,
        )

    print()
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Live feed inspection: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )