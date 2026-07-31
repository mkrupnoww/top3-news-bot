import argparse
import asyncio
import re

import asyncpg

from app.collectors.feed_collector import (
    collect_feed,
)
from app.collectors.feed_http import (
    FeedDownloadError,
)
from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


_SOURCE_CODE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_-]{1,63}$"
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает настройки источника и сборщика."""

    parser = argparse.ArgumentParser(
        description=(
            "Загружает RSS/Atom-ленту и сохраняет "
            "новые записи в PostgreSQL. "
            "Telegram не используется."
        ),
    )

    parser.add_argument(
        "feed_url",
        help="Публичный URL RSS/Atom-ленты.",
    )

    parser.add_argument(
        "--source-code",
        required=True,
        help=(
            "Уникальный технический код источника, "
            "например variety_film."
        ),
    )

    parser.add_argument(
        "--source-name",
        required=True,
        help="Отображаемое название источника.",
    )

    parser.add_argument(
        "--base-url",
        default=None,
        help="Основной URL сайта источника.",
    )

    parser.add_argument(
        "--language-code",
        default="en",
        help=(
            "Язык новостей. По умолчанию: en."
        ),
    )

    parser.add_argument(
        "--collection-priority",
        type=int,
        default=100,
        help=(
            "Приоритет источника. "
            "По умолчанию: 100."
        ),
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=20,
        help=(
            "Максимальное число записей за запуск. "
            "По умолчанию: 20."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="HTTP-таймаут. По умолчанию: 15.",
    )

    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=2_000_000,
        help=(
            "Максимальный размер XML. "
            "По умолчанию: 2000000."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет локальные параметры CLI."""

    arguments.source_code = (
        arguments.source_code.strip().lower()
    )

    arguments.source_name = (
        arguments.source_name.strip()
    )

    arguments.language_code = (
        arguments.language_code.strip().lower()
    )

    if not _SOURCE_CODE_PATTERN.fullmatch(
        arguments.source_code
    ):
        raise ValueError(
            "--source-code должен содержать только "
            "строчные латинские буквы, цифры, "
            "дефис и нижнее подчёркивание."
        )

    if not arguments.source_name:
        raise ValueError(
            "--source-name не может быть пустым."
        )

    if arguments.collection_priority < 0:
        raise ValueError(
            "--collection-priority не может "
            "быть отрицательным."
        )

    if arguments.max_entries <= 0:
        raise ValueError(
            "--max-entries должен быть больше нуля."
        )


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет один запуск сборщика."""

    try:
        validate_arguments(
            arguments
        )
    except ValueError as error:
        print("Feed collection refused")
        print(error)
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        result = await collect_feed(
            database_pool,
            source_code=arguments.source_code,
            source_name=arguments.source_name,
            feed_url=arguments.feed_url,
            base_url=arguments.base_url,
            language_code=(
                arguments.language_code
            ),
            collection_priority=(
                arguments.collection_priority
            ),
            max_entries=arguments.max_entries,
            timeout_seconds=(
                arguments.timeout_seconds
            ),
            max_response_bytes=(
                arguments.max_response_bytes
            ),
        )

    except (
        FeedDownloadError,
        ValueError,
        asyncpg.PostgresError,
    ) as error:
        print("Feed collection failed")
        print(
            f"{type(error).__name__}: {error}"
        )
        print(
            "Collection run status: failed"
        )
        print("Telegram publication: not performed")
        return 1

    finally:
        await close_database_pool(
            database_pool
        )

    print("Feed collection completed")
    print(f"source_id={result.source_id}")
    print(
        "collection_run_id="
        f"{result.collection_run_id}"
    )
    print(
        f"run_status={result.run_status}"
    )
    print(
        f"feed_title="
        f"{result.feed_title or 'not specified'}"
    )
    print(f"feed_type={result.feed_type}")
    print(
        f"requested_url="
        f"{result.requested_url}"
    )
    print(f"final_url={result.final_url}")
    print(
        f"bytes_downloaded="
        f"{result.bytes_downloaded}"
    )
    print(
        f"redirect_count="
        f"{result.redirect_count}"
    )
    print(
        f"fetched_count="
        f"{result.fetched_count}"
    )
    print(
        f"inserted_count="
        f"{result.inserted_count}"
    )
    print(
        f"duplicate_count="
        f"{result.duplicate_count}"
    )
    print(
        f"rejected_count="
        f"{result.rejected_count}"
    )
    print(
        "news_ids="
        + (
            ",".join(
                str(news_id)
                for news_id in result.news_ids
            )
            or "none"
        )
    )
    print("Telegram publication: not performed")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )