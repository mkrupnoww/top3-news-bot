import argparse
import asyncio

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


COLLECTOR_LOCK_ID = 7_320_031


def parse_arguments() -> argparse.Namespace:
    """Разбирает общие параметры запуска сборщика."""

    parser = argparse.ArgumentParser(
        description=(
            "Последовательно загружает все активные "
            "RSS/Atom-источники из PostgreSQL. "
            "Telegram не используется."
        ),
    )

    parser.add_argument(
        "--max-entries",
        type=int,
        default=100,
        help=(
            "Максимальное число записей из одной "
            "ленты за запуск. По умолчанию: 100."
        ),
    )

    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help=(
            "HTTP-таймаут одной ленты. "
            "По умолчанию: 15 секунд."
        ),
    )

    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=2_000_000,
        help=(
            "Максимальный размер одного XML-документа. "
            "По умолчанию: 2000000 байт."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет параметры запуска."""

    if arguments.max_entries <= 0:
        raise ValueError(
            "--max-entries должен быть больше нуля."
        )

    if arguments.timeout_seconds <= 0:
        raise ValueError(
            "--timeout-seconds должен быть больше нуля."
        )

    if arguments.max_response_bytes <= 0:
        raise ValueError(
            "--max-response-bytes должен быть "
            "больше нуля."
        )


async def load_active_sources(
    connection: asyncpg.Connection,
) -> tuple[asyncpg.Record, ...]:
    """Загружает активные RSS/Atom-источники."""

    records = await connection.fetch(
        """
        SELECT
            source_id,
            source_code,
            source_name,
            feed_url,
            base_url,
            default_language,
            collection_priority
        FROM top3_news.sources
        WHERE is_active = true
          AND feed_url IS NOT NULL
          AND BTRIM(feed_url) <> ''
          AND COALESCE(
                settings ->> 'collector',
                ''
              ) = 'rss_atom_http'
        ORDER BY
            collection_priority DESC,
            source_code
        """
    )

    return tuple(records)


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет один сбор всех активных лент."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print("Active feed collection refused")
        print(error)
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    pool: asyncpg.Pool | None = None

    try:
        pool = await create_database_pool(
            get_settings()
        )

        async with pool.acquire() as lock_connection:
            lock_acquired = await lock_connection.fetchval(
                """
                SELECT pg_try_advisory_lock($1)
                """,
                COLLECTOR_LOCK_ID,
            )

            if not lock_acquired:
                print("Active feed collection skipped")
                print(
                    "reason=another_collection_is_running"
                )
                print("Database changes: not performed")
                print("Telegram publication: not performed")
                return 0

            try:
                sources = await load_active_sources(
                    lock_connection
                )

                if not sources:
                    print("Active feed collection refused")
                    print(
                        "No active RSS/Atom sources "
                        "were found."
                    )
                    print(
                        "Database changes: not performed"
                    )
                    print(
                        "Telegram publication: "
                        "not performed"
                    )
                    return 2

                completed_count = 0
                failed_count = 0
                fetched_count = 0
                inserted_count = 0
                duplicate_count = 0
                rejected_count = 0

                print("Active feed collection started")
                print(
                    f"source_count={len(sources)}"
                )

                for source in sources:
                    source_code = str(
                        source["source_code"]
                    )

                    print()
                    print(
                        "Feed collection started: "
                        f"source_code={source_code}"
                    )

                    try:
                        result = await collect_feed(
                            pool,
                            source_code=source_code,
                            source_name=str(
                                source["source_name"]
                            ),
                            feed_url=str(
                                source["feed_url"]
                            ),
                            base_url=source["base_url"],
                            language_code=str(
                                source[
                                    "default_language"
                                ]
                                or "en"
                            ),
                            collection_priority=int(
                                source[
                                    "collection_priority"
                                ]
                            ),
                            max_entries=(
                                arguments.max_entries
                            ),
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
                        failed_count += 1

                        print("Feed collection failed")
                        print(
                            f"source_code={source_code}"
                        )
                        print(
                            f"error_type="
                            f"{type(error).__name__}"
                        )
                        print(f"error={error}")
                        continue

                    except Exception as error:
                        failed_count += 1

                        print("Feed collection failed")
                        print(
                            f"source_code={source_code}"
                        )
                        print(
                            f"error_type="
                            f"{type(error).__name__}"
                        )
                        print(f"error={error}")
                        continue

                    completed_count += 1
                    fetched_count += result.fetched_count
                    inserted_count += result.inserted_count
                    duplicate_count += (
                        result.duplicate_count
                    )
                    rejected_count += (
                        result.rejected_count
                    )

                    print("Feed collection completed")
                    print(
                        f"source_code={source_code}"
                    )
                    print(
                        "collection_run_id="
                        f"{result.collection_run_id}"
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

                print()
                print(
                    "Active feed collection summary"
                )
                print(
                    f"source_count={len(sources)}"
                )
                print(
                    f"completed_count={completed_count}"
                )
                print(
                    f"failed_count={failed_count}"
                )
                print(
                    f"fetched_count={fetched_count}"
                )
                print(
                    f"inserted_count={inserted_count}"
                )
                print(
                    f"duplicate_count={duplicate_count}"
                )
                print(
                    f"rejected_count={rejected_count}"
                )
                print(
                    "Telegram publication: "
                    "not performed"
                )

                if failed_count:
                    print(
                        "Active feed collection: "
                        "completed_with_errors"
                    )
                    return 1

                print(
                    "Active feed collection: OK"
                )
                return 0

            finally:
                await lock_connection.execute(
                    """
                    SELECT pg_advisory_unlock($1)
                    """,
                    COLLECTOR_LOCK_ID,
                )

    except asyncpg.PostgresError as error:
        print("Active feed collection failed")
        print(
            f"{type(error).__name__}: {error}"
        )
        print(
            "Telegram publication: not performed"
        )
        return 1

    finally:
        await close_database_pool(pool)


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )
