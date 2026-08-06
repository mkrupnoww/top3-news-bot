import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


CLEANUP_LOCK_ID = 7_320_032

DEFAULT_NEWS_RETENTION_DAYS = 7
DEFAULT_RANKING_RETENTION_DAYS = 7
DEFAULT_COLLECTION_RUN_RETENTION_DAYS = 30

TERMINAL_RUN_STATUSES = (
    "completed",
    "failed",
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры очистки старых данных."""

    parser = argparse.ArgumentParser(
        description=(
            "Удаляет устаревшие служебные данные "
            "TOP 3 NEWS. По умолчанию работает "
            "в безопасном режиме dry-run."
        ),
    )

    parser.add_argument(
        "--news-retention-days",
        type=int,
        default=DEFAULT_NEWS_RETENTION_DAYS,
        help=(
            "Срок хранения непубликованных новостей. "
            "По умолчанию: 7 суток."
        ),
    )

    parser.add_argument(
        "--ranking-retention-days",
        type=int,
        default=DEFAULT_RANKING_RETENTION_DAYS,
        help=(
            "Срок хранения завершённых запусков "
            "ранжирования. По умолчанию: 7 суток."
        ),
    )

    parser.add_argument(
        "--collection-run-retention-days",
        type=int,
        default=(
            DEFAULT_COLLECTION_RUN_RETENTION_DAYS
        ),
        help=(
            "Срок хранения журналов сборщика. "
            "По умолчанию: 30 суток."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Выполнить реальное удаление. "
            "Без этого флага изменения "
            "не производятся."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет сроки хранения."""

    for field_name, value in (
        (
            "news_retention_days",
            arguments.news_retention_days,
        ),
        (
            "ranking_retention_days",
            arguments.ranking_retention_days,
        ),
        (
            "collection_run_retention_days",
            arguments.collection_run_retention_days,
        ),
    ):
        if value <= 0:
            option_name = field_name.replace(
                "_",
                "-",
            )

            raise ValueError(
                f"--{option_name} должен быть "
                "больше нуля."
            )


def calculate_cutoffs(
    arguments: argparse.Namespace,
) -> tuple[
    datetime,
    datetime,
    datetime,
    datetime,
]:
    """Вычисляет единое время запуска и границы."""

    as_of = datetime.now(timezone.utc)

    news_cutoff = as_of - timedelta(
        days=arguments.news_retention_days,
    )

    ranking_cutoff = as_of - timedelta(
        days=arguments.ranking_retention_days,
    )

    collection_run_cutoff = as_of - timedelta(
        days=(
            arguments.collection_run_retention_days
        ),
    )

    return (
        as_of,
        news_cutoff,
        ranking_cutoff,
        collection_run_cutoff,
    )


async def load_cleanup_preview(
    connection: asyncpg.Connection,
    *,
    news_cutoff: datetime,
    ranking_cutoff: datetime,
    collection_run_cutoff: datetime,
) -> asyncpg.Record:
    """
    Рассчитывает ожидаемый результат очистки.

    При подсчёте новостей учитывается, что старые
    ranking_runs будут удалены раньше news_items.
    """

    return await connection.fetchrow(
        """
        WITH deletable_ranking_runs AS (
            SELECT
                rr.ranking_run_id
            FROM top3_news.ranking_runs AS rr
            WHERE rr.run_status = ANY($1::text[])
              AND COALESCE(
                    rr.finished_at,
                    rr.updated_at,
                    rr.started_at
                  ) < $2
              AND NOT EXISTS (
                    SELECT 1
                    FROM top3_news.publication_batches
                        AS pb
                    WHERE pb.ranking_run_id
                          = rr.ranking_run_id
                      AND pb.published_at IS NULL
              )
        ),
        deletable_news_items AS (
            SELECT
                n.news_id
            FROM top3_news.news_items AS n
            WHERE COALESCE(
                    n.source_published_at,
                    n.collected_at,
                    n.created_at
                  ) < $3
              AND NOT EXISTS (
                    SELECT 1
                    FROM top3_news.batch_items AS bi
                    WHERE bi.news_id = n.news_id
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM top3_news.ranking_event_members
                        AS rem
                    WHERE rem.news_id = n.news_id
                      AND NOT EXISTS (
                            SELECT 1
                            FROM deletable_ranking_runs
                                AS drr
                            WHERE drr.ranking_run_id
                                  = rem.ranking_run_id
                      )
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM top3_news.ranking_events AS re
                    WHERE re.representative_news_id
                          = n.news_id
                      AND NOT EXISTS (
                            SELECT 1
                            FROM deletable_ranking_runs
                                AS drr
                            WHERE drr.ranking_run_id
                                  = re.ranking_run_id
                      )
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM top3_news.news_scores AS ns
                    WHERE ns.news_id = n.news_id
                      AND NOT EXISTS (
                            SELECT 1
                            FROM deletable_ranking_runs
                                AS drr
                            WHERE drr.ranking_run_id
                                  = ns.ranking_run_id
                      )
              )
        )
        SELECT
            (
                SELECT COUNT(*)
                FROM top3_news.news_items
            ) AS total_news_items,
            (
                SELECT COUNT(*)
                FROM top3_news.news_items AS n
                WHERE COALESCE(
                        n.source_published_at,
                        n.collected_at,
                        n.created_at
                      ) < $3
            ) AS old_news_items,
            (
                SELECT COUNT(*)
                FROM deletable_news_items
            ) AS news_items_to_delete,
            (
                SELECT COUNT(*)
                FROM deletable_ranking_runs
            ) AS ranking_runs_to_delete,
            (
                SELECT COUNT(*)
                FROM top3_news.collection_runs AS cr
                WHERE cr.run_status = ANY($1::text[])
                  AND COALESCE(
                        cr.finished_at,
                        cr.updated_at,
                        cr.started_at
                      ) < $4
            ) AS collection_runs_to_delete
        """,
        list(TERMINAL_RUN_STATUSES),
        ranking_cutoff,
        news_cutoff,
        collection_run_cutoff,
    )


def print_configuration(
    *,
    mode: str,
    as_of: datetime,
    news_cutoff: datetime,
    ranking_cutoff: datetime,
    collection_run_cutoff: datetime,
) -> None:
    """Печатает режим и рассчитанные границы."""

    print("Old data cleanup started")
    print(f"mode={mode}")
    print(f"as_of={as_of.isoformat()}")
    print(
        f"news_cutoff={news_cutoff.isoformat()}"
    )
    print(
        "ranking_cutoff="
        f"{ranking_cutoff.isoformat()}"
    )
    print(
        "collection_run_cutoff="
        f"{collection_run_cutoff.isoformat()}"
    )


def print_preview(
    preview: asyncpg.Record,
) -> None:
    """Печатает ожидаемый объём очистки."""

    print()
    print("Old data cleanup preview")
    print(
        "total_news_items="
        f"{preview['total_news_items']}"
    )
    print(
        f"old_news_items="
        f"{preview['old_news_items']}"
    )
    print(
        "news_items_to_delete="
        f"{preview['news_items_to_delete']}"
    )
    print(
        "ranking_runs_to_delete="
        f"{preview['ranking_runs_to_delete']}"
    )
    print(
        "collection_runs_to_delete="
        f"{preview['collection_runs_to_delete']}"
    )


async def execute_cleanup(
    connection: asyncpg.Connection,
    *,
    news_cutoff: datetime,
    ranking_cutoff: datetime,
    collection_run_cutoff: datetime,
) -> tuple[int, int, int] | None:
    """Удаляет старые данные одной транзакцией."""

    async with connection.transaction():
        lock_acquired = await connection.fetchval(
            """
            SELECT pg_try_advisory_xact_lock($1)
            """,
            CLEANUP_LOCK_ID,
        )

        if not lock_acquired:
            return None

        deleted_ranking_runs = await connection.fetchval(
            """
            WITH deleted AS (
                DELETE FROM top3_news.ranking_runs
                    AS rr
                WHERE rr.run_status
                      = ANY($1::text[])
                  AND COALESCE(
                        rr.finished_at,
                        rr.updated_at,
                        rr.started_at
                      ) < $2
                  AND NOT EXISTS (
                        SELECT 1
                        FROM top3_news.publication_batches
                            AS pb
                        WHERE pb.ranking_run_id
                              = rr.ranking_run_id
                          AND pb.published_at IS NULL
                  )
                RETURNING rr.ranking_run_id
            )
            SELECT COUNT(*)
            FROM deleted
            """,
            list(TERMINAL_RUN_STATUSES),
            ranking_cutoff,
        )

        deleted_news_items = await connection.fetchval(
            """
            WITH deleted AS (
                DELETE FROM top3_news.news_items AS n
                WHERE COALESCE(
                        n.source_published_at,
                        n.collected_at,
                        n.created_at
                      ) < $1
                  AND NOT EXISTS (
                        SELECT 1
                        FROM top3_news.batch_items AS bi
                        WHERE bi.news_id = n.news_id
                  )
                  AND NOT EXISTS (
                        SELECT 1
                        FROM top3_news.ranking_event_members
                            AS rem
                        WHERE rem.news_id = n.news_id
                  )
                  AND NOT EXISTS (
                        SELECT 1
                        FROM top3_news.ranking_events AS re
                        WHERE re.representative_news_id
                              = n.news_id
                  )
                  AND NOT EXISTS (
                        SELECT 1
                        FROM top3_news.news_scores AS ns
                        WHERE ns.news_id = n.news_id
                  )
                RETURNING n.news_id
            )
            SELECT COUNT(*)
            FROM deleted
            """,
            news_cutoff,
        )

        deleted_collection_runs = (
            await connection.fetchval(
                """
                WITH deleted AS (
                    DELETE
                    FROM top3_news.collection_runs
                        AS cr
                    WHERE cr.run_status
                          = ANY($1::text[])
                      AND COALESCE(
                            cr.finished_at,
                            cr.updated_at,
                            cr.started_at
                          ) < $2
                    RETURNING cr.collection_run_id
                )
                SELECT COUNT(*)
                FROM deleted
                """,
                list(TERMINAL_RUN_STATUSES),
                collection_run_cutoff,
            )
        )

        return (
            int(deleted_ranking_runs),
            int(deleted_news_items),
            int(deleted_collection_runs),
        )


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет preview или реальную очистку."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print("Old data cleanup refused")
        print(error)
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    (
        as_of,
        news_cutoff,
        ranking_cutoff,
        collection_run_cutoff,
    ) = calculate_cutoffs(arguments)

    mode = (
        "execute"
        if arguments.execute
        else "dry_run"
    )

    print_configuration(
        mode=mode,
        as_of=as_of,
        news_cutoff=news_cutoff,
        ranking_cutoff=ranking_cutoff,
        collection_run_cutoff=(
            collection_run_cutoff
        ),
    )

    pool: asyncpg.Pool | None = None

    try:
        pool = await create_database_pool(
            get_settings()
        )

        async with pool.acquire() as connection:
            preview = await load_cleanup_preview(
                connection,
                news_cutoff=news_cutoff,
                ranking_cutoff=ranking_cutoff,
                collection_run_cutoff=(
                    collection_run_cutoff
                ),
            )

            print_preview(preview)

            if not arguments.execute:
                print()
                print(
                    "Database changes: not performed"
                )
                print(
                    "Telegram publication: "
                    "not performed"
                )
                print("Old data cleanup dry-run: OK")
                return 0

            result = await execute_cleanup(
                connection,
                news_cutoff=news_cutoff,
                ranking_cutoff=ranking_cutoff,
                collection_run_cutoff=(
                    collection_run_cutoff
                ),
            )

            if result is None:
                print()
                print("Old data cleanup skipped")
                print(
                    "reason=another_cleanup_is_running"
                )
                print(
                    "Database changes: not performed"
                )
                print(
                    "Telegram publication: "
                    "not performed"
                )
                return 0

            (
                deleted_ranking_runs,
                deleted_news_items,
                deleted_collection_runs,
            ) = result

            print()
            print("Old data cleanup result")
            print(
                "deleted_ranking_runs="
                f"{deleted_ranking_runs}"
            )
            print(
                "deleted_news_items="
                f"{deleted_news_items}"
            )
            print(
                "deleted_collection_runs="
                f"{deleted_collection_runs}"
            )
            print(
                "Database changes: old data deleted"
            )
            print(
                "Telegram publication: not performed"
            )
            print("Old data cleanup: OK")
            return 0

    except asyncpg.PostgresError as error:
        print()
        print("Old data cleanup failed")
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