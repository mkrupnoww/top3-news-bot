import asyncio

import asyncpg

from app.config import get_settings
from app.db.daily_workflow_selection_attempts import (
    ensure_initial_daily_workflow_selection,
    load_active_daily_workflow_selection,
    load_daily_workflow_selection_attempts,
    load_used_daily_workflow_combination_ids,
    replace_daily_workflow_selection_after_moderation,
)
from app.db.generation_selection import (
    choose_next_generation_combination,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


RANKING_RUN_ID = 142
EXPECTED_WORKFLOW_ID = 16

WINNER_COMBINATION_ID = 1844
FIRST_REPLACEMENT_ID = 1845
SECOND_REPLACEMENT_ID = 1846

WINNER_NEWS_IDS = (
    1029,
    1037,
    986,
)


class _SingleConnectionAcquire:
    """Context manager одной connection."""

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    async def __aenter__(
        self,
    ) -> asyncpg.Connection:
        return self._connection

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None


class _SingleConnectionPool:
    """Pool-like wrapper одной connection для rollback test."""

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    def acquire(
        self,
    ) -> _SingleConnectionAcquire:
        return _SingleConnectionAcquire(
            self._connection
        )


async def _assert_migration_applied(
    pool: asyncpg.Pool,
) -> None:
    """Проверяет migration 014."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                version,
                description
            FROM top3_news.schema_migrations
            WHERE version = '014'
            """
        )

    if record is None:
        raise AssertionError(
            "Migration 014 не применена."
        )

    print(
        "Migration 014 applied: OK"
    )


async def _load_production_fixture(
    pool,
) -> tuple[int, int]:
    """Читает workflow 142 и один definitive moderation block."""

    async with pool.acquire() as connection:
        workflow = await connection.fetchrow(
            """
            SELECT
                daily_workflow_run_id,
                ranking_run_id,
                batch_id,
                generated_post_id
            FROM top3_news.daily_workflow_runs
            WHERE ranking_run_id = $1
            """,
            RANKING_RUN_ID,
        )

        if workflow is None:
            raise AssertionError(
                "Production workflow fixture "
                "ranking_run_id=142 не найден."
            )

        workflow_id = int(
            workflow["daily_workflow_run_id"]
        )

        if workflow_id != EXPECTED_WORKFLOW_ID:
            raise AssertionError(
                "Неожиданный production workflow: "
                f"{workflow_id}"
            )

        failed_image = await connection.fetchrow(
            """
            SELECT
                igr.image_generation_id,
                igr.batch_id,
                igr.generated_post_id
            FROM
                top3_news
                .image_generation_requests AS igr
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = igr.batch_id
            WHERE b.ranking_run_id = $1
              AND igr.request_kind = 'initial'
              AND igr.image_status = 'failed'
              AND igr.failed_at IS NOT NULL
              AND igr.error_type = 'BadRequestError'
              AND lower(
                    COALESCE(
                        igr.error_message,
                        ''
                    )
                  ) LIKE '%moderation_blocked%'
            ORDER BY igr.image_generation_id
            LIMIT 1
            """,
            RANKING_RUN_ID,
        )

    if failed_image is None:
        raise AssertionError(
            "Для ranking_run_id=142 не найден "
            "definitive moderation_blocked image."
        )

    return (
        workflow_id,
        int(
            failed_image[
                "image_generation_id"
            ]
        ),
    )


async def main() -> int:
    """Проверяет restart-safe selection history без OpenAI/Telegram."""

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        await _assert_migration_applied(
            database_pool
        )

        (
            workflow_id,
            failed_image_generation_id,
        ) = await _load_production_fixture(
            database_pool
        )

        async with database_pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()

            pool = _SingleConnectionPool(
                connection
            )

            try:
                # Тест полностью откатывается. Удаление здесь
                # позволяет тесту оставаться повторяемым даже
                # после будущего production использования таблицы.
                await connection.execute(
                    """
                    DELETE FROM
                        top3_news
                        .daily_workflow_selection_attempts
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )

                initial = (
                    await
                    ensure_initial_daily_workflow_selection(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        combination_id=(
                            WINNER_COMBINATION_ID
                        ),
                    )
                )

                assert initial.created_new is True
                assert initial.attempt_number == 1
                assert initial.selection_kind == "winner"
                assert initial.active is True
                assert (
                    initial.combination_id
                    == WINNER_COMBINATION_ID
                )

                print(
                    "Initial winner selection: OK"
                )

                repeated_initial = (
                    await
                    ensure_initial_daily_workflow_selection(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        combination_id=(
                            WINNER_COMBINATION_ID
                        ),
                    )
                )

                assert (
                    repeated_initial.selection_attempt_id
                    == initial.selection_attempt_id
                )
                assert (
                    repeated_initial.created_new
                    is False
                )

                print(
                    "Initial selection idempotency: OK"
                )

                replacement = (
                    await
                    replace_daily_workflow_selection_after_moderation(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        current_selection_attempt_id=(
                            initial.selection_attempt_id
                        ),
                        replacement_combination_id=(
                            FIRST_REPLACEMENT_ID
                        ),
                        failed_image_generation_id=(
                            failed_image_generation_id
                        ),
                    )
                )

                assert replacement.created_new is True
                assert replacement.attempt_number == 2
                assert (
                    replacement.selection_kind
                    == "replacement"
                )
                assert replacement.active is True
                assert (
                    replacement.source_selection_attempt_id
                    == initial.selection_attempt_id
                )
                assert (
                    replacement.combination_id
                    == FIRST_REPLACEMENT_ID
                )

                print(
                    "Atomic moderation replacement: OK"
                )

                history = (
                    await
                    load_daily_workflow_selection_attempts(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                    )
                )

                assert len(history) == 2

                blocked = history[0]

                assert blocked.moderation_blocked is True
                assert (
                    blocked.image_generation_id
                    == failed_image_generation_id
                )
                assert blocked.ended_at is not None

                assert (
                    history[1].selection_attempt_id
                    == replacement.selection_attempt_id
                )

                active = (
                    await
                    load_active_daily_workflow_selection(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                    )
                )

                if active is None:
                    raise AssertionError(
                        "Active replacement selection "
                        "не найдена."
                    )

                assert (
                    active.selection_attempt_id
                    == replacement.selection_attempt_id
                )

                print(
                    "Single active selection: OK"
                )

                repeated_replacement = (
                    await
                    replace_daily_workflow_selection_after_moderation(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        current_selection_attempt_id=(
                            initial.selection_attempt_id
                        ),
                        replacement_combination_id=(
                            FIRST_REPLACEMENT_ID
                        ),
                        failed_image_generation_id=(
                            failed_image_generation_id
                        ),
                    )
                )

                assert (
                    repeated_replacement
                    .selection_attempt_id
                    == replacement.selection_attempt_id
                )
                assert (
                    repeated_replacement.created_new
                    is False
                )

                print(
                    "Replacement idempotency: OK"
                )

                used_ids = (
                    await
                    load_used_daily_workflow_combination_ids(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                    )
                )

                assert used_ids == (
                    WINNER_COMBINATION_ID,
                    FIRST_REPLACEMENT_ID,
                )

                print(
                    "Used combination history: OK"
                )

                next_replacement = (
                    await
                    choose_next_generation_combination(
                        pool,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        current_news_ids=(
                            WINNER_NEWS_IDS
                        ),
                        excluded_combination_ids=(
                            used_ids
                        ),
                    )
                )

                if next_replacement is None:
                    raise AssertionError(
                        "Следующая replacement "
                        "combination не найдена."
                    )

                assert (
                    next_replacement
                    .combination
                    .combination_id
                    == SECOND_REPLACEMENT_ID
                )

                print(
                    "Restart-safe exclusion feeds "
                    "next selector: OK"
                )
                print(
                    "next_combination_id="
                    f"{SECOND_REPLACEMENT_ID}"
                )

            finally:
                await transaction.rollback()

        async with database_pool.acquire() as connection:
            remaining = await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM
                    top3_news
                    .daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
            )

        if int(remaining) != 0:
            raise AssertionError(
                "Rollback test оставил selection "
                "history в production БД."
            )

        print()
        print(
            "Database changes=rolled_back"
        )
        print(
            "OpenAI requests=not_performed"
        )
        print(
            "Telegram requests=not_performed"
        )
        print(
            "Daily workflow selection history "
            "test: OK"
        )

        return 0

    finally:
        await close_database_pool(
            database_pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )