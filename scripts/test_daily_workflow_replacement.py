import asyncio

import asyncpg

from app.config import get_settings
from app.db.daily_workflow_replacement import (
    replace_daily_workflow_after_image_moderation,
)
from app.db.daily_workflow_selection_attempts import (
    ensure_initial_daily_workflow_selection,
    load_active_daily_workflow_selection,
    load_daily_workflow_selection_attempts,
    load_used_daily_workflow_combination_ids,
)
from app.db.generation_selection import (
    choose_next_generation_combination,
    load_generation_combination,
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
    """Проверяет migration 015 и новый batch status."""

    async with pool.acquire() as connection:
        migration = await connection.fetchrow(
            """
            SELECT
                version,
                description
            FROM top3_news.schema_migrations
            WHERE version = '015'
            """
        )

        constraint_definition = (
            await connection.fetchval(
                """
                SELECT pg_get_constraintdef(
                    oid,
                    true
                )
                FROM pg_constraint
                WHERE conrelid =
                    'top3_news.publication_batches'
                    ::regclass
                  AND conname =
                    'publication_batches_status_chk'
                """
            )
        )

    if migration is None:
        raise AssertionError(
            "Migration 015 не применена."
        )

    if (
        constraint_definition is None
        or "superseded"
        not in constraint_definition
    ):
        raise AssertionError(
            "publication_batches_status_chk "
            "не разрешает superseded."
        )

    print(
        "Migration 015 applied: OK"
    )


async def _load_fixture(
    pool,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """
    Возвращает production fixture:
    workflow, batch, post, failed moderation image.
    """

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
        int(failed_image["batch_id"]),
        int(
            failed_image[
                "generated_post_id"
            ]
        ),
        int(
            failed_image[
                "image_generation_id"
            ]
        ),
    )


async def main() -> int:
    """Проверяет atomic workflow replacement в полном rollback."""

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
            batch_id,
            generated_post_id,
            failed_image_generation_id,
        ) = await _load_fixture(
            database_pool
        )

        async with database_pool.acquire() as connection:
            original_selection_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .daily_workflow_selection_attempts
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )
            )

        async with database_pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()

            pool = _SingleConnectionPool(
                connection
            )

            try:
                # ------------------------------------------------------------
                # Подготавливаем исторический production fixture к состоянию,
                # которое реально существует перед replacement transition.
                # Всё ниже откатывается внешней транзакцией.
                # ------------------------------------------------------------

                await connection.execute(
                    """
                    DELETE FROM
                        top3_news
                        .daily_workflow_selection_attempts
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )

                await connection.execute(
                    """
                    UPDATE top3_news.daily_workflow_runs
                    SET
                        workflow_status = 'running',
                        current_stage = 'image',
                        batch_id = $2,
                        generated_post_id = $3,
                        image_generation_id = $4,
                        error_type = NULL,
                        error_message = NULL,
                        finished_at = NULL
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                    batch_id,
                    generated_post_id,
                    failed_image_generation_id,
                )

                await connection.execute(
                    """
                    UPDATE top3_news.publication_batches
                    SET
                        batch_status = 'awaiting_review',
                        approved_at = NULL,
                        published_at = NULL,
                        approved_by_telegram_user_id = NULL,
                        error_message = NULL
                    WHERE batch_id = $1
                    """,
                    batch_id,
                )

                await connection.execute(
                    """
                    UPDATE top3_news.generated_posts
                    SET
                        post_status = 'awaiting_review',
                        image_path = NULL,
                        image_sha256 = NULL,
                        image_prompt = NULL,
                        image_model_name = NULL,
                        image_prompt_version = NULL
                    WHERE generated_post_id = $1
                      AND batch_id = $2
                    """,
                    generated_post_id,
                    batch_id,
                )

                await connection.execute(
                    """
                    DELETE FROM
                        top3_news.image_generation_requests
                    WHERE batch_id = $1
                      AND generated_post_id = $2
                      AND image_generation_id <> $3
                      AND image_status IN (
                            'reserved',
                            'completed'
                      )
                    """,
                    batch_id,
                    generated_post_id,
                    failed_image_generation_id,
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

                assert initial.attempt_number == 1
                assert initial.active is True
                assert (
                    initial.combination_id
                    == WINNER_COMBINATION_ID
                )

                print(
                    "Pre-replacement fixture: OK"
                )

                result = (
                    await
                    replace_daily_workflow_after_image_moderation(
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

                assert result.created_new is True
                assert (
                    result.source_combination_id
                    == WINNER_COMBINATION_ID
                )
                assert (
                    result.replacement_combination_id
                    == FIRST_REPLACEMENT_ID
                )
                assert (
                    result.superseded_batch_id
                    == batch_id
                )
                assert (
                    result.superseded_generated_post_id
                    == generated_post_id
                )

                print(
                    "Atomic workflow replacement: OK"
                )

                state = await connection.fetchrow(
                    """
                    SELECT
                        dw.workflow_status,
                        dw.current_stage,
                        dw.ranking_run_id,
                        dw.batch_id,
                        dw.generated_post_id,
                        dw.image_generation_id,
                        dw.error_type,
                        dw.error_message,
                        dw.finished_at,
                        b.batch_status,
                        gp.post_status
                    FROM top3_news.daily_workflow_runs AS dw
                    JOIN top3_news.publication_batches AS b
                      ON b.batch_id = $2
                    JOIN top3_news.generated_posts AS gp
                      ON gp.generated_post_id = $3
                    WHERE dw.daily_workflow_run_id = $1
                    """,
                    workflow_id,
                    batch_id,
                    generated_post_id,
                )

                if state is None:
                    raise AssertionError(
                        "Не удалось прочитать "
                        "post-replacement state."
                    )

                assert (
                    state["workflow_status"]
                    == "running"
                )
                assert (
                    state["current_stage"]
                    == "generation"
                )
                assert (
                    state["ranking_run_id"]
                    == RANKING_RUN_ID
                )
                assert state["batch_id"] is None
                assert (
                    state["generated_post_id"]
                    is None
                )
                assert (
                    state["image_generation_id"]
                    is None
                )
                assert state["error_type"] is None
                assert (
                    state["error_message"] is None
                )
                assert state["finished_at"] is None

                assert (
                    state["batch_status"]
                    == "superseded"
                )
                assert (
                    state["post_status"]
                    == "superseded"
                )

                print(
                    "Source batch/post superseded: OK"
                )
                print(
                    "Workflow reset to generation: OK"
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

                source = history[0]
                replacement = history[1]

                assert (
                    source.moderation_blocked
                    is True
                )
                assert source.batch_id == batch_id
                assert (
                    source.generated_post_id
                    == generated_post_id
                )
                assert (
                    source.image_generation_id
                    == failed_image_generation_id
                )
                assert source.ended_at is not None

                assert replacement.active is True
                assert (
                    replacement.combination_id
                    == FIRST_REPLACEMENT_ID
                )
                assert (
                    replacement.source_selection_attempt_id
                    == initial.selection_attempt_id
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
                        "Replacement active selection "
                        "не найдена."
                    )

                assert (
                    active.selection_attempt_id
                    == replacement.selection_attempt_id
                )

                print(
                    "Selection provenance: OK"
                )

                repeated = (
                    await
                    replace_daily_workflow_after_image_moderation(
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

                assert repeated.created_new is False
                assert (
                    repeated
                    .replacement_selection_attempt_id
                    == result
                    .replacement_selection_attempt_id
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

                replacement_combination = (
                    await load_generation_combination(
                        pool,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        combination_id=(
                            FIRST_REPLACEMENT_ID
                        ),
                    )
                )

                next_replacement = (
                    await
                    choose_next_generation_combination(
                        pool,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        current_news_ids=(
                            replacement_combination
                            .news_ids
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
                    "Restart continues with next "
                    "combination: OK"
                )
                print(
                    "next_combination_id="
                    f"{SECOND_REPLACEMENT_ID}"
                )

            finally:
                await transaction.rollback()

        async with database_pool.acquire() as connection:
            remaining_selection_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .daily_workflow_selection_attempts
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )
            )

        assert (
            int(remaining_selection_count)
            == int(original_selection_count)
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
            "Daily workflow replacement "
            "transition test: OK"
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