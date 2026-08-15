import asyncio

import asyncpg

from app.config import get_settings
from app.db.daily_workflow_replacement import (
    replace_daily_workflow_after_image_moderation,
)
from app.db.daily_workflow_selection_attempts import (
    load_daily_workflow_selection_attempts,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.workflows.daily_production import (
    MAX_TOP3_REPLACEMENTS,
    _choose_workflow_replacement,
    _resolve_active_selection,
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

FIRST_REPLACEMENT_NEWS_IDS = (
    1029,
    1030,
    986,
)


class _SingleConnectionAcquire:
    """Context manager одной asyncpg connection."""

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
    """Pool-like wrapper одной connection для полного rollback."""

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


async def _load_fixture(
    pool: asyncpg.Pool,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """
    Возвращает historical production fixture:
    workflow, batch, generated_post и moderation-blocked image.
    """

    async with pool.acquire() as connection:
        workflow = await connection.fetchrow(
            """
            SELECT
                daily_workflow_run_id,
                batch_id,
                generated_post_id
            FROM top3_news.daily_workflow_runs
            WHERE ranking_run_id = $1
            """,
            RANKING_RUN_ID,
        )

        if workflow is None:
            raise AssertionError(
                "Workflow fixture для ranking_run_id=142 "
                "не найден."
            )

        workflow_id = int(
            workflow["daily_workflow_run_id"]
        )

        if workflow_id != EXPECTED_WORKFLOW_ID:
            raise AssertionError(
                "Неожиданный workflow fixture: "
                f"{workflow_id}"
            )

        failed_image = await connection.fetchrow(
            """
            SELECT
                igr.image_generation_id,
                igr.batch_id,
                igr.generated_post_id
            FROM
                top3_news.image_generation_requests
                AS igr
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
            "Для ranking_run_id=142 "
            "не найден moderation_blocked image."
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


async def _prepare_replacement_state(
    connection: asyncpg.Connection,
    *,
    workflow_id: int,
    batch_id: int,
    generated_post_id: int,
    failed_image_generation_id: int,
) -> None:
    """
    Внешней rollback-транзакцией восстанавливает исторический fixture
    в точное pre-replacement состояние нового orchestrator.
    """

    await connection.execute(
        """
        DELETE FROM
            top3_news.daily_workflow_selection_attempts
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

    # Historical run может уже иметь successful later image.
    # Для pre-replacement fixture оставляем только failed attempts.
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


async def main() -> int:
    """Проверяет orchestration-level replacement без OpenAI/Telegram."""

    if MAX_TOP3_REPLACEMENTS != 3:
        raise AssertionError(
            "Неожиданный production replacement limit: "
            f"{MAX_TOP3_REPLACEMENTS}"
        )

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
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
                await _prepare_replacement_state(
                    connection,
                    workflow_id=workflow_id,
                    batch_id=batch_id,
                    generated_post_id=(
                        generated_post_id
                    ),
                    failed_image_generation_id=(
                        failed_image_generation_id
                    ),
                )

                (
                    active_winner,
                    winner_selection,
                ) = await _resolve_active_selection(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                    ranking_run_id=(
                        RANKING_RUN_ID
                    ),
                )

                assert (
                    active_winner.combination_id
                    == WINNER_COMBINATION_ID
                )
                assert (
                    active_winner.attempt_number
                    == 1
                )
                assert (
                    winner_selection.news_ids
                    == WINNER_NEWS_IDS
                )

                print(
                    "Orchestrator resolves winner selection: OK"
                )

                first_candidate = (
                    await _choose_workflow_replacement(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        active_selection=(
                            active_winner
                        ),
                        current_selection=(
                            winner_selection
                        ),
                    )
                )

                if first_candidate is None:
                    raise AssertionError(
                        "Первый replacement candidate "
                        "не найден."
                    )

                assert (
                    first_candidate
                    .combination
                    .combination_id
                    == FIRST_REPLACEMENT_ID
                )
                assert (
                    first_candidate.overlap_count
                    == 2
                )
                assert (
                    first_candidate.removed_news_ids
                    == (1037,)
                )
                assert (
                    first_candidate.added_news_ids
                    == (1030,)
                )

                print(
                    "Orchestrator chooses "
                    "overlap=2 replacement: OK"
                )

                transition = (
                    await
                    replace_daily_workflow_after_image_moderation(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        current_selection_attempt_id=(
                            active_winner
                            .selection_attempt_id
                        ),
                        replacement_combination_id=(
                            FIRST_REPLACEMENT_ID
                        ),
                        failed_image_generation_id=(
                            failed_image_generation_id
                        ),
                    )
                )

                assert transition.created_new is True
                assert (
                    transition
                    .replacement_combination_id
                    == FIRST_REPLACEMENT_ID
                )

                (
                    active_replacement,
                    replacement_selection,
                ) = await _resolve_active_selection(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                    ranking_run_id=(
                        RANKING_RUN_ID
                    ),
                )

                assert (
                    active_replacement
                    .selection_attempt_id
                    == transition
                    .replacement_selection_attempt_id
                )
                assert (
                    active_replacement.combination_id
                    == FIRST_REPLACEMENT_ID
                )
                assert (
                    active_replacement.attempt_number
                    == 2
                )
                assert (
                    replacement_selection.news_ids
                    == FIRST_REPLACEMENT_NEWS_IDS
                )

                print(
                    "Restart resolves active "
                    "replacement selection: OK"
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
                        b.batch_status,
                        gp.post_status
                    FROM
                        top3_news.daily_workflow_runs AS dw
                    JOIN
                        top3_news.publication_batches AS b
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
                        "Post-replacement state "
                        "не найден."
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
                assert (
                    state["batch_status"]
                    == "superseded"
                )
                assert (
                    state["post_status"]
                    == "superseded"
                )

                print(
                    "Workflow ready for replacement "
                    "text generation: OK"
                )

                second_candidate = (
                    await _choose_workflow_replacement(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        active_selection=(
                            active_replacement
                        ),
                        current_selection=(
                            replacement_selection
                        ),
                    )
                )

                if second_candidate is None:
                    raise AssertionError(
                        "Второй replacement candidate "
                        "не найден."
                    )

                assert (
                    second_candidate
                    .combination
                    .combination_id
                    == SECOND_REPLACEMENT_ID
                )

                print(
                    "Cascade next combination: OK"
                )
                print(
                    "next_combination_id="
                    f"{SECOND_REPLACEMENT_ID}"
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
                assert (
                    history[0].combination_id
                    == WINNER_COMBINATION_ID
                )
                assert (
                    history[0].moderation_blocked
                    is True
                )
                assert (
                    history[1].combination_id
                    == FIRST_REPLACEMENT_ID
                )
                assert history[1].active is True

                print(
                    "Restart-safe selection history: OK"
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
            "Daily production replacement "
            "cascade integration test: OK"
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