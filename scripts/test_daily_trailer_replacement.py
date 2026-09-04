import asyncio
import json

import asyncpg

from app.config import get_settings
from app.db.daily_workflow_selection_attempts import (
    ensure_initial_daily_workflow_selection,
)
from app.db.daily_workflow_trailer_replacement import (
    TRAILER_REJECTION_REASON,
    replace_daily_workflow_after_trailer_unverified,
)
from app.db.generation_selection import (
    choose_next_generation_combination,
    load_generation_combination,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


class _SingleConnectionAcquire:
    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> asyncpg.Connection:
        return self._connection

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None


class _SingleConnectionPool:
    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    def acquire(self) -> _SingleConnectionAcquire:
        return _SingleConnectionAcquire(self._connection)


async def _snapshot(pool: asyncpg.Pool, workflow_id: int) -> str:
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            SELECT jsonb_build_object(
                'workflow', to_jsonb(dw),
                'attempts', COALESCE((
                    SELECT jsonb_agg(to_jsonb(dws) ORDER BY dws.selection_attempt_id)
                    FROM top3_news.daily_workflow_selection_attempts AS dws
                    WHERE dws.daily_workflow_run_id = dw.daily_workflow_run_id
                ), '[]'::jsonb)
            )::text AS snapshot
            FROM top3_news.daily_workflow_runs AS dw
            WHERE dw.daily_workflow_run_id = $1
            """,
            workflow_id,
        )
    if row is None:
        raise AssertionError("Workflow fixture исчез.")
    return json.dumps(json.loads(row["snapshot"]), sort_keys=True)


async def main() -> int:
    settings = get_settings()
    pool = await create_database_pool(settings)

    try:
        async with pool.acquire() as connection:
            migration = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM top3_news.schema_migrations
                    WHERE version = '016'
                )
                """
            )
            if migration is not True:
                raise AssertionError("Migration 016 не применена.")

            fixture = await connection.fetchrow(
                """
                SELECT
                    dw.daily_workflow_run_id,
                    dw.ranking_run_id,
                    rc.combination_id
                FROM top3_news.daily_workflow_runs AS dw
                JOIN top3_news.ranking_runs AS rr
                  ON rr.ranking_run_id = dw.ranking_run_id
                JOIN top3_news.ranking_combinations AS rc
                  ON rc.ranking_run_id = rr.ranking_run_id
                 AND rc.is_winner = true
                WHERE rr.run_status = 'completed'
                  AND (
                      SELECT COUNT(*)
                      FROM top3_news.ranking_combinations AS rc2
                      WHERE rc2.ranking_run_id = rr.ranking_run_id
                  ) > 1
                ORDER BY dw.daily_workflow_run_id DESC
                LIMIT 1
                """
            )

        if fixture is None:
            raise AssertionError("Не найден rollback-safe workflow fixture.")

        workflow_id = int(fixture["daily_workflow_run_id"])
        ranking_run_id = int(fixture["ranking_run_id"])
        winner_combination_id = int(fixture["combination_id"])
        snapshot_before = await _snapshot(pool, workflow_id)

        async with pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()
            single_pool = _SingleConnectionPool(connection)

            try:
                await connection.execute(
                    """
                    DELETE FROM top3_news.daily_workflow_selection_attempts
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )

                await connection.execute(
                    """
                    UPDATE top3_news.daily_workflow_runs
                    SET
                        workflow_status = 'running',
                        current_stage = 'generation',
                        batch_id = NULL,
                        generated_post_id = NULL,
                        image_generation_id = NULL,
                        error_type = NULL,
                        error_message = NULL,
                        finished_at = NULL
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )

                initial = await ensure_initial_daily_workflow_selection(
                    single_pool,
                    daily_workflow_run_id=workflow_id,
                    combination_id=winner_combination_id,
                )

                current = await load_generation_combination(
                    single_pool,
                    ranking_run_id=ranking_run_id,
                    combination_id=winner_combination_id,
                )

                rejected_news_id = current.news_ids[1]
                replacement = await choose_next_generation_combination(
                    single_pool,
                    ranking_run_id=ranking_run_id,
                    current_news_ids=current.news_ids,
                    excluded_combination_ids=(winner_combination_id,),
                    excluded_news_ids=(rejected_news_id,),
                )

                if replacement is None:
                    raise AssertionError(
                        "Fixture не имеет replacement без rejected news_id."
                    )

                result = await replace_daily_workflow_after_trailer_unverified(
                    single_pool,
                    daily_workflow_run_id=workflow_id,
                    current_selection_attempt_id=initial.selection_attempt_id,
                    replacement_combination_id=(
                        replacement.combination.combination_id
                    ),
                    rejected_news_ids=(rejected_news_id,),
                )
                assert result.created_new is True

                source = await connection.fetchrow(
                    """
                    SELECT
                        selection_status,
                        rejection_reason,
                        rejected_news_ids
                    FROM top3_news.daily_workflow_selection_attempts
                    WHERE selection_attempt_id = $1
                    """,
                    initial.selection_attempt_id,
                )
                assert source["selection_status"] == "trailer_unverified"
                assert source["rejection_reason"] == TRAILER_REJECTION_REASON
                assert tuple(source["rejected_news_ids"]) == (rejected_news_id,)

                child = await connection.fetchrow(
                    """
                    SELECT selection_status, combination_id
                    FROM top3_news.daily_workflow_selection_attempts
                    WHERE selection_attempt_id = $1
                    """,
                    result.replacement_selection_attempt_id,
                )
                assert child["selection_status"] == "active"
                assert int(child["combination_id"]) == (
                    replacement.combination.combination_id
                )

                repeated = await replace_daily_workflow_after_trailer_unverified(
                    single_pool,
                    daily_workflow_run_id=workflow_id,
                    current_selection_attempt_id=initial.selection_attempt_id,
                    replacement_combination_id=(
                        replacement.combination.combination_id
                    ),
                    rejected_news_ids=(rejected_news_id,),
                )
                assert repeated.created_new is False
                assert (
                    repeated.replacement_selection_attempt_id
                    == result.replacement_selection_attempt_id
                )

                print("Trailer selection replacement: OK")
                print("Replacement idempotency: OK")
                print(f"fixture_workflow_id={workflow_id}")
                print(f"rejected_news_id={rejected_news_id}")
                print(
                    "replacement_combination_id="
                    f"{replacement.combination.combination_id}"
                )

            finally:
                await transaction.rollback()

        snapshot_after = await _snapshot(pool, workflow_id)
        if snapshot_after != snapshot_before:
            raise AssertionError("Production fixture изменился после rollback.")

        print("Production fixture restored after rollback: OK")
        print("Database changes=rolled_back")
        print("OpenAI requests=not_performed")
        print("Telegram requests=not_performed")
        print("Daily trailer replacement test: OK")
        return 0

    finally:
        await close_database_pool(pool)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
