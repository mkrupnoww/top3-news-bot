from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.workflows.daily_production import (
    run_daily_production_workflow,
)


EXPECTED_ERROR_PREFIX = (
    "Integrity gate не пройден после "
    "ограниченного числа revision-попыток:"
)


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    daily_workflow_run_id: int
    publication_date: date
    as_of: datetime
    ranking_run_id: int
    failed_batch_id: int


def _progress(message: str) -> None:
    print(message, flush=True)


async def _reopen_failed_integrity_workflow(
    pool,
    *,
    daily_workflow_run_id: int,
) -> RecoveryContext:
    """
    Атомарно reopen'ит только доказанный старый
    text-integrity exhaustion до generated_post/image.
    """

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await connection.fetchrow(
                """
                SELECT
                    daily_workflow_run_id,
                    publication_date,
                    workflow_status,
                    current_stage,
                    as_of,
                    ranking_run_id,
                    batch_id,
                    generated_post_id,
                    image_generation_id,
                    error_type,
                    error_message
                FROM top3_news.daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                daily_workflow_run_id,
            )

            if workflow is None:
                raise LookupError("daily_workflow_run не найден.")

            if (
                workflow["workflow_status"] != "failed"
                or workflow["current_stage"] != "failed"
            ):
                raise ValueError(
                    "Recovery разрешён только для failed/failed workflow."
                )

            error_type = str(workflow["error_type"] or "").strip()
            error_message = str(workflow["error_message"] or "").strip()

            if error_type != "ValueError":
                raise ValueError(
                    "Recovery запрещён: error_type не ValueError: "
                    f"{error_type!r}"
                )

            if not error_message.startswith(EXPECTED_ERROR_PREFIX):
                raise ValueError(
                    "Recovery запрещён: failure не является старым "
                    "text-integrity exhaustion."
                )

            if workflow["ranking_run_id"] is None:
                raise ValueError("Workflow не содержит ranking_run_id.")
            if workflow["batch_id"] is None:
                raise ValueError("Workflow не содержит failed batch_id.")
            if workflow["generated_post_id"] is not None:
                raise ValueError(
                    "Recovery запрещён: generated_post_id уже существует."
                )
            if workflow["image_generation_id"] is not None:
                raise ValueError(
                    "Recovery запрещён: image_generation_id уже существует."
                )

            failed_batch_id = int(workflow["batch_id"])

            batch = await connection.fetchrow(
                """
                SELECT batch_id, ranking_run_id, batch_status
                FROM top3_news.publication_batches
                WHERE batch_id = $1
                FOR UPDATE
                """,
                failed_batch_id,
            )

            if batch is None:
                raise LookupError("Failed publication batch не найден.")
            if batch["batch_status"] != "failed":
                raise ValueError(
                    "Recovery запрещён: связанный batch не failed."
                )
            if int(batch["ranking_run_id"]) != int(workflow["ranking_run_id"]):
                raise ValueError(
                    "Recovery запрещён: batch связан с другим ranking_run."
                )

            generated_post_count = await connection.fetchval(
                """
                SELECT count(*)
                FROM top3_news.generated_posts
                WHERE batch_id = $1
                """,
                failed_batch_id,
            )

            if int(generated_post_count) != 0:
                raise ValueError(
                    "Recovery запрещён: failed batch уже содержит generated_post."
                )

            updated = await connection.execute(
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
                  AND workflow_status = 'failed'
                  AND current_stage = 'failed'
                """,
                daily_workflow_run_id,
            )

            if updated != "UPDATE 1":
                raise RuntimeError("Не удалось reopen daily workflow.")

            return RecoveryContext(
                daily_workflow_run_id=int(workflow["daily_workflow_run_id"]),
                publication_date=workflow["publication_date"],
                as_of=workflow["as_of"],
                ranking_run_id=int(workflow["ranking_run_id"]),
                failed_batch_id=failed_batch_id,
            )


async def async_main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a daily workflow failed specifically by the old "
            "terminal text-integrity gate."
        )
    )
    parser.add_argument(
        "--daily-workflow-run-id",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    if args.daily_workflow_run_id <= 0:
        raise ValueError("daily-workflow-run-id должен быть больше нуля.")

    settings = get_settings()
    pool = await create_database_pool(settings)

    try:
        context = await _reopen_failed_integrity_workflow(
            pool,
            daily_workflow_run_id=args.daily_workflow_run_id,
        )

        _progress("Text integrity workflow reopened")
        _progress(f"daily_workflow_run_id={context.daily_workflow_run_id}")
        _progress(f"publication_date={context.publication_date}")
        _progress(f"as_of={context.as_of.isoformat()}")
        _progress(f"ranking_run_id={context.ranking_run_id}")
        _progress(f"historical_failed_batch_id={context.failed_batch_id}")

        result = await run_daily_production_workflow(
            pool,
            settings=settings,
            publication_date=context.publication_date,
            as_of=context.as_of,
            candidate_limit=args.candidate_limit,
            progress=_progress,
        )

        _progress("Recovery workflow: OK")
        _progress(f"result_status={result.workflow_status}")
        return 0
    finally:
        await close_database_pool(pool)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
