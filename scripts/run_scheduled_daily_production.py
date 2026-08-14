import asyncio
from datetime import datetime, timezone

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.workflows.daily_production import (
    run_daily_production_workflow,
)


CANDIDATE_LIMIT = 500


def _progress(
    message: str,
) -> None:
    """Печатает этап немедленно для systemd journal."""

    print(
        message,
        flush=True,
    )


async def async_main() -> int:
    """Запускает ежедневный production workflow по systemd timer."""

    as_of = datetime.now(
        timezone.utc
    )

    publication_date = as_of.date()

    settings = get_settings()

    _progress(
        "Scheduled daily production started"
    )
    _progress(
        "publication_date="
        f"{publication_date}"
    )
    _progress(
        "as_of="
        f"{as_of.isoformat()}"
    )
    _progress(
        "candidate_limit="
        f"{CANDIDATE_LIMIT}"
    )

    pool = await create_database_pool(
        settings
    )

    try:
        result = (
            await run_daily_production_workflow(
                pool,
                settings=settings,
                publication_date=(
                    publication_date
                ),
                as_of=as_of,
                candidate_limit=(
                    CANDIDATE_LIMIT
                ),
                progress=_progress,
            )
        )
    finally:
        await close_database_pool(
            pool
        )

    print(
        "Scheduled daily production: OK",
        flush=True,
    )
    print(
        "daily_workflow_run_id="
        f"{result.daily_workflow_run_id}",
        flush=True,
    )
    print(
        "publication_date="
        f"{result.publication_date}",
        flush=True,
    )
    print(
        "as_of="
        f"{result.as_of.isoformat()}",
        flush=True,
    )
    print(
        "workflow_status="
        f"{result.workflow_status}",
        flush=True,
    )
    print(
        "ranking_run_id="
        f"{result.ranking_run_id}",
        flush=True,
    )
    print(
        "batch_id="
        f"{result.batch_id}",
        flush=True,
    )
    print(
        "generated_post_id="
        f"{result.generated_post_id}",
        flush=True,
    )
    print(
        "image_generation_id="
        f"{result.image_generation_id}",
        flush=True,
    )
    print(
        "ranking_model_called="
        f"{str(result.ranking_model_called).lower()}",
        flush=True,
    )
    print(
        "generation_model_called="
        f"{str(result.generation_model_called).lower()}",
        flush=True,
    )
    print(
        "image_model_called="
        f"{str(result.image_model_called).lower()}",
        flush=True,
    )
    print(
        "reviewer_count="
        f"{result.reviewer_count}",
        flush=True,
    )
    print(
        "review_sent_count="
        f"{result.review_sent_count}",
        flush=True,
    )
    print(
        "review_failed_count="
        f"{result.review_failed_count}",
        flush=True,
    )
    print(
        "review_unknown_count="
        f"{result.review_unknown_count}",
        flush=True,
    )
    print(
        "review_skipped_count="
        f"{result.review_skipped_count}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            async_main()
        )
    )