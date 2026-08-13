import asyncio

from app.config import get_settings
from app.db.daily_workflow_state import (
    load_generation_workflow_state,
    load_ranking_workflow_state,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


RANKING_RUN_ID = 137
SUCCESS_BATCH_ID = 60
FAILED_BATCH_ID = 59


async def main() -> int:
    """Проверяет resume-state на production данных."""

    print("Daily workflow state probe")
    print("Database changes=not_performed")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print()

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    try:
        ranking = (
            await load_ranking_workflow_state(
                pool,
                ranking_run_id=(
                    RANKING_RUN_ID
                ),
            )
        )

        print("Ranking state:")
        print(
            f"ranking_run_id="
            f"{ranking.ranking_run_id}"
        )
        print(
            f"run_status="
            f"{ranking.run_status}"
        )
        print(
            "ready_for_generation="
            f"{ranking.ready_for_generation}"
        )
        print(
            f"top3_news_ids="
            f"{ranking.top3_news_ids}"
        )

        assert ranking.run_status == "completed"
        assert ranking.ready_for_generation is True
        assert len(ranking.top3_news_ids) == 3

        print("Completed ranking resume: OK")
        print()

        generation = (
            await load_generation_workflow_state(
                pool,
                batch_id=SUCCESS_BATCH_ID,
            )
        )

        print("Successful generation state:")
        print(
            f"batch_id={generation.batch_id}"
        )
        print(
            f"batch_status="
            f"{generation.batch_status}"
        )
        print(
            "generated_post_id="
            f"{generation.generated_post_id}"
        )
        print(
            f"has_image="
            f"{generation.has_image}"
        )
        print(
            "ready_for_review_delivery="
            f"{generation.ready_for_review_delivery}"
        )

        assert generation.batch_status == (
            "awaiting_review"
        )
        assert generation.generated_post_id == 60
        assert generation.has_image is True
        assert (
            generation.ready_for_review_delivery
            is True
        )

        print(
            "Completed generation/image resume: OK"
        )
        print()

        failed = (
            await load_generation_workflow_state(
                pool,
                batch_id=FAILED_BATCH_ID,
            )
        )

        print("Failed generation state:")
        print(
            f"batch_id={failed.batch_id}"
        )
        print(
            f"batch_status="
            f"{failed.batch_status}"
        )
        print(
            f"failed={failed.failed}"
        )
        print(
            "ready_for_review_delivery="
            f"{failed.ready_for_review_delivery}"
        )

        assert failed.batch_status == "failed"
        assert failed.failed is True
        assert (
            failed.ready_for_review_delivery
            is False
        )

        print("Failed generation blocking: OK")

    finally:
        await close_database_pool(
            pool
        )

    print()
    print("Database changes=not_performed")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print("Daily workflow state probe: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
