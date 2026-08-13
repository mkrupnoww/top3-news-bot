import asyncio
from datetime import date, datetime, timezone

import asyncpg

from app.config import get_settings
from app.db.daily_workflow import (
    DAILY_WORKFLOW_VERSION,
    attach_daily_workflow_generation,
    attach_daily_workflow_image,
    attach_daily_workflow_ranking,
    complete_daily_workflow,
    mark_daily_workflow_stage,
    reserve_daily_workflow,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


PUBLICATION_DATE = date(
    2026,
    8,
    13,
)

AS_OF = datetime(
    2026,
    8,
    12,
    11,
    23,
    tzinfo=timezone.utc,
)

RANKING_RUN_ID = 137
BATCH_ID = 60
GENERATED_POST_ID = 60
IMAGE_GENERATION_ID = 20


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
    """Pool-like wrapper для rollback test."""

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


async def main() -> int:
    """Проверяет lifecycle с полным rollback."""

    print("Daily workflow lifecycle test")
    print("Database changes=rolled_back")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print()

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        async with (
            database_pool.acquire()
            as connection
        ):
            transaction = connection.transaction()

            await transaction.start()

            try:
                pool = _SingleConnectionPool(
                    connection
                )

                workflow = await reserve_daily_workflow(
                    pool,
                    publication_date=PUBLICATION_DATE,
                    as_of=AS_OF,
                    target_telegram_chat_id=(
                        settings.telegram_channel_id
                    ),
                    workflow_version=(
                        DAILY_WORKFLOW_VERSION
                    ),
                )

                assert workflow.created_new is True
                assert workflow.workflow_status == (
                    "running"
                )
                assert workflow.current_stage == (
                    "reserved"
                )

                print("Workflow reservation: OK")

                duplicate = await reserve_daily_workflow(
                    pool,
                    publication_date=PUBLICATION_DATE,
                    as_of=AS_OF,
                    target_telegram_chat_id=(
                        settings.telegram_channel_id
                    ),
                    workflow_version=(
                        DAILY_WORKFLOW_VERSION
                    ),
                )

                assert duplicate.created_new is False
                assert (
                    duplicate.daily_workflow_run_id
                    == workflow.daily_workflow_run_id
                )

                print(
                    "Workflow duplicate reservation: OK"
                )

                workflow = (
                    await mark_daily_workflow_stage(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        stage="ranking",
                    )
                )

                assert workflow.current_stage == "ranking"

                workflow = (
                    await attach_daily_workflow_ranking(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                    )
                )

                assert (
                    workflow.ranking_run_id
                    == RANKING_RUN_ID
                )

                print("Ranking attachment: OK")

                workflow = (
                    await mark_daily_workflow_stage(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        stage="generation",
                    )
                )

                workflow = (
                    await attach_daily_workflow_generation(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        batch_id=BATCH_ID,
                        generated_post_id=(
                            GENERATED_POST_ID
                        ),
                    )
                )

                assert workflow.batch_id == BATCH_ID
                assert (
                    workflow.generated_post_id
                    == GENERATED_POST_ID
                )

                print("Generation attachment: OK")

                workflow = (
                    await mark_daily_workflow_stage(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        stage="image",
                    )
                )

                workflow = (
                    await attach_daily_workflow_image(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        image_generation_id=(
                            IMAGE_GENERATION_ID
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == IMAGE_GENERATION_ID
                )

                print("Image attachment: OK")

                workflow = (
                    await mark_daily_workflow_stage(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        stage="review_delivery",
                    )
                )

                assert (
                    workflow.current_stage
                    == "review_delivery"
                )

                workflow = (
                    await complete_daily_workflow(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                    )
                )

                assert workflow.awaiting_review is True
                assert (
                    workflow.current_stage
                    == "awaiting_review"
                )

                print("Workflow completion: OK")

            finally:
                await transaction.rollback()

    finally:
        await close_database_pool(
            database_pool
        )

    print()
    print("Database changes=rolled_back")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print("Daily workflow lifecycle test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
