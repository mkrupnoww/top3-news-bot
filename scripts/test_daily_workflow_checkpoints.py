import asyncio
from datetime import date, datetime, timezone

import asyncpg

from app.config import get_settings
from app.db.daily_workflow import (
    DAILY_WORKFLOW_VERSION,
    complete_daily_workflow,
    mark_daily_workflow_stage,
    reserve_daily_workflow,
)
from app.db.daily_workflow_checkpoints import (
    checkpoint_generated_post,
    checkpoint_generation_reservation,
    checkpoint_image_reservation,
    checkpoint_ranking_reservation,
    recover_batch_id,
    recover_generated_post_id,
    recover_image_generation_id,
    recover_ranking_run_id,
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
    """Проверяет checkpoint/recovery с rollback."""

    print("Daily workflow checkpoint test")
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

                ranking_run_id = (
                    await recover_ranking_run_id(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                    )
                )

                assert (
                    ranking_run_id
                    == RANKING_RUN_ID
                )

                print(
                    "Ranking reservation recovery: OK"
                )

                workflow = (
                    await checkpoint_ranking_reservation(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        ranking_run_id=(
                            ranking_run_id
                        ),
                    )
                )

                assert (
                    workflow.ranking_run_id
                    == RANKING_RUN_ID
                )
                assert (
                    workflow.current_stage
                    == "ranking"
                )

                print(
                    "Ranking reservation checkpoint: OK"
                )

                batch_id = await recover_batch_id(
                    pool,
                    daily_workflow_run_id=(
                        workflow.daily_workflow_run_id
                    ),
                )

                assert batch_id == BATCH_ID

                print(
                    "Generation reservation recovery: OK"
                )

                workflow = (
                    await checkpoint_generation_reservation(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        batch_id=batch_id,
                    )
                )

                assert workflow.batch_id == BATCH_ID
                assert (
                    workflow.current_stage
                    == "generation"
                )

                print(
                    "Generation reservation checkpoint: OK"
                )

                generated_post_id = (
                    await recover_generated_post_id(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                    )
                )

                assert (
                    generated_post_id
                    == GENERATED_POST_ID
                )

                workflow = (
                    await checkpoint_generated_post(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        generated_post_id=(
                            generated_post_id
                        ),
                    )
                )

                assert (
                    workflow.generated_post_id
                    == GENERATED_POST_ID
                )

                print(
                    "Generated post recovery/checkpoint: OK"
                )

                image_generation_id = (
                    await recover_image_generation_id(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                    )
                )

                assert (
                    image_generation_id
                    == IMAGE_GENERATION_ID
                )

                print(
                    "Image reservation recovery: OK"
                )

                workflow = (
                    await checkpoint_image_reservation(
                        pool,
                        daily_workflow_run_id=(
                            workflow
                            .daily_workflow_run_id
                        ),
                        image_generation_id=(
                            image_generation_id
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == IMAGE_GENERATION_ID
                )
                assert (
                    workflow.current_stage
                    == "image"
                )

                print(
                    "Image reservation checkpoint: OK"
                )

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

                print(
                    "Workflow final transition: OK"
                )

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
    print("Daily workflow checkpoint test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
