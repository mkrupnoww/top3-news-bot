import asyncio

import asyncpg

from app.config import get_settings
from app.db.daily_workflow import (
    DailyWorkflowImageModerationRetryNotAllowedError,
    load_daily_workflow,
    reopen_daily_workflow_for_image_moderation_retry,
    require_daily_workflow_image_moderation_retry,
)
from app.db.daily_workflow_checkpoints import (
    checkpoint_image_reservation,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


WORKFLOW_ID = 12
RANKING_RUN_ID = 141
BATCH_ID = 65
GENERATED_POST_ID = 61
FAILED_IMAGE_GENERATION_ID = 21


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


async def _assert_production_fixture(
    pool,
) -> None:
    """Проверяет, что production fixture всё ещё соответствует incident."""

    workflow = await load_daily_workflow(
        pool,
        daily_workflow_run_id=WORKFLOW_ID,
    )

    assert workflow.workflow_status == "failed"
    assert workflow.current_stage == "failed"
    assert workflow.ranking_run_id == RANKING_RUN_ID
    assert workflow.batch_id == BATCH_ID
    assert (
        workflow.generated_post_id
        == GENERATED_POST_ID
    )
    assert (
        workflow.image_generation_id
        == FAILED_IMAGE_GENERATION_ID
    )

    async with pool.acquire() as connection:
        state = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                p.post_status,
                p.image_path,
                p.image_sha256,
                i.image_status,
                i.request_kind,
                i.error_type,
                i.error_message,
                i.failed_at,
                (
                    SELECT COUNT(*)::integer
                    FROM image_generation_requests AS x
                    WHERE x.batch_id = $1
                      AND x.generated_post_id = $2
                      AND x.request_kind = 'initial'
                ) AS initial_attempt_count
            FROM publication_batches AS b
            JOIN generated_posts AS p
              ON p.batch_id = b.batch_id
            JOIN image_generation_requests AS i
              ON i.image_generation_id = $3
            WHERE b.batch_id = $1
              AND p.generated_post_id = $2
            """,
            BATCH_ID,
            GENERATED_POST_ID,
            FAILED_IMAGE_GENERATION_ID,
        )

    if state is None:
        raise AssertionError(
            "Production moderation fixture не найдена."
        )

    assert state["batch_status"] == "awaiting_review"
    assert state["post_status"] == "awaiting_review"
    assert state["image_path"] is None
    assert state["image_sha256"] is None
    assert state["image_status"] == "failed"
    assert state["request_kind"] == "initial"
    assert state["error_type"] == "BadRequestError"
    assert "moderation_blocked" in (
        str(state["error_message"]).lower()
    )
    assert state["failed_at"] is not None
    assert int(state["initial_attempt_count"]) == 1


async def _insert_synthetic_retry_reservation(
    pool,
) -> int:
    """
    Создаёт вторую reserved image attempt только внутри rollback test.

    Поля точного Image API request копируются из failed attempt 21.
    Никакого Image API вызова не выполняется.
    """

    async with pool.acquire() as connection:
        image_generation_id = await connection.fetchval(
            """
            INSERT INTO image_generation_requests (
                batch_id,
                generated_post_id,
                review_action_id,
                image_request_key,
                request_key_version,
                image_status,
                request_kind,
                editorial_comment,
                issues,
                model_name,
                generator_version,
                prompt_version,
                image_size,
                image_quality,
                output_format,
                background,
                moderation,
                image_count,
                request_payload
            )
            SELECT
                batch_id,
                generated_post_id,
                review_action_id,
                image_request_key,
                request_key_version,
                'reserved',
                request_kind,
                editorial_comment,
                issues,
                model_name,
                generator_version,
                prompt_version,
                image_size,
                image_quality,
                output_format,
                background,
                moderation,
                image_count,
                request_payload
            FROM image_generation_requests
            WHERE image_generation_id = $1
            RETURNING image_generation_id
            """,
            FAILED_IMAGE_GENERATION_ID,
        )

    if image_generation_id is None:
        raise RuntimeError(
            "Не удалось создать synthetic retry reservation."
        )

    return int(image_generation_id)


async def main() -> int:
    """Проверяет reopen + replacement + one-retry limit с полным rollback."""

    settings = get_settings()
    database_pool = await create_database_pool(
        settings
    )

    try:
        async with database_pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()

            pool = _SingleConnectionPool(
                connection
            )

            try:
                await _assert_production_fixture(
                    pool
                )

                eligible_image_id = (
                    await
                    require_daily_workflow_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                    )
                )

                assert (
                    eligible_image_id
                    == FAILED_IMAGE_GENERATION_ID
                )

                print(
                    "Definitive moderation failure "
                    "eligibility: OK"
                )

                workflow = (
                    await
                    reopen_daily_workflow_for_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                    )
                )

                assert workflow.running is True
                assert workflow.current_stage == "image"
                assert (
                    workflow.image_generation_id
                    == FAILED_IMAGE_GENERATION_ID
                )

                print(
                    "Failed workflow reopen to image: OK"
                )

                retry_image_id = (
                    await _insert_synthetic_retry_reservation(
                        pool
                    )
                )

                assert (
                    retry_image_id
                    != FAILED_IMAGE_GENERATION_ID
                )

                workflow = (
                    await checkpoint_image_reservation(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        image_generation_id=(
                            retry_image_id
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == retry_image_id
                )
                assert workflow.current_stage == "image"

                print(
                    "Failed image checkpoint replacement: OK"
                )

                try:
                    await (
                        require_daily_workflow_image_moderation_retry(
                            pool,
                            daily_workflow_run_id=(
                                WORKFLOW_ID
                            ),
                        )
                    )
                except (
                    DailyWorkflowImageModerationRetryNotAllowedError
                ):
                    print(
                        "Third automatic image attempt blocked: OK"
                    )
                else:
                    raise AssertionError(
                        "После второй initial attempt "
                        "третий automatic retry не был заблокирован."
                    )

            finally:
                await transaction.rollback()

        # Проверяем rollback уже через обычный pool.
        await _assert_production_fixture(
            database_pool
        )

        print()
        print("Database changes=rolled_back")
        print("OpenAI requests=not_performed")
        print("Telegram requests=not_performed")
        print(
            "Daily workflow image moderation retry test: OK"
        )

        return 0

    finally:
        await close_database_pool(
            database_pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )