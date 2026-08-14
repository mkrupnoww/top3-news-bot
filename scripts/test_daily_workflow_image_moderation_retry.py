import asyncio
import hashlib

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
from app.generation.image_generator import (
    OPENAI_IMAGE_PROMPT_VERSION,
)


WORKFLOW_ID = 12
RANKING_RUN_ID = 141
BATCH_ID = 65
GENERATED_POST_ID = 61
LINKED_FAILED_IMAGE_ID = 23
OLD_PROMPT_VERSION = "movie_news_image_v1"
EXPECTED_CURRENT_PROMPT_VERSION = "movie_news_image_v2"


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


def _synthetic_request_key(
    *,
    attempt_number: int,
) -> str:
    """Создаёт уникальный synthetic request key."""

    payload = (
        "daily-workflow-v2-retry-test:"
        f"{WORKFLOW_ID}:"
        f"{attempt_number}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


async def _assert_production_fixture(
    pool,
) -> None:
    """Проверяет неизменный production incident fixture."""

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
        == LINKED_FAILED_IMAGE_ID
    )

    async with pool.acquire() as connection:
        generation = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                p.post_status,
                p.image_path,
                p.image_sha256
            FROM publication_batches AS b
            JOIN generated_posts AS p
              ON p.batch_id = b.batch_id
            WHERE b.batch_id = $1
              AND p.generated_post_id = $2
            """,
            BATCH_ID,
            GENERATED_POST_ID,
        )

        attempts = await connection.fetch(
            """
            SELECT
                image_generation_id,
                image_status,
                prompt_version,
                error_type,
                error_message,
                failed_at
            FROM image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
            ORDER BY image_generation_id
            """,
            BATCH_ID,
            GENERATED_POST_ID,
        )

    if generation is None:
        raise AssertionError(
            "Production batch/post fixture не найдена."
        )

    assert generation["batch_status"] == "awaiting_review"
    assert generation["post_status"] == "awaiting_review"
    assert generation["image_path"] is None
    assert generation["image_sha256"] is None

    old_attempts = [
        row
        for row in attempts
        if row["prompt_version"] == OLD_PROMPT_VERSION
    ]

    current_attempts = [
        row
        for row in attempts
        if (
            row["prompt_version"]
            == EXPECTED_CURRENT_PROMPT_VERSION
        )
    ]

    assert len(old_attempts) == 2
    assert len(current_attempts) == 0

    for row in old_attempts:
        assert row["image_status"] == "failed"
        assert row["error_type"] == "BadRequestError"
        assert row["failed_at"] is not None
        assert "moderation_blocked" in (
            str(row["error_message"]).lower()
        )


async def _insert_synthetic_v2_reservation(
    pool,
    *,
    attempt_number: int,
) -> int:
    """Создаёт synthetic v2 reservation внутри rollback transaction."""

    request_key = _synthetic_request_key(
        attempt_number=attempt_number
    )

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
                $2,
                request_key_version,
                'reserved',
                request_kind,
                editorial_comment,
                issues,
                model_name,
                generator_version,
                $3,
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
            LINKED_FAILED_IMAGE_ID,
            request_key,
            EXPECTED_CURRENT_PROMPT_VERSION,
        )

    if image_generation_id is None:
        raise RuntimeError(
            "Не удалось создать synthetic v2 reservation."
        )

    return int(image_generation_id)


async def _mark_synthetic_moderation_failed(
    pool,
    *,
    image_generation_id: int,
) -> None:
    """Переводит synthetic reservation в definitive moderation failure."""

    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            UPDATE image_generation_requests
            SET
                image_status = 'failed',
                error_type = 'BadRequestError',
                error_message = (
                    'Synthetic Error code: 400 '
                    'moderation_blocked'
                ),
                failed_at = now()
            WHERE image_generation_id = $1
              AND image_status = 'reserved'
            """,
            image_generation_id,
        )

    if result != "UPDATE 1":
        raise RuntimeError(
            "Не удалось перевести synthetic image "
            "reservation в failed."
        )


async def main() -> int:
    """Проверяет version-aware budget без OpenAI/Telegram."""

    if (
        OPENAI_IMAGE_PROMPT_VERSION
        != EXPECTED_CURRENT_PROMPT_VERSION
    ):
        raise AssertionError(
            "Тест требует current prompt_version="
            f"{EXPECTED_CURRENT_PROMPT_VERSION}, "
            f"actual={OPENAI_IMAGE_PROMPT_VERSION}"
        )

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

                attempts_used = (
                    await
                    require_daily_workflow_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_PROMPT_VERSION
                        ),
                    )
                )

                assert attempts_used == 0

                print(
                    "New prompt version has fresh budget: OK"
                )

                workflow = (
                    await
                    reopen_daily_workflow_for_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_PROMPT_VERSION
                        ),
                    )
                )

                assert workflow.running is True
                assert workflow.current_stage == "image"
                assert (
                    workflow.image_generation_id
                    == LINKED_FAILED_IMAGE_ID
                )

                print(
                    "Failed v1 workflow reopens for v2: OK"
                )

                first_v2_id = (
                    await _insert_synthetic_v2_reservation(
                        pool,
                        attempt_number=1,
                    )
                )

                workflow = (
                    await checkpoint_image_reservation(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        image_generation_id=(
                            first_v2_id
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == first_v2_id
                )

                await _mark_synthetic_moderation_failed(
                    pool,
                    image_generation_id=(
                        first_v2_id
                    ),
                )

                attempts_used = (
                    await
                    require_daily_workflow_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_PROMPT_VERSION
                        ),
                    )
                )

                assert attempts_used == 1

                print(
                    "Second v2 attempt allowed after "
                    "first moderation block: OK"
                )

                second_v2_id = (
                    await _insert_synthetic_v2_reservation(
                        pool,
                        attempt_number=2,
                    )
                )

                workflow = (
                    await checkpoint_image_reservation(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        image_generation_id=(
                            second_v2_id
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == second_v2_id
                )

                await _mark_synthetic_moderation_failed(
                    pool,
                    image_generation_id=(
                        second_v2_id
                    ),
                )

                try:
                    await (
                        require_daily_workflow_image_moderation_retry(
                            pool,
                            daily_workflow_run_id=(
                                WORKFLOW_ID
                            ),
                            prompt_version=(
                                OPENAI_IMAGE_PROMPT_VERSION
                            ),
                        )
                    )
                except (
                    DailyWorkflowImageModerationRetryNotAllowedError
                ):
                    print(
                        "Third v2 attempt blocked: OK"
                    )
                else:
                    raise AssertionError(
                        "После двух v2 moderation failures "
                        "третья попытка не была заблокирована."
                    )

            finally:
                await transaction.rollback()

        await _assert_production_fixture(
            database_pool
        )

        print()
        print("Database changes=rolled_back")
        print("OpenAI requests=not_performed")
        print("Telegram requests=not_performed")
        print(
            "Version-aware image moderation retry test: OK"
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