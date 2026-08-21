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
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    OPENAI_IMAGE_PROMPT_VERSION,
    ImageGenerationNewsItem,
    ImageModelRequest,
    ImageModelResponse,
    OpenAIMovieNewsImageGenerator,
)


WORKFLOW_ID = 16
RANKING_RUN_ID = 142
BATCH_ID = 67
GENERATED_POST_ID = 64

CURRENT_NORMAL_PROMPT_VERSION = "movie_news_image_v3"
HISTORICAL_NORMAL_PROMPT_VERSION = "movie_news_image_v2"
HISTORICAL_FALLBACK_PROMPT_VERSION = (
    "movie_news_image_moderation_fallback_v1"
)
EXPECTED_FALLBACK_PROMPT_VERSION = (
    "movie_news_image_moderation_fallback_v5"
)

SENSITIVE_FALLBACK_TERMS = (
    "X-Men",
    "Marvel",
    "Frozen",
    "Disney",
    "Anna",
    "Kristoff",
    "Olaf",
    "Paramount",
    "Warner Bros",
)


class _NeverCalledImageClient:
    """Image client, который не должен вызываться в rollback test."""

    async def create_image(
        self,
        request: ImageModelRequest,
    ) -> ImageModelResponse:
        raise AssertionError(
            "Image API не должен вызываться."
        )


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
        "daily-workflow-moderation-fallback-v5-test:"
        f"{WORKFLOW_ID}:"
        f"{attempt_number}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _assert_generator_fallback_identity() -> None:
    """
    Проверяет новую fallback identity и обезличивание prompt.

    Normal prompt обязан сохранить исходные новости.
    Fallback v5 обязан передавать только semantic_visual_brief.
    """

    generator = OpenAIMovieNewsImageGenerator(
        client=_NeverCalledImageClient(),
        model_name="gpt-image-2",
        size="1024x1536",
        quality="medium",
    )

    items = (
        ImageGenerationNewsItem(
            position=1,
            news_id=1029,
            title=(
                "Marvel’s ‘X-Men’ Reboot Sets "
                "May 2028 Release Date"
            ),
            summary=(
                "Marvel announced the X-Men reboot "
                "and its May 2028 release date."
            ),
        ),
        ImageGenerationNewsItem(
            position=2,
            news_id=1037,
            title=(
                "‘Frozen 3’ First Details: Anna and "
                "Kristoff Get Married, Olaf Gets "
                "a Girlfriend"
            ),
            summary=(
                "Disney revealed the first story details "
                "for Frozen 3."
            ),
        ),
        ImageGenerationNewsItem(
            position=3,
            news_id=986,
            title=(
                "Paramount-Warner Bros. Merger Gets "
                "Mexico’s Approval"
            ),
            summary=(
                "The Paramount-Warner Bros. transaction "
                "received regulatory approval in Mexico."
            ),
        ),
    )

    normal_metadata = generator.metadata
    normal_request = generator.build_request(
        items=items
    )

    assert (
        normal_metadata.prompt_version
        == CURRENT_NORMAL_PROMPT_VERSION
    )

    for term in SENSITIVE_FALLBACK_TERMS:
        if term not in normal_request.prompt:
            raise AssertionError(
                "Normal prompt неожиданно не содержит "
                f"исходный термин: {term!r}"
            )

    generator.set_moderation_safe_editorial_fallback(
        True
    )

    fallback_metadata = generator.metadata
    fallback_request = generator.build_request(
        items=items
    )

    assert (
        fallback_metadata.prompt_version
        == EXPECTED_FALLBACK_PROMPT_VERSION
    )

    assert (
        fallback_request.prompt
        != normal_request.prompt
    )

    assert (
        '"mode":"semantic_visual_brief_v5"'
        in fallback_request.prompt
    )

    assert (
        '"semantic_visual_brief":'
        in fallback_request.prompt
    )

    assert (
        '"moderation_safe_editorial_fallback":'
        in fallback_request.prompt
    )

    assert (
        '"title":'
        not in fallback_request.prompt
    )

    assert (
        '"summary":'
        not in fallback_request.prompt
    )

    for term in SENSITIVE_FALLBACK_TERMS:
        if term in fallback_request.prompt:
            raise AssertionError(
                "Fallback prompt содержит "
                "исходный чувствительный термин: "
                f"{term!r}"
            )

    if (
        "Если новость относится к конкретному "
        "фильму или франшизе"
        in fallback_request.prompt
    ):
        raise AssertionError(
            "Fallback v5 не должен включать "
            "основной permissive IMAGE_PROMPT_INSTRUCTIONS."
        )

    print(
        "Fallback v5 changes prompt identity: OK"
    )
    print(
        "Fallback v5 removes title/summary from Image API prompt: OK"
    )
    print(
        "Fallback v5 removes franchise/person/company terms: OK"
    )


async def _load_attempts(
    pool,
) -> list[asyncpg.Record]:
    """Читает image attempts текущего production incident."""

    async with pool.acquire() as connection:
        rows = await connection.fetch(
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

    return list(rows)


async def _assert_production_fixture(
    pool,
) -> int:
    """
    Проверяет incident 2026-08-15.

    Возвращает текущий linked failed image_generation_id.
    """

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

    linked_failed_image_id = (
        workflow.image_generation_id
    )

    if (
        not isinstance(
            linked_failed_image_id,
            int,
        )
        or isinstance(
            linked_failed_image_id,
            bool,
        )
        or linked_failed_image_id <= 0
    ):
        raise AssertionError(
            "Workflow должен ссылаться "
            "на failed image_generation_id."
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

    if generation is None:
        raise AssertionError(
            "Production batch/post fixture не найдена."
        )

    assert generation["batch_status"] == "awaiting_review"
    assert generation["post_status"] == "awaiting_review"
    assert generation["image_path"] is None
    assert generation["image_sha256"] is None

    attempts = await _load_attempts(
        pool
    )

    normal_attempts = [
        row
        for row in attempts
        if (
            row["prompt_version"]
            == HISTORICAL_NORMAL_PROMPT_VERSION
        )
    ]

    fallback_v1_attempts = [
        row
        for row in attempts
        if (
            row["prompt_version"]
            == HISTORICAL_FALLBACK_PROMPT_VERSION
        )
    ]

    current_fallback_attempts = [
        row
        for row in attempts
        if (
            row["prompt_version"]
            == EXPECTED_FALLBACK_PROMPT_VERSION
        )
    ]

    assert len(normal_attempts) == 1
    assert len(fallback_v1_attempts) == 2
    assert len(current_fallback_attempts) == 0

    for row in (
        *normal_attempts,
        *fallback_v1_attempts,
    ):
        assert row["image_status"] == "failed"
        assert row["error_type"] == "BadRequestError"
        assert row["failed_at"] is not None

        if "moderation_blocked" not in (
            str(row["error_message"]).lower()
        ):
            raise AssertionError(
                "Ожидался moderation_blocked: "
                f"image_generation_id="
                f"{row['image_generation_id']}"
            )

    linked_row = next(
        (
            row
            for row in attempts
            if (
                row["image_generation_id"]
                == linked_failed_image_id
            )
        ),
        None,
    )

    if linked_row is None:
        raise AssertionError(
            "Linked image_generation_id "
            "не найден среди attempts."
        )

    assert (
        linked_row["prompt_version"]
        == HISTORICAL_FALLBACK_PROMPT_VERSION
    )

    assert (
        linked_row["image_status"]
        == "failed"
    )

    print(
        "Current production moderation incident fixture: OK"
    )
    print(
        "normal_v2_failed_attempts=1"
    )
    print(
        "fallback_v1_failed_attempts=2"
    )
    print(
        "current_fallback_v5_existing_attempts=0"
    )

    return linked_failed_image_id


async def _insert_synthetic_fallback_reservation(
    pool,
    *,
    source_image_generation_id: int,
    attempt_number: int,
) -> int:
    """
    Создаёт synthetic fallback-v5 reservation.

    Все изменения выполняются внутри внешней rollback transaction.
    """

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
            source_image_generation_id,
            request_key,
            EXPECTED_FALLBACK_PROMPT_VERSION,
        )

    if image_generation_id is None:
        raise RuntimeError(
            "Не удалось создать synthetic "
            "fallback-v5 reservation."
        )

    return int(
        image_generation_id
    )


async def _mark_synthetic_moderation_failed(
    pool,
    *,
    image_generation_id: int,
) -> None:
    """Переводит synthetic reservation в moderation failure."""

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
            "Не удалось перевести synthetic "
            "fallback-v5 reservation в failed."
        )


async def main() -> int:
    """Проверяет fallback-v5 prompt и retry budget без OpenAI/Telegram."""

    if (
        OPENAI_IMAGE_PROMPT_VERSION
        != CURRENT_NORMAL_PROMPT_VERSION
    ):
        raise AssertionError(
            "Неожиданная normal prompt_version: "
            f"{OPENAI_IMAGE_PROMPT_VERSION}"
        )

    if (
        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
        != EXPECTED_FALLBACK_PROMPT_VERSION
    ):
        raise AssertionError(
            "Неожиданная fallback prompt_version: "
            f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}"
        )

    _assert_generator_fallback_identity()

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
                linked_failed_image_id = (
                    await _assert_production_fixture(
                        pool
                    )
                )

                attempts_used = (
                    await
                    require_daily_workflow_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                        ),
                    )
                )

                assert attempts_used == 0

                print(
                    "Fallback v5 prompt version "
                    "has fresh budget: OK"
                )

                workflow = (
                    await
                    reopen_daily_workflow_for_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                        ),
                    )
                )

                assert workflow.running is True
                assert workflow.current_stage == "image"
                assert (
                    workflow.image_generation_id
                    == linked_failed_image_id
                )

                print(
                    "Failed fallback-v1 workflow "
                    "reopens for fallback v5: OK"
                )

                first_fallback_id = (
                    await
                    _insert_synthetic_fallback_reservation(
                        pool,
                        source_image_generation_id=(
                            linked_failed_image_id
                        ),
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
                            first_fallback_id
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == first_fallback_id
                )

                await _mark_synthetic_moderation_failed(
                    pool,
                    image_generation_id=(
                        first_fallback_id
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
                            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                        ),
                    )
                )

                assert attempts_used == 1

                print(
                    "Second fallback-v5 attempt allowed: OK"
                )

                second_fallback_id = (
                    await
                    _insert_synthetic_fallback_reservation(
                        pool,
                        source_image_generation_id=(
                            linked_failed_image_id
                        ),
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
                            second_fallback_id
                        ),
                    )
                )

                assert (
                    workflow.image_generation_id
                    == second_fallback_id
                )

                await _mark_synthetic_moderation_failed(
                    pool,
                    image_generation_id=(
                        second_fallback_id
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
                                OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                            ),
                        )
                    )
                except (
                    DailyWorkflowImageModerationRetryNotAllowedError
                ):
                    print(
                        "Third fallback-v5 attempt blocked: OK"
                    )
                else:
                    raise AssertionError(
                        "После двух fallback-v5 failures "
                        "третья попытка не была заблокирована."
                    )

            finally:
                await transaction.rollback()

        await _assert_production_fixture(
            database_pool
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
            "Moderation-safe image fallback v5 test: OK"
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