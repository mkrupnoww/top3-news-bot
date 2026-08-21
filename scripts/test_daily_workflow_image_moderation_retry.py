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
    """Читает historical image attempts production fixture."""

    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                image_generation_id,
                image_status,
                prompt_version,
                error_type,
                error_message,
                failed_at,
                completed_at
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


async def _load_fixture_snapshot(
    pool,
) -> dict[str, object]:
    """
    Снимает фактическое production-состояние fixture.

    Снимок используется только для проверки полного rollback.
    Тест не требует, чтобы workflow навсегда оставался failed.
    """

    async with pool.acquire() as connection:
        workflow = await connection.fetchrow(
            """
            SELECT
                workflow_status,
                current_stage,
                ranking_run_id,
                batch_id,
                generated_post_id,
                image_generation_id,
                error_type,
                error_message,
                finished_at
            FROM daily_workflow_runs
            WHERE daily_workflow_run_id = $1
            """,
            WORKFLOW_ID,
        )

        generation = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                b.approved_at,
                b.published_at,
                b.approved_by_telegram_user_id,
                b.error_message AS batch_error_message,
                p.post_status,
                p.image_path,
                p.image_sha256,
                p.image_prompt,
                p.image_model_name,
                p.image_prompt_version
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
                image_path,
                image_sha256,
                error_type,
                error_message,
                failed_at,
                completed_at
            FROM image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
            ORDER BY image_generation_id
            """,
            BATCH_ID,
            GENERATED_POST_ID,
        )

    if workflow is None:
        raise AssertionError(
            "Production workflow fixture не найден."
        )

    if generation is None:
        raise AssertionError(
            "Production batch/post fixture не найдена."
        )

    return {
        "workflow": dict(workflow),
        "generation": dict(generation),
        "attempts": [
            dict(row)
            for row in attempts
        ],
    }


async def _assert_historical_fixture(
    pool,
) -> int:
    """
    Проверяет только неизменяемую историческую часть incident 2026-08-15.

    Текущий workflow уже мог быть успешно восстановлен, approved или
    опубликован. Для теста важны:
    - исходные ranking/batch/post;
    - historical normal-v2 moderation block;
    - два historical fallback-v1 moderation blocks.

    Возвращает historical normal failed image_generation_id, который
    используется как безопасный source для synthetic rollback-сценария.
    """

    workflow = await load_daily_workflow(
        pool,
        daily_workflow_run_id=WORKFLOW_ID,
    )

    assert workflow.ranking_run_id == RANKING_RUN_ID
    assert workflow.batch_id == BATCH_ID
    assert (
        workflow.generated_post_id
        == GENERATED_POST_ID
    )

    attempts = await _load_attempts(
        pool
    )

    normal_attempts = [
        row
        for row in attempts
        if (
            row["prompt_version"]
            == HISTORICAL_NORMAL_PROMPT_VERSION
            and row["image_status"] == "failed"
        )
    ]

    fallback_v1_attempts = [
        row
        for row in attempts
        if (
            row["prompt_version"]
            == HISTORICAL_FALLBACK_PROMPT_VERSION
            and row["image_status"] == "failed"
        )
    ]

    if len(normal_attempts) != 1:
        raise AssertionError(
            "Ожидалась ровно одна historical normal-v2 "
            "failed image attempt."
        )

    if len(fallback_v1_attempts) != 2:
        raise AssertionError(
            "Ожидались ровно две historical fallback-v1 "
            "failed image attempts."
        )

    for row in (
        *normal_attempts,
        *fallback_v1_attempts,
    ):
        assert row["error_type"] == "BadRequestError"
        assert row["failed_at"] is not None

        if "moderation_blocked" not in (
            str(row["error_message"]).lower()
        ):
            raise AssertionError(
                "Historical attempt не является "
                "moderation_blocked: "
                f"image_generation_id="
                f"{row['image_generation_id']}"
            )

    normal_failed_image_id = int(
        normal_attempts[0]["image_generation_id"]
    )

    print(
        "Historical production moderation fixture: OK"
    )
    print(
        "current_workflow_status="
        f"{workflow.workflow_status}"
    )
    print(
        "historical_normal_v2_failed_attempts=1"
    )
    print(
        "historical_fallback_v1_failed_attempts=2"
    )

    return normal_failed_image_id


async def _prepare_retry_fixture(
    connection: asyncpg.Connection,
    *,
    normal_failed_image_id: int,
) -> None:
    """
    Создаёт failed/failed retry-состояние только внутри rollback transaction.

    Production workflow может в реальности быть awaiting_review/approved/
    published и иметь successful historical fallback. Сначала переключаем
    workflow на historical failed normal image, затем временно убираем
    active/completed initial image rows и attempts текущего fallback-v5.
    После rollback исходное production-состояние восстанавливается PostgreSQL.
    """

    updated = await connection.execute(
        """
        UPDATE daily_workflow_runs
        SET
            workflow_status = 'failed',
            current_stage = 'failed',
            ranking_run_id = $2,
            batch_id = $3,
            generated_post_id = $4,
            image_generation_id = $5,
            error_type = 'BadRequestError',
            error_message = (
                'Synthetic rollback fixture: '
                'moderation_blocked'
            ),
            finished_at = now()
        WHERE daily_workflow_run_id = $1
        """,
        WORKFLOW_ID,
        RANKING_RUN_ID,
        BATCH_ID,
        GENERATED_POST_ID,
        normal_failed_image_id,
    )

    if updated != "UPDATE 1":
        raise AssertionError(
            "Не удалось подготовить workflow rollback fixture."
        )

    await connection.execute(
        """
        UPDATE publication_batches
        SET
            batch_status = 'awaiting_review',
            approved_at = NULL,
            published_at = NULL,
            approved_by_telegram_user_id = NULL,
            error_message = NULL
        WHERE batch_id = $1
        """,
        BATCH_ID,
    )

    await connection.execute(
        """
        UPDATE generated_posts
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
        GENERATED_POST_ID,
        BATCH_ID,
    )

    # Successful historical fallback (например fallback-v2) блокирует
    # новый moderation retry по production-правилам. Для synthetic test
    # временно убираем active/completed initial requests независимо от
    # их исторической версии. Одновременно очищаем attempts текущего v5,
    # чтобы его version-aware budget начинался с нуля.
    await connection.execute(
        """
        DELETE FROM image_generation_requests
        WHERE batch_id = $1
          AND generated_post_id = $2
          AND request_kind = 'initial'
          AND (
                image_status IN (
                    'reserved',
                    'completed'
                )
                OR prompt_version = $3
              )
        """,
        BATCH_ID,
        GENERATED_POST_ID,
        EXPECTED_FALLBACK_PROMPT_VERSION,
    )

    prepared = await connection.fetchrow(
        """
        SELECT
            workflow_status,
            current_stage,
            image_generation_id
        FROM daily_workflow_runs
        WHERE daily_workflow_run_id = $1
        """,
        WORKFLOW_ID,
    )

    if prepared is None:
        raise AssertionError(
            "Prepared workflow fixture исчез."
        )

    assert prepared["workflow_status"] == "failed"
    assert prepared["current_stage"] == "failed"
    assert (
        int(prepared["image_generation_id"])
        == normal_failed_image_id
    )


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
        snapshot_before = await _load_fixture_snapshot(
            database_pool
        )

        normal_failed_image_id = (
            await _assert_historical_fixture(
                database_pool
            )
        )

        async with database_pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()

            pool = _SingleConnectionPool(
                connection
            )

            try:
                await _prepare_retry_fixture(
                    connection,
                    normal_failed_image_id=(
                        normal_failed_image_id
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

                assert attempts_used == 0

                print(
                    "Synthetic failed/image fixture prepared: OK"
                )
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
                    == normal_failed_image_id
                )

                print(
                    "Synthetic failed workflow "
                    "reopens for fallback v5: OK"
                )

                first_fallback_id = (
                    await
                    _insert_synthetic_fallback_reservation(
                        pool,
                        source_image_generation_id=(
                            normal_failed_image_id
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
                            normal_failed_image_id
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

        snapshot_after = await _load_fixture_snapshot(
            database_pool
        )

        if snapshot_after != snapshot_before:
            raise AssertionError(
                "Production fixture изменился после rollback."
            )

        print(
            "Production fixture restored after rollback: OK"
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