import asyncio
import hashlib
from tempfile import TemporaryDirectory

import asyncpg

from app.config import get_settings
from app.db.daily_workflow import (
    DailyWorkflowImageModerationRetryNotAllowedError,
    require_daily_workflow_image_moderation_retry,
)
from app.db.daily_workflow_checkpoints import (
    checkpoint_image_reservation,
)
from app.db.daily_workflow_selection_attempts import (
    load_active_daily_workflow_selection,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.image_generator import (
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    OPENAI_IMAGE_PROMPT_VERSION,
    OPENAI_IMAGE_VERSATILE_PROMPT_VERSION,
)
from app.generation.openai_image_pipeline import (
    run_reserved_openai_image_generation,
)
from app.workflows.daily_production import (
    VERSATILE_OPTION_IMAGE_PATH,
    _create_versatile_option_generator,
    _fallback_budget_is_exhausted,
    _resolve_active_selection,
)


WORKFLOW_ID = 16
RANKING_RUN_ID = 142
BATCH_ID = 67
GENERATED_POST_ID = 64

CURRENT_NORMAL_PROMPT_VERSION = "movie_news_image_v5"
HISTORICAL_NORMAL_PROMPT_VERSION = "movie_news_image_v2"

WINNER_COMBINATION_ID = 1844

WINNER_NEWS_IDS = (
    1029,
    1037,
    986,
)


class _SingleConnectionAcquire:
    """Context manager одной asyncpg connection."""

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
    """Pool-like wrapper одной connection для полного rollback."""

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
    """Создаёт уникальный request key synthetic fallback."""

    payload = (
        "daily-production-versatile-budget-test:"
        f"{WORKFLOW_ID}:"
        f"{attempt_number}:"
        f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


async def _load_fixture_snapshot(
    pool,
) -> dict[str, object]:
    """Снимает production fixture для строгой проверки rollback."""

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
            FROM top3_news.daily_workflow_runs
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
                gp.post_status,
                gp.image_path,
                gp.image_sha256,
                gp.image_prompt,
                gp.image_model_name,
                gp.image_prompt_version
            FROM top3_news.publication_batches AS b
            JOIN top3_news.generated_posts AS gp
              ON gp.batch_id = b.batch_id
            WHERE b.batch_id = $1
              AND gp.generated_post_id = $2
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
            FROM top3_news.image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
            ORDER BY image_generation_id
            """,
            BATCH_ID,
            GENERATED_POST_ID,
        )

        selection_attempts = await connection.fetch(
            """
            SELECT
                selection_attempt_id,
                attempt_number,
                combination_id,
                selection_status,
                batch_id,
                generated_post_id,
                image_generation_id
            FROM top3_news.daily_workflow_selection_attempts
            WHERE daily_workflow_run_id = $1
            ORDER BY selection_attempt_id
            """,
            WORKFLOW_ID,
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
        "selection_attempts": [
            dict(row)
            for row in selection_attempts
        ],
    }


async def _assert_historical_fixture(
    pool,
) -> int:
    """
    Проверяет неизменяемую часть incident 2026-08-15.

    Текущий workflow может быть уже успешно восстановлен. Для branch-test
    нужен только historical normal-v2 moderation-blocked request.
    """

    async with pool.acquire() as connection:
        workflow = await connection.fetchrow(
            """
            SELECT
                ranking_run_id,
                batch_id,
                generated_post_id,
                workflow_status
            FROM top3_news.daily_workflow_runs
            WHERE daily_workflow_run_id = $1
            """,
            WORKFLOW_ID,
        )

        if workflow is None:
            raise AssertionError(
                "Historical workflow fixture не найден."
            )

        assert int(workflow["ranking_run_id"]) == RANKING_RUN_ID
        assert int(workflow["batch_id"]) == BATCH_ID
        assert int(workflow["generated_post_id"]) == GENERATED_POST_ID

        normal_failed_image_id = (
            await _load_normal_failed_image_id(
                connection
            )
        )

    print(
        "Historical replacement fixture: OK"
    )
    print(
        "current_workflow_status="
        f"{workflow['workflow_status']}"
    )
    print(
        "historical_normal_prompt_version="
        f"{HISTORICAL_NORMAL_PROMPT_VERSION}"
    )

    return normal_failed_image_id


async def _load_normal_failed_image_id(
    connection: asyncpg.Connection,
) -> int:
    """Находит исторический normal moderation-blocked image."""

    image_generation_id = await connection.fetchval(
        """
        SELECT image_generation_id
        FROM top3_news.image_generation_requests
        WHERE batch_id = $1
          AND generated_post_id = $2
          AND request_kind = 'initial'
          AND prompt_version = $3
          AND image_status = 'failed'
          AND failed_at IS NOT NULL
          AND error_type = 'BadRequestError'
          AND lower(
                COALESCE(
                    error_message,
                    ''
                )
              ) LIKE '%moderation_blocked%'
        ORDER BY image_generation_id
        LIMIT 1
        """,
        BATCH_ID,
        GENERATED_POST_ID,
        HISTORICAL_NORMAL_PROMPT_VERSION,
    )

    if image_generation_id is None:
        raise AssertionError(
            "Historical normal moderation-blocked "
            "image fixture не найден."
        )

    return int(image_generation_id)


async def _prepare_fixture(
    connection: asyncpg.Connection,
    *,
    normal_failed_image_id: int,
) -> None:
    """
    В rollback-транзакции восстанавливает состояние:
    normal image -> moderation_blocked, current fallback-v7 budget ещё свежий.
    """

    await connection.execute(
        """
        DELETE FROM
            top3_news.daily_workflow_selection_attempts
        WHERE daily_workflow_run_id = $1
        """,
        WORKFLOW_ID,
    )

    # Сначала убираем workflow FK с возможного successful fallback.
    await connection.execute(
        """
        UPDATE top3_news.daily_workflow_runs
        SET
            workflow_status = 'running',
            current_stage = 'image',
            ranking_run_id = $2,
            batch_id = $3,
            generated_post_id = $4,
            image_generation_id = $5,
            error_type = NULL,
            error_message = NULL,
            finished_at = NULL
        WHERE daily_workflow_run_id = $1
        """,
        WORKFLOW_ID,
        RANKING_RUN_ID,
        BATCH_ID,
        GENERATED_POST_ID,
        normal_failed_image_id,
    )

    await connection.execute(
        """
        UPDATE top3_news.publication_batches
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
        UPDATE top3_news.generated_posts
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

    # Production fixture уже мог иметь successful historical fallback
    # (сейчас это fallback-v2). Он корректно блокирует новый retry.
    # Для synthetic branch-test временно удаляем любой active/completed
    # initial image request, а также attempts текущего fallback-v7,
    # чтобы его version-aware budget начинался с нуля.
    # Всё изменение находится во внешней rollback transaction.
    await connection.execute(
        """
        DELETE FROM top3_news.image_generation_requests
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
        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    )


async def _insert_synthetic_fallback_reservation(
    pool,
    *,
    source_image_generation_id: int,
    attempt_number: int,
) -> int:
    """Создаёт synthetic fallback reservation без Image API."""

    request_key = _synthetic_request_key(
        attempt_number=attempt_number
    )

    async with pool.acquire() as connection:
        image_generation_id = await connection.fetchval(
            """
            INSERT INTO top3_news.image_generation_requests (
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
            FROM top3_news.image_generation_requests
            WHERE image_generation_id = $1
            RETURNING image_generation_id
            """,
            source_image_generation_id,
            request_key,
            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
        )

    if image_generation_id is None:
        raise RuntimeError(
            "Не удалось создать synthetic "
            "fallback reservation."
        )

    return int(image_generation_id)


async def _mark_synthetic_moderation_failed(
    pool,
    *,
    image_generation_id: int,
) -> None:
    """Переводит synthetic fallback в definitive moderation failure."""

    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            UPDATE top3_news.image_generation_requests
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
            "fallback reservation в failed."
        )


async def main() -> int:
    """
    Доказывает новую критическую production-развилку:

    normal block
    -> fallback #1 block
    -> fallback #2 block
    -> budget exhausted
    -> local versatile_option.png
    -> исходный TOP-3/batch/post сохраняются.
    """

    if (
        OPENAI_IMAGE_PROMPT_VERSION
        != CURRENT_NORMAL_PROMPT_VERSION
    ):
        raise AssertionError(
            "Неожиданная current normal prompt_version: "
            f"{OPENAI_IMAGE_PROMPT_VERSION}"
        )

    if not VERSATILE_OPTION_IMAGE_PATH.is_file():
        raise AssertionError(
            "Universal image asset не найден: "
            f"{VERSATILE_OPTION_IMAGE_PATH}"
        )

    versatile_generator = (
        _create_versatile_option_generator()
    )

    if (
        versatile_generator.metadata.prompt_version
        != OPENAI_IMAGE_VERSATILE_PROMPT_VERSION
    ):
        raise AssertionError(
            "Unexpected versatile prompt_version: "
            f"{versatile_generator.metadata.prompt_version}"
        )

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        snapshot_before = await _load_fixture_snapshot(
            database_pool
        )

        historical_normal_failed_image_id = (
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
                normal_failed_image_id = (
                    historical_normal_failed_image_id
                )

                await _prepare_fixture(
                    connection,
                    normal_failed_image_id=(
                        normal_failed_image_id
                    ),
                )

                (
                    active_selection,
                    selection,
                ) = await _resolve_active_selection(
                    pool,
                    daily_workflow_run_id=(
                        WORKFLOW_ID
                    ),
                    ranking_run_id=(
                        RANKING_RUN_ID
                    ),
                )

                assert (
                    active_selection.combination_id
                    == WINNER_COMBINATION_ID
                )
                assert (
                    active_selection.attempt_number
                    == 1
                )
                assert (
                    selection.news_ids
                    == WINNER_NEWS_IDS
                )

                print(
                    "Winner selection fixture: OK"
                )
                print(
                    "Normal moderation_blocked linked: OK"
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
                    "Fallback budget starts at 0: OK"
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

                workflow = await checkpoint_image_reservation(
                    pool,
                    daily_workflow_run_id=(
                        WORKFLOW_ID
                    ),
                    image_generation_id=(
                        first_fallback_id
                    ),
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
                    "Fallback #1 moderation_blocked: OK"
                )
                print(
                    "Fallback #2 allowed: OK"
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

                workflow = await checkpoint_image_reservation(
                    pool,
                    daily_workflow_run_id=(
                        WORKFLOW_ID
                    ),
                    image_generation_id=(
                        second_fallback_id
                    ),
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
                    pass
                else:
                    raise AssertionError(
                        "Третья fallback attempt "
                        "не была заблокирована."
                    )

                exhausted = (
                    await _fallback_budget_is_exhausted(
                        pool,
                        batch_id=BATCH_ID,
                        generated_post_id=(
                            GENERATED_POST_ID
                        ),
                    )
                )

                assert exhausted is True

                print(
                    "Fallback #2 moderation_blocked: OK"
                )
                print(
                    "Fallback budget exhausted: OK"
                )
                print(
                    "Third fallback attempt blocked: OK"
                )

                async def image_observer(
                    reservation,
                ) -> None:
                    await checkpoint_image_reservation(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                        image_generation_id=(
                            reservation
                            .image_generation_id
                        ),
                    )

                with TemporaryDirectory(
                    prefix="top3-versatile-test-"
                ) as output_dir:
                    versatile_result = (
                        await run_reserved_openai_image_generation(
                            pool,
                            generator=(
                                versatile_generator
                            ),
                            selection=selection,
                            batch_id=BATCH_ID,
                            generated_post_id=(
                                GENERATED_POST_ID
                            ),
                            request_kind="initial",
                            output_dir=output_dir,
                            cost_estimator=None,
                            reservation_observer=(
                                image_observer
                            ),
                        )
                    )

                    assert versatile_result.completed is True
                    assert versatile_result.artifact is not None
                    assert versatile_result.generation is not None
                    assert (
                        versatile_result
                        .generation
                        .model_response
                        .usage
                        is None
                    )
                    stored_path = (
                        versatile_result
                        .artifact
                        .image_path
                    )

                    if not stored_path:
                        raise AssertionError(
                            "Versatile artifact path пуст."
                        )

                    print(
                        "Versatile local PNG completed: OK"
                    )

                active_after = (
                    await
                    load_active_daily_workflow_selection(
                        pool,
                        daily_workflow_run_id=(
                            WORKFLOW_ID
                        ),
                    )
                )

                if active_after is None:
                    raise AssertionError(
                        "Active original selection не найдена."
                    )

                assert (
                    active_after.combination_id
                    == WINNER_COMBINATION_ID
                )
                assert active_after.attempt_number == 1

                state = await connection.fetchrow(
                    """
                    SELECT
                        dw.workflow_status,
                        dw.current_stage,
                        dw.batch_id,
                        dw.generated_post_id,
                        dw.image_generation_id,
                        b.batch_status,
                        gp.post_status,
                        gp.image_path,
                        gp.image_model_name,
                        gp.image_prompt_version,
                        igr.image_status,
                        igr.model_name,
                        igr.prompt_version
                    FROM
                        top3_news.daily_workflow_runs AS dw
                    JOIN
                        top3_news.publication_batches AS b
                      ON b.batch_id = dw.batch_id
                    JOIN top3_news.generated_posts AS gp
                      ON gp.generated_post_id = dw.generated_post_id
                    JOIN top3_news.image_generation_requests AS igr
                      ON igr.image_generation_id = dw.image_generation_id
                    WHERE dw.daily_workflow_run_id = $1
                    """,
                    WORKFLOW_ID,
                )

                if state is None:
                    raise AssertionError(
                        "Post-versatile state не найден."
                    )

                assert state["workflow_status"] == "running"
                assert state["current_stage"] == "image"
                assert int(state["batch_id"]) == BATCH_ID
                assert (
                    int(state["generated_post_id"])
                    == GENERATED_POST_ID
                )
                assert state["batch_status"] == "awaiting_review"
                assert state["post_status"] == "awaiting_review"
                assert state["image_status"] == "completed"
                assert (
                    state["model_name"]
                    == "local_static_png_asset"
                )
                assert (
                    state["prompt_version"]
                    == OPENAI_IMAGE_VERSATILE_PROMPT_VERSION
                )
                assert (
                    state["image_model_name"]
                    == "local_static_png_asset"
                )
                assert (
                    state["image_prompt_version"]
                    == OPENAI_IMAGE_VERSATILE_PROMPT_VERSION
                )

                print(
                    "Original TOP-3 preserved: OK"
                )
                print(
                    "Original batch/post preserved: OK"
                )
                print(
                    "No image-driven replacement: OK"
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
            "Daily production moderation to versatile option test: OK"
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
