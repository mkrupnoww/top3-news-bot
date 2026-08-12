import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg

from app.config import get_settings
from app.db.image_generation_completion import (
    complete_reserved_image_generation,
    fail_reserved_image_generation,
)
from app.db.image_generation_reservation import (
    ImageGenerationReservation,
    reserve_image_generation,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.image_generator import (
    ImageGenerationNewsItem,
    ImageModelRequest,
    OpenAIMovieNewsImageGenerator,
)
from app.generation.image_request_key import (
    ImageRequestKey,
    create_image_request_key,
)
from scripts import (
    test_event_ranking_run_completion as ranking_fixture,
)
from scripts import (
    test_image_generation_reservation as reservation_fixture,
)


INITIAL_IMAGE_PATH = (
    "data/images/generated/"
    "synthetic-initial-completed.png"
)

INITIAL_IMAGE_SHA256 = "b" * 64

REGENERATED_IMAGE_PATH = (
    "data/images/generated/"
    "synthetic-regenerated-completed.png"
)

REGENERATED_IMAGE_SHA256 = "c" * 64

SYNTHETIC_RESPONSE_METADATA = {
    "synthetic": True,
    "created": 1786518000,
    "output_format": "png",
    "quality": "medium",
    "size": reservation_fixture.TEST_IMAGE_SIZE,
    "background": "opaque",
}

SYNTHETIC_FAILURE_TYPE = (
    "SyntheticImageGenerationError"
)

SYNTHETIC_FAILURE_MESSAGE = (
    "Synthetic image completion failure test."
)


@dataclass(frozen=True, slots=True)
class ReservedImageContext:
    """Контекст одной временной image reservation."""

    batch_id: int
    generated_post_id: int
    review_action_id: int | None
    generator: OpenAIMovieNewsImageGenerator
    items: tuple[
        ImageGenerationNewsItem,
        ImageGenerationNewsItem,
        ImageGenerationNewsItem,
    ]
    model_request: ImageModelRequest
    request_key: ImageRequestKey
    reservation: ImageGenerationReservation


def build_test_generator() -> OpenAIMovieNewsImageGenerator:
    """Создаёт image generator без сетевых вызовов."""

    return OpenAIMovieNewsImageGenerator(
        client=(
            reservation_fixture
            .NoCallImageGenerationClient()
        ),
        model_name=(
            reservation_fixture
            .TEST_IMAGE_MODEL_NAME
        ),
        size=(
            reservation_fixture
            .TEST_IMAGE_SIZE
        ),
    )


async def reserve_test_image(
    pool: asyncpg.Pool,
    *,
    selection: Any,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int | None,
    request_kind: str,
    test_name: str,
    created_batch_ids: set[int],
) -> ReservedImageContext:
    """Создаёт временный batch/post и image reservation."""

    if request_kind not in {
        "initial",
        "regenerate",
    }:
        raise ValueError(
            "request_kind должен быть initial "
            "или regenerate."
        )

    existing_image = (
        request_kind == "regenerate"
    )

    (
        batch_id,
        generated_post_id,
    ) = await (
        reservation_fixture
        .create_test_batch_and_post(
            pool,
            selection=selection,
            telegram_chat_id=telegram_chat_id,
            existing_image=existing_image,
            test_name=test_name,
            created_batch_ids=created_batch_ids,
        )
    )

    review_action_id: int | None

    if request_kind == "initial":
        review_action_id = None
    else:
        if reviewer_telegram_user_id is None:
            raise ValueError(
                "Для regenerate требуется "
                "reviewer_telegram_user_id."
            )

        review_action_id = await (
            reservation_fixture
            .create_regenerate_review_action(
                pool,
                generated_post_id=(
                    generated_post_id
                ),
                reviewer_telegram_user_id=(
                    reviewer_telegram_user_id
                ),
                test_name=test_name,
            )
        )

    generator = build_test_generator()

    items = (
        reservation_fixture
        .build_image_items(selection)
    )

    if request_kind == "initial":
        model_request = generator.build_request(
            items=items,
        )
        editorial_comment = None
        issues: tuple[str, ...] = ()
    else:
        model_request = generator.build_request(
            items=items,
            editorial_comment=(
                reservation_fixture
                .EDITORIAL_COMMENT
            ),
            issues=(
                reservation_fixture
                .IMAGE_ISSUES
            ),
        )
        editorial_comment = (
            reservation_fixture
            .EDITORIAL_COMMENT
        )
        issues = (
            reservation_fixture
            .IMAGE_ISSUES
        )

    request_key = create_image_request_key(
        batch_id=batch_id,
        ranking_run_id=selection.ranking_run_id,
        request_kind=request_kind,
        review_action_id=review_action_id,
        metadata=generator.metadata,
        model_request=model_request,
        items=items,
    )

    reservation = await reserve_image_generation(
        pool,
        request_key=request_key,
        batch_id=batch_id,
        generated_post_id=generated_post_id,
        ranking_run_id=(
            selection.ranking_run_id
        ),
        request_kind=request_kind,
        review_action_id=review_action_id,
        editorial_comment=editorial_comment,
        issues=issues,
        metadata=generator.metadata,
        model_request=model_request,
        items=items,
    )

    assert reservation.created_new is True
    assert reservation.should_call_model is True
    assert reservation.image_status == "reserved"

    return ReservedImageContext(
        batch_id=batch_id,
        generated_post_id=generated_post_id,
        review_action_id=review_action_id,
        generator=generator,
        items=items,
        model_request=model_request,
        request_key=request_key,
        reservation=reservation,
    )


async def load_completion_state(
    pool: asyncpg.Pool,
    *,
    context: ReservedImageContext,
) -> asyncpg.Record:
    """Читает image request, post и batch одним запросом."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                igr.image_generation_id,
                igr.image_status,
                igr.request_kind,
                igr.review_action_id,
                igr.response_metadata,
                igr.openai_usage,
                igr.openai_cost,
                igr.image_path
                    AS request_image_path,
                igr.image_sha256
                    AS request_image_sha256,
                igr.error_type,
                igr.error_message,
                igr.completed_at,
                igr.failed_at,

                b.batch_status,

                gp.post_status,
                gp.image_path
                    AS post_image_path,
                gp.image_sha256
                    AS post_image_sha256,
                gp.image_prompt
                    AS post_image_prompt,
                gp.image_model_name
                    AS post_image_model_name,
                gp.image_prompt_version
                    AS post_image_prompt_version,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp2
                    WHERE gp2.batch_id = b.batch_id
                ) AS generated_post_count

            FROM top3_news.image_generation_requests AS igr
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = igr.batch_id
            JOIN top3_news.generated_posts AS gp
              ON gp.generated_post_id =
                 igr.generated_post_id
            WHERE igr.image_generation_id = $1
              AND igr.image_request_key = $2
              AND igr.batch_id = $3
              AND igr.generated_post_id = $4
            """,
            context.reservation.image_generation_id,
            context.request_key.value,
            context.batch_id,
            context.generated_post_id,
        )

    if record is None:
        raise AssertionError(
            "Не найдено состояние временной "
            "image-generation."
        )

    return record


async def test_initial_completion(
    pool: asyncpg.Pool,
    *,
    selection: Any,
    telegram_chat_id: int,
    created_batch_ids: set[int],
) -> None:
    """Проверяет успешный initial completion."""

    context = await reserve_test_image(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        reviewer_telegram_user_id=None,
        request_kind="initial",
        test_name=(
            "image_generation_initial_completion"
        ),
        created_batch_ids=created_batch_ids,
    )

    result = await (
        complete_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            image_path=INITIAL_IMAGE_PATH,
            image_sha256=INITIAL_IMAGE_SHA256,
            response_metadata=(
                SYNTHETIC_RESPONSE_METADATA
            ),
        )
    )

    assert result.already_completed is False
    assert result.image_status == "completed"
    assert result.request_kind == "initial"
    assert result.review_action_id is None
    assert result.batch_status == "awaiting_review"
    assert result.post_status == "awaiting_review"
    assert result.image_path == INITIAL_IMAGE_PATH
    assert result.image_sha256 == INITIAL_IMAGE_SHA256

    record = await load_completion_state(
        pool,
        context=context,
    )

    assert record["image_status"] == "completed"
    assert record["request_kind"] == "initial"
    assert record["review_action_id"] is None
    assert (
        record["request_image_path"]
        == INITIAL_IMAGE_PATH
    )
    assert (
        record["request_image_sha256"]
        == INITIAL_IMAGE_SHA256
    )
    assert record["openai_usage"] is None
    assert record["openai_cost"] is None
    assert record["error_type"] is None
    assert record["error_message"] is None
    assert record["completed_at"] is not None
    assert record["failed_at"] is None
    assert record["batch_status"] == "awaiting_review"
    assert record["post_status"] == "awaiting_review"
    assert record["generated_post_count"] == 1
    assert (
        record["post_image_path"]
        == INITIAL_IMAGE_PATH
    )
    assert (
        record["post_image_sha256"]
        == INITIAL_IMAGE_SHA256
    )
    assert (
        record["post_image_prompt"]
        == context.model_request.prompt
    )
    assert (
        record["post_image_model_name"]
        == context.generator.metadata.model_name
    )
    assert (
        record["post_image_prompt_version"]
        == context.generator.metadata.prompt_version
    )

    response_metadata = (
        reservation_fixture.decode_json_object(
            record["response_metadata"],
            field_name="response_metadata",
        )
    )

    assert (
        response_metadata
        == SYNTHETIC_RESPONSE_METADATA
    )

    repeated = await (
        complete_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            image_path=INITIAL_IMAGE_PATH,
            image_sha256=INITIAL_IMAGE_SHA256,
            response_metadata=(
                SYNTHETIC_RESPONSE_METADATA
            ),
        )
    )

    assert repeated.already_completed is True
    assert repeated.image_status == "completed"

    try:
        await complete_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            image_path=INITIAL_IMAGE_PATH,
            image_sha256=("d" * 64),
            response_metadata=(
                SYNTHETIC_RESPONSE_METADATA
            ),
        )
    except ValueError as error:
        if "не соответствует повторному" not in str(
            error
        ):
            raise AssertionError(
                "Получена неожиданная ошибка "
                "при mismatch completion."
            ) from error
    else:
        raise AssertionError(
            "Повторный completion с другим SHA "
            "не был заблокирован."
        )

    try:
        await fail_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            error_type=SYNTHETIC_FAILURE_TYPE,
            error_message=(
                SYNTHETIC_FAILURE_MESSAGE
            ),
        )
    except ValueError as error:
        if "completed" not in str(error):
            raise AssertionError(
                "Получена неожиданная ошибка "
                "при completed -> failed."
            ) from error
    else:
        raise AssertionError(
            "Completed image-generation удалось "
            "перевести в failed."
        )

    print("Initial image completion: OK")
    print(
        "image_generation_id="
        f"{context.reservation.image_generation_id}"
    )
    print(f"batch_id={context.batch_id}")
    print(
        f"generated_post_id={context.generated_post_id}"
    )
    print("image_status=completed")
    print("generated_post_image_fields_saved=true")
    print("repeated_completion_idempotent=true")
    print("mismatched_completion_blocked=true")
    print("completed_to_failed_blocked=true")


async def test_initial_failure(
    pool: asyncpg.Pool,
    *,
    selection: Any,
    telegram_chat_id: int,
    created_batch_ids: set[int],
) -> None:
    """Проверяет failed initial без изменения post."""

    context = await reserve_test_image(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        reviewer_telegram_user_id=None,
        request_kind="initial",
        test_name=(
            "image_generation_initial_failure"
        ),
        created_batch_ids=created_batch_ids,
    )

    result = await fail_reserved_image_generation(
        pool,
        image_generation_id=(
            context.reservation.image_generation_id
        ),
        request_key=context.request_key.value,
        error_type=SYNTHETIC_FAILURE_TYPE,
        error_message=SYNTHETIC_FAILURE_MESSAGE,
    )

    assert result.already_failed is False
    assert result.image_status == "failed"
    assert result.request_kind == "initial"
    assert result.review_action_id is None
    assert result.batch_status == "awaiting_review"
    assert result.post_status == "awaiting_review"

    record = await load_completion_state(
        pool,
        context=context,
    )

    assert record["image_status"] == "failed"
    assert record["response_metadata"] is None
    assert record["openai_usage"] is None
    assert record["openai_cost"] is None
    assert record["request_image_path"] is None
    assert record["request_image_sha256"] is None
    assert record["error_type"] == SYNTHETIC_FAILURE_TYPE
    assert (
        record["error_message"]
        == SYNTHETIC_FAILURE_MESSAGE
    )
    assert record["completed_at"] is None
    assert record["failed_at"] is not None
    assert record["batch_status"] == "awaiting_review"
    assert record["post_status"] == "awaiting_review"
    assert record["generated_post_count"] == 1
    assert record["post_image_path"] is None
    assert record["post_image_sha256"] is None
    assert record["post_image_prompt"] is None
    assert record["post_image_model_name"] is None
    assert record["post_image_prompt_version"] is None

    repeated = await fail_reserved_image_generation(
        pool,
        image_generation_id=(
            context.reservation.image_generation_id
        ),
        request_key=context.request_key.value,
        error_type=SYNTHETIC_FAILURE_TYPE,
        error_message=SYNTHETIC_FAILURE_MESSAGE,
    )

    assert repeated.already_failed is True
    assert repeated.image_status == "failed"

    try:
        await complete_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            image_path=INITIAL_IMAGE_PATH,
            image_sha256=INITIAL_IMAGE_SHA256,
            response_metadata=(
                SYNTHETIC_RESPONSE_METADATA
            ),
        )
    except ValueError as error:
        if "failed" not in str(error):
            raise AssertionError(
                "Получена неожиданная ошибка "
                "при failed -> completed."
            ) from error
    else:
        raise AssertionError(
            "Failed image-generation удалось "
            "завершить как completed."
        )

    print()
    print("Initial image failure: OK")
    print(
        "image_generation_id="
        f"{context.reservation.image_generation_id}"
    )
    print("image_status=failed")
    print("generated_post_image_fields_unchanged=true")
    print("repeated_failure_idempotent=true")
    print("failed_to_completed_blocked=true")


async def test_regenerate_completion(
    pool: asyncpg.Pool,
    *,
    selection: Any,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    created_batch_ids: set[int],
) -> None:
    """Проверяет успешный regenerate completion."""

    context = await reserve_test_image(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        reviewer_telegram_user_id=(
            reviewer_telegram_user_id
        ),
        request_kind="regenerate",
        test_name=(
            "image_generation_regenerate_completion"
        ),
        created_batch_ids=created_batch_ids,
    )

    result = await (
        complete_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            image_path=REGENERATED_IMAGE_PATH,
            image_sha256=(
                REGENERATED_IMAGE_SHA256
            ),
            response_metadata=(
                SYNTHETIC_RESPONSE_METADATA
            ),
        )
    )

    assert result.already_completed is False
    assert result.image_status == "completed"
    assert result.request_kind == "regenerate"
    assert (
        result.review_action_id
        == context.review_action_id
    )
    assert result.batch_status == "awaiting_review"
    assert result.post_status == "awaiting_review"

    record = await load_completion_state(
        pool,
        context=context,
    )

    assert record["image_status"] == "completed"
    assert record["request_kind"] == "regenerate"
    assert (
        record["review_action_id"]
        == context.review_action_id
    )
    assert (
        record["request_image_path"]
        == REGENERATED_IMAGE_PATH
    )
    assert (
        record["request_image_sha256"]
        == REGENERATED_IMAGE_SHA256
    )
    assert record["completed_at"] is not None
    assert record["failed_at"] is None
    assert record["batch_status"] == "awaiting_review"
    assert record["post_status"] == "awaiting_review"
    assert record["generated_post_count"] == 1
    assert (
        record["post_image_path"]
        == REGENERATED_IMAGE_PATH
    )
    assert (
        record["post_image_sha256"]
        == REGENERATED_IMAGE_SHA256
    )
    assert (
        record["post_image_prompt"]
        == context.model_request.prompt
    )
    assert (
        record["post_image_model_name"]
        == context.generator.metadata.model_name
    )
    assert (
        record["post_image_prompt_version"]
        == context.generator.metadata.prompt_version
    )
    assert (
        record["post_image_path"]
        != reservation_fixture.EXISTING_IMAGE_PATH
    )
    assert (
        record["post_image_sha256"]
        != reservation_fixture.EXISTING_IMAGE_SHA256
    )

    repeated = await (
        complete_reserved_image_generation(
            pool,
            image_generation_id=(
                context.reservation
                .image_generation_id
            ),
            request_key=(
                context.request_key.value
            ),
            image_path=REGENERATED_IMAGE_PATH,
            image_sha256=(
                REGENERATED_IMAGE_SHA256
            ),
            response_metadata=(
                SYNTHETIC_RESPONSE_METADATA
            ),
        )
    )

    assert repeated.already_completed is True

    print()
    print("Regenerate image completion: OK")
    print(
        "image_generation_id="
        f"{context.reservation.image_generation_id}"
    )
    print(f"review_action_id={context.review_action_id}")
    print("image_status=completed")
    print("same_generated_post_updated=true")
    print("generated_post_count=1")
    print("previous_image_replaced=true")
    print("repeated_completion_idempotent=true")


async def test_regenerate_failure(
    pool: asyncpg.Pool,
    *,
    selection: Any,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    created_batch_ids: set[int],
) -> None:
    """Проверяет failed regenerate без потери старой картинки."""

    context = await reserve_test_image(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        reviewer_telegram_user_id=(
            reviewer_telegram_user_id
        ),
        request_kind="regenerate",
        test_name=(
            "image_generation_regenerate_failure"
        ),
        created_batch_ids=created_batch_ids,
    )

    result = await fail_reserved_image_generation(
        pool,
        image_generation_id=(
            context.reservation.image_generation_id
        ),
        request_key=context.request_key.value,
        error_type=SYNTHETIC_FAILURE_TYPE,
        error_message=SYNTHETIC_FAILURE_MESSAGE,
        response_metadata={
            "synthetic": True,
            "stage": "after_api_before_storage",
        },
    )

    assert result.already_failed is False
    assert result.image_status == "failed"
    assert result.request_kind == "regenerate"
    assert (
        result.review_action_id
        == context.review_action_id
    )
    assert result.batch_status == "awaiting_review"
    assert result.post_status == "awaiting_review"

    record = await load_completion_state(
        pool,
        context=context,
    )

    assert record["image_status"] == "failed"
    assert record["request_image_path"] is None
    assert record["request_image_sha256"] is None
    assert record["completed_at"] is None
    assert record["failed_at"] is not None
    assert record["batch_status"] == "awaiting_review"
    assert record["post_status"] == "awaiting_review"
    assert record["generated_post_count"] == 1
    assert (
        record["post_image_path"]
        == reservation_fixture.EXISTING_IMAGE_PATH
    )
    assert (
        record["post_image_sha256"]
        == reservation_fixture.EXISTING_IMAGE_SHA256
    )
    assert (
        record["post_image_prompt"]
        == reservation_fixture.EXISTING_IMAGE_PROMPT
    )
    assert (
        record["post_image_model_name"]
        == reservation_fixture.EXISTING_IMAGE_MODEL_NAME
    )
    assert (
        record["post_image_prompt_version"]
        == reservation_fixture.EXISTING_IMAGE_PROMPT_VERSION
    )

    response_metadata = (
        reservation_fixture.decode_json_object(
            record["response_metadata"],
            field_name="response_metadata",
        )
    )

    assert response_metadata == {
        "synthetic": True,
        "stage": "after_api_before_storage",
    }

    repeated = await fail_reserved_image_generation(
        pool,
        image_generation_id=(
            context.reservation.image_generation_id
        ),
        request_key=context.request_key.value,
        error_type=SYNTHETIC_FAILURE_TYPE,
        error_message=SYNTHETIC_FAILURE_MESSAGE,
    )

    assert repeated.already_failed is True

    print()
    print("Regenerate image failure: OK")
    print(
        "image_generation_id="
        f"{context.reservation.image_generation_id}"
    )
    print(f"review_action_id={context.review_action_id}")
    print("image_status=failed")
    print("existing_image_preserved=true")
    print("same_generated_post_preserved=true")
    print("repeated_failure_idempotent=true")


async def main() -> int:
    """Запускает интеграционный completion/failure тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    created_batch_ids: set[int] = set()
    created_run_ids: set[int] = set()
    created_news_ids: tuple[int, ...] = ()
    fixture_ranking_run_id: int | None = None

    try:
        await (
            reservation_fixture
            .assert_migration_applied(pool)
        )

        created_news_ids = await (
            ranking_fixture
            .create_test_news_items(pool)
        )

        ranking_fixture.configure_test_news_ids(
            created_news_ids
        )

        selection = await (
            reservation_fixture
            .create_test_ranking_selection(
                pool,
                created_run_ids=(
                    created_run_ids
                ),
            )
        )

        fixture_ranking_run_id = (
            selection.ranking_run_id
        )

        reviewer_telegram_user_id = await (
            reservation_fixture
            .load_test_reviewer(pool)
        )

        await test_initial_completion(
            pool,
            selection=selection,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )

        await test_initial_failure(
            pool,
            selection=selection,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )

        await test_regenerate_completion(
            pool,
            selection=selection,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )

        await test_regenerate_failure(
            pool,
            selection=selection,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )
    finally:
        try:
            if created_batch_ids:
                if fixture_ranking_run_id is None:
                    raise RuntimeError(
                        "Неизвестен ranking_run_id "
                        "для cleanup временных "
                        "publication_batches."
                    )

                await (
                    reservation_fixture
                    .cleanup_test_batches(
                        pool,
                        created_batch_ids=(
                            created_batch_ids
                        ),
                        ranking_run_id=(
                            fixture_ranking_run_id
                        ),
                    )
                )
        finally:
            try:
                await (
                    ranking_fixture
                    .cleanup_test_runs(
                        pool,
                        created_run_ids=(
                            created_run_ids
                        ),
                    )
                )
            finally:
                try:
                    await (
                        ranking_fixture
                        .cleanup_test_news_items(
                            pool,
                            news_ids=(
                                created_news_ids
                            ),
                        )
                    )
                finally:
                    await close_database_pool(
                        pool
                    )

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("OpenAI Image requests: not performed")
    print("PNG files created: 0")
    print(
        "Database changes: temporary ranking "
        "fixture and image completion/failure "
        "data inserted and deleted"
    )
    print("Permanent generated_posts created: 0")
    print("publication_attempts created: 0")
    print("Telegram publication: not performed")
    print(
        "Image generation completion test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )