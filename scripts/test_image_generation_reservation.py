import asyncio
from datetime import date, timedelta
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.generation_selection import (
    GenerationTop3Selection,
    load_generation_top3,
)
from app.db.image_generation_reservation import (
    reserve_image_generation,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.image_generator import (
    ImageGenerationNewsItem,
    ImageModelRequest,
    ImageModelResponse,
    OpenAIMovieNewsImageGenerator,
)
from app.generation.image_request_key import (
    create_image_request_key,
)


TEST_RANKING_RUN_ID = 18

TEST_IMAGE_MODEL_NAME = (
    "synthetic-image-model"
)

TEST_IMAGE_SIZE = "1024x1536"

TEST_TEXT_MODEL_NAME = (
    "synthetic-text-model"
)

TEST_TEXT_PROMPT_VERSION = (
    "synthetic-text-prompt-v1"
)

SOURCE_POST_TEXT = (
    "**TOP-3 НОВОСТЕЙ КИНО "
    "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n"
    "_______________\n\n"
    "1️⃣ **Тестовый заголовок первой новости**\n\n"
    "Тестовый текст первой новости.\n\n"
    "2️⃣ **Тестовый заголовок второй новости**\n\n"
    "Тестовый текст второй новости.\n\n"
    "3️⃣ **Тестовый заголовок третьей новости**\n\n"
    "Тестовый текст третьей новости."
)

EXISTING_IMAGE_PATH = (
    "data/images/generated/"
    "synthetic-existing-image.png"
)

EXISTING_IMAGE_SHA256 = "a" * 64

EXISTING_IMAGE_PROMPT = (
    "Synthetic existing image prompt."
)

EXISTING_IMAGE_MODEL_NAME = (
    "synthetic-existing-image-model"
)

EXISTING_IMAGE_PROMPT_VERSION = (
    "movie_news_image_v0"
)

EDITORIAL_COMMENT = (
    "Исправить визуальные замечания, "
    "сохранив смысл трёх исходных новостей."
)

IMAGE_ISSUES = (
    "Экран ноутбука должен находиться "
    "с правильной стороны устройства.",
    "Не смешивать объекты разных новостей "
    "между горизонтальными зонами.",
)


class NoCallImageGenerationClient:
    """Клиент, запрещающий вызов Image API."""

    async def create_image(
        self,
        request: ImageModelRequest,
    ) -> ImageModelResponse:
        """Блокирует любой неожиданный API-вызов."""

        raise AssertionError(
            "OpenAI Image API не должен "
            "вызываться в reservation-тесте."
        )


def build_test_publication_date() -> date:
    """Создаёт изолированную дату выпуска."""

    random_offset = (
        int(uuid4().hex[:8], 16)
        % 30000
    )

    return (
        date(2200, 1, 1)
        + timedelta(days=random_offset)
    )


def build_test_batch_request_key() -> str:
    """Создаёт уникальный ключ временного batch."""

    return sha256(
        uuid4().bytes
    ).hexdigest()


def decode_json_array(
    value: Any,
    *,
    field_name: str,
) -> list[Any]:
    """Преобразует jsonb-массив в Python list."""

    decoded_value: Any

    if isinstance(value, str):
        try:
            decoded_value = json.loads(
                value
            )
        except json.JSONDecodeError as error:
            raise AssertionError(
                f"{field_name} содержит "
                "некорректный JSON."
            ) from error
    else:
        decoded_value = value

    if not isinstance(
        decoded_value,
        list,
    ):
        raise AssertionError(
            f"{field_name} не является "
            "JSON-массивом."
        )

    return decoded_value


def decode_json_object(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """Преобразует jsonb-объект в Python dict."""

    decoded_value: Any

    if isinstance(value, str):
        try:
            decoded_value = json.loads(
                value
            )
        except json.JSONDecodeError as error:
            raise AssertionError(
                f"{field_name} содержит "
                "некорректный JSON."
            ) from error
    else:
        decoded_value = value

    if not isinstance(
        decoded_value,
        dict,
    ):
        raise AssertionError(
            f"{field_name} не является "
            "JSON-объектом."
        )

    return decoded_value


def build_image_items(
    selection: GenerationTop3Selection,
) -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """Строит factual input image-generator."""

    if len(selection.items) != 3:
        raise ValueError(
            "Для image-теста требуется "
            "ровно три новости."
        )

    items = tuple(
        ImageGenerationNewsItem(
            position=item.position,
            news_id=item.news_id,
            title=item.title,
            summary=item.summary,
        )
        for item in selection.items
    )

    return (
        items[0],
        items[1],
        items[2],
    )


async def assert_migration_applied(
    pool: asyncpg.Pool,
) -> None:
    """Проверяет применение migration 010."""

    async with pool.acquire() as connection:
        migration_exists = (
            await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM top3_news.schema_migrations
                    WHERE version = '010'
                )
                """
            )
        )

        table_exists = (
            await connection.fetchval(
                """
                SELECT to_regclass(
                    'top3_news.image_generation_requests'
                ) IS NOT NULL
                """
            )
        )

    if migration_exists is not True:
        raise RuntimeError(
            "Migration 010 не зарегистрирована "
            "в schema_migrations."
        )

    if table_exists is not True:
        raise RuntimeError(
            "Таблица image_generation_requests "
            "не существует."
        )


async def load_test_reviewer(
    pool: asyncpg.Pool,
) -> int:
    """Возвращает активного администратора."""

    async with pool.acquire() as connection:
        reviewer_telegram_user_id = (
            await connection.fetchval(
                """
                SELECT telegram_user_id
                FROM top3_news.bot_users
                WHERE is_active = true
                  AND user_role = 'admin'
                ORDER BY telegram_user_id
                LIMIT 1
                """
            )
        )

    if reviewer_telegram_user_id is None:
        raise LookupError(
            "Для regenerate-теста не найден "
            "активный bot_users admin."
        )

    return int(
        reviewer_telegram_user_id
    )


async def create_test_batch_and_post(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    telegram_chat_id: int,
    existing_image: bool,
    test_name: str,
    created_batch_ids: set[int],
) -> tuple[int, int]:
    """Создаёт временный batch и generated_post."""

    publication_date = (
        build_test_publication_date()
    )

    batch_request_key = (
        build_test_batch_request_key()
    )

    if existing_image:
        image_path: str | None = (
            EXISTING_IMAGE_PATH
        )
        image_sha256: str | None = (
            EXISTING_IMAGE_SHA256
        )
        image_prompt: str | None = (
            EXISTING_IMAGE_PROMPT
        )
        image_model_name: str | None = (
            EXISTING_IMAGE_MODEL_NAME
        )
        image_prompt_version: str | None = (
            EXISTING_IMAGE_PROMPT_VERSION
        )
    else:
        image_path = None
        image_sha256 = None
        image_prompt = None
        image_model_name = None
        image_prompt_version = None

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    $1::bigint
                )
                """,
                publication_date.toordinal(),
            )

            edition = await connection.fetchval(
                """
                SELECT
                    COALESCE(
                        MAX(edition),
                        0
                    )::integer + 1
                FROM top3_news.publication_batches
                WHERE publication_date = $1
                """,
                publication_date,
            )

            batch_id = await connection.fetchval(
                """
                INSERT INTO
                    top3_news.publication_batches (
                        publication_date,
                        edition,
                        ranking_run_id,
                        batch_status,
                        target_telegram_chat_id,
                        metadata,
                        generation_request_key
                    )
                VALUES (
                    $1,
                    $2,
                    $3,
                    'awaiting_review',
                    $4,
                    $5::jsonb,
                    $6
                )
                RETURNING batch_id
                """,
                publication_date,
                edition,
                selection.ranking_run_id,
                telegram_chat_id,
                json.dumps(
                    {
                        "test_mode": True,
                        "test_name": test_name,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                batch_request_key,
            )

            for item, score_id in zip(
                selection.items,
                selection.score_ids,
                strict=True,
            ):
                await connection.execute(
                    """
                    INSERT INTO
                        top3_news.batch_items (
                            batch_id,
                            news_id,
                            score_id,
                            position,
                            selection_reason
                        )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5
                    )
                    """,
                    batch_id,
                    item.news_id,
                    score_id,
                    item.position,
                    item.selection_reason,
                )

            generated_post_id = (
                await connection.fetchval(
                    """
                    INSERT INTO
                        top3_news.generated_posts (
                            batch_id,
                            version_number,
                            post_status,
                            post_text,
                            text_format,
                            image_path,
                            image_sha256,
                            image_prompt,
                            text_model_name,
                            image_model_name,
                            text_prompt_version,
                            image_prompt_version,
                            generation_metadata
                        )
                    VALUES (
                        $1,
                        1,
                        'awaiting_review',
                        $2,
                        'markdown',
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10::jsonb
                    )
                    RETURNING generated_post_id
                    """,
                    batch_id,
                    SOURCE_POST_TEXT,
                    image_path,
                    image_sha256,
                    image_prompt,
                    TEST_TEXT_MODEL_NAME,
                    image_model_name,
                    TEST_TEXT_PROMPT_VERSION,
                    image_prompt_version,
                    json.dumps(
                        {
                            "test_mode": True,
                            "test_name": test_name,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

    created_batch_ids.add(
        int(batch_id)
    )

    return (
        int(batch_id),
        int(generated_post_id),
    )


async def create_regenerate_review_action(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
    reviewer_telegram_user_id: int,
    test_name: str,
) -> int:
    """Создаёт human regenerate_image action."""

    async with pool.acquire() as connection:
        review_action_id = (
            await connection.fetchval(
                """
                INSERT INTO
                    top3_news.review_actions (
                        generated_post_id,
                        reviewer_type,
                        reviewer_telegram_user_id,
                        decision,
                        requested_action,
                        requires_human_review,
                        comment_text,
                        issues,
                        review_details
                    )
                VALUES (
                    $1,
                    'human',
                    $2,
                    'changes_required',
                    'regenerate_image',
                    true,
                    $3,
                    $4::jsonb,
                    $5::jsonb
                )
                RETURNING review_action_id
                """,
                generated_post_id,
                reviewer_telegram_user_id,
                EDITORIAL_COMMENT,
                json.dumps(
                    list(IMAGE_ISSUES),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                json.dumps(
                    {
                        "test_mode": True,
                        "test_name": test_name,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        )

    return int(
        review_action_id
    )


async def delete_test_batch(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> None:
    """Удаляет временный тестовый выпуск."""

    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            DELETE FROM top3_news.publication_batches
            WHERE batch_id = $1
            """,
            batch_id,
        )

    if result != "DELETE 1":
        raise RuntimeError(
            "Не удалось удалить test "
            f"publication_batch: {result}"
        )


async def assert_test_batch_deleted(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> None:
    """Проверяет каскадное удаление."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                EXISTS (
                    SELECT 1
                    FROM top3_news.publication_batches
                    WHERE batch_id = $1
                ) AS batch_exists,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.batch_items
                    WHERE batch_id = $1
                ) AS batch_item_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts
                    WHERE batch_id = $1
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.image_generation_requests
                    WHERE batch_id = $1
                ) AS image_request_count,

                EXISTS (
                    SELECT 1
                    FROM top3_news.ranking_runs
                    WHERE ranking_run_id = $2
                ) AS ranking_run_exists
            """,
            batch_id,
            TEST_RANKING_RUN_ID,
        )

    if record is None:
        raise AssertionError(
            "Не получен результат проверки "
            "удаления тестового batch."
        )

    assert record["batch_exists"] is False
    assert record["batch_item_count"] == 0
    assert record["generated_post_count"] == 0
    assert record["image_request_count"] == 0
    assert record["ranking_run_exists"] is True


async def test_initial_reservation(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    telegram_chat_id: int,
    created_batch_ids: set[int],
) -> None:
    """Проверяет initial reservation и failed retry."""

    generator = OpenAIMovieNewsImageGenerator(
        client=NoCallImageGenerationClient(),
        model_name=TEST_IMAGE_MODEL_NAME,
        size=TEST_IMAGE_SIZE,
    )

    items = build_image_items(
        selection
    )

    (
        batch_id,
        generated_post_id,
    ) = await create_test_batch_and_post(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        existing_image=False,
        test_name=(
            "image_generation_initial_reservation"
        ),
        created_batch_ids=created_batch_ids,
    )

    model_request = generator.build_request(
        items=items,
    )

    request_key = create_image_request_key(
        batch_id=batch_id,
        ranking_run_id=selection.ranking_run_id,
        request_kind="initial",
        review_action_id=None,
        metadata=generator.metadata,
        model_request=model_request,
        items=items,
    )

    first_reservation = (
        await reserve_image_generation(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="initial",
            review_action_id=None,
            editorial_comment=None,
            issues=(),
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    assert first_reservation.created_new is True
    assert (
        first_reservation.should_call_model
        is True
    )
    assert (
        first_reservation.image_status
        == "reserved"
    )
    assert first_reservation.batch_id == batch_id
    assert (
        first_reservation.generated_post_id
        == generated_post_id
    )
    assert (
        first_reservation.ranking_run_id
        == selection.ranking_run_id
    )
    assert (
        first_reservation.request_kind
        == "initial"
    )
    assert (
        first_reservation.review_action_id
        is None
    )
    assert (
        first_reservation.request_key
        == request_key.value
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                image_generation_id,
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
                request_payload,
                response_metadata,
                openai_usage,
                openai_cost,
                image_path,
                image_sha256,
                completed_at,
                failed_at,
                error_type,
                error_message
            FROM top3_news.image_generation_requests
            WHERE image_generation_id = $1
            """,
            first_reservation.image_generation_id,
        )

    if record is None:
        raise AssertionError(
            "Не найдена initial image reservation."
        )

    assert record["batch_id"] == batch_id
    assert (
        record["generated_post_id"]
        == generated_post_id
    )
    assert record["review_action_id"] is None
    assert (
        record["image_request_key"]
        == request_key.value
    )
    assert (
        record["request_key_version"]
        == request_key.version
    )
    assert record["image_status"] == "reserved"
    assert record["request_kind"] == "initial"
    assert record["editorial_comment"] is None
    assert (
        decode_json_array(
            record["issues"],
            field_name="issues",
        )
        == []
    )
    assert (
        record["model_name"]
        == generator.metadata.model_name
    )
    assert (
        record["generator_version"]
        == generator.metadata.generator_version
    )
    assert (
        record["prompt_version"]
        == generator.metadata.prompt_version
    )
    assert (
        record["image_size"]
        == model_request.size
    )
    assert (
        record["image_quality"]
        == model_request.quality
    )
    assert record["output_format"] == "png"
    assert record["background"] == "opaque"
    assert record["moderation"] == "auto"
    assert record["image_count"] == 1
    assert record["response_metadata"] is None
    assert record["openai_usage"] is None
    assert record["openai_cost"] is None
    assert record["image_path"] is None
    assert record["image_sha256"] is None
    assert record["completed_at"] is None
    assert record["failed_at"] is None
    assert record["error_type"] is None
    assert record["error_message"] is None

    request_payload = decode_json_object(
        record["request_payload"],
        field_name="request_payload",
    )

    assert (
        request_payload[
            "image_request_key_version"
        ]
        == request_key.version
    )
    assert request_payload["batch_id"] == batch_id
    assert (
        request_payload["ranking_run_id"]
        == selection.ranking_run_id
    )
    assert (
        request_payload["request_kind"]
        == "initial"
    )
    assert (
        request_payload["review_action_id"]
        is None
    )
    assert (
        request_payload["top3_news_ids"]
        == list(selection.news_ids)
    )
    assert "top3" not in request_payload
    assert (
        request_payload["model_request"][
            "prompt"
        ]
        == model_request.prompt
    )

    print("Initial image reservation: OK")
    print(
        "image_generation_id="
        f"{first_reservation.image_generation_id}"
    )
    print(f"batch_id={batch_id}")
    print(
        f"generated_post_id={generated_post_id}"
    )
    print("image_status=reserved")
    print("request_kind=initial")
    print("created_new=true")
    print("should_call_model=true")

    repeated_reservation = (
        await reserve_image_generation(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="initial",
            review_action_id=None,
            editorial_comment=None,
            issues=(),
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    assert (
        repeated_reservation.image_generation_id
        == first_reservation.image_generation_id
    )
    assert (
        repeated_reservation.created_new
        is False
    )
    assert (
        repeated_reservation.should_call_model
        is False
    )
    assert (
        repeated_reservation.image_status
        == "reserved"
    )

    print()
    print("Repeated initial reservation: OK")
    print("created_new=false")
    print("should_call_model=false")

    conflict_generator = (
        OpenAIMovieNewsImageGenerator(
            client=NoCallImageGenerationClient(),
            model_name=TEST_IMAGE_MODEL_NAME,
            size=TEST_IMAGE_SIZE,
            quality="high",
        )
    )

    conflict_model_request = (
        conflict_generator.build_request(
            items=items,
        )
    )

    conflict_request_key = (
        create_image_request_key(
            batch_id=batch_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="initial",
            review_action_id=None,
            metadata=conflict_generator.metadata,
            model_request=(
                conflict_model_request
            ),
            items=items,
        )
    )

    assert (
        conflict_request_key.value
        != request_key.value
    )

    try:
        await reserve_image_generation(
            pool,
            request_key=conflict_request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="initial",
            review_action_id=None,
            editorial_comment=None,
            issues=(),
            metadata=conflict_generator.metadata,
            model_request=(
                conflict_model_request
            ),
            items=items,
        )
    except ValueError as error:
        if "active reservation" not in str(error):
            raise AssertionError(
                "Получена неожиданная ошибка "
                "при проверке конфликта."
            ) from error
    else:
        raise AssertionError(
            "Конфликтующая initial reservation "
            "не была заблокирована."
        )

    async with pool.acquire() as connection:
        conflict_row_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.image_generation_requests
                WHERE image_request_key = $1
                """,
                conflict_request_key.value,
            )
        )

    assert conflict_row_count == 0

    print()
    print("Conflicting initial reservation: OK")
    print("conflicting_request_inserted=false")

    async with pool.acquire() as connection:
        failed_update = (
            await connection.execute(
                """
                UPDATE
                    top3_news.image_generation_requests
                SET
                    image_status = 'failed',
                    error_type = (
                        'SyntheticImageReservationError'
                    ),
                    error_message = (
                        'Synthetic image reservation '
                        'retry test'
                    ),
                    failed_at = now(),
                    updated_at = now()
                WHERE image_generation_id = $1
                  AND image_status = 'reserved'
                """,
                first_reservation.image_generation_id,
            )
        )

    if failed_update != "UPDATE 1":
        raise RuntimeError(
            "Не удалось перевести initial "
            "reservation в failed: "
            f"{failed_update}"
        )

    retry_reservation = (
        await reserve_image_generation(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="initial",
            review_action_id=None,
            editorial_comment=None,
            issues=(),
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    assert retry_reservation.created_new is True
    assert (
        retry_reservation.should_call_model
        is True
    )
    assert (
        retry_reservation.image_status
        == "reserved"
    )
    assert (
        retry_reservation.image_generation_id
        != first_reservation.image_generation_id
    )
    assert (
        retry_reservation.request_key
        == request_key.value
    )

    print()
    print("Failed initial reservation retry: OK")
    print(
        "failed_image_generation_id="
        f"{first_reservation.image_generation_id}"
    )
    print(
        "retry_image_generation_id="
        f"{retry_reservation.image_generation_id}"
    )
    print("retry_created_new=true")
    print("retry_should_call_model=true")

    repeated_retry = (
        await reserve_image_generation(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="initial",
            review_action_id=None,
            editorial_comment=None,
            issues=(),
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    assert (
        repeated_retry.image_generation_id
        == retry_reservation.image_generation_id
    )
    assert repeated_retry.created_new is False
    assert (
        repeated_retry.should_call_model is False
    )

    async with pool.acquire() as connection:
        retry_rows = await connection.fetch(
            """
            SELECT
                image_generation_id,
                image_status
            FROM top3_news.image_generation_requests
            WHERE image_request_key = $1
            ORDER BY image_generation_id
            """,
            request_key.value,
        )

        post_record = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                gp.post_status,
                gp.image_path,
                gp.image_sha256,
                gp.image_prompt,
                gp.image_model_name,
                gp.image_prompt_version
            FROM top3_news.publication_batches AS b
            JOIN top3_news.generated_posts AS gp
              ON gp.generated_post_id = $2
            WHERE b.batch_id = $1
            """,
            batch_id,
            generated_post_id,
        )

    assert len(retry_rows) == 2
    assert (
        retry_rows[0]["image_status"]
        == "failed"
    )
    assert (
        retry_rows[1]["image_status"]
        == "reserved"
    )

    if post_record is None:
        raise AssertionError(
            "Не найден generated_post "
            "после failed retry."
        )

    assert (
        post_record["batch_status"]
        == "awaiting_review"
    )
    assert (
        post_record["post_status"]
        == "awaiting_review"
    )
    assert post_record["image_path"] is None
    assert post_record["image_sha256"] is None
    assert post_record["image_prompt"] is None
    assert (
        post_record["image_model_name"]
        is None
    )
    assert (
        post_record["image_prompt_version"]
        is None
    )

    print()
    print("Repeated failed retry reservation: OK")
    print("image_request_row_count=2")
    print("failed_image_request_count=1")
    print("reserved_image_request_count=1")
    print("generated_post_image_fields_unchanged=true")


async def test_regenerate_reservation(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    created_batch_ids: set[int],
) -> None:
    """Проверяет regenerate_image reservation."""

    generator = OpenAIMovieNewsImageGenerator(
        client=NoCallImageGenerationClient(),
        model_name=TEST_IMAGE_MODEL_NAME,
        size=TEST_IMAGE_SIZE,
    )

    items = build_image_items(
        selection
    )

    (
        batch_id,
        generated_post_id,
    ) = await create_test_batch_and_post(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        existing_image=True,
        test_name=(
            "image_generation_regenerate_reservation"
        ),
        created_batch_ids=created_batch_ids,
    )

    review_action_id = (
        await create_regenerate_review_action(
            pool,
            generated_post_id=generated_post_id,
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            test_name=(
                "image_generation_regenerate_"
                "reservation"
            ),
        )
    )

    model_request = generator.build_request(
        items=items,
        editorial_comment=EDITORIAL_COMMENT,
        issues=IMAGE_ISSUES,
    )

    request_key = create_image_request_key(
        batch_id=batch_id,
        ranking_run_id=selection.ranking_run_id,
        request_kind="regenerate",
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
        ranking_run_id=selection.ranking_run_id,
        request_kind="regenerate",
        review_action_id=review_action_id,
        editorial_comment=EDITORIAL_COMMENT,
        issues=IMAGE_ISSUES,
        metadata=generator.metadata,
        model_request=model_request,
        items=items,
    )

    assert reservation.created_new is True
    assert reservation.should_call_model is True
    assert reservation.image_status == "reserved"
    assert reservation.batch_id == batch_id
    assert (
        reservation.generated_post_id
        == generated_post_id
    )
    assert (
        reservation.review_action_id
        == review_action_id
    )
    assert (
        reservation.request_kind
        == "regenerate"
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                igr.image_generation_id,
                igr.image_status,
                igr.request_kind,
                igr.review_action_id,
                igr.editorial_comment,
                igr.issues,
                igr.request_payload,

                gp.image_path,
                gp.image_sha256,
                gp.image_prompt,
                gp.image_model_name,
                gp.image_prompt_version

            FROM top3_news.image_generation_requests AS igr
            JOIN top3_news.generated_posts AS gp
              ON gp.generated_post_id =
                 igr.generated_post_id
            WHERE igr.image_generation_id = $1
            """,
            reservation.image_generation_id,
        )

    if record is None:
        raise AssertionError(
            "Не найдена regenerate reservation."
        )

    assert record["image_status"] == "reserved"
    assert record["request_kind"] == "regenerate"
    assert (
        record["review_action_id"]
        == review_action_id
    )
    assert (
        record["editorial_comment"]
        == EDITORIAL_COMMENT
    )
    assert (
        decode_json_array(
            record["issues"],
            field_name="issues",
        )
        == list(IMAGE_ISSUES)
    )

    payload = decode_json_object(
        record["request_payload"],
        field_name="request_payload",
    )

    assert (
        payload["request_kind"]
        == "regenerate"
    )
    assert (
        payload["review_action_id"]
        == review_action_id
    )
    assert (
        payload["top3_news_ids"]
        == list(selection.news_ids)
    )
    assert "top3" not in payload
    assert (
        '"editorial_revision"'
        in payload["model_request"]["prompt"]
    )

    assert (
        record["image_path"]
        == EXISTING_IMAGE_PATH
    )
    assert (
        record["image_sha256"]
        == EXISTING_IMAGE_SHA256
    )
    assert (
        record["image_prompt"]
        == EXISTING_IMAGE_PROMPT
    )
    assert (
        record["image_model_name"]
        == EXISTING_IMAGE_MODEL_NAME
    )
    assert (
        record["image_prompt_version"]
        == EXISTING_IMAGE_PROMPT_VERSION
    )

    print()
    print("Regenerate image reservation: OK")
    print(
        "image_generation_id="
        f"{reservation.image_generation_id}"
    )
    print(f"batch_id={batch_id}")
    print(
        f"generated_post_id={generated_post_id}"
    )
    print(
        f"review_action_id={review_action_id}"
    )
    print("request_kind=regenerate")
    print("image_status=reserved")
    print("existing_image_unchanged=true")
    print("should_call_model=true")

    repeated_reservation = (
        await reserve_image_generation(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="regenerate",
            review_action_id=review_action_id,
            editorial_comment=EDITORIAL_COMMENT,
            issues=IMAGE_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    assert (
        repeated_reservation.image_generation_id
        == reservation.image_generation_id
    )
    assert (
        repeated_reservation.created_new
        is False
    )
    assert (
        repeated_reservation.should_call_model
        is False
    )

    print()
    print("Repeated regenerate reservation: OK")
    print("created_new=false")
    print("should_call_model=false")

    second_review_action_id = (
        await create_regenerate_review_action(
            pool,
            generated_post_id=generated_post_id,
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            test_name=(
                "image_generation_regenerate_"
                "conflict"
            ),
        )
    )

    second_request_key = (
        create_image_request_key(
            batch_id=batch_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="regenerate",
            review_action_id=(
                second_review_action_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    assert (
        second_request_key.value
        != request_key.value
    )

    try:
        await reserve_image_generation(
            pool,
            request_key=second_request_key,
            batch_id=batch_id,
            generated_post_id=generated_post_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind="regenerate",
            review_action_id=(
                second_review_action_id
            ),
            editorial_comment=EDITORIAL_COMMENT,
            issues=IMAGE_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    except ValueError as error:
        if "active reservation" not in str(error):
            raise AssertionError(
                "Получена неожиданная ошибка "
                "при regenerate conflict."
            ) from error
    else:
        raise AssertionError(
            "Вторая одновременная regenerate "
            "reservation не была заблокирована."
        )

    async with pool.acquire() as connection:
        conflicting_row_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.image_generation_requests
                WHERE image_request_key = $1
                """,
                second_request_key.value,
            )
        )

    assert conflicting_row_count == 0

    print()
    print("Concurrent regenerate conflict: OK")
    print(
        "second_review_action_id="
        f"{second_review_action_id}"
    )
    print("second_image_request_inserted=false")


async def cleanup_test_batches(
    pool: asyncpg.Pool,
    *,
    created_batch_ids: set[int],
) -> None:
    """Удаляет все созданные тестовые выпуски."""

    for batch_id in sorted(
        created_batch_ids
    ):
        await delete_test_batch(
            pool,
            batch_id=batch_id,
        )

        await assert_test_batch_deleted(
            pool,
            batch_id=batch_id,
        )

        print()
        print("Test data cleanup: OK")
        print(
            f"temporary_batch_id={batch_id}"
        )
        print("temporary_batch_deleted=true")
        print(
            "temporary_image_requests_deleted=true"
        )
        print(
            "ranking_run_18_preserved=true"
        )


async def main() -> int:
    """Запускает интеграционный reservation-тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    created_batch_ids: set[int] = set()

    try:
        await assert_migration_applied(
            pool
        )

        selection = await load_generation_top3(
            pool,
            ranking_run_id=TEST_RANKING_RUN_ID,
        )

        reviewer_telegram_user_id = (
            await load_test_reviewer(
                pool
            )
        )

        await test_initial_reservation(
            pool,
            selection=selection,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )

        await test_regenerate_reservation(
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
            await cleanup_test_batches(
                pool,
                created_batch_ids=(
                    created_batch_ids
                ),
            )
        finally:
            await close_database_pool(
                pool
            )

    print()
    print("API key required: no")
    print("OpenAI Image requests: not performed")
    print("PNG files created: 0")
    print(
        "Database changes: temporary image "
        "reservation data inserted and deleted"
    )
    print("Permanent generated_posts created: 0")
    print("publication_attempts created: 0")
    print("Telegram publication: not performed")
    print(
        "Image generation reservation test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )