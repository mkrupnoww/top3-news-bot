import asyncio
from dataclasses import replace
from datetime import date, timedelta
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.generation_revision_reservation import (
    reserve_generation_revision,
)
from app.db.generation_selection import (
    GenerationTop3Selection,
    load_generation_top3,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    OPENAI_POST_REVISION_PROMPT_VERSION,
    OpenAITelegramPostGenerator,
)
from app.generation.revision_request_key import (
    create_generation_revision_request_key,
)


TEST_RANKING_RUN_ID = 18

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

EDITORIAL_COMMENT = (
    "Исправить редакционные замечания "
    "и не добавлять неподтверждённые факты."
)

REVISION_ISSUES = (
    "Использовать только факты из title и summary.",
    "Имена людей в русском тексте передавать кириллицей.",
)


class NoCallGenerationClient:
    """Клиент, запрещающий вызов модели."""

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Блокирует любой неожиданный API-вызов."""

        raise AssertionError(
            "OpenAI не должен вызываться "
            "в тесте revision reservation."
        )


def build_test_publication_date() -> date:
    """Создаёт изолированную дату тестового выпуска."""

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


async def load_test_reviewer(
    pool: asyncpg.Pool,
) -> int:
    """Возвращает активного администратора бота."""

    async with pool.acquire() as connection:
        reviewer_telegram_user_id = (
            await connection.fetchval(
                """
                SELECT telegram_user_id
                FROM top3_news.bot_users
                WHERE is_active = true
                  AND role = 'admin'
                ORDER BY telegram_user_id
                LIMIT 1
                """
            )
        )

    if reviewer_telegram_user_id is None:
        raise LookupError(
            "Для теста не найден активный "
            "bot_users admin."
        )

    return int(
        reviewer_telegram_user_id
    )


async def create_test_review_context(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    generator: OpenAITelegramPostGenerator,
    created_batch_ids: set[int],
) -> tuple[int, int, int]:
    """
    Создаёт временные batch, generated_post v1
    и changes_required.
    """

    publication_date = (
        build_test_publication_date()
    )

    batch_request_key = (
        build_test_batch_request_key()
    )

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
                        "test_name": (
                            "generation_revision_"
                            "reservation"
                        ),
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
                    INSERT INTO top3_news.batch_items (
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

            source_generated_post_id = (
                await connection.fetchval(
                    """
                    INSERT INTO
                        top3_news.generated_posts (
                            batch_id,
                            version_number,
                            post_status,
                            post_text,
                            text_format,
                            text_model_name,
                            text_prompt_version,
                            generation_metadata
                        )
                    VALUES (
                        $1,
                        1,
                        'awaiting_review',
                        $2,
                        $3,
                        $4,
                        $5,
                        $6::jsonb
                    )
                    RETURNING generated_post_id
                    """,
                    batch_id,
                    SOURCE_POST_TEXT,
                    generator.metadata.text_format,
                    generator.metadata.model_name,
                    generator.metadata.prompt_version,
                    json.dumps(
                        {
                            "test_mode": True,
                            "test_name": (
                                "generation_revision_"
                                "reservation"
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

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
                        'regenerate_text',
                        true,
                        $3,
                        $4::jsonb,
                        $5::jsonb
                    )
                    RETURNING review_action_id
                    """,
                    source_generated_post_id,
                    reviewer_telegram_user_id,
                    EDITORIAL_COMMENT,
                    json.dumps(
                        list(REVISION_ISSUES),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "test_mode": True,
                            "test_name": (
                                "generation_revision_"
                                "reservation"
                            ),
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
        int(source_generated_post_id),
        int(review_action_id),
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
            "Не удалось удалить тестовый "
            f"publication_batch: {result}"
        )


async def assert_test_batch_deleted(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> None:
    """Проверяет каскадное удаление тестовых данных."""

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
                    FROM top3_news.generation_revision_requests
                    WHERE batch_id = $1
                ) AS revision_request_count,

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
            "удаления тестового выпуска."
        )

    assert record["batch_exists"] is False
    assert record["batch_item_count"] == 0
    assert record["generated_post_count"] == 0
    assert record["revision_request_count"] == 0
    assert record["ranking_run_exists"] is True


async def test_revision_reservation(
    pool: asyncpg.Pool,
    *,
    telegram_chat_id: int,
    model_name: str,
    created_batch_ids: set[int],
) -> None:
    """Проверяет защищённое резервирование ревизии."""

    selection = await load_generation_top3(
        pool,
        ranking_run_id=TEST_RANKING_RUN_ID,
    )

    generator = OpenAITelegramPostGenerator(
        client=NoCallGenerationClient(),
        model_name=model_name,
    )

    reviewer_telegram_user_id = (
        await load_test_reviewer(pool)
    )

    (
        batch_id,
        source_generated_post_id,
        review_action_id,
    ) = await create_test_review_context(
        pool,
        selection=selection,
        telegram_chat_id=telegram_chat_id,
        reviewer_telegram_user_id=(
            reviewer_telegram_user_id
        ),
        generator=generator,
        created_batch_ids=created_batch_ids,
    )

    target_version_number = 2

    model_request = (
        generator.build_revision_request(
            selection.items,
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
        )
    )

    request_key = (
        create_generation_revision_request_key(
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=review_action_id,
            target_version_number=(
                target_version_number
            ),
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    first_reservation = (
        await reserve_generation_revision(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=review_action_id,
            target_version_number=(
                target_version_number
            ),
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    assert first_reservation.created_new is True
    assert first_reservation.should_call_model is True
    assert (
        first_reservation.revision_status
        == "reserved"
    )
    assert first_reservation.batch_id == batch_id
    assert (
        first_reservation.source_generated_post_id
        == source_generated_post_id
    )
    assert (
        first_reservation.review_action_id
        == review_action_id
    )
    assert (
        first_reservation.target_version_number
        == target_version_number
    )
    assert (
        first_reservation.request_key
        == request_key.value
    )

    print("Initial generation revision reservation: OK")
    print(
        "generation_revision_id="
        f"{first_reservation.generation_revision_id}"
    )
    print(f"batch_id={batch_id}")
    print(
        "source_generated_post_id="
        f"{source_generated_post_id}"
    )
    print(
        f"review_action_id={review_action_id}"
    )
    print("target_version_number=2")
    print("revision_status=reserved")
    print("created_new=true")
    print("should_call_model=true")

    second_reservation = (
        await reserve_generation_revision(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=review_action_id,
            target_version_number=(
                target_version_number
            ),
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    assert (
        second_reservation.generation_revision_id
        == first_reservation.generation_revision_id
    )
    assert second_reservation.created_new is False
    assert second_reservation.should_call_model is False
    assert (
        second_reservation.revision_status
        == "reserved"
    )

    print()
    print("Repeated generation revision reservation: OK")
    print("created_new=false")
    print("should_call_model=false")
    print("Duplicate paid request: blocked")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                grr.generation_revision_id,
                grr.batch_id,
                grr.source_generated_post_id,
                grr.review_action_id,
                grr.target_version_number,
                grr.revision_request_key,
                grr.request_key_version,
                grr.revision_status,
                grr.requested_action,
                grr.editorial_comment,
                grr.issues,
                grr.model_name,
                grr.generator_version,
                grr.prompt_version,
                grr.text_format,
                grr.request_payload,
                grr.openai_usage,
                grr.openai_cost,
                grr.generated_post_id,
                grr.error_type,
                grr.error_message,
                grr.completed_at,
                grr.failed_at,

                b.batch_status,

                gp.post_status
                    AS source_post_status,
                gp.version_number
                    AS source_version_number,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp2
                    WHERE gp2.batch_id = grr.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.publication_attempts AS pa
                    JOIN top3_news.generated_posts AS gp3
                      ON gp3.generated_post_id =
                         pa.generated_post_id
                    WHERE gp3.batch_id = grr.batch_id
                ) AS publication_attempt_count

            FROM top3_news.generation_revision_requests
                AS grr
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = grr.batch_id
            JOIN top3_news.generated_posts AS gp
              ON gp.generated_post_id =
                 grr.source_generated_post_id
            WHERE grr.generation_revision_id = $1
            """,
            first_reservation.generation_revision_id,
        )

        active_key_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.generation_revision_requests
                WHERE revision_request_key = $1
                  AND revision_status IN (
                      'reserved',
                      'completed'
                  )
                """,
                request_key.value,
            )
        )

    if record is None:
        raise AssertionError(
            "Revision reservation "
            "не найдена."
        )

    persisted_issues = decode_json_array(
        record["issues"],
        field_name="issues",
    )

    request_payload = decode_json_object(
        record["request_payload"],
        field_name="request_payload",
    )

    assert (
        record["generation_revision_id"]
        == first_reservation.generation_revision_id
    )
    assert record["batch_id"] == batch_id
    assert (
        record["source_generated_post_id"]
        == source_generated_post_id
    )
    assert (
        record["review_action_id"]
        == review_action_id
    )
    assert (
        record["target_version_number"]
        == 2
    )
    assert (
        record["revision_request_key"]
        == request_key.value
    )
    assert (
        record["request_key_version"]
        == request_key.version
    )
    assert (
        record["revision_status"]
        == "reserved"
    )
    assert (
        record["requested_action"]
        == "regenerate_text"
    )
    assert (
        record["editorial_comment"]
        == EDITORIAL_COMMENT
    )
    assert (
        persisted_issues
        == list(REVISION_ISSUES)
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
        == OPENAI_POST_REVISION_PROMPT_VERSION
    )
    assert (
        record["text_format"]
        == generator.metadata.text_format
    )
    assert (
        request_payload
        == json.loads(
            request_key.canonical_json
        )
    )
    assert record["openai_usage"] is None
    assert record["openai_cost"] is None
    assert record["generated_post_id"] is None
    assert record["error_type"] is None
    assert record["error_message"] is None
    assert record["completed_at"] is None
    assert record["failed_at"] is None
    assert (
        record["batch_status"]
        == "awaiting_review"
    )
    assert (
        record["source_post_status"]
        == "awaiting_review"
    )
    assert record["source_version_number"] == 1
    assert record["generated_post_count"] == 1
    assert record["publication_attempt_count"] == 0
    assert active_key_count == 1

    print()
    print("Persisted generation revision reservation: OK")
    print("batch_status=awaiting_review")
    print("source_post_status=awaiting_review")
    print("source_version_number=1")
    print("generated_post_count=1")
    print("publication_attempt_count=0")
    print("active_request_key_count=1")

    changed_request = replace(
        model_request,
        input_text=(
            model_request.input_text
            + "\nchanged-test-input"
        ),
    )

    try:
        await reserve_generation_revision(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=review_action_id,
            target_version_number=2,
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=changed_request,
            items=selection.items,
        )
    except ValueError as error:
        assert (
            "generation revision request_key "
            "не соответствует"
            in str(error)
        )

        print()
        print(
            "Changed prepared revision request "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Изменённый revision request "
            "не был заблокирован."
        )

    async with pool.acquire() as connection:
        failed_update = await connection.execute(
            """
            UPDATE top3_news.generation_revision_requests
            SET
                revision_status = 'failed',
                error_type = 'SyntheticTestFailure',
                error_message = (
                    'Synthetic reservation retry test'
                ),
                failed_at = now(),
                updated_at = now()
            WHERE generation_revision_id = $1
              AND revision_status = 'reserved'
            """,
            first_reservation.generation_revision_id,
        )

    if failed_update != "UPDATE 1":
        raise RuntimeError(
            "Не удалось перевести первую "
            "test reservation в failed: "
            f"{failed_update}"
        )

    retry_reservation = (
        await reserve_generation_revision(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=review_action_id,
            target_version_number=2,
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    assert retry_reservation.created_new is True
    assert retry_reservation.should_call_model is True
    assert (
        retry_reservation.revision_status
        == "reserved"
    )
    assert (
        retry_reservation.generation_revision_id
        != first_reservation.generation_revision_id
    )
    assert (
        retry_reservation.request_key
        == request_key.value
    )

    print()
    print("Failed generation revision retry: OK")
    print(
        "failed_generation_revision_id="
        f"{first_reservation.generation_revision_id}"
    )
    print(
        "retry_generation_revision_id="
        f"{retry_reservation.generation_revision_id}"
    )
    print("retry_created_new=true")
    print("retry_should_call_model=true")

    repeated_retry = (
        await reserve_generation_revision(
            pool,
            request_key=request_key,
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=review_action_id,
            target_version_number=2,
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    assert (
        repeated_retry.generation_revision_id
        == retry_reservation.generation_revision_id
    )
    assert repeated_retry.created_new is False
    assert repeated_retry.should_call_model is False

    async with pool.acquire() as connection:
        retry_rows = await connection.fetch(
            """
            SELECT
                generation_revision_id,
                revision_status
            FROM top3_news.generation_revision_requests
            WHERE revision_request_key = $1
            ORDER BY generation_revision_id
            """,
            request_key.value,
        )

        post_and_batch_status = (
            await connection.fetchrow(
                """
                SELECT
                    b.batch_status,
                    gp.post_status,
                    (
                        SELECT COUNT(*)::integer
                        FROM top3_news.generated_posts AS gp2
                        WHERE gp2.batch_id = b.batch_id
                    ) AS generated_post_count
                FROM top3_news.publication_batches AS b
                JOIN top3_news.generated_posts AS gp
                  ON gp.generated_post_id = $2
                WHERE b.batch_id = $1
                """,
                batch_id,
                source_generated_post_id,
            )
        )

    assert len(retry_rows) == 2
    assert (
        retry_rows[0]["revision_status"]
        == "failed"
    )
    assert (
        retry_rows[1]["revision_status"]
        == "reserved"
    )

    if post_and_batch_status is None:
        raise AssertionError(
            "Не удалось проверить статусы "
            "после failed retry."
        )

    assert (
        post_and_batch_status["batch_status"]
        == "awaiting_review"
    )
    assert (
        post_and_batch_status["post_status"]
        == "awaiting_review"
    )
    assert (
        post_and_batch_status[
            "generated_post_count"
        ]
        == 1
    )

    print()
    print("Repeated retry reservation: OK")
    print("retry_created_new=false")
    print("retry_should_call_model=false")
    print("revision_request_row_count=2")
    print("failed_revision_count=1")
    print("reserved_revision_count=1")
    print("batch_status_after_retry=awaiting_review")
    print("source_post_status_after_retry=awaiting_review")
    print("generated_post_count_after_retry=1")


async def cleanup_test_batches(
    pool: asyncpg.Pool,
    *,
    created_batch_ids: set[int],
) -> None:
    """Удаляет все созданные тестом выпуски."""

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
            "temporary_revision_requests_deleted=true"
        )
        print(
            "ranking_run_18_preserved=true"
        )


async def main() -> int:
    """Запускает интеграционный тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    created_batch_ids: set[int] = set()

    try:
        await test_revision_reservation(
            pool,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            model_name=(
                settings.openai_generation_model
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
            await close_database_pool(pool)

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print(
        "Database changes: temporary revision "
        "reservation data inserted and deleted"
    )
    print(
        "Permanent generated_posts created: 0"
    )
    print("publication_attempts created: 0")
    print("Telegram publication: not performed")
    print(
        "Generation revision reservation "
        "test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )