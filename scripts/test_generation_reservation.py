import asyncio
from dataclasses import replace
from datetime import date, timedelta
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.generation_reservation import (
    reserve_generation,
)
from app.db.generation_selection import (
    load_generation_top3,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    OpenAITelegramPostGenerator,
)
from app.generation.request_key import (
    create_generation_request_key,
)


TEST_RANKING_RUN_ID = 18


class NoCallGenerationClient:
    """Клиент, запрещающий вызов модели."""

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Блокирует любой неожиданный API-вызов."""

        raise AssertionError(
            "OpenAI не должен вызываться "
            "в тесте резервирования."
        )


def build_test_publication_date() -> date:
    """Создаёт изолированную дату тестового выпуска."""

    random_offset = (
        int(uuid4().hex[:8], 16)
        % 30000
    )

    return (
        date(2100, 1, 1)
        + timedelta(days=random_offset)
    )


def decode_jsonb_integer_array(
    value: Any,
    *,
    field_name: str,
) -> list[int]:
    """Преобразует jsonb-массив в список int."""

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

    result: list[int] = []

    for item in decoded_value:
        if isinstance(item, bool):
            raise AssertionError(
                f"{field_name} содержит bool."
            )

        if not isinstance(item, int):
            raise AssertionError(
                f"{field_name} содержит "
                "значение не типа int."
            )

        result.append(item)

    return result


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
    """Проверяет каскадное удаление выпуска."""

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

    assert (
        record["generated_post_count"]
        == 0
    )

    assert (
        record["ranking_run_exists"]
        is True
    )


async def test_reservation(
    pool: asyncpg.Pool,
    *,
    telegram_chat_id: int,
    model_name: str,
    created_batch_ids: set[int],
) -> None:
    """Проверяет резервирование генерации."""

    selection = await load_generation_top3(
        pool,
        ranking_run_id=(
            TEST_RANKING_RUN_ID
        ),
    )

    generator = (
        OpenAITelegramPostGenerator(
            client=NoCallGenerationClient(),
            model_name=model_name,
        )
    )

    model_request = generator.build_request(
        selection.items
    )

    publication_date = (
        build_test_publication_date()
    )

    request_key = (
        create_generation_request_key(
            ranking_run_id=(
                selection.ranking_run_id
            ),
            publication_date=publication_date,
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    first_reservation = (
        await reserve_generation(
            pool,
            request_key=request_key,
            selection=selection,
            publication_date=publication_date,
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
        )
    )

    # Сохраняем ID сразу, чтобы finally
    # удалил запись при любом дальнейшем сбое.
    created_batch_ids.add(
        first_reservation.batch_id
    )

    assert (
        first_reservation.created_new
        is True
    )

    assert (
        first_reservation.should_call_model
        is True
    )

    assert (
        first_reservation.batch_status
        == "ranked"
    )

    assert (
        first_reservation.ranking_run_id
        == TEST_RANKING_RUN_ID
    )

    assert (
        first_reservation.request_key
        == request_key.value
    )

    assert (
        first_reservation.publication_date
        == publication_date
    )

    assert first_reservation.edition > 0

    assert (
        first_reservation.news_ids
        == selection.news_ids
    )

    assert (
        first_reservation.score_ids
        == selection.score_ids
    )

    print("Initial generation reservation: OK")
    print(
        "batch_id="
        f"{first_reservation.batch_id}"
    )
    print(
        "publication_date="
        f"{first_reservation.publication_date}"
    )
    print(
        f"edition={first_reservation.edition}"
    )
    print(
        "ranking_run_id="
        f"{first_reservation.ranking_run_id}"
    )
    print(
        "generation_request_key="
        f"{first_reservation.request_key}"
    )
    print("created_new=true")
    print("should_call_model=true")
    print("batch_status=ranked")

    second_reservation = (
        await reserve_generation(
            pool,
            request_key=request_key,
            selection=selection,
            publication_date=publication_date,
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
        )
    )

    assert (
        second_reservation.batch_id
        == first_reservation.batch_id
    )

    assert (
        second_reservation.created_new
        is False
    )

    assert (
        second_reservation.should_call_model
        is False
    )

    assert (
        second_reservation.batch_status
        == "ranked"
    )

    assert (
        second_reservation.news_ids
        == selection.news_ids
    )

    assert (
        second_reservation.score_ids
        == selection.score_ids
    )

    print()
    print("Repeated generation reservation: OK")
    print(
        "batch_id="
        f"{second_reservation.batch_id}"
    )
    print("created_new=false")
    print("should_call_model=false")
    print("Duplicate paid request: blocked")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_id,
                b.publication_date,
                b.edition,
                b.batch_status,
                b.ranking_run_id,
                b.target_telegram_chat_id,
                b.generation_request_key,

                b.metadata->>'generation_mode'
                    AS generation_mode,

                b.metadata->>'generator_name'
                    AS generator_name,

                b.metadata->>'generator_version'
                    AS generator_version,

                b.metadata->>'prompt_version'
                    AS prompt_version,

                b.metadata->>'model_name'
                    AS model_name,

                b.metadata->>'text_format'
                    AS text_format,

                b.metadata->>'generation_request_key'
                    AS metadata_request_key,

                b.metadata->>'generation_request_key_version'
                    AS request_key_version,

                (
                    b.metadata->>'ranking_run_id'
                )::bigint
                    AS metadata_ranking_run_id,

                (
                    b.metadata->>'news_count'
                )::integer
                    AS news_count,

                (
                    b.metadata->>'idempotency_reserved'
                )::boolean
                    AS idempotency_reserved,

                b.metadata->'news_ids'
                    AS metadata_news_ids,

                b.metadata->'score_ids'
                    AS metadata_score_ids,

                ARRAY(
                    SELECT bi.position
                    FROM top3_news.batch_items AS bi
                    WHERE bi.batch_id = b.batch_id
                    ORDER BY bi.position
                ) AS positions,

                ARRAY(
                    SELECT bi.news_id
                    FROM top3_news.batch_items AS bi
                    WHERE bi.batch_id = b.batch_id
                    ORDER BY bi.position
                ) AS news_ids,

                ARRAY(
                    SELECT bi.score_id
                    FROM top3_news.batch_items AS bi
                    WHERE bi.batch_id = b.batch_id
                    ORDER BY bi.position
                ) AS score_ids,

                ARRAY(
                    SELECT bi.selection_reason
                    FROM top3_news.batch_items AS bi
                    WHERE bi.batch_id = b.batch_id
                    ORDER BY bi.position
                ) AS selection_reasons,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp
                    WHERE gp.batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.publication_attempts AS pa
                    JOIN top3_news.generated_posts AS gp
                        ON gp.generated_post_id =
                            pa.generated_post_id
                    WHERE gp.batch_id = b.batch_id
                ) AS publication_attempt_count

            FROM top3_news.publication_batches AS b
            WHERE b.batch_id = $1
            """,
            first_reservation.batch_id,
        )

        duplicate_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.publication_batches
                WHERE generation_request_key = $1
                """,
                request_key.value,
            )
        )

    if record is None:
        raise AssertionError(
            "Зарезервированный выпуск "
            "не найден."
        )

    metadata_news_ids = (
        decode_jsonb_integer_array(
            record["metadata_news_ids"],
            field_name=(
                "metadata.news_ids"
            ),
        )
    )

    metadata_score_ids = (
        decode_jsonb_integer_array(
            record["metadata_score_ids"],
            field_name=(
                "metadata.score_ids"
            ),
        )
    )

    persisted_positions = tuple(
        int(value)
        for value in record["positions"]
    )

    persisted_news_ids = tuple(
        int(value)
        for value in record["news_ids"]
    )

    persisted_score_ids = tuple(
        int(value)
        for value in record["score_ids"]
    )

    persisted_reasons = tuple(
        record["selection_reasons"]
    )

    expected_reasons = tuple(
        item.selection_reason
        for item in selection.items
    )

    assert (
        record["batch_id"]
        == first_reservation.batch_id
    )

    assert (
        record["publication_date"]
        == publication_date
    )

    assert (
        record["edition"]
        == first_reservation.edition
    )

    assert (
        record["batch_status"]
        == "ranked"
    )

    assert (
        record["ranking_run_id"]
        == TEST_RANKING_RUN_ID
    )

    assert (
        record["target_telegram_chat_id"]
        == telegram_chat_id
    )

    assert (
        record["generation_request_key"]
        == request_key.value
    )

    assert (
        record["generation_mode"]
        == "openai"
    )

    assert (
        record["generator_name"]
        == generator.metadata.generator_name
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
        record["model_name"]
        == generator.metadata.model_name
    )

    assert (
        record["text_format"]
        == generator.metadata.text_format
    )

    assert (
        record["metadata_request_key"]
        == request_key.value
    )

    assert (
        record["request_key_version"]
        == request_key.version
    )

    assert (
        record["metadata_ranking_run_id"]
        == TEST_RANKING_RUN_ID
    )

    assert record["news_count"] == 3

    assert (
        record["idempotency_reserved"]
        is True
    )

    assert (
        metadata_news_ids
        == list(selection.news_ids)
    )

    assert (
        metadata_score_ids
        == list(selection.score_ids)
    )

    assert (
        persisted_positions
        == (1, 2, 3)
    )

    assert (
        persisted_news_ids
        == selection.news_ids
    )

    assert (
        persisted_score_ids
        == selection.score_ids
    )

    assert (
        persisted_reasons
        == expected_reasons
    )

    assert (
        record["generated_post_count"]
        == 0
    )

    assert (
        record["publication_attempt_count"]
        == 0
    )

    assert duplicate_count == 1

    print()
    print("Persisted generation reservation: OK")
    print(
        "positions="
        + ",".join(
            str(value)
            for value in persisted_positions
        )
    )
    print(
        "news_ids="
        + ",".join(
            str(value)
            for value in persisted_news_ids
        )
    )
    print(
        "score_ids="
        + ",".join(
            str(value)
            for value in persisted_score_ids
        )
    )
    print(
        "generated_post_count="
        f"{record['generated_post_count']}"
    )
    print(
        "publication_attempt_count="
        f"{record['publication_attempt_count']}"
    )
    print(
        "request_key_row_count="
        f"{duplicate_count}"
    )

    changed_request = replace(
        model_request,
        input_text=(
            model_request.input_text
            + "\nchanged-test-input"
        ),
    )

    try:
        await reserve_generation(
            pool,
            request_key=request_key,
            selection=selection,
            publication_date=publication_date,
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=generator.metadata,
            model_request=changed_request,
        )
    except ValueError as error:
        assert (
            "generation request_key не "
            "соответствует"
            in str(error)
        )

        print()
        print(
            "Changed prepared request "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Изменённый подготовленный запрос "
            "не был заблокирован."
        )

    async with pool.acquire() as connection:
        final_duplicate_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.publication_batches
                WHERE generation_request_key = $1
                """,
                request_key.value,
            )
        )

    assert final_duplicate_count == 1

    print()
    print(
        "Reservation row count after "
        "blocking: OK"
    )
    print(
        "generation_request_key_rows="
        f"{final_duplicate_count}"
    )


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
            "temporary_batch_id="
            f"{batch_id}"
        )
        print(
            "temporary_batch_deleted=true"
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
        await test_reservation(
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
        "Database changes: temporary batch "
        "and batch_items inserted and deleted"
    )
    print("generated_posts created: 0")
    print("publication_attempts created: 0")
    print("Telegram publication: not performed")
    print(
        "Generation reservation test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )