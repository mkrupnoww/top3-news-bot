import asyncio
from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.generation_completion import (
    GENERATION_COMPLETION_VERSION,
    GENERATION_COST_ACCOUNTING_VERSION,
    GENERATION_FAILURE_VERSION,
    WEB_SEARCH_TOOL_PRICE_USD_PER_CALL,
    WEB_SEARCH_TOOL_PRICING_VERSION,
    calculate_web_search_tool_cost,
    complete_reserved_generation,
    fail_reserved_generation,
)
from app.db.generation_reservation import (
    GenerationReservation,
    reserve_generation,
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
    OpenAIGeneratedNewsPayload,
    OpenAIGeneratedPostPayload,
    OpenAIPostGenerationResult,
    OpenAITelegramPostGenerator,
)
from app.generation.request_key import (
    GenerationRequestKey,
    create_generation_request_key,
)
from app.ranking.openai_usage import (
    OpenAITokenUsage,
    calculate_openai_cost,
    get_model_pricing,
)


class NoCallGenerationClient:
    """Клиент, запрещающий вызов OpenAI."""

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Блокирует неожиданный API-вызов."""

        raise AssertionError(
            "OpenAI не должен вызываться "
            "в тесте завершения генерации."
        )


def test_web_search_cost_calculation() -> None:
    """Проверяет стоимость 0, 1 и нескольких поисков."""

    assert (
        calculate_web_search_tool_cost(0)
        == Decimal("0.00")
    )

    assert (
        calculate_web_search_tool_cost(1)
        == Decimal("0.01")
    )

    assert (
        calculate_web_search_tool_cost(3)
        == Decimal("0.03")
    )

    print(
        "Web Search cost calculation: OK"
    )


def build_test_publication_date() -> date:
    """Создаёт изолированную дату теста."""

    random_offset = (
        int(uuid4().hex[:8], 16)
        % 20000
    )

    return (
        date(2300, 1, 1)
        + timedelta(days=random_offset)
    )


async def load_latest_completed_top3_selection(
    pool: asyncpg.Pool,
) -> GenerationTop3Selection:
    """Выбирает свежий completed ranking с полным TOP-3."""

    async with pool.acquire() as connection:
        ranking_run_id = await connection.fetchval(
            """
            SELECT rr.ranking_run_id
            FROM top3_news.ranking_runs AS rr
            WHERE rr.run_status = 'completed'
              AND (
                    SELECT COUNT(*)
                    FROM top3_news.news_scores AS ns
                    WHERE ns.ranking_run_id = rr.ranking_run_id
                      AND ns.selected_for_top3 IS TRUE
                      AND ns.top3_position BETWEEN 1 AND 3
                  ) = 3
              AND (
                    SELECT COUNT(DISTINCT ns.top3_position)
                    FROM top3_news.news_scores AS ns
                    WHERE ns.ranking_run_id = rr.ranking_run_id
                      AND ns.selected_for_top3 IS TRUE
                      AND ns.top3_position BETWEEN 1 AND 3
                  ) = 3
            ORDER BY rr.ranking_run_id DESC
            LIMIT 1
            """
        )

    if ranking_run_id is None:
        raise LookupError(
            "Не найден completed ranking_run "
            "с полноценным сохранённым TOP-3."
        )

    normalized_ranking_run_id = int(
        ranking_run_id
    )

    selection = await load_generation_top3(
        pool,
        ranking_run_id=(
            normalized_ranking_run_id
        ),
    )

    if selection.run_status != "completed":
        raise AssertionError(
            "Динамический ranking fixture "
            "не имеет статус completed."
        )

    if len(selection.items) != 3:
        raise AssertionError(
            "Динамический ranking fixture "
            "не содержит ровно три новости."
        )

    if len(set(selection.news_ids)) != 3:
        raise AssertionError(
            "Динамический ranking fixture "
            "содержит дублирующиеся news_id."
        )

    positions = tuple(
        item.position
        for item in selection.items
    )

    if positions != (1, 2, 3):
        raise AssertionError(
            "Динамический ranking fixture "
            "не содержит позиции 1, 2, 3."
        )

    print(
        "Dynamic ranking fixture: OK"
    )
    print(
        "test_ranking_run_id="
        f"{selection.ranking_run_id}"
    )
    print(
        "test_news_ids="
        + ",".join(
            str(news_id)
            for news_id in selection.news_ids
        )
    )

    return selection


def build_generation_result(
    selection: GenerationTop3Selection,
    *,
    post_suffix: str = "",
) -> OpenAIPostGenerationResult:
    """Создаёт локальный результат модели."""

    items = [
        OpenAIGeneratedNewsPayload(
            position=1,
            news_id=(
                selection.items[0].news_id
            ),
            headline=(
                "Sony Pictures отчиталась "
                "о снижении выручки"
            ),
            body=(
                "Киноподразделение Sony "
                "сообщило о снижении выручки "
                "за квартал, тогда как другие "
                "направления компании показали "
                "разную динамику."
            ),
        ),
        OpenAIGeneratedNewsPayload(
            position=2,
            news_id=(
                selection.items[1].news_id
            ),
            headline=(
                "Кинофестиваль добавил "
                "конкурс AI-фильмов"
            ),
            body=(
                "Международный фестиваль "
                "короткометражного кино "
                "расширил программу и добавил "
                "категории для AI-фильмов "
                "и screen dance."
            ),
        ),
        OpenAIGeneratedNewsPayload(
            position=3,
            news_id=(
                selection.items[2].news_id
            ),
            headline=(
                "Element Pictures создала "
                "новую руководящую должность"
            ),
            body=(
                "Компания назначила первого "
                "руководителя по коммуникациям "
                "и маркетингу, который будет "
                "работать с её проектами "
                "и корпоративными коммуникациями."
            ),
        ),
    ]

    post_text = (
        "**TOP-3 киноновости дня**\n\n"
        "**1. Sony Pictures отчиталась "
        "о снижении выручки**\n"
        "Киноподразделение Sony сообщило "
        "о снижении выручки за квартал, "
        "тогда как другие направления "
        "компании показали разную динамику.\n\n"
        "**2. Кинофестиваль добавил "
        "конкурс AI-фильмов**\n"
        "Международный фестиваль "
        "короткометражного кино расширил "
        "программу и добавил категории "
        "для AI-фильмов и screen dance.\n\n"
        "**3. Element Pictures создала "
        "новую руководящую должность**\n"
        "Компания назначила первого "
        "руководителя по коммуникациям "
        "и маркетингу, который будет "
        "работать с её проектами "
        "и корпоративными коммуникациями.\n\n"
        "__Какую из новостей обсудим "
        "подробнее?__"
        f"{post_suffix}"
    )

    payload = OpenAIGeneratedPostPayload(
        post_text=post_text,
        items=items,
    )

    usage = OpenAITokenUsage(
        input_tokens=1200,
        cached_input_tokens=200,
        cache_write_tokens=300,
        output_tokens=240,
        reasoning_tokens=40,
        total_tokens=1440,
    )

    pricing = get_model_pricing(
        "gpt-5.6-terra"
    )

    cost_estimate = calculate_openai_cost(
        usage,
        pricing,
    )

    return OpenAIPostGenerationResult(
        payload=payload,
        model_response=(
            GenerationModelResponse(
                output_text=(
                    payload.model_dump_json()
                ),
                usage=usage,
                cost_estimate=cost_estimate,
                web_search_used=True,
                web_search_call_count=2,
                web_source_urls=(
                    "https://example.com/source-1",
                    "https://example.com/source-2",
                ),
            )
        ),
    )


async def create_test_reservation(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    generator: OpenAITelegramPostGenerator,
    publication_date: date,
    telegram_chat_id: int,
) -> tuple[
    GenerationReservation,
    GenerationRequestKey,
]:
    """Создаёт тестовое резервирование."""

    model_request = generator.build_request(
        selection.items
    )

    request_key = (
        create_generation_request_key(
            ranking_run_id=(
                selection.ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    reservation = await reserve_generation(
        pool,
        request_key=request_key,
        selection=selection,
        publication_date=publication_date,
        telegram_chat_id=telegram_chat_id,
        metadata=generator.metadata,
        model_request=model_request,
    )

    assert reservation.created_new is True

    assert (
        reservation.should_call_model
        is True
    )

    assert (
        reservation.batch_status
        == "ranked"
    )

    return reservation, request_key


async def delete_test_batch(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> None:
    """Удаляет временный выпуск."""

    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            DELETE FROM
                top3_news.publication_batches
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
    ranking_run_id: int,
) -> None:
    """Проверяет каскадную очистку."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                EXISTS (
                    SELECT 1
                    FROM
                        top3_news
                        .publication_batches
                    WHERE batch_id = $1
                ) AS batch_exists,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.batch_items
                    WHERE batch_id = $1
                ) AS batch_item_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.generated_posts
                    WHERE batch_id = $1
                ) AS generated_post_count,

                EXISTS (
                    SELECT 1
                    FROM top3_news.ranking_runs
                    WHERE ranking_run_id = $2
                ) AS ranking_run_exists
            """,
            batch_id,
            ranking_run_id,
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


async def test_successful_completion(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    generator: OpenAITelegramPostGenerator,
    telegram_chat_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> None:
    """Проверяет успешное завершение."""

    reservation, request_key = (
        await create_test_reservation(
            pool,
            selection=selection,
            generator=generator,
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
        )
    )

    created_batch_ids.add(
        reservation.batch_id
    )

    result = build_generation_result(
        selection
    )

    completion = (
        await complete_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=request_key.value,
            metadata=generator.metadata,
            result=result,
        )
    )

    assert (
        completion.batch_id
        == reservation.batch_id
    )

    assert (
        completion.request_key
        == request_key.value
    )

    assert (
        completion.batch_status
        == "awaiting_review"
    )

    assert (
        completion.post_status
        == "awaiting_review"
    )

    assert completion.version_number == 1

    assert (
        completion.text_format
        == generator.metadata.text_format
    )

    assert (
        completion.news_ids
        == selection.news_ids
    )

    assert (
        completion.already_completed
        is False
    )

    print("Generation completion: OK")
    print(
        f"batch_id={completion.batch_id}"
    )
    print(
        "generated_post_id="
        f"{completion.generated_post_id}"
    )
    print(
        "batch_status="
        f"{completion.batch_status}"
    )
    print(
        "post_status="
        f"{completion.post_status}"
    )
    print("already_completed=false")

    repeated_completion = (
        await complete_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=request_key.value,
            metadata=generator.metadata,
            result=result,
        )
    )

    assert (
        repeated_completion
        .generated_post_id
        == completion.generated_post_id
    )

    assert (
        repeated_completion
        .already_completed
        is True
    )

    assert (
        repeated_completion.batch_status
        == "awaiting_review"
    )

    assert (
        repeated_completion.post_status
        == "awaiting_review"
    )

    print()
    print(
        "Repeated generation completion: OK"
    )
    print(
        "generated_post_id="
        f"{repeated_completion.generated_post_id}"
    )
    print("already_completed=true")
    print(
        "Duplicate generated post: blocked"
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                b.error_message
                    AS batch_error_message,

                (
                    b.metadata
                    ->>'generation_completed'
                )::boolean
                    AS generation_completed,

                (
                    b.metadata
                    ->>'generated_post_id'
                )::bigint
                    AS metadata_generated_post_id,

                b.metadata
                    ->>'generation_completion_version'
                    AS batch_completion_version,

                (
                    b.metadata
                    ->'openai_usage'
                    ->>'total_tokens'
                )::integer
                    AS batch_total_tokens,

                b.metadata
                    ->'openai_cost'
                    ->>'total_cost_usd'
                    AS batch_total_cost_usd,

                (
                    b.metadata
                    ->'openai_web_search'
                    ->>'used'
                )::boolean
                    AS batch_web_search_used,

                (
                    b.metadata
                    ->'openai_web_search'
                    ->>'call_count'
                )::integer
                    AS batch_web_search_call_count,

                b.metadata
                    ->'openai_web_search'
                    ->>'pricing_version'
                    AS batch_web_search_pricing_version,

                b.metadata
                    ->'openai_web_search'
                    ->>'tool_price_usd_per_call'
                    AS batch_web_search_price_per_call,

                b.metadata
                    ->'openai_web_search'
                    ->>'tool_cost_usd'
                    AS batch_web_search_tool_cost_usd,

                jsonb_array_length(
                    b.metadata
                    ->'openai_web_search'
                    ->'source_urls'
                ) AS batch_web_source_url_count,

                b.metadata
                    ->'openai_web_search'
                    ->'source_urls'
                    ->>0
                    AS batch_first_web_source_url,

                b.metadata
                    ->'openai_web_search'
                    ->'source_urls'
                    ->>1
                    AS batch_second_web_source_url,

                b.metadata
                    ->>'generation_total_cost_usd'
                    AS generation_total_cost_usd,

                b.metadata
                    ->>'generation_cost_accounting_version'
                    AS generation_cost_accounting_version,

                gp.generated_post_id,
                gp.version_number,
                gp.post_status,
                gp.post_text,
                gp.text_format,
                gp.text_model_name,
                gp.text_prompt_version,

                gp.generation_metadata
                    ->>'generation_request_key'
                    AS post_request_key,

                gp.generation_metadata
                    ->>'generation_request_key_version'
                    AS post_request_key_version,

                (
                    gp.generation_metadata
                    ->>'ranking_run_id'
                )::bigint
                    AS post_ranking_run_id,

                (
                    gp.generation_metadata
                    ->>'news_count'
                )::integer
                    AS post_news_count,

                (
                    gp.generation_metadata
                    ->>'post_length'
                )::integer
                    AS post_length,

                gp.generation_metadata
                    ->>'completion_version'
                    AS post_completion_version,

                jsonb_array_length(
                    gp.generation_metadata
                    ->'generated_items'
                ) AS generated_item_count,

                (
                    gp.generation_metadata
                    ->'openai_usage'
                    ->>'total_tokens'
                )::integer
                    AS post_total_tokens,

                gp.generation_metadata
                    ->'openai_cost'
                    ->>'total_cost_usd'
                    AS post_total_cost_usd,

                (
                    gp.generation_metadata
                    ? 'openai_web_search'
                ) AS post_has_web_search_metadata,

                (
                    gp.generation_metadata
                    ? 'generation_total_cost_usd'
                ) AS post_has_generation_total_cost,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.generated_posts
                    WHERE batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .publication_attempts AS pa
                    JOIN
                        top3_news
                        .generated_posts AS p
                      ON p.generated_post_id =
                         pa.generated_post_id
                    WHERE p.batch_id = b.batch_id
                ) AS publication_attempt_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.review_actions AS ra
                    JOIN
                        top3_news
                        .generated_posts AS p
                      ON p.generated_post_id =
                         ra.generated_post_id
                    WHERE p.batch_id = b.batch_id
                ) AS review_action_count

            FROM
                top3_news
                .publication_batches AS b
            JOIN
                top3_news.generated_posts AS gp
              ON gp.batch_id = b.batch_id
             AND gp.version_number = 1
            WHERE b.batch_id = $1
            """,
            reservation.batch_id,
        )

    if record is None:
        raise AssertionError(
            "Завершённый тестовый выпуск "
            "не найден."
        )

    usage = result.model_response.usage
    cost = result.model_response.cost_estimate

    if usage is None or cost is None:
        raise AssertionError(
            "Тестовый результат не содержит "
            "телеметрию."
        )

    assert (
        record["batch_status"]
        == "awaiting_review"
    )

    assert (
        record["batch_error_message"]
        is None
    )

    assert (
        record["generation_completed"]
        is True
    )

    assert (
        record["metadata_generated_post_id"]
        == completion.generated_post_id
    )

    assert (
        record["batch_completion_version"]
        == GENERATION_COMPLETION_VERSION
    )

    assert (
        record["batch_total_tokens"]
        == usage.total_tokens
    )

    assert (
        record["batch_total_cost_usd"]
        == str(cost.total_cost_usd)
    )

    assert (
        record["batch_web_search_used"]
        is True
    )

    assert (
        record["batch_web_search_call_count"]
        == 2
    )

    assert (
        record[
            "batch_web_search_pricing_version"
        ]
        == WEB_SEARCH_TOOL_PRICING_VERSION
    )

    assert (
        record[
            "batch_web_search_price_per_call"
        ]
        == str(
            WEB_SEARCH_TOOL_PRICE_USD_PER_CALL
        )
    )

    assert (
        record[
            "batch_web_search_tool_cost_usd"
        ]
        == "0.02"
    )

    assert (
        record["batch_web_source_url_count"]
        == 2
    )

    assert (
        record["batch_first_web_source_url"]
        == "https://example.com/source-1"
    )

    assert (
        record["batch_second_web_source_url"]
        == "https://example.com/source-2"
    )

    expected_generation_total_cost = (
        cost.total_cost_usd
        + Decimal("0.02")
    )

    assert (
        record["generation_total_cost_usd"]
        == str(expected_generation_total_cost)
    )

    assert (
        record[
            "generation_cost_accounting_version"
        ]
        == GENERATION_COST_ACCOUNTING_VERSION
    )

    assert (
        record["generated_post_id"]
        == completion.generated_post_id
    )

    assert record["version_number"] == 1

    assert (
        record["post_status"]
        == "awaiting_review"
    )

    assert (
        record["post_text"]
        == result.payload.post_text
    )

    assert (
        record["text_format"]
        == generator.metadata.text_format
    )

    assert (
        record["text_model_name"]
        == generator.metadata.model_name
    )

    assert (
        record["text_prompt_version"]
        == generator.metadata.prompt_version
    )

    assert (
        record["post_request_key"]
        == request_key.value
    )

    assert (
        record["post_request_key_version"]
        == request_key.version
    )

    assert (
        record["post_ranking_run_id"]
        == selection.ranking_run_id
    )

    assert record["post_news_count"] == 3

    assert (
        record["post_length"]
        == len(result.payload.post_text)
    )

    assert (
        record["post_completion_version"]
        == GENERATION_COMPLETION_VERSION
    )

    assert (
        record["generated_item_count"]
        == 3
    )

    assert (
        record["post_total_tokens"]
        == usage.total_tokens
    )

    assert (
        record["post_total_cost_usd"]
        == str(cost.total_cost_usd)
    )

    assert (
        record["post_has_web_search_metadata"]
        is False
    )

    assert (
        record[
            "post_has_generation_total_cost"
        ]
        is False
    )

    assert (
        record["generated_post_count"]
        == 1
    )

    assert (
        record["publication_attempt_count"]
        == 0
    )

    assert record["review_action_count"] == 0

    print()
    print(
        "Persisted generation completion: OK"
    )
    print(
        "generated_post_count="
        f"{record['generated_post_count']}"
    )
    print(
        "generated_item_count="
        f"{record['generated_item_count']}"
    )
    print(
        "total_tokens="
        f"{record['post_total_tokens']}"
    )
    print(
        "estimated_cost_usd="
        f"{record['post_total_cost_usd']}"
    )
    print(
        "web_search_tool_cost_usd="
        f"{record['batch_web_search_tool_cost_usd']}"
    )
    print(
        "generation_total_cost_usd="
        f"{record['generation_total_cost_usd']}"
    )
    print(
        "publication_attempt_count="
        f"{record['publication_attempt_count']}"
    )
    print(
        "review_action_count="
        f"{record['review_action_count']}"
    )

    changed_result = build_generation_result(
        selection,
        post_suffix=(
            "\n\nТестовое изменение."
        ),
    )

    try:
        await complete_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=request_key.value,
            metadata=generator.metadata,
            result=changed_result,
        )
    except ValueError as error:
        assert (
            "Существующий generated_post "
            "не соответствует результату"
            in str(error)
        )

        print()
        print(
            "Conflicting completion "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Изменённый результат повторного "
            "завершения не был заблокирован."
        )

    try:
        await fail_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=request_key.value,
            error_message=(
                "Ошибка после завершения"
            ),
            error_type="RuntimeError",
        )
    except ValueError as error:
        assert (
            "из статуса: awaiting_review"
            in str(error)
        )

        print()
        print(
            "Completed batch failure "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Завершённый выпуск был ошибочно "
            "переведён в failed."
        )


async def test_failure_completion(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    generator: OpenAITelegramPostGenerator,
    telegram_chat_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> None:
    """Проверяет фиксацию ошибки."""

    reservation, request_key = (
        await create_test_reservation(
            pool,
            selection=selection,
            generator=generator,
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
        )
    )

    created_batch_ids.add(
        reservation.batch_id
    )

    failure = await fail_reserved_generation(
        pool,
        batch_id=reservation.batch_id,
        request_key=request_key.value,
        error_message=(
            "Тестовая ошибка генерации"
        ),
        error_type="RuntimeError",
    )

    assert (
        failure.batch_id
        == reservation.batch_id
    )

    assert (
        failure.request_key
        == request_key.value
    )

    assert (
        failure.batch_status
        == "failed"
    )

    assert failure.already_failed is False

    assert (
        failure.error_message
        == "Тестовая ошибка генерации"
    )

    print()
    print("Generation failure handling: OK")
    print(
        f"batch_id={failure.batch_id}"
    )
    print("batch_status=failed")
    print("already_failed=false")

    repeated_failure = (
        await fail_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=request_key.value,
            error_message=(
                "Тестовая ошибка генерации"
            ),
            error_type="RuntimeError",
        )
    )

    assert (
        repeated_failure.already_failed
        is True
    )

    assert (
        repeated_failure.error_message
        == "Тестовая ошибка генерации"
    )

    print()
    print(
        "Repeated generation failure: OK"
    )
    print("already_failed=true")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                b.error_message,

                (
                    b.metadata
                    ->>'generation_failed'
                )::boolean
                    AS generation_failed,

                b.metadata
                    ->'failure'
                    ->>'error_type'
                    AS error_type,

                b.metadata
                    ->'failure'
                    ->>'error_message'
                    AS metadata_error_message,

                b.metadata
                    ->>'generation_failure_version'
                    AS failure_version,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.generated_posts
                    WHERE batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .publication_attempts AS pa
                    JOIN
                        top3_news
                        .generated_posts AS gp
                      ON gp.generated_post_id =
                         pa.generated_post_id
                    WHERE gp.batch_id = b.batch_id
                ) AS publication_attempt_count

            FROM
                top3_news
                .publication_batches AS b
            WHERE b.batch_id = $1
            """,
            reservation.batch_id,
        )

    if record is None:
        raise AssertionError(
            "Failed-выпуск не найден."
        )

    assert record["batch_status"] == "failed"

    assert (
        record["error_message"]
        == "Тестовая ошибка генерации"
    )

    assert (
        record["generation_failed"]
        is True
    )

    assert (
        record["error_type"]
        == "RuntimeError"
    )

    assert (
        record["metadata_error_message"]
        == "Тестовая ошибка генерации"
    )

    assert (
        record["failure_version"]
        == GENERATION_FAILURE_VERSION
    )

    assert (
        record["generated_post_count"]
        == 0
    )

    assert (
        record["publication_attempt_count"]
        == 0
    )

    print()
    print(
        "Persisted generation failure: OK"
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
        "failure_version="
        f"{record['failure_version']}"
    )

    result = build_generation_result(
        selection
    )

    try:
        await complete_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=request_key.value,
            metadata=generator.metadata,
            result=result,
        )
    except ValueError as error:
        assert (
            "со статусом failed"
            in str(error)
        )

        print()
        print(
            "Failed batch completion "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Failed-выпуск был ошибочно "
            "завершён."
        )


async def cleanup_test_batches(
    pool: asyncpg.Pool,
    *,
    created_batch_ids: set[int],
    ranking_run_id: int,
) -> None:
    """Удаляет созданные тестом выпуски."""

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
            ranking_run_id=ranking_run_id,
        )

        print()
        print("Test data cleanup: OK")
        print(
            f"temporary_batch_id={batch_id}"
        )
        print(
            "temporary_batch_deleted=true"
        )
        print(
            "ranking_run_preserved="
            f"{ranking_run_id}"
        )


async def main() -> int:
    """Запускает интеграционный тест."""

    test_web_search_cost_calculation()

    settings = get_settings()

    if (
        settings.openai_generation_model
        != "gpt-5.6-terra"
    ):
        raise ValueError(
            "Тестовая телеметрия настроена "
            "для модели gpt-5.6-terra."
        )

    pool = await create_database_pool(
        settings
    )

    created_batch_ids: set[int] = set()
    selected_ranking_run_id: int | None = None

    try:
        selection = (
            await load_latest_completed_top3_selection(
                pool
            )
        )

        selected_ranking_run_id = (
            selection.ranking_run_id
        )

        generator = (
            OpenAITelegramPostGenerator(
                client=(
                    NoCallGenerationClient()
                ),
                model_name=(
                    settings
                    .openai_generation_model
                ),
            )
        )

        publication_date = (
            build_test_publication_date()
        )

        await test_successful_completion(
            pool,
            selection=selection,
            generator=generator,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            publication_date=(
                publication_date
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )

        await test_failure_completion(
            pool,
            selection=selection,
            generator=generator,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            publication_date=(
                publication_date
                + timedelta(days=1)
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )
    finally:
        try:
            if created_batch_ids:
                if selected_ranking_run_id is None:
                    raise RuntimeError(
                        "Неизвестен ranking_run_id "
                        "для cleanup тестовых batches."
                    )

                await cleanup_test_batches(
                    pool,
                    created_batch_ids=(
                        created_batch_ids
                    ),
                    ranking_run_id=(
                        selected_ranking_run_id
                    ),
                )
        finally:
            await close_database_pool(pool)

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print(
        "Database changes: temporary batches, "
        "batch_items and generated_post "
        "inserted and deleted"
    )
    print(
        "publication_attempts created: 0"
    )
    print("Telegram publication: not performed")
    print(
        "Generation completion test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )