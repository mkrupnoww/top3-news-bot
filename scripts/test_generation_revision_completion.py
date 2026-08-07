import asyncio
from datetime import date, timedelta
from hashlib import sha256
import json
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.generation_revision_completion import (
    GENERATION_REVISION_COMPLETION_VERSION,
    GENERATION_REVISION_FAILURE_VERSION,
    complete_reserved_generation_revision,
    fail_reserved_generation_revision,
)
from app.db.generation_revision_reservation import (
    GenerationRevisionReservation,
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
    OpenAIGeneratedNewsPayload,
    OpenAIGeneratedPostPayload,
    OpenAIPostGenerationResult,
    OpenAITelegramPostGenerator,
)
from app.generation.revision_request_key import (
    GenerationRevisionRequestKey,
    create_generation_revision_request_key,
)
from app.ranking.openai_usage import (
    OpenAITokenUsage,
    calculate_openai_cost,
    get_model_pricing,
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

TEST_IMAGE_PATH = (
    "/tmp/top3-news-test-revision-image.png"
)

TEST_IMAGE_SHA256 = sha256(
    b"top3-news-test-revision-image"
).hexdigest()

TEST_IMAGE_PROMPT = (
    "Тестовая иллюстрация для проверки "
    "наследования при regenerate_text."
)

TEST_IMAGE_MODEL_NAME = "test-image-model"

TEST_IMAGE_PROMPT_VERSION = (
    "test_image_prompt_v1"
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
            "в тесте revision completion."
        )


def build_test_publication_date() -> date:
    """Создаёт изолированную дату теста."""

    random_offset = (
        int(uuid4().hex[:8], 16)
        % 20000
    )

    return (
        date(2400, 1, 1)
        + timedelta(days=random_offset)
    )


def build_test_batch_request_key() -> str:
    """Создаёт уникальный ключ временного batch."""

    return sha256(
        uuid4().bytes
    ).hexdigest()


def build_revision_result(
    selection: GenerationTop3Selection,
    *,
    post_suffix: str = "",
) -> OpenAIPostGenerationResult:
    """Создаёт локальный результат revision-модели."""

    items = [
        OpenAIGeneratedNewsPayload(
            position=1,
            news_id=(
                selection.items[0].news_id
            ),
            headline=(
                "Sony Pictures сообщила "
                "о снижении выручки"
            ),
            body=(
                "Киноподразделение Sony "
                "сообщило о снижении квартальной "
                "выручки. В тексте использованы "
                "только сведения из title "
                "и summary."
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
                "добавил категории для AI-фильмов "
                "и screen dance."
            ),
        ),
        OpenAIGeneratedNewsPayload(
            position=3,
            news_id=(
                selection.items[2].news_id
            ),
            headline=(
                "Element Pictures назначила "
                "руководителя по коммуникациям"
            ),
            body=(
                "Компания назначила первого "
                "руководителя по коммуникациям "
                "и маркетингу."
            ),
        ),
    ]

    post_text = (
        "**TOP-3 НОВОСТЕЙ КИНО "
        "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n"
        "_______________\n\n"
        "1️⃣ **Sony Pictures сообщила "
        "о снижении выручки**\n\n"
        "Киноподразделение Sony сообщило "
        "о снижении квартальной выручки.\n\n"
        "2️⃣ **Кинофестиваль добавил "
        "конкурс AI-фильмов**\n\n"
        "Международный фестиваль "
        "короткометражного кино добавил "
        "категории для AI-фильмов "
        "и screen dance.\n\n"
        "3️⃣ **Element Pictures назначила "
        "руководителя по коммуникациям**\n\n"
        "Компания назначила первого "
        "руководителя по коммуникациям "
        "и маркетингу."
        f"{post_suffix}"
    )

    payload = OpenAIGeneratedPostPayload(
        post_text=post_text,
        items=items,
    )

    usage = OpenAITokenUsage(
        input_tokens=1500,
        cached_input_tokens=300,
        cache_write_tokens=200,
        output_tokens=260,
        reasoning_tokens=40,
        total_tokens=1760,
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
            )
        ),
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
    publication_date: date,
    created_batch_ids: set[int],
) -> tuple[int, int, int]:
    """
    Создаёт batch, generated_post v1
    и changes_required.
    """

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
                            "completion"
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
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10,
                        $11::jsonb
                    )
                    RETURNING generated_post_id
                    """,
                    batch_id,
                    SOURCE_POST_TEXT,
                    generator.metadata.text_format,
                    TEST_IMAGE_PATH,
                    TEST_IMAGE_SHA256,
                    TEST_IMAGE_PROMPT,
                    generator.metadata.model_name,
                    TEST_IMAGE_MODEL_NAME,
                    generator.metadata.prompt_version,
                    TEST_IMAGE_PROMPT_VERSION,
                    json.dumps(
                        {
                            "test_mode": True,
                            "test_name": (
                                "generation_revision_"
                                "completion"
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
                                "completion"
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


async def create_test_revision_reservation(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    generator: OpenAITelegramPostGenerator,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> tuple[
    GenerationRevisionReservation,
    GenerationRevisionRequestKey,
]:
    """Создаёт защищённую test revision reservation."""

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
        publication_date=publication_date,
        created_batch_ids=created_batch_ids,
    )

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
            target_version_number=2,
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    reservation = (
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

    assert reservation.created_new is True
    assert reservation.should_call_model is True
    assert (
        reservation.revision_status
        == "reserved"
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
) -> None:
    """Проверяет каскадную очистку."""

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
                    FROM
                        top3_news
                        .generation_revision_requests
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


async def test_successful_revision_completion(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    generator: OpenAITelegramPostGenerator,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> None:
    """Проверяет успешное завершение ревизии."""

    reservation, request_key = (
        await create_test_revision_reservation(
            pool,
            selection=selection,
            generator=generator,
            telegram_chat_id=(
                telegram_chat_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            publication_date=publication_date,
            created_batch_ids=(
                created_batch_ids
            ),
        )
    )

    result = build_revision_result(
        selection
    )

    completion = (
        await complete_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
            request_key=request_key.value,
            metadata=generator.metadata,
            result=result,
        )
    )

    assert (
        completion.generation_revision_id
        == reservation.generation_revision_id
    )

    assert (
        completion.batch_id
        == reservation.batch_id
    )

    assert (
        completion.source_generated_post_id
        == reservation.source_generated_post_id
    )

    assert (
        completion.review_action_id
        == reservation.review_action_id
    )

    assert (
        completion.request_key
        == request_key.value
    )

    assert (
        completion.revision_status
        == "completed"
    )

    assert (
        completion.source_post_status
        == "superseded"
    )

    assert (
        completion.post_status
        == "awaiting_review"
    )

    assert completion.version_number == 2

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

    print("Generation revision completion: OK")
    print(
        "generation_revision_id="
        f"{completion.generation_revision_id}"
    )
    print(
        f"batch_id={completion.batch_id}"
    )
    print(
        "source_generated_post_id="
        f"{completion.source_generated_post_id}"
    )
    print(
        "generated_post_id="
        f"{completion.generated_post_id}"
    )
    print("source_post_status=superseded")
    print("post_status=awaiting_review")
    print("version_number=2")
    print("already_completed=false")

    repeated_completion = (
        await complete_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
            request_key=request_key.value,
            metadata=generator.metadata,
            result=result,
        )
    )

    assert (
        repeated_completion.generated_post_id
        == completion.generated_post_id
    )

    assert (
        repeated_completion.already_completed
        is True
    )

    assert (
        repeated_completion.version_number
        == 2
    )

    print()
    print(
        "Repeated generation revision "
        "completion: OK"
    )
    print(
        "generated_post_id="
        f"{repeated_completion.generated_post_id}"
    )
    print("already_completed=true")
    print("Duplicate target version: blocked")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                grr.revision_status,
                grr.generated_post_id
                    AS revision_generated_post_id,
                grr.openai_usage,
                grr.openai_cost,
                grr.error_type,
                grr.error_message,
                grr.completed_at,
                grr.failed_at,

                b.batch_status,

                source_gp.version_number
                    AS source_version_number,
                source_gp.post_status
                    AS source_post_status,
                source_gp.image_path
                    AS source_image_path,
                source_gp.image_sha256
                    AS source_image_sha256,
                source_gp.image_prompt
                    AS source_image_prompt,
                source_gp.image_model_name
                    AS source_image_model_name,
                source_gp.image_prompt_version
                    AS source_image_prompt_version,

                target_gp.version_number
                    AS target_version_number,
                target_gp.post_status
                    AS target_post_status,
                target_gp.post_text
                    AS target_post_text,
                target_gp.text_format
                    AS target_text_format,
                target_gp.text_model_name
                    AS target_text_model_name,
                target_gp.text_prompt_version
                    AS target_text_prompt_version,
                target_gp.image_path
                    AS target_image_path,
                target_gp.image_sha256
                    AS target_image_sha256,
                target_gp.image_prompt
                    AS target_image_prompt,
                target_gp.image_model_name
                    AS target_image_model_name,
                target_gp.image_prompt_version
                    AS target_image_prompt_version,

                target_gp.generation_metadata
                    ->>'generation_mode'
                    AS generation_mode,

                (
                    target_gp.generation_metadata
                    ->>'generation_revision_id'
                )::bigint
                    AS metadata_revision_id,

                target_gp.generation_metadata
                    ->>'generation_revision_request_key'
                    AS metadata_request_key,

                target_gp.generation_metadata
                    ->>'generation_revision_request_key_version'
                    AS metadata_request_key_version,

                (
                    target_gp.generation_metadata
                    ->>'source_generated_post_id'
                )::bigint
                    AS metadata_source_post_id,

                (
                    target_gp.generation_metadata
                    ->>'review_action_id'
                )::bigint
                    AS metadata_review_action_id,

                (
                    target_gp.generation_metadata
                    ->>'target_version_number'
                )::integer
                    AS metadata_target_version,

                target_gp.generation_metadata
                    ->>'requested_action'
                    AS metadata_requested_action,

                target_gp.generation_metadata
                    ->>'revision_prompt_version'
                    AS metadata_revision_prompt_version,

                (
                    target_gp.generation_metadata
                    ->>'news_count'
                )::integer
                    AS metadata_news_count,

                jsonb_array_length(
                    target_gp.generation_metadata
                    ->'generated_items'
                ) AS generated_item_count,

                (
                    target_gp.generation_metadata
                    ->'openai_usage'
                    ->>'total_tokens'
                )::integer
                    AS post_total_tokens,

                target_gp.generation_metadata
                    ->'openai_cost'
                    ->>'total_cost_usd'
                    AS post_total_cost_usd,

                target_gp.generation_metadata
                    ->>'completion_version'
                    AS post_completion_version,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts
                    WHERE batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.publication_attempts
                            AS pa
                    JOIN top3_news.generated_posts
                            AS gp
                      ON gp.generated_post_id =
                         pa.generated_post_id
                    WHERE gp.batch_id = b.batch_id
                ) AS publication_attempt_count

            FROM
                top3_news
                .generation_revision_requests AS grr
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = grr.batch_id
            JOIN top3_news.generated_posts AS source_gp
              ON source_gp.generated_post_id =
                 grr.source_generated_post_id
            JOIN top3_news.generated_posts AS target_gp
              ON target_gp.generated_post_id =
                 grr.generated_post_id
            WHERE grr.generation_revision_id = $1
            """,
            reservation.generation_revision_id,
        )

    if record is None:
        raise AssertionError(
            "Completed revision не найдена."
        )

    usage = result.model_response.usage
    cost = result.model_response.cost_estimate

    if usage is None or cost is None:
        raise AssertionError(
            "Тестовый revision result "
            "не содержит телеметрию."
        )

    assert (
        record["revision_status"]
        == "completed"
    )

    assert (
        record["revision_generated_post_id"]
        == completion.generated_post_id
    )

    assert record["openai_usage"] is not None
    assert record["openai_cost"] is not None
    assert record["error_type"] is None
    assert record["error_message"] is None
    assert record["completed_at"] is not None
    assert record["failed_at"] is None

    assert (
        record["batch_status"]
        == "awaiting_review"
    )

    assert record["source_version_number"] == 1
    assert (
        record["source_post_status"]
        == "superseded"
    )

    assert record["target_version_number"] == 2
    assert (
        record["target_post_status"]
        == "awaiting_review"
    )

    assert (
        record["target_post_text"]
        == result.payload.post_text
    )

    assert (
        record["target_text_format"]
        == generator.metadata.text_format
    )

    assert (
        record["target_text_model_name"]
        == generator.metadata.model_name
    )

    assert (
        record["target_text_prompt_version"]
        == (
            "movie_news_telegram_post_"
            "revision_prompt_v1"
        )
    )

    assert (
        record["source_image_path"]
        == TEST_IMAGE_PATH
    )

    assert (
        record["source_image_sha256"]
        == TEST_IMAGE_SHA256
    )

    assert (
        record["source_image_prompt"]
        == TEST_IMAGE_PROMPT
    )

    assert (
        record["source_image_model_name"]
        == TEST_IMAGE_MODEL_NAME
    )

    assert (
        record["source_image_prompt_version"]
        == TEST_IMAGE_PROMPT_VERSION
    )

    assert (
        record["target_image_path"]
        == record["source_image_path"]
    )

    assert (
        record["target_image_sha256"]
        == record["source_image_sha256"]
    )

    assert (
        record["target_image_prompt"]
        == record["source_image_prompt"]
    )

    assert (
        record["target_image_model_name"]
        == record["source_image_model_name"]
    )

    assert (
        record["target_image_prompt_version"]
        == record["source_image_prompt_version"]
    )

    assert (
        record["generation_mode"]
        == "openai_revision"
    )

    assert (
        record["metadata_revision_id"]
        == reservation.generation_revision_id
    )

    assert (
        record["metadata_request_key"]
        == request_key.value
    )

    assert (
        record["metadata_request_key_version"]
        == request_key.version
    )

    assert (
        record["metadata_source_post_id"]
        == reservation.source_generated_post_id
    )

    assert (
        record["metadata_review_action_id"]
        == reservation.review_action_id
    )

    assert (
        record["metadata_target_version"]
        == 2
    )

    assert (
        record["metadata_requested_action"]
        == "regenerate_text"
    )

    assert (
        record["metadata_revision_prompt_version"]
        == (
            "movie_news_telegram_post_"
            "revision_prompt_v1"
        )
    )

    assert record["metadata_news_count"] == 3
    assert record["generated_item_count"] == 3

    assert (
        record["post_total_tokens"]
        == usage.total_tokens
    )

    assert (
        record["post_total_cost_usd"]
        == str(cost.total_cost_usd)
    )

    assert (
        record["post_completion_version"]
        == GENERATION_REVISION_COMPLETION_VERSION
    )

    assert record["generated_post_count"] == 2
    assert (
        record["publication_attempt_count"]
        == 0
    )

    print()
    print(
        "Persisted generation revision "
        "completion: OK"
    )
    print("generated_post_count=2")
    print("source_version_status=superseded")
    print("target_version_status=awaiting_review")
    print("image_fields_inherited=true")
    print(
        f"total_tokens={record['post_total_tokens']}"
    )
    print(
        "estimated_cost_usd="
        f"{record['post_total_cost_usd']}"
    )
    print(
        "publication_attempt_count="
        f"{record['publication_attempt_count']}"
    )

    changed_result = build_revision_result(
        selection,
        post_suffix=(
            "\n\nТестовое изменение."
        ),
    )

    try:
        await complete_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
            request_key=request_key.value,
            metadata=generator.metadata,
            result=changed_result,
        )
    except ValueError as error:
        assert (
            "Существующая target-версия "
            "не соответствует результату"
            in str(error)
        )

        print()
        print(
            "Conflicting revision completion "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Изменённый результат повторной "
            "revision completion "
            "не был заблокирован."
        )

    try:
        await fail_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
            request_key=request_key.value,
            error_message=(
                "Ошибка после completion"
            ),
            error_type="RuntimeError",
        )
    except ValueError as error:
        assert (
            "completed revision"
            in str(error)
        )

        print()
        print(
            "Completed revision failure "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Completed revision была ошибочно "
            "переведена в failed."
        )


async def test_revision_failure(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    generator: OpenAITelegramPostGenerator,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> None:
    """Проверяет безопасную фиксацию ошибки."""

    reservation, request_key = (
        await create_test_revision_reservation(
            pool,
            selection=selection,
            generator=generator,
            telegram_chat_id=(
                telegram_chat_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            publication_date=publication_date,
            created_batch_ids=(
                created_batch_ids
            ),
        )
    )

    failure = (
        await fail_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
            request_key=request_key.value,
            error_message=(
                "Тестовая ошибка revision"
            ),
            error_type="RuntimeError",
        )
    )

    assert (
        failure.generation_revision_id
        == reservation.generation_revision_id
    )

    assert (
        failure.batch_id
        == reservation.batch_id
    )

    assert (
        failure.source_generated_post_id
        == reservation.source_generated_post_id
    )

    assert (
        failure.review_action_id
        == reservation.review_action_id
    )

    assert (
        failure.request_key
        == request_key.value
    )

    assert (
        failure.revision_status
        == "failed"
    )

    assert (
        failure.batch_status
        == "awaiting_review"
    )

    assert (
        failure.source_post_status
        == "awaiting_review"
    )

    assert failure.already_failed is False

    assert (
        failure.error_type
        == "RuntimeError"
    )

    assert (
        failure.error_message
        == "Тестовая ошибка revision"
    )

    print()
    print("Generation revision failure: OK")
    print(
        "generation_revision_id="
        f"{failure.generation_revision_id}"
    )
    print("revision_status=failed")
    print("batch_status=awaiting_review")
    print("source_post_status=awaiting_review")
    print("already_failed=false")

    repeated_failure = (
        await fail_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
            request_key=request_key.value,
            error_message=(
                "Тестовая ошибка revision"
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
        == "Тестовая ошибка revision"
    )

    print()
    print("Repeated generation revision failure: OK")
    print("already_failed=true")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                grr.revision_status,
                grr.generated_post_id,
                grr.openai_usage,
                grr.openai_cost,
                grr.error_type,
                grr.error_message,
                grr.completed_at,
                grr.failed_at,

                b.batch_status,

                gp.version_number,
                gp.post_status,
                gp.image_path,
                gp.image_sha256,
                gp.image_prompt,
                gp.image_model_name,
                gp.image_prompt_version,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp2
                    WHERE gp2.batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.publication_attempts
                            AS pa
                    JOIN top3_news.generated_posts
                            AS gp3
                      ON gp3.generated_post_id =
                         pa.generated_post_id
                    WHERE gp3.batch_id = b.batch_id
                ) AS publication_attempt_count

            FROM
                top3_news
                .generation_revision_requests AS grr
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = grr.batch_id
            JOIN top3_news.generated_posts AS gp
              ON gp.generated_post_id =
                 grr.source_generated_post_id
            WHERE grr.generation_revision_id = $1
            """,
            reservation.generation_revision_id,
        )

    if record is None:
        raise AssertionError(
            "Failed revision не найдена."
        )

    assert (
        record["revision_status"]
        == "failed"
    )

    assert record["generated_post_id"] is None
    assert record["openai_usage"] is None
    assert record["openai_cost"] is None

    assert (
        record["error_type"]
        == "RuntimeError"
    )

    assert (
        record["error_message"]
        == "Тестовая ошибка revision"
    )

    assert record["completed_at"] is None
    assert record["failed_at"] is not None

    assert (
        record["batch_status"]
        == "awaiting_review"
    )

    assert record["version_number"] == 1

    assert (
        record["post_status"]
        == "awaiting_review"
    )

    assert (
        record["image_path"]
        == TEST_IMAGE_PATH
    )

    assert (
        record["image_sha256"]
        == TEST_IMAGE_SHA256
    )

    assert (
        record["image_prompt"]
        == TEST_IMAGE_PROMPT
    )

    assert (
        record["image_model_name"]
        == TEST_IMAGE_MODEL_NAME
    )

    assert (
        record["image_prompt_version"]
        == TEST_IMAGE_PROMPT_VERSION
    )

    assert record["generated_post_count"] == 1

    assert (
        record["publication_attempt_count"]
        == 0
    )

    print()
    print(
        "Persisted generation revision "
        "failure: OK"
    )
    print("generated_post_count=1")
    print("source_version_status=awaiting_review")
    print("image_fields_preserved=true")
    print("publication_attempt_count=0")
    print(
        "failure_version="
        f"{GENERATION_REVISION_FAILURE_VERSION}"
    )

    result = build_revision_result(
        selection
    )

    try:
        await complete_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation.generation_revision_id
            ),
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
            "Failed revision completion "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Failed revision была ошибочно "
            "завершена."
        )


async def cleanup_test_batches(
    pool: asyncpg.Pool,
    *,
    created_batch_ids: set[int],
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
            "temporary_revision_requests_deleted=true"
        )
        print(
            "ranking_run_18_preserved=true"
        )


async def main() -> int:
    """Запускает интеграционный тест."""

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

    try:
        selection = (
            await load_generation_top3(
                pool,
                ranking_run_id=(
                    TEST_RANKING_RUN_ID
                ),
            )
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

        reviewer_telegram_user_id = (
            await load_test_reviewer(pool)
        )

        publication_date = (
            build_test_publication_date()
        )

        await test_successful_revision_completion(
            pool,
            selection=selection,
            generator=generator,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            publication_date=publication_date,
            created_batch_ids=(
                created_batch_ids
            ),
        )

        await test_revision_failure(
            pool,
            selection=selection,
            generator=generator,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
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
        "completion/failure data inserted "
        "and deleted"
    )
    print(
        "publication_attempts created: 0"
    )
    print("Telegram publication: not performed")
    print(
        "Generation revision completion "
        "test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )