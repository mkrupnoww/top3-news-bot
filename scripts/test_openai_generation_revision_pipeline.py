import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
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
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    OpenAITelegramPostGenerator,
)
from app.generation.openai_revision_pipeline import (
    run_reserved_openai_generation_revision,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


TEST_RANKING_RUN_ID = 18
TEST_SUITE_ID = uuid4().hex

SOURCE_POST_TEXT = (
    "**TOP-3 НОВОСТЕЙ КИНО "
    "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n"
    "_______________\n\n"
    "1️⃣ **Тестовая первая версия**\n\n"
    "Исходный текст первой новости.\n\n"
    "2️⃣ **Тестовая вторая версия**\n\n"
    "Исходный текст второй новости.\n\n"
    "3️⃣ **Тестовая третья версия**\n\n"
    "Исходный текст третьей новости.\n\n"
    "……………\n"
    "Подписаться на VIP канал - @kkm_vip_bot"
)

EDITORIAL_COMMENT = (
    "Перепиши пост по фактам из title и summary "
    "и исправь редакционные замечания."
)

REVISION_ISSUES = (
    "Не добавлять неподтверждённые факты.",
    "Имена людей в русском тексте передавать кириллицей.",
)

TEST_IMAGE_PATH = (
    "/tmp/top3-news-test-revision-pipeline-image.png"
)

TEST_IMAGE_SHA256 = sha256(
    b"top3-news-test-revision-pipeline-image"
).hexdigest()

TEST_IMAGE_PROMPT = (
    "Тестовая иллюстрация для revision pipeline."
)

TEST_IMAGE_MODEL_NAME = (
    "test-revision-pipeline-image-model"
)

TEST_IMAGE_PROMPT_VERSION = (
    "test_revision_pipeline_image_prompt_v1"
)


class SyntheticGenerationRevisionError(
    RuntimeError
):
    """Тестовая ошибка revision-модели."""


@dataclass(frozen=True, slots=True)
class TestRevisionTelemetry:
    """Синтетические usage и стоимость."""

    usage: OpenAITokenUsage
    cost_estimate: OpenAICostEstimate


class FakeRevisionGenerationClient:
    """Поддельный revision-клиент без сети."""

    def __init__(
        self,
        *,
        failures_remaining: int = 0,
    ) -> None:
        self._failures_remaining = (
            failures_remaining
        )

        self.requests: list[
            GenerationModelRequest
        ] = []

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Возвращает revision-пост или тестовую ошибку."""

        self.requests.append(request)

        input_payload = json.loads(
            request.input_text
        )

        if (
            input_payload.get("task")
            != (
                "revise_russian_telegram_"
                "movie_news_top3"
            )
        ):
            raise AssertionError(
                "Revision-запрос содержит "
                "неверный task."
            )

        if (
            input_payload.get(
                "source_post_text"
            )
            != SOURCE_POST_TEXT
        ):
            raise AssertionError(
                "Revision-запрос содержит "
                "неверный source_post_text."
            )

        if (
            input_payload.get(
                "editorial_comment"
            )
            != EDITORIAL_COMMENT
        ):
            raise AssertionError(
                "Revision-запрос содержит "
                "неверный editorial_comment."
            )

        if (
            input_payload.get("issues")
            != list(REVISION_ISSUES)
        ):
            raise AssertionError(
                "Revision-запрос содержит "
                "неверный issues."
            )

        news_items = input_payload.get(
            "news"
        )

        if not isinstance(
            news_items,
            list,
        ):
            raise AssertionError(
                "Revision-запрос не содержит "
                "список news."
            )

        if len(news_items) != 3:
            raise AssertionError(
                "Revision-запрос должен содержать "
                "ровно три новости."
            )

        expected_news_keys = {
            "position",
            "news_id",
            "title",
            "summary",
        }

        generated_items: list[
            dict[str, object]
        ] = []

        draft_sections: list[str] = []

        for expected_position, item in enumerate(
            news_items,
            start=1,
        ):
            if not isinstance(item, dict):
                raise AssertionError(
                    "Новость revision-запроса "
                    "не является объектом."
                )

            if set(item) != expected_news_keys:
                raise AssertionError(
                    "Revision news содержит "
                    "лишние или отсутствующие поля: "
                    f"actual={sorted(item)}"
                )

            position = item.get(
                "position"
            )

            news_id = item.get(
                "news_id"
            )

            title = item.get(
                "title"
            )

            summary = item.get(
                "summary"
            )

            if position != expected_position:
                raise AssertionError(
                    "Нарушен порядок позиций "
                    "revision news."
                )

            if isinstance(news_id, bool):
                raise AssertionError(
                    "news_id не может быть bool."
                )

            if not isinstance(news_id, int):
                raise AssertionError(
                    "news_id должен быть int."
                )

            if not isinstance(title, str):
                raise AssertionError(
                    "title должен быть str."
                )

            if not isinstance(summary, str):
                raise AssertionError(
                    "summary должен быть str."
                )

            normalized_title = title.strip()
            normalized_summary = summary.strip()

            if not normalized_title:
                raise AssertionError(
                    "title не может быть пустым."
                )

            if not normalized_summary:
                raise AssertionError(
                    "summary не может быть пустым."
                )

            headline = normalized_title[:220]
            body = normalized_summary[:700]

            generated_items.append(
                {
                    "position": position,
                    "news_id": news_id,
                    "headline": headline,
                    "body": body,
                }
            )

            draft_sections.append(
                f"**{position}. {headline}**\n"
                f"{body}"
            )

        if "official_trailer_url" not in (
            request.instructions
        ):
            raise AssertionError(
                "Основные инструкции не содержат "
                "правило official_trailer_url."
            )

        if (
            "Дополнительные правила ревизии"
            not in request.instructions
        ):
            raise AssertionError(
                "Revision-инструкции не добавлены "
                "к основному prompt."
            )

        if self._failures_remaining > 0:
            self._failures_remaining -= 1

            raise SyntheticGenerationRevisionError(
                "Synthetic generation revision "
                "failure for pipeline test."
            )

        draft_post_text = (
            "**TOP-3 киноновости дня**\n\n"
            + "\n\n".join(
                draft_sections
            )
        )

        telemetry = build_telemetry(
            model_name=request.model
        )

        return GenerationModelResponse(
            output_text=json.dumps(
                {
                    "post_text": (
                        draft_post_text
                    ),
                    "items": generated_items,
                },
                ensure_ascii=False,
            ),
            usage=telemetry.usage,
            cost_estimate=(
                telemetry.cost_estimate
            ),
        )


def build_telemetry(
    *,
    model_name: str,
) -> TestRevisionTelemetry:
    """Создаёт согласованную телеметрию."""

    usage = OpenAITokenUsage(
        input_tokens=1600,
        cached_input_tokens=300,
        cache_write_tokens=100,
        output_tokens=320,
        reasoning_tokens=60,
        total_tokens=1920,
    )

    cost_estimate = OpenAICostEstimate(
        model_name=model_name,
        pricing_version=(
            "synthetic_generation_revision_"
            "pipeline_pricing_v1"
        ),
        regular_input_cost_usd=(
            Decimal("0.00240000")
        ),
        cached_input_cost_usd=(
            Decimal("0.00006000")
        ),
        cache_write_cost_usd=(
            Decimal("0.00025000")
        ),
        output_cost_usd=(
            Decimal("0.00384000")
        ),
        total_cost_usd=(
            Decimal("0.00655000")
        ),
    )

    return TestRevisionTelemetry(
        usage=usage,
        cost_estimate=cost_estimate,
    )


def build_model_name(
    *,
    scenario: str,
) -> str:
    """Создаёт уникальное имя тестовой модели."""

    return (
        "test-openai-revision-pipeline-"
        f"{TEST_SUITE_ID}-"
        f"{scenario}"
    )


def build_publication_date() -> date:
    """Создаёт изолированную дату теста."""

    random_offset = (
        int(TEST_SUITE_ID[:8], 16)
        % 20000
    )

    return (
        date(2600, 1, 1)
        + timedelta(days=random_offset)
    )


def build_batch_request_key() -> str:
    """Создаёт уникальный ключ test batch."""

    return sha256(
        uuid4().bytes
    ).hexdigest()


def decode_jsonb(
    value: Any,
) -> Any:
    """Декодирует jsonb из asyncpg."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(
                "Поле jsonb содержит "
                "некорректный JSON."
            ) from error

    return value


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
    Создаёт batch, source post v1
    и human changes_required.
    """

    batch_request_key = (
        build_batch_request_key()
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
                            "openai_generation_"
                            "revision_pipeline"
                        ),
                        "model_name": (
                            generator
                            .metadata
                            .model_name
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
                                "openai_generation_"
                                "revision_pipeline"
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
                                "openai_generation_"
                                "revision_pipeline"
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


async def load_revision_state(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
    review_action_id: int,
) -> asyncpg.Record:
    """Загружает состояние test revision."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_id,
                b.batch_status,
                b.ranking_run_id,

                source_gp.generated_post_id
                    AS source_generated_post_id,
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

                target_gp.generated_post_id
                    AS target_generated_post_id,
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
                    AS target_generation_mode,

                target_gp.generation_metadata
                    ->>'generation_revision_request_key'
                    AS target_request_key,

                target_gp.generation_metadata
                    ->>'generation_revision_request_key_version'
                    AS target_request_key_version,

                target_gp.generation_metadata
                    ->>'completion_version'
                    AS target_completion_version,

                target_gp.generation_metadata
                    ->'openai_usage'
                    AS target_openai_usage,

                target_gp.generation_metadata
                    ->'openai_cost'
                    AS target_openai_cost,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp
                    WHERE gp.batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.review_actions AS ra
                    JOIN top3_news.generated_posts AS gp
                      ON gp.generated_post_id =
                         ra.generated_post_id
                    WHERE gp.batch_id = b.batch_id
                ) AS review_action_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.publication_attempts AS pa
                    JOIN top3_news.generated_posts AS gp
                      ON gp.generated_post_id =
                         pa.generated_post_id
                    WHERE gp.batch_id = b.batch_id
                ) AS publication_attempt_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                ) AS revision_request_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'reserved'
                ) AS reserved_revision_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'completed'
                ) AS completed_revision_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'failed'
                ) AS failed_revision_count,

                (
                    SELECT COUNT(
                        DISTINCT grr.revision_request_key
                    )::integer
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                ) AS distinct_revision_request_key_count,

                (
                    SELECT MAX(
                        grr.generation_revision_id
                    )::bigint
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'completed'
                ) AS completed_generation_revision_id,

                (
                    SELECT MAX(
                        grr.generation_revision_id
                    )::bigint
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'failed'
                ) AS failed_generation_revision_id,

                (
                    SELECT grr.revision_request_key
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'completed'
                    ORDER BY
                        grr.generation_revision_id DESC
                    LIMIT 1
                ) AS completed_request_key,

                (
                    SELECT grr.openai_usage
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'completed'
                    ORDER BY
                        grr.generation_revision_id DESC
                    LIMIT 1
                ) AS revision_openai_usage,

                (
                    SELECT grr.openai_cost
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'completed'
                    ORDER BY
                        grr.generation_revision_id DESC
                    LIMIT 1
                ) AS revision_openai_cost,

                (
                    SELECT grr.error_type
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'failed'
                    ORDER BY
                        grr.generation_revision_id DESC
                    LIMIT 1
                ) AS failed_error_type,

                (
                    SELECT grr.error_message
                    FROM
                        top3_news
                        .generation_revision_requests AS grr
                    WHERE grr.batch_id = b.batch_id
                      AND grr.revision_status = 'failed'
                    ORDER BY
                        grr.generation_revision_id DESC
                    LIMIT 1
                ) AS failed_error_message

            FROM top3_news.publication_batches AS b
            JOIN top3_news.generated_posts AS source_gp
              ON source_gp.batch_id = b.batch_id
             AND source_gp.version_number = 1
            LEFT JOIN top3_news.generated_posts AS target_gp
              ON target_gp.batch_id = b.batch_id
             AND target_gp.version_number = 2
            WHERE b.batch_id = $1
              AND EXISTS (
                    SELECT 1
                    FROM top3_news.review_actions AS ra
                    WHERE ra.review_action_id = $2
                      AND ra.generated_post_id =
                          source_gp.generated_post_id
              )
            """,
            batch_id,
            review_action_id,
        )

    if record is None:
        raise AssertionError(
            "Не найден test revision state."
        )

    return record


async def test_successful_revision_pipeline(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> None:
    """Проверяет успешный полный revision pipeline."""

    model_name = build_model_name(
        scenario="success"
    )

    fake_client = (
        FakeRevisionGenerationClient()
    )

    generator = (
        OpenAITelegramPostGenerator(
            client=fake_client,
            model_name=model_name,
        )
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
        publication_date=publication_date,
        created_batch_ids=created_batch_ids,
    )

    first_result = (
        await run_reserved_openai_generation_revision(
            pool,
            generator=generator,
            review_action_id=review_action_id,
        )
    )

    assert (
        first_result.review_action_id
        == review_action_id
    )

    assert (
        first_result.batch_id
        == batch_id
    )

    assert (
        first_result.source_generated_post_id
        == source_generated_post_id
    )

    assert (
        first_result
        .revision_selection
        .ranking_run_id
        == TEST_RANKING_RUN_ID
    )

    assert (
        first_result
        .revision_selection
        .news_ids
        == selection.news_ids
    )

    assert (
        first_result.target_version_number
        == 2
    )

    assert (
        first_result.reservation.created_new
        is True
    )

    assert (
        first_result
        .reservation
        .should_call_model
        is True
    )

    assert first_result.model_called is True

    assert (
        first_result
        .duplicate_request_blocked
        is False
    )

    assert first_result.completed is True

    assert (
        first_result.revision_status
        == "completed"
    )

    assert first_result.generation is not None
    assert first_result.completion is not None

    assert (
        first_result.generated_post_id
        is not None
    )

    assert (
        first_result
        .completion
        .source_post_status
        == "superseded"
    )

    assert (
        first_result
        .completion
        .post_status
        == "awaiting_review"
    )

    assert len(fake_client.requests) == 1

    print(
        "Successful protected OpenAI "
        "generation revision pipeline: OK"
    )
    print(
        "generation_revision_id="
        f"{first_result.generation_revision_id}"
    )
    print(f"batch_id={batch_id}")
    print(
        "source_generated_post_id="
        f"{source_generated_post_id}"
    )
    print(
        "generated_post_id="
        f"{first_result.generated_post_id}"
    )
    print(
        "revision_request_key="
        f"{first_result.request_key.value}"
    )
    print("model_called=true")
    print("revision_status=completed")
    print("source_post_status=superseded")
    print("target_post_status=awaiting_review")
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )

    record = await load_revision_state(
        pool,
        batch_id=batch_id,
        review_action_id=review_action_id,
    )

    revision_usage = decode_jsonb(
        record["revision_openai_usage"]
    )

    revision_cost = decode_jsonb(
        record["revision_openai_cost"]
    )

    target_usage = decode_jsonb(
        record["target_openai_usage"]
    )

    target_cost = decode_jsonb(
        record["target_openai_cost"]
    )

    assert isinstance(revision_usage, dict)
    assert isinstance(revision_cost, dict)
    assert isinstance(target_usage, dict)
    assert isinstance(target_cost, dict)

    assert (
        record["batch_status"]
        == "awaiting_review"
    )

    assert (
        record["ranking_run_id"]
        == TEST_RANKING_RUN_ID
    )

    assert (
        record["source_generated_post_id"]
        == source_generated_post_id
    )

    assert record["source_version_number"] == 1

    assert (
        record["source_post_status"]
        == "superseded"
    )

    assert (
        record["target_generated_post_id"]
        == first_result.generated_post_id
    )

    assert record["target_version_number"] == 2

    assert (
        record["target_post_status"]
        == "awaiting_review"
    )

    assert (
        record["target_post_text"]
        == (
            first_result
            .generation
            .payload
            .post_text
        )
    )

    assert (
        record["target_text_format"]
        == generator.metadata.text_format
    )

    assert (
        record["target_text_model_name"]
        == model_name
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
        record["target_generation_mode"]
        == "openai_revision"
    )

    assert (
        record["target_request_key"]
        == first_result.request_key.value
    )

    assert (
        record["target_request_key_version"]
        == first_result.request_key.version
    )

    assert (
        record["target_completion_version"]
        == "generation_revision_completion_v1"
    )

    assert record["generated_post_count"] == 2
    assert record["review_action_count"] == 1
    assert (
        record["publication_attempt_count"]
        == 0
    )

    assert record["revision_request_count"] == 1
    assert record["reserved_revision_count"] == 0
    assert record["completed_revision_count"] == 1
    assert record["failed_revision_count"] == 0

    assert (
        record["distinct_revision_request_key_count"]
        == 1
    )

    assert (
        record["completed_generation_revision_id"]
        == first_result.generation_revision_id
    )

    assert (
        record["completed_request_key"]
        == first_result.request_key.value
    )

    assert (
        revision_usage["input_tokens"]
        == 1600
    )

    assert (
        revision_usage[
            "regular_input_tokens"
        ]
        == 1200
    )

    assert (
        revision_usage[
            "cached_input_tokens"
        ]
        == 300
    )

    assert (
        revision_usage[
            "cache_write_tokens"
        ]
        == 100
    )

    assert (
        revision_usage["output_tokens"]
        == 320
    )

    assert (
        revision_usage["total_tokens"]
        == 1920
    )

    assert target_usage == revision_usage

    assert (
        revision_cost["model_name"]
        == model_name
    )

    assert (
        revision_cost["total_cost_usd"]
        == "0.00655000"
    )

    assert target_cost == revision_cost

    print()
    print(
        "Persisted generation revision "
        "pipeline data: OK"
    )
    print("generated_post_count=2")
    print("revision_request_count=1")
    print("completed_revision_count=1")
    print("failed_revision_count=0")
    print("image_fields_inherited=true")
    print(
        "input_tokens="
        f"{revision_usage['input_tokens']}"
    )
    print(
        "output_tokens="
        f"{revision_usage['output_tokens']}"
    )
    print(
        "estimated_cost_usd="
        f"{revision_cost['total_cost_usd']}"
    )
    print("publication_attempt_count=0")

    second_result = (
        await run_reserved_openai_generation_revision(
            pool,
            generator=generator,
            review_action_id=review_action_id,
        )
    )

    assert (
        second_result.generation_revision_id
        == first_result.generation_revision_id
    )

    assert (
        second_result.request_key.value
        == first_result.request_key.value
    )

    assert (
        second_result.reservation.created_new
        is False
    )

    assert (
        second_result
        .reservation
        .should_call_model
        is False
    )

    assert second_result.model_called is False

    assert (
        second_result
        .duplicate_request_blocked
        is True
    )

    assert (
        second_result.revision_status
        == "completed"
    )

    assert second_result.completed is True

    assert second_result.generation is None
    assert second_result.completion is None

    assert len(fake_client.requests) == 1

    repeated_record = (
        await load_revision_state(
            pool,
            batch_id=batch_id,
            review_action_id=review_action_id,
        )
    )

    assert (
        repeated_record["generated_post_count"]
        == 2
    )

    assert (
        repeated_record["revision_request_count"]
        == 1
    )

    assert (
        repeated_record["completed_revision_count"]
        == 1
    )

    print()
    print(
        "Repeated completed revision "
        "pipeline request: OK"
    )
    print("model_called=false")
    print("duplicate_request_blocked=true")
    print("source_superseded_loader=true")
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )
    print("duplicate_generated_post=false")


async def test_failed_retry_revision_pipeline(
    pool: asyncpg.Pool,
    *,
    selection: GenerationTop3Selection,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    publication_date: date,
    created_batch_ids: set[int],
) -> None:
    """Проверяет failed -> retry -> completed."""

    model_name = build_model_name(
        scenario="failed-retry"
    )

    fake_client = (
        FakeRevisionGenerationClient(
            failures_remaining=1
        )
    )

    generator = (
        OpenAITelegramPostGenerator(
            client=fake_client,
            model_name=model_name,
        )
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
        publication_date=publication_date,
        created_batch_ids=created_batch_ids,
    )

    try:
        await run_reserved_openai_generation_revision(
            pool,
            generator=generator,
            review_action_id=review_action_id,
        )
    except SyntheticGenerationRevisionError as error:
        assert (
            "Synthetic generation revision failure"
            in str(error)
        )

        print()
        print(
            "Failed protected OpenAI generation "
            "revision pipeline: OK"
        )
        print(
            "raised_error_type="
            f"{type(error).__name__}"
        )
        print(
            "fake_model_call_count="
            f"{len(fake_client.requests)}"
        )
    else:
        raise AssertionError(
            "Тестовая revision-ошибка "
            "не была передана вызывающему коду."
        )

    assert len(fake_client.requests) == 1

    failed_record = await load_revision_state(
        pool,
        batch_id=batch_id,
        review_action_id=review_action_id,
    )

    assert (
        failed_record["batch_status"]
        == "awaiting_review"
    )

    assert (
        failed_record["source_post_status"]
        == "awaiting_review"
    )

    assert (
        failed_record["target_generated_post_id"]
        is None
    )

    assert (
        failed_record["generated_post_count"]
        == 1
    )

    assert (
        failed_record["revision_request_count"]
        == 1
    )

    assert (
        failed_record["reserved_revision_count"]
        == 0
    )

    assert (
        failed_record["completed_revision_count"]
        == 0
    )

    assert (
        failed_record["failed_revision_count"]
        == 1
    )

    assert (
        failed_record[
            "distinct_revision_request_key_count"
        ]
        == 1
    )

    assert (
        failed_record["failed_error_type"]
        == "SyntheticGenerationRevisionError"
    )

    assert (
        "Synthetic generation revision failure"
        in failed_record["failed_error_message"]
    )

    assert (
        failed_record["publication_attempt_count"]
        == 0
    )

    failed_generation_revision_id = int(
        failed_record[
            "failed_generation_revision_id"
        ]
    )

    print()
    print(
        "Persisted revision pipeline "
        "failure: OK"
    )
    print(
        "failed_generation_revision_id="
        f"{failed_generation_revision_id}"
    )
    print("revision_status=failed")
    print("batch_status=awaiting_review")
    print("source_post_status=awaiting_review")
    print("generated_post_count=1")
    print("publication_attempt_count=0")

    retry_result = (
        await run_reserved_openai_generation_revision(
            pool,
            generator=generator,
            review_action_id=review_action_id,
        )
    )

    assert (
        retry_result.reservation.created_new
        is True
    )

    assert (
        retry_result
        .reservation
        .should_call_model
        is True
    )

    assert retry_result.model_called is True

    assert (
        retry_result
        .duplicate_request_blocked
        is False
    )

    assert retry_result.completed is True

    assert (
        retry_result.revision_status
        == "completed"
    )

    assert (
        retry_result.generation_revision_id
        != failed_generation_revision_id
    )

    assert (
        retry_result.generated_post_id
        is not None
    )

    assert len(fake_client.requests) == 2

    retry_record = await load_revision_state(
        pool,
        batch_id=batch_id,
        review_action_id=review_action_id,
    )

    assert (
        retry_record["batch_status"]
        == "awaiting_review"
    )

    assert (
        retry_record["source_post_status"]
        == "superseded"
    )

    assert (
        retry_record["target_post_status"]
        == "awaiting_review"
    )

    assert (
        retry_record["generated_post_count"]
        == 2
    )

    assert (
        retry_record["revision_request_count"]
        == 2
    )

    assert (
        retry_record["reserved_revision_count"]
        == 0
    )

    assert (
        retry_record["completed_revision_count"]
        == 1
    )

    assert (
        retry_record["failed_revision_count"]
        == 1
    )

    assert (
        retry_record[
            "distinct_revision_request_key_count"
        ]
        == 1
    )

    assert (
        retry_record[
            "completed_generation_revision_id"
        ]
        == retry_result.generation_revision_id
    )

    assert (
        retry_record["completed_request_key"]
        == retry_result.request_key.value
    )

    assert (
        retry_record["publication_attempt_count"]
        == 0
    )

    print()
    print(
        "Failed revision retry: OK"
    )
    print(
        "retry_generation_revision_id="
        f"{retry_result.generation_revision_id}"
    )
    print("retry_created_new=true")
    print("retry_model_called=true")
    print("revision_request_count=2")
    print("failed_revision_count=1")
    print("completed_revision_count=1")
    print("generated_post_count=2")
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )

    repeated_result = (
        await run_reserved_openai_generation_revision(
            pool,
            generator=generator,
            review_action_id=review_action_id,
        )
    )

    assert (
        repeated_result.generation_revision_id
        == retry_result.generation_revision_id
    )

    assert (
        repeated_result.reservation.created_new
        is False
    )

    assert (
        repeated_result
        .reservation
        .should_call_model
        is False
    )

    assert repeated_result.model_called is False

    assert (
        repeated_result
        .duplicate_request_blocked
        is True
    )

    assert (
        repeated_result.revision_status
        == "completed"
    )

    assert repeated_result.completed is True

    assert len(fake_client.requests) == 2

    final_record = await load_revision_state(
        pool,
        batch_id=batch_id,
        review_action_id=review_action_id,
    )

    assert (
        final_record["revision_request_count"]
        == 2
    )

    assert (
        final_record["generated_post_count"]
        == 2
    )

    print()
    print(
        "Repeated completed retry request: OK"
    )
    print("model_called=false")
    print("duplicate_request_blocked=true")
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )


async def delete_test_batch(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> None:
    """Удаляет временный test batch."""

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
            "Не удалось удалить test batch: "
            f"batch_id={batch_id}, "
            f"result={result}"
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
            "удаления test batch."
        )

    assert record["batch_exists"] is False
    assert record["batch_item_count"] == 0
    assert record["generated_post_count"] == 0
    assert record["revision_request_count"] == 0
    assert record["ranking_run_exists"] is True


async def cleanup_test_batches(
    pool: asyncpg.Pool,
    *,
    created_batch_ids: set[int],
) -> None:
    """Удаляет временные данные pipeline."""

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

        reviewer_telegram_user_id = (
            await load_test_reviewer(pool)
        )

        publication_date = (
            build_publication_date()
        )

        await test_successful_revision_pipeline(
            pool,
            selection=selection,
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

        await test_failed_retry_revision_pipeline(
            pool,
            selection=selection,
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
        "Fake model calls: local only"
    )
    print(
        "Database changes: temporary batches, "
        "generated posts, review actions and "
        "revision requests inserted and deleted"
    )
    print("publication_attempts created: 0")
    print("Telegram publication: not performed")
    print(
        "Protected OpenAI generation revision "
        "pipeline test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )