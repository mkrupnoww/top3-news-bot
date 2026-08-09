import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.ranking_run_completion import (
    complete_reserved_ranking_run,
)
from app.db.ranking_run_reservation import (
    reserve_ranking_run,
)
from app.db.ranking_scores import (
    ManualNewsAssessment,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    OpenAITelegramPostGenerator,
)
from app.generation.openai_pipeline import (
    run_reserved_openai_generation,
)
from app.generation.official_trailer_enrichment import (
    OfficialTrailerEnrichmentResult,
)
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)
from app.ranking.request_key import (
    REQUEST_KEY_VERSION,
    RankingRequestKey,
)
from app.ranking.score_formula import (
    FORMULA_VERSION,
)


TEST_NEWS_IDS = (
    11,
    9,
    10,
)

TEST_SUITE_ID = uuid4().hex

OFFICIAL_TRAILER_URL = (
    "https://www.youtube.com/watch?v=5fHXyqQOKL8"
)


class SyntheticGenerationError(
    RuntimeError
):
    """Тестовая ошибка генератора."""


@dataclass(frozen=True, slots=True)
class TestGenerationTelemetry:
    """Синтетические usage и стоимость."""

    usage: OpenAITokenUsage
    cost_estimate: OpenAICostEstimate


class FakeOfficialTrailerEnricher:
    """Поддельный trailer enrichment без HTTP."""

    def __init__(self) -> None:
        self.calls: list[
            dict[str, str]
        ] = []

    async def __call__(
        self,
        *,
        source_url: str,
        source_title: str,
        source_summary: str,
    ) -> OfficialTrailerEnrichmentResult:
        """Подтверждает трейлер только для TOP-2."""

        self.calls.append(
            {
                "source_url": source_url,
                "source_title": source_title,
                "source_summary": source_summary,
            }
        )

        position_in_pass = (
            (len(self.calls) - 1) % 3
        ) + 1

        if position_in_pass == 2:
            return OfficialTrailerEnrichmentResult(
                attempted=True,
                verified=True,
                official_trailer_url=(
                    OFFICIAL_TRAILER_URL
                ),
                reason=(
                    "verified_official_trailer"
                ),
                article_final_url=source_url,
                youtube_candidate_urls=(
                    OFFICIAL_TRAILER_URL,
                ),
                checked_video_urls=(
                    OFFICIAL_TRAILER_URL,
                ),
                verification_reasons=(
                    "verified_official_trailer",
                ),
                oembed_error_count=0,
                error_type=None,
            )

        return OfficialTrailerEnrichmentResult(
            attempted=False,
            verified=False,
            official_trailer_url=None,
            reason="source_is_not_trailer_news",
            article_final_url=None,
            youtube_candidate_urls=(),
            checked_video_urls=(),
            verification_reasons=(),
            oembed_error_count=0,
            error_type=None,
        )


class FakeStructuredGenerationClient:
    """Поддельный клиент генерации без сети."""

    def __init__(
        self,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        if (
            fail_on_call is not None
            and fail_on_call <= 0
        ):
            raise ValueError(
                "fail_on_call должен быть "
                "больше нуля."
            )

        self._fail_on_call = fail_on_call

        self.requests: list[
            GenerationModelRequest
        ] = []

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Возвращает пост или тестовую ошибку."""

        self.requests.append(request)

        call_number = len(self.requests)

        if (
            self._fail_on_call is not None
            and call_number
            == self._fail_on_call
        ):
            raise SyntheticGenerationError(
                "Synthetic generation failure "
                "for pipeline integration test "
                f"on call {call_number}."
            )

        input_payload = json.loads(
            request.input_text
        )

        task = input_payload.get("task")

        expected_tasks = {
            (
                "generate_russian_telegram_"
                "movie_news_top3"
            ),
            (
                "self_review_russian_telegram_"
                "movie_news_top3"
            ),
        }

        if task not in expected_tasks:
            raise AssertionError(
                "Неожиданный task в запросе "
                f"модели: {task!r}"
            )

        is_self_review = (
            task
            == (
                "self_review_russian_telegram_"
                "movie_news_top3"
            )
        )

        if is_self_review:
            source_post_text = (
                input_payload.get(
                    "source_post_text"
                )
            )

            if not isinstance(
                source_post_text,
                str,
            ):
                raise AssertionError(
                    "Self-review запрос не "
                    "содержит source_post_text."
                )

            if not source_post_text.strip():
                raise AssertionError(
                    "source_post_text self-review "
                    "не может быть пустым."
                )

        news_items = input_payload.get(
            "news"
        )

        if not isinstance(
            news_items,
            list,
        ):
            raise AssertionError(
                "Запрос модели не содержит "
                "список news."
            )

        if len(news_items) != 3:
            raise AssertionError(
                "Запрос модели должен содержать "
                "ровно три новости."
            )

        generated_items: list[
            dict[str, object]
        ] = []

        post_sections: list[str] = []

        for expected_position, item in enumerate(
            news_items,
            start=1,
        ):
            if not isinstance(item, dict):
                raise AssertionError(
                    "Новость модели "
                    "не является объектом."
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
                    "во входном запросе."
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

            normalized_summary = (
                summary.strip()
            )

            if not normalized_title:
                raise AssertionError(
                    "title не может быть пустым."
                )

            if not normalized_summary:
                raise AssertionError(
                    "summary не может быть пустым."
                )

            headline = (
                normalized_title[:220]
            )

            body = (
                normalized_summary[:700]
            )

            if is_self_review:
                headline = (
                    f"{headline} — self-review"
                )

            generated_items.append(
                {
                    "position": position,
                    "news_id": news_id,
                    "headline": headline,
                    "body": body,
                }
            )

            post_sections.append(
                f"**{position}. {headline}**\n"
                f"{body}"
            )

        post_text = (
            "**TOP-3 киноновости дня**\n\n"
            + "\n\n".join(
                post_sections
            )
            + (
                "\n\n"
                "__Какую из новостей "
                "обсудим подробнее?__"
            )
        )

        telemetry = build_telemetry(
            model_name=request.model
        )

        output_payload = {
            "post_text": post_text,
            "items": generated_items,
        }

        return GenerationModelResponse(
            output_text=json.dumps(
                output_payload,
                ensure_ascii=False,
            ),
            usage=telemetry.usage,
            cost_estimate=(
                telemetry.cost_estimate
            ),
            web_search_used=is_self_review,
            web_search_call_count=(
                1 if is_self_review else 0
            ),
            web_source_urls=(
                (
                    "https://example.test/"
                    "self-review-source"
                ),
            )
            if is_self_review
            else (),
        )


def build_telemetry(
    *,
    model_name: str,
) -> TestGenerationTelemetry:
    """Создаёт согласованную телеметрию."""

    usage = OpenAITokenUsage(
        input_tokens=1400,
        cached_input_tokens=200,
        cache_write_tokens=100,
        output_tokens=300,
        reasoning_tokens=50,
        total_tokens=1700,
    )

    cost_estimate = OpenAICostEstimate(
        model_name=model_name,
        pricing_version=(
            "synthetic_generation_"
            "pipeline_pricing_v1"
        ),
        regular_input_cost_usd=(
            Decimal("0.00220000")
        ),
        cached_input_cost_usd=(
            Decimal("0.00004000")
        ),
        cache_write_cost_usd=(
            Decimal("0.00025000")
        ),
        output_cost_usd=(
            Decimal("0.00360000")
        ),
        total_cost_usd=(
            Decimal("0.00609000")
        ),
    )

    return TestGenerationTelemetry(
        usage=usage,
        cost_estimate=cost_estimate,
    )


def build_model_name(
    *,
    scenario: str,
) -> str:
    """Создаёт уникальное имя тестовой модели."""

    return (
        "test-openai-generation-pipeline-"
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
        date(2500, 1, 1)
        + timedelta(days=random_offset)
    )


def build_ranking_metadata() -> RankingEvaluatorMetadata:
    """Создаёт метаданные временной ranking-фикстуры."""

    return RankingEvaluatorMetadata(
        run_mode="openai_ranking",
        evaluator_name=(
            "TestOpenAIGenerationPipelineRankingFixture"
        ),
        evaluator_version=(
            "test_openai_generation_pipeline_"
            "ranking_fixture_v1"
        ),
        prompt_version=(
            "test_openai_generation_pipeline_"
            "ranking_prompt_v1"
        ),
        model_name="gpt-5.6-terra",
    )


def build_ranking_request_key() -> RankingRequestKey:
    """Создаёт уникальный request_key фикстуры."""

    payload = {
        "test": (
            "openai_generation_pipeline_"
            "ranking_fixture"
        ),
        "test_suite_id": TEST_SUITE_ID,
        "news_ids": list(TEST_NEWS_IDS),
    }

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    value = sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    return RankingRequestKey(
        value=value,
        version=REQUEST_KEY_VERSION,
        canonical_json=canonical_json,
    )


def build_ranking_assessments() -> tuple[
    ManualNewsAssessment,
    ...,
]:
    """Фиксирует legacy TOP-3 в порядке 11, 9, 10."""

    return (
        ManualNewsAssessment(
            news_id=11,
            f_score=Decimal("10.000000"),
            m_score=Decimal("10.000000"),
            r_score=Decimal("10.000000"),
            h_score=Decimal("10.000000"),
            q_score=Decimal("1.000000"),
            explanation=(
                "Тестовая новость с максимальным "
                "баллом для позиции 1."
            ),
        ),
        ManualNewsAssessment(
            news_id=9,
            f_score=Decimal("9.000000"),
            m_score=Decimal("9.000000"),
            r_score=Decimal("9.000000"),
            h_score=Decimal("9.000000"),
            q_score=Decimal("1.000000"),
            explanation=(
                "Тестовая новость со вторым "
                "баллом для позиции 2."
            ),
        ),
        ManualNewsAssessment(
            news_id=10,
            f_score=Decimal("8.000000"),
            m_score=Decimal("8.000000"),
            r_score=Decimal("8.000000"),
            h_score=Decimal("8.000000"),
            q_score=Decimal("1.000000"),
            explanation=(
                "Тестовая новость с третьим "
                "баллом для позиции 3."
            ),
        ),
    )


async def create_test_ranking_run(
    pool: asyncpg.Pool,
    *,
    created_ranking_run_ids: set[int],
) -> int:
    """Создаёт и завершает временный ranking_run."""

    metadata = build_ranking_metadata()
    request_key = build_ranking_request_key()

    window_finished_at = datetime(
        2026,
        8,
        9,
        12,
        0,
        tzinfo=timezone.utc,
    )

    window_started_at = (
        window_finished_at
        - timedelta(hours=24)
    )

    reservation = await reserve_ranking_run(
        pool,
        request_key=request_key,
        formula_version=FORMULA_VERSION,
        metadata=metadata,
        window_started_at=window_started_at,
        window_finished_at=window_finished_at,
        news_ids=TEST_NEWS_IDS,
    )

    created_ranking_run_ids.add(
        reservation.ranking_run_id
    )

    assert reservation.created_new is True
    assert reservation.should_call_model is True
    assert reservation.run_status == "running"

    telemetry = build_telemetry(
        model_name="gpt-5.6-terra"
    )

    completion = (
        await complete_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=metadata,
            assessments=(
                build_ranking_assessments()
            ),
            usage=telemetry.usage,
            cost_estimate=(
                telemetry.cost_estimate
            ),
        )
    )

    assert completion.run_status == "completed"
    assert completion.already_completed is False
    assert completion.candidate_count == 3
    assert completion.scored_count == 3
    assert completion.eligible_count == 3

    assert [
        score.news_id
        for score in completion.scores
    ] == [
        11,
        9,
        10,
    ]

    assert [
        score.rank_position
        for score in completion.scores
    ] == [
        1,
        2,
        3,
    ]

    async with pool.acquire() as connection:
        selected_for_top3_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.news_scores
                WHERE ranking_run_id = $1
                  AND selected_for_top3 = true
                """,
                reservation.ranking_run_id,
            )
        )

    assert selected_for_top3_count == 0

    print(
        "Temporary ranking fixture: OK"
    )
    print(
        "temporary_ranking_run_id="
        f"{reservation.ranking_run_id}"
    )
    print("ranking_fixture_news_ids=11,9,10")
    print("selection_mode=legacy_rank_position")

    return reservation.ranking_run_id


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


async def load_batch_by_model(
    pool: asyncpg.Pool,
    *,
    model_name: str,
) -> asyncpg.Record:
    """Загружает тестовый publication_batch."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                b.batch_id,
                b.publication_date,
                b.edition,
                b.batch_status,
                b.ranking_run_id,
                b.target_telegram_chat_id,
                b.generation_request_key,
                b.error_message,

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

                b.metadata->'openai_usage'
                    AS batch_openai_usage,

                b.metadata->'openai_cost'
                    AS batch_openai_cost,

                b.metadata->'failure'
                    AS failure,

                gp.generated_post_id,
                gp.version_number,
                gp.post_status,
                gp.post_text,
                gp.text_format
                    AS post_text_format,
                gp.text_model_name,
                gp.text_prompt_version,

                gp.generation_metadata
                    ->'openai_usage'
                    AS post_openai_usage,

                gp.generation_metadata
                    ->'openai_cost'
                    AS post_openai_cost,

                gp.generation_metadata
                    ->'generated_items'
                    AS generated_items,

                gp.generation_metadata
                    ->>'generation_request_key'
                    AS post_request_key,

                gp.generation_metadata
                    ->>'completion_version'
                    AS completion_version,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.batch_items AS bi
                    WHERE bi.batch_id = b.batch_id
                ) AS batch_item_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.generated_posts AS p
                    WHERE p.batch_id = b.batch_id
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
            LEFT JOIN
                top3_news.generated_posts AS gp
              ON gp.batch_id = b.batch_id
             AND gp.version_number = 1
            WHERE
                b.metadata->>'model_name' = $1
            ORDER BY b.batch_id
            """,
            model_name,
        )

    if len(records) != 1:
        raise AssertionError(
            "Ожидался ровно один "
            "publication_batch для модели "
            f"{model_name!r}, "
            f"получено: {len(records)}"
        )

    return records[0]


async def test_successful_pipeline(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    telegram_chat_id: int,
    publication_date: date,
    test_model_names: set[str],
) -> None:
    """Проверяет успешный полный pipeline."""

    model_name = build_model_name(
        scenario="success"
    )

    test_model_names.add(
        model_name
    )

    fake_client = (
        FakeStructuredGenerationClient()
    )

    fake_trailer_enricher = (
        FakeOfficialTrailerEnricher()
    )

    generator = (
        OpenAITelegramPostGenerator(
            client=fake_client,
            model_name=model_name,
        )
    )

    first_result = (
        await run_reserved_openai_generation(
            pool,
            generator=generator,
            trailer_enricher=(
                fake_trailer_enricher
            ),
            ranking_run_id=(
                ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
        )
    )

    assert (
        first_result.selection.ranking_run_id
        == ranking_run_id
    )

    assert (
        first_result.selection.news_ids
        == (11, 9, 10)
    )

    assert all(
        item.official_trailer_url is None
        for item in first_result.selection.items
    )

    assert len(
        fake_trailer_enricher.calls
    ) == 3

    reserved_input = json.loads(
        first_result.model_request.input_text
    )

    assert all(
        "official_trailer_url"
        not in news_item
        for news_item in reserved_input["news"]
    )

    request_key_payload = json.loads(
        first_result.request_key.canonical_json
    )

    assert all(
        "official_trailer_url"
        not in news_item
        for news_item in request_key_payload["top3"]
    )

    assert (
        OFFICIAL_TRAILER_URL
        not in first_result.request_key.canonical_json
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
        first_result.batch_status
        == "awaiting_review"
    )

    assert first_result.generation is not None

    assert first_result.completion is not None

    assert (
        first_result.generated_post_id
        is not None
    )

    assert len(fake_client.requests) == 2

    primary_request = (
        fake_client.requests[0]
    )

    self_review_request = (
        fake_client.requests[1]
    )

    assert (
        primary_request.allow_web_search
        is False
    )

    assert (
        self_review_request.allow_web_search
        is True
    )

    primary_input = json.loads(
        primary_request.input_text
    )

    self_review_input = json.loads(
        self_review_request.input_text
    )

    assert (
        primary_input["task"]
        == (
            "generate_russian_telegram_"
            "movie_news_top3"
        )
    )

    assert (
        self_review_input["task"]
        == (
            "self_review_russian_telegram_"
            "movie_news_top3"
        )
    )

    assert (
        "official_trailer_url"
        not in primary_input["news"][0]
    )

    assert primary_input[
        "news"
    ][1][
        "official_trailer_url"
    ] == OFFICIAL_TRAILER_URL

    assert (
        "official_trailer_url"
        not in primary_input["news"][2]
    )

    assert (
        "official_trailer_url"
        not in self_review_input["news"][0]
    )

    assert self_review_input[
        "news"
    ][1][
        "official_trailer_url"
    ] == OFFICIAL_TRAILER_URL

    assert (
        "official_trailer_url"
        not in self_review_input["news"][2]
    )

    assert isinstance(
        self_review_input["source_post_text"],
        str,
    )

    assert (
        self_review_input[
            "source_post_text"
        ].strip()
    )

    assert (
        first_result
        .generation
        .model_response
        .usage
        is not None
    )

    assert (
        first_result
        .generation
        .model_response
        .cost_estimate
        is not None
    )

    assert (
        first_result
        .generation
        .model_response
        .web_search_used
        is True
    )

    assert (
        first_result
        .generation
        .model_response
        .web_search_call_count
        == 1
    )

    assert (
        first_result
        .generation
        .model_response
        .web_source_urls
        == (
            (
                "https://example.test/"
                "self-review-source"
            ),
        )
    )

    print(
        "Successful protected "
        "generation pipeline: OK"
    )
    print(
        f"batch_id={first_result.batch_id}"
    )
    print(
        "generated_post_id="
        f"{first_result.generated_post_id}"
    )
    print(
        "generation_request_key="
        f"{first_result.request_key.value}"
    )
    print(
        "news_ids="
        + ",".join(
            str(news_id)
            for news_id
            in first_result.selection.news_ids
        )
    )
    print("model_called=true")
    print("batch_status=awaiting_review")
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )

    record = await load_batch_by_model(
        pool,
        model_name=model_name,
    )

    batch_usage = decode_jsonb(
        record["batch_openai_usage"]
    )

    batch_cost = decode_jsonb(
        record["batch_openai_cost"]
    )

    post_usage = decode_jsonb(
        record["post_openai_usage"]
    )

    post_cost = decode_jsonb(
        record["post_openai_cost"]
    )

    generated_items = decode_jsonb(
        record["generated_items"]
    )

    assert isinstance(batch_usage, dict)
    assert isinstance(batch_cost, dict)
    assert isinstance(post_usage, dict)
    assert isinstance(post_cost, dict)

    assert isinstance(
        generated_items,
        list,
    )

    assert (
        record["batch_id"]
        == first_result.batch_id
    )

    assert (
        record["publication_date"]
        == publication_date
    )

    assert record["edition"] > 0

    assert (
        record["batch_status"]
        == "awaiting_review"
    )

    assert (
        record["ranking_run_id"]
        == ranking_run_id
    )

    assert (
        record["target_telegram_chat_id"]
        == telegram_chat_id
    )

    assert (
        record["generation_request_key"]
        == first_result.request_key.value
    )

    assert record["error_message"] is None

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
        == model_name
    )

    assert (
        record["text_format"]
        == "markdown"
    )

    assert (
        record["generated_post_id"]
        == first_result.generated_post_id
    )

    assert record["version_number"] == 1

    assert (
        record["post_status"]
        == "awaiting_review"
    )

    assert (
        record["post_text"]
        == (
            first_result
            .generation
            .payload
            .post_text
        )
    )

    assert (
        record["post_text_format"]
        == "markdown"
    )

    assert (
        record["text_model_name"]
        == model_name
    )

    assert (
        record["text_prompt_version"]
        == generator.metadata.prompt_version
    )

    assert (
        record["post_request_key"]
        == first_result.request_key.value
    )

    assert (
        record["completion_version"]
        == (
            "reserved_generation_"
            "completion_v1"
        )
    )

    assert record["batch_item_count"] == 3

    assert (
        record["generated_post_count"]
        == 1
    )

    assert (
        record["publication_attempt_count"]
        == 0
    )

    assert record["review_action_count"] == 0

    assert len(generated_items) == 3

    assert [
        item["news_id"]
        for item in generated_items
    ] == [
        11,
        9,
        10,
    ]

    assert all(
        isinstance(
            item.get("headline"),
            str,
        )
        and item["headline"].endswith(
            " — self-review"
        )
        for item in generated_items
    )

    assert (
        "— self-review"
        in record["post_text"]
    )

    assert (
        batch_usage["input_tokens"]
        == 2800
    )

    assert (
        batch_usage[
            "regular_input_tokens"
        ]
        == 2200
    )

    assert (
        batch_usage[
            "cached_input_tokens"
        ]
        == 400
    )

    assert (
        batch_usage[
            "cache_write_tokens"
        ]
        == 200
    )

    assert (
        batch_usage["output_tokens"]
        == 600
    )

    assert (
        batch_usage["total_tokens"]
        == 3400
    )

    assert post_usage == batch_usage

    assert (
        batch_cost["model_name"]
        == model_name
    )

    assert (
        batch_cost["total_cost_usd"]
        == "0.01218000"
    )

    assert post_cost == batch_cost

    print()
    print(
        "Persisted generation pipeline "
        "data: OK"
    )
    print(
        "batch_item_count="
        f"{record['batch_item_count']}"
    )
    print(
        "generated_post_count="
        f"{record['generated_post_count']}"
    )
    print(
        "generated_item_count="
        f"{len(generated_items)}"
    )
    print(
        "input_tokens="
        f"{batch_usage['input_tokens']}"
    )
    print(
        "output_tokens="
        f"{batch_usage['output_tokens']}"
    )
    print(
        "estimated_cost_usd="
        f"{batch_cost['total_cost_usd']}"
    )
    print(
        "publication_attempt_count="
        f"{record['publication_attempt_count']}"
    )
    print(
        "review_action_count="
        f"{record['review_action_count']}"
    )

    second_result = (
        await run_reserved_openai_generation(
            pool,
            generator=generator,
            trailer_enricher=(
                fake_trailer_enricher
            ),
            ranking_run_id=(
                ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
        )
    )

    assert (
        second_result.batch_id
        == first_result.batch_id
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

    assert second_result.generation is None
    assert second_result.completion is None

    assert (
        second_result.batch_status
        == "awaiting_review"
    )

    assert (
        second_result.generated_post_id
        is None
    )

    assert len(fake_client.requests) == 2

    assert len(
        fake_trailer_enricher.calls
    ) == 3

    repeated_record = (
        await load_batch_by_model(
            pool,
            model_name=model_name,
        )
    )

    assert (
        repeated_record["batch_id"]
        == first_result.batch_id
    )

    assert (
        repeated_record[
            "generated_post_count"
        ]
        == 1
    )

    print()
    print(
        "Repeated generation pipeline "
        "request: OK"
    )
    print("model_called=false")
    print(
        "duplicate_request_blocked=true"
    )
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )
    print(
        "duplicate_generated_post=false"
    )


async def test_failed_pipeline(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    telegram_chat_id: int,
    publication_date: date,
    test_model_names: set[str],
) -> None:
    """Проверяет ошибку self-review и failed."""

    model_name = build_model_name(
        scenario="failure"
    )

    test_model_names.add(
        model_name
    )

    fake_client = (
        FakeStructuredGenerationClient(
            fail_on_call=2
        )
    )

    fake_trailer_enricher = (
        FakeOfficialTrailerEnricher()
    )

    generator = (
        OpenAITelegramPostGenerator(
            client=fake_client,
            model_name=model_name,
        )
    )

    try:
        await run_reserved_openai_generation(
            pool,
            generator=generator,
            trailer_enricher=(
                fake_trailer_enricher
            ),
            ranking_run_id=(
                ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
        )
    except SyntheticGenerationError as error:
        assert (
            "Synthetic generation failure"
            in str(error)
        )

        print()
        print(
            "Failed protected generation "
            "pipeline: OK"
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
            "Тестовая ошибка генератора "
            "не была передана вызывающему коду."
        )

    assert len(fake_client.requests) == 2

    assert len(
        fake_trailer_enricher.calls
    ) == 3

    assert (
        fake_client
        .requests[0]
        .allow_web_search
        is False
    )

    assert (
        fake_client
        .requests[1]
        .allow_web_search
        is True
    )

    failed_self_review_input = json.loads(
        fake_client.requests[1].input_text
    )

    assert (
        failed_self_review_input["task"]
        == (
            "self_review_russian_telegram_"
            "movie_news_top3"
        )
    )

    record = await load_batch_by_model(
        pool,
        model_name=model_name,
    )

    batch_id = int(
        record["batch_id"]
    )

    assert (
        record["batch_status"]
        == "failed"
    )

    assert (
        record["ranking_run_id"]
        == ranking_run_id
    )

    assert (
        record["generation_request_key"]
        is not None
    )

    assert (
        "Synthetic generation failure"
        in record["error_message"]
    )

    assert record["batch_item_count"] == 3

    assert (
        record["generated_post_count"]
        == 0
    )

    assert (
        record["publication_attempt_count"]
        == 0
    )

    assert record["review_action_count"] == 0

    failure_payload = decode_jsonb(
        record["failure"]
    )

    assert isinstance(
        failure_payload,
        dict,
    )

    assert (
        failure_payload["error_type"]
        == "SyntheticGenerationError"
    )

    assert (
        "Synthetic generation failure"
        in failure_payload["error_message"]
    )

    print()
    print(
        "Persisted generation pipeline "
        "failure: OK"
    )
    print(f"batch_id={batch_id}")
    print("batch_status=failed")
    print(
        "batch_item_count="
        f"{record['batch_item_count']}"
    )
    print("generated_post_count=0")
    print(
        "failure_error_type="
        f"{failure_payload['error_type']}"
    )

    repeated_result = (
        await run_reserved_openai_generation(
            pool,
            generator=generator,
            trailer_enricher=(
                fake_trailer_enricher
            ),
            ranking_run_id=(
                ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
        )
    )

    assert (
        repeated_result.batch_id
        == batch_id
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
        repeated_result.batch_status
        == "failed"
    )

    assert repeated_result.generation is None
    assert repeated_result.completion is None

    assert len(fake_client.requests) == 2

    assert len(
        fake_trailer_enricher.calls
    ) == 3

    print()
    print(
        "Repeated failed generation "
        "pipeline request: OK"
    )
    print("model_called=false")
    print(
        "duplicate_request_blocked=true"
    )
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )


async def cleanup_test_batches(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Удаляет временные данные pipeline."""

    if not test_model_names:
        return

    model_names = sorted(
        test_model_names
    )

    async with pool.acquire() as connection:
        batch_records = (
            await connection.fetch(
                """
                SELECT batch_id
                FROM
                    top3_news
                    .publication_batches
                WHERE
                    metadata->>'model_name'
                    = ANY($1::text[])
                ORDER BY batch_id
                """,
                model_names,
            )
        )

    batch_ids = [
        int(record["batch_id"])
        for record in batch_records
    ]

    if batch_ids:
        async with pool.acquire() as connection:
            deleted_records = (
                await connection.fetch(
                    """
                    DELETE FROM
                        top3_news
                        .publication_batches
                    WHERE batch_id =
                        ANY($1::bigint[])
                    RETURNING batch_id
                    """,
                    batch_ids,
                )
            )

        deleted_batch_ids = sorted(
            int(record["batch_id"])
            for record in deleted_records
        )
    else:
        deleted_batch_ids = []

    async with pool.acquire() as connection:
        remaining_batch_count = (
            await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM
                    top3_news
                    .publication_batches
                WHERE
                    metadata->>'model_name'
                    = ANY($1::text[])
                """,
                model_names,
            )
        )

        if batch_ids:
            remaining_item_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM top3_news.batch_items
                    WHERE batch_id =
                        ANY($1::bigint[])
                    """,
                    batch_ids,
                )
            )

            remaining_post_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.generated_posts
                    WHERE batch_id =
                        ANY($1::bigint[])
                    """,
                    batch_ids,
                )
            )
        else:
            remaining_item_count = 0
            remaining_post_count = 0

    assert remaining_batch_count == 0
    assert remaining_item_count == 0
    assert remaining_post_count == 0

    print()
    print("Test data cleanup: OK")
    print(
        "deleted_batch_ids="
        + (
            ",".join(
                str(batch_id)
                for batch_id
                in deleted_batch_ids
            )
            if deleted_batch_ids
            else "none"
        )
    )
    print(
        "temporary_batches_items_posts_"
        "deleted=true"
    )



async def cleanup_test_ranking_runs(
    pool: asyncpg.Pool,
    *,
    created_ranking_run_ids: set[int],
) -> None:
    """Удаляет временные ranking runs и scores."""

    deleted_ranking_run_ids: list[int] = []

    for ranking_run_id in sorted(
        created_ranking_run_ids
    ):
        async with pool.acquire() as connection:
            result = await connection.execute(
                """
                DELETE FROM top3_news.ranking_runs
                WHERE ranking_run_id = $1
                """,
                ranking_run_id,
            )

        if result != "DELETE 1":
            raise RuntimeError(
                "Не удалось удалить временный "
                "ranking_run: "
                f"ranking_run_id={ranking_run_id}, "
                f"result={result}"
            )

        async with pool.acquire() as connection:
            run_exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM top3_news.ranking_runs
                    WHERE ranking_run_id = $1
                )
                """,
                ranking_run_id,
            )

            score_count = await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.news_scores
                WHERE ranking_run_id = $1
                """,
                ranking_run_id,
            )

        assert run_exists is False
        assert score_count == 0

        deleted_ranking_run_ids.append(
            ranking_run_id
        )

    print()
    print("Ranking fixture cleanup: OK")
    print(
        "deleted_ranking_run_ids="
        + (
            ",".join(
                str(ranking_run_id)
                for ranking_run_id
                in deleted_ranking_run_ids
            )
            if deleted_ranking_run_ids
            else "none"
        )
    )
    print("temporary_ranking_data_deleted=true")


async def main() -> int:
    """Запускает интеграционный тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    test_model_names: set[str] = set()

    created_ranking_run_ids: set[int] = set()

    publication_date = (
        build_publication_date()
    )

    try:
        ranking_run_id = (
            await create_test_ranking_run(
                pool,
                created_ranking_run_ids=(
                    created_ranking_run_ids
                ),
            )
        )

        await test_successful_pipeline(
            pool,
            ranking_run_id=ranking_run_id,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            publication_date=(
                publication_date
            ),
            test_model_names=(
                test_model_names
            ),
        )

        await test_failed_pipeline(
            pool,
            ranking_run_id=ranking_run_id,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            publication_date=(
                publication_date
                + timedelta(days=1)
            ),
            test_model_names=(
                test_model_names
            ),
        )
    finally:
        try:
            await cleanup_test_batches(
                pool,
                test_model_names=(
                    test_model_names
                ),
            )
        finally:
            try:
                await cleanup_test_ranking_runs(
                    pool,
                    created_ranking_run_ids=(
                        created_ranking_run_ids
                    ),
                )
            finally:
                await close_database_pool(pool)

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print(
        "Trailer HTTP requests: fake enricher only"
    )
    print(
        "Database changes: temporary ranking_run, "
        "news_scores, batches, batch_items and "
        "generated_post inserted and deleted"
    )
    print(
        "publication_attempts created: 0"
    )
    print("Telegram publication: not performed")
    print(
        "Protected OpenAI generation "
        "pipeline test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )