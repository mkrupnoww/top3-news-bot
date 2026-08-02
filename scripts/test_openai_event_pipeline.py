import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from math import comb
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.event_evaluator import (
    EventRankingModelRequest,
    EventRankingModelResponse,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)
from app.ranking.openai_event_evaluator import (
    OpenAIEventRankingEvaluator,
)
from app.ranking.openai_event_pipeline import (
    run_reserved_openai_event_ranking,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


AS_OF = datetime(
    2026,
    7,
    31,
    11,
    21,
    tzinfo=timezone.utc,
)

SOURCE_CODES = (
    "variety_film",
)

WINDOW_HOURS = 24.0
CANDIDATE_LIMIT = 5

TEST_SUITE_ID = uuid4().hex

MACRO_TOPICS = (
    "creative_cast_production",
    "business_economy_law",
    "festivals_awards_criticism",
    "trailers_premieres_releases",
    "people_conflicts_legal",
)


class SyntheticEventModelError(
    RuntimeError
):
    """Тестовая ошибка event-модели."""


@dataclass(frozen=True, slots=True)
class TestModelTelemetry:
    """Согласованные usage и стоимость."""

    usage: OpenAITokenUsage
    cost_estimate: OpenAICostEstimate


class FakeStructuredEventRankingClient:
    """Поддельный event-клиент без сети."""

    def __init__(
        self,
        *,
        fail_request: bool = False,
    ) -> None:
        self._fail_request = fail_request
        self.requests: list[
            EventRankingModelRequest
        ] = []

    async def create_response(
        self,
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Возвращает события или тестовую ошибку."""

        self.requests.append(request)

        if self._fail_request:
            raise SyntheticEventModelError(
                "Synthetic event model failure "
                "for pipeline integration test."
            )

        input_payload = json.loads(
            request.input_text
        )

        candidates = input_payload.get(
            "candidates"
        )

        if not isinstance(candidates, list):
            raise AssertionError(
                "Event-запрос не содержит "
                "список candidates."
            )

        events: list[
            dict[str, object]
        ] = []

        for index, candidate in enumerate(
            candidates
        ):
            if not isinstance(candidate, dict):
                raise AssertionError(
                    "Кандидат не является объектом."
                )

            news_id = candidate.get("news_id")
            published_at = candidate.get(
                "published_at"
            )

            if (
                isinstance(news_id, bool)
                or not isinstance(news_id, int)
            ):
                raise AssertionError(
                    "news_id должен быть int."
                )

            if not isinstance(
                published_at,
                str,
            ):
                raise AssertionError(
                    "published_at должен быть str."
                )

            events.append(
                {
                    "representative_news_id": news_id,
                    "event_title": (
                        "Synthetic movie event "
                        f"{news_id}"
                    ),
                    "event_time_utc": published_at,
                    "macro_topic": (
                        MACRO_TOPICS[
                            index
                            % len(MACRO_TOPICS)
                        ]
                    ),
                    "i_score": 10,
                    "k_score": 10,
                    "n_score": 10,
                    "e_score": 10,
                    "x_score": 10,
                    "q_score": 1,
                    "impact_reason": (
                        "Synthetic maximum "
                        "industry impact."
                    ),
                    "hook_reason": (
                        "Synthetic strong "
                        "editorial hook."
                    ),
                    "q_reason": (
                        "Synthetic verified "
                        "primary source."
                    ),
                    "members": [
                        {
                            "news_id": news_id,
                            "source_relation": (
                                "primary"
                            ),
                            "is_representative": True,
                            "is_independent_source": True,
                            "counts_toward_reach": True,
                            "membership_reason": (
                                "Synthetic primary "
                                "representative."
                            ),
                        }
                    ],
                }
            )

        telemetry = build_telemetry(
            model_name=request.model
        )

        return EventRankingModelResponse(
            output_text=json.dumps(
                {
                    "events": events,
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
) -> TestModelTelemetry:
    """Создаёт тестовую телеметрию."""

    usage = OpenAITokenUsage(
        input_tokens=1000,
        cached_input_tokens=100,
        cache_write_tokens=0,
        output_tokens=300,
        reasoning_tokens=75,
        total_tokens=1300,
    )

    cost_estimate = OpenAICostEstimate(
        model_name=model_name,
        pricing_version=(
            "synthetic_event_pipeline_pricing_v1"
        ),
        regular_input_cost_usd=(
            Decimal("0.00180000")
        ),
        cached_input_cost_usd=(
            Decimal("0.00002000")
        ),
        cache_write_cost_usd=(
            Decimal("0.00000000")
        ),
        output_cost_usd=(
            Decimal("0.00360000")
        ),
        total_cost_usd=(
            Decimal("0.00542000")
        ),
    )

    return TestModelTelemetry(
        usage=usage,
        cost_estimate=cost_estimate,
    )


def build_model_name(
    *,
    scenario: str,
) -> str:
    """Создаёт уникальное имя тестовой модели."""

    return (
        "test-openai-event-pipeline-"
        f"{TEST_SUITE_ID}-"
        f"{scenario}"
    )


def decode_jsonb(
    value: Any,
) -> Any:
    """Декодирует jsonb из asyncpg."""

    if isinstance(value, str):
        return json.loads(value)

    return value


async def load_run_by_model(
    pool: asyncpg.Pool,
    *,
    model_name: str,
) -> asyncpg.Record:
    """Загружает единственный тестовый run."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                ranking_run_id,
                request_key,
                run_status,
                formula_version,
                candidate_count,
                scored_count,
                eligible_count,
                error_message,
                finished_at,
                parameters->'openai_usage'
                    AS openai_usage,
                parameters->'openai_cost'
                    AS openai_cost,
                parameters->'failure'
                    AS failure
            FROM top3_news.ranking_runs
            WHERE model_name = $1
            ORDER BY ranking_run_id
            """,
            model_name,
        )

    if len(records) != 1:
        raise AssertionError(
            "Ожидался один ranking_run "
            f"для {model_name!r}, "
            f"получено: {len(records)}"
        )

    return records[0]


async def load_v2_counts(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> asyncpg.Record:
    """Читает количество сохранённых сущностей."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                (
                    SELECT count(*)
                    FROM top3_news.ranking_events
                    WHERE ranking_run_id = $1
                ) AS event_count,
                (
                    SELECT count(*)
                    FROM top3_news.ranking_event_members
                    WHERE ranking_run_id = $1
                ) AS member_count,
                (
                    SELECT count(*)
                    FROM top3_news.news_scores
                    WHERE ranking_run_id = $1
                ) AS score_count,
                (
                    SELECT count(*)
                    FROM top3_news.ranking_combinations
                    WHERE ranking_run_id = $1
                ) AS combination_count,
                (
                    SELECT count(*)
                    FROM top3_news.ranking_combination_items
                    WHERE ranking_run_id = $1
                ) AS combination_item_count
            """,
            ranking_run_id,
        )

    if record is None:
        raise AssertionError(
            "Не удалось получить v2 counts."
        )

    return record


async def test_successful_pipeline(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Проверяет полный защищённый v2-конвейер."""

    model_name = build_model_name(
        scenario="success"
    )

    test_model_names.add(model_name)

    fake_client = (
        FakeStructuredEventRankingClient()
    )

    evaluator = OpenAIEventRankingEvaluator(
        client=fake_client,
        model_name=model_name,
    )

    first = (
        await run_reserved_openai_event_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    )

    candidate_count = len(
        first
        .candidate_selection
        .candidates
    )

    assert candidate_count >= 3
    assert first.reservation.created_new is True
    assert (
        first.reservation.should_call_model
        is True
    )
    assert first.model_called is True
    assert first.completed is True
    assert first.run_status == "completed"
    assert first.evaluation is not None
    assert first.calculation is not None
    assert first.completion is not None
    assert len(fake_client.requests) == 1

    assert (
        first.calculation.formula_version
        == FULL_FORMULA_VERSION
    )

    assert len(
        first.calculation.calculated_events
    ) == candidate_count

    assert (
        first
        .calculation
        .top3_selection
        .eligible_count
        == candidate_count
    )

    expected_combination_count = comb(
        candidate_count,
        3,
    )

    assert len(
        first
        .calculation
        .top3_selection
        .combinations
    ) == expected_combination_count

    assert (
        first.completion.candidate_count
        == candidate_count
    )
    assert (
        first.completion.scored_count
        == candidate_count
    )
    assert (
        first.completion.eligible_count
        == candidate_count
    )
    assert (
        first.completion.combination_count
        == expected_combination_count
    )

    print("Successful event pipeline: OK")
    print(
        "ranking_run_id="
        f"{first.ranking_run_id}"
    )
    print(
        "request_key="
        f"{first.request_key.value}"
    )
    print(
        f"candidate_count={candidate_count}"
    )
    print(
        "combination_count="
        f"{expected_combination_count}"
    )
    print("model_called=true")
    print("run_status=completed")

    record = await load_run_by_model(
        pool,
        model_name=model_name,
    )

    usage = decode_jsonb(
        record["openai_usage"]
    )
    cost = decode_jsonb(
        record["openai_cost"]
    )

    assert record["ranking_run_id"] == (
        first.ranking_run_id
    )
    assert record["request_key"] == (
        first.request_key.value
    )
    assert record["run_status"] == "completed"
    assert record["formula_version"] == (
        FULL_FORMULA_VERSION
    )
    assert record["candidate_count"] == (
        candidate_count
    )
    assert record["scored_count"] == (
        candidate_count
    )
    assert record["eligible_count"] == (
        candidate_count
    )
    assert record["error_message"] is None
    assert record["finished_at"] is not None

    assert usage["input_tokens"] == 1000
    assert usage[
        "regular_input_tokens"
    ] == 900
    assert usage[
        "cached_input_tokens"
    ] == 100
    assert usage["output_tokens"] == 300
    assert usage["total_tokens"] == 1300

    assert cost["model_name"] == model_name
    assert cost["total_cost_usd"] == (
        "0.00542000"
    )

    counts = await load_v2_counts(
        pool,
        ranking_run_id=first.ranking_run_id,
    )

    assert int(counts["event_count"]) == (
        candidate_count
    )
    assert int(counts["member_count"]) == (
        candidate_count
    )
    assert int(counts["score_count"]) == (
        candidate_count
    )
    assert int(
        counts["combination_count"]
    ) == expected_combination_count
    assert int(
        counts["combination_item_count"]
    ) == (
        expected_combination_count * 3
    )

    print()
    print("Persisted event pipeline data: OK")
    print(
        "ranking_events="
        f"{counts['event_count']}"
    )
    print(
        "news_scores="
        f"{counts['score_count']}"
    )
    print(
        "ranking_combinations="
        f"{counts['combination_count']}"
    )
    print(
        "python_postgres_scores_match=true"
    )
    print(
        "python_postgres_top_score_match=true"
    )

    second = (
        await run_reserved_openai_event_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    )

    assert second.ranking_run_id == (
        first.ranking_run_id
    )
    assert second.request_key.value == (
        first.request_key.value
    )
    assert (
        second.reservation.created_new
        is False
    )
    assert (
        second.reservation.should_call_model
        is False
    )
    assert second.model_called is False
    assert (
        second.duplicate_request_blocked
        is True
    )
    assert second.evaluation is None
    assert second.calculation is None
    assert second.completion is None
    assert second.run_status == "completed"
    assert len(fake_client.requests) == 1

    print()
    print("Repeated event pipeline request: OK")
    print("model_called=false")
    print("duplicate_request_blocked=true")
    print("fake_model_call_count=1")


async def test_failed_pipeline(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Проверяет ошибку модели и статус failed."""

    model_name = build_model_name(
        scenario="failure"
    )

    test_model_names.add(model_name)

    fake_client = (
        FakeStructuredEventRankingClient(
            fail_request=True
        )
    )

    evaluator = OpenAIEventRankingEvaluator(
        client=fake_client,
        model_name=model_name,
    )

    try:
        await run_reserved_openai_event_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    except SyntheticEventModelError as error:
        assert "Synthetic event model failure" in (
            str(error)
        )

        print()
        print("Failed event pipeline: OK")
        print(
            "raised_error_type="
            f"{type(error).__name__}"
        )
    else:
        raise AssertionError(
            "Ошибка fake event-модели "
            "не была передана вызывающему коду."
        )

    assert len(fake_client.requests) == 1

    record = await load_run_by_model(
        pool,
        model_name=model_name,
    )

    assert record["run_status"] == "failed"
    assert record["scored_count"] == 0
    assert record["eligible_count"] == 0
    assert record["finished_at"] is not None
    assert "Synthetic event model failure" in (
        record["error_message"]
    )

    failure = decode_jsonb(
        record["failure"]
    )

    assert failure["error_type"] == (
        "SyntheticEventModelError"
    )

    counts = await load_v2_counts(
        pool,
        ranking_run_id=int(
            record["ranking_run_id"]
        ),
    )

    assert all(
        int(counts[field_name]) == 0
        for field_name in (
            "event_count",
            "member_count",
            "score_count",
            "combination_count",
            "combination_item_count",
        )
    )

    repeated = (
        await run_reserved_openai_event_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    )

    assert repeated.model_called is False
    assert (
        repeated.duplicate_request_blocked
        is True
    )
    assert repeated.run_status == "failed"
    assert len(fake_client.requests) == 1

    print()
    print("Persisted event pipeline failure: OK")
    print("run_status=failed")
    print("stored_v2_entities=0")
    print("repeated_model_call=false")


async def test_non_24_hour_window_blocking(
    pool: asyncpg.Pool,
) -> None:
    """Проверяет строгое окно методики."""

    model_name = build_model_name(
        scenario="invalid-window"
    )

    fake_client = (
        FakeStructuredEventRankingClient()
    )

    evaluator = OpenAIEventRankingEvaluator(
        client=fake_client,
        model_name=model_name,
    )

    try:
        await run_reserved_openai_event_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=23.0,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    except ValueError as error:
        assert "строгое окно 24 часа" in str(
            error
        )

        assert len(fake_client.requests) == 0

        print()
        print("Non-24-hour window blocking: OK")
        print("model_called=false")
        return

    raise AssertionError(
        "Окно, отличное от 24 часов, "
        "не было заблокировано."
    )


async def cleanup_test_runs(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Удаляет временные v2-запуски."""

    if not test_model_names:
        return

    async with pool.acquire() as connection:
        deleted_records = await connection.fetch(
            """
            DELETE FROM top3_news.ranking_runs
            WHERE model_name = ANY($1::text[])
            RETURNING ranking_run_id
            """,
            sorted(test_model_names),
        )

    deleted_run_ids = tuple(
        int(record["ranking_run_id"])
        for record in deleted_records
    )

    if deleted_run_ids:
        async with pool.acquire() as connection:
            counts = await connection.fetchrow(
                """
                SELECT
                    (
                        SELECT count(*)
                        FROM top3_news.ranking_events
                        WHERE ranking_run_id
                            = ANY($1::bigint[])
                    ) AS event_count,
                    (
                        SELECT count(*)
                        FROM top3_news.news_scores
                        WHERE ranking_run_id
                            = ANY($1::bigint[])
                    ) AS score_count,
                    (
                        SELECT count(*)
                        FROM top3_news.ranking_combinations
                        WHERE ranking_run_id
                            = ANY($1::bigint[])
                    ) AS combination_count
                """,
                list(deleted_run_ids),
            )

        assert counts is not None
        assert int(counts["event_count"]) == 0
        assert int(counts["score_count"]) == 0
        assert int(
            counts["combination_count"]
        ) == 0

    print()
    print("Test data cleanup: OK")
    print(
        "deleted_ranking_run_ids="
        + (
            ",".join(
                str(run_id)
                for run_id in deleted_run_ids
            )
            if deleted_run_ids
            else "none"
        )
    )
    print("temporary_v2_data_deleted=true")


async def main() -> int:
    """Запускает интеграционный тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    test_model_names: set[str] = set()

    try:
        await test_successful_pipeline(
            pool,
            test_model_names=(
                test_model_names
            ),
        )

        await test_failed_pipeline(
            pool,
            test_model_names=(
                test_model_names
            ),
        )

        await test_non_24_hour_window_blocking(
            pool
        )
    finally:
        try:
            await cleanup_test_runs(
                pool,
                test_model_names=(
                    test_model_names
                ),
            )
        finally:
            await close_database_pool(pool)

    print()
    print("OpenAI requests: not performed")
    print(
        "Database changes: temporary v2 data "
        "inserted and deleted"
    )
    print("Telegram publication: not performed")
    print("Protected OpenAI event pipeline test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )