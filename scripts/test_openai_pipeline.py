import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.openai_evaluator import (
    OpenAIRankingEvaluator,
    RankingModelRequest,
    RankingModelResponse,
)
from app.ranking.openai_pipeline import (
    run_reserved_openai_ranking,
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


class SyntheticModelError(
    RuntimeError
):
    """Тестовая ошибка модели."""


@dataclass(frozen=True, slots=True)
class TestModelTelemetry:
    """Тестовые usage и стоимость."""

    usage: OpenAITokenUsage
    cost_estimate: OpenAICostEstimate


class FakeStructuredRankingClient:
    """Поддельный клиент модели без сети."""

    def __init__(
        self,
        *,
        fail_request: bool = False,
    ) -> None:
        self._fail_request = fail_request
        self.requests: list[
            RankingModelRequest
        ] = []

    async def create_response(
        self,
        request: RankingModelRequest,
    ) -> RankingModelResponse:
        """Возвращает оценки или тестовую ошибку."""

        self.requests.append(request)

        if self._fail_request:
            raise SyntheticModelError(
                "Synthetic model failure "
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
                "Запрос модели не содержит "
                "список candidates."
            )

        scores: list[
            dict[str, object]
        ] = []

        for index, candidate in enumerate(
            candidates
        ):
            if not isinstance(candidate, dict):
                raise AssertionError(
                    "Кандидат модели "
                    "не является объектом."
                )

            news_id = candidate.get(
                "news_id"
            )

            if isinstance(news_id, bool):
                raise AssertionError(
                    "news_id не может быть bool."
                )

            if not isinstance(news_id, int):
                raise AssertionError(
                    "news_id должен быть int."
                )

            scores.append(
                {
                    "news_id": news_id,
                    "f_score": round(
                        9.5 - index * 0.1,
                        2,
                    ),
                    "m_score": round(
                        8.0 - index * 0.2,
                        2,
                    ),
                    "r_score": round(
                        7.0 - index * 0.15,
                        2,
                    ),
                    "h_score": round(
                        6.0 - index * 0.1,
                        2,
                    ),
                    "q_score": 0.95,
                    "explanation": (
                        "Синтетическая оценка "
                        "для интеграционного теста: "
                        f"news_id={news_id}."
                    ),
                }
            )

        telemetry = build_telemetry(
            model_name=request.model
        )

        return RankingModelResponse(
            output_text=json.dumps(
                {
                    "scores": scores,
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
    """Создаёт согласованную телеметрию."""

    usage = OpenAITokenUsage(
        input_tokens=1000,
        cached_input_tokens=100,
        cache_write_tokens=0,
        output_tokens=200,
        reasoning_tokens=50,
        total_tokens=1200,
    )

    cost_estimate = OpenAICostEstimate(
        model_name=model_name,
        pricing_version=(
            "synthetic_pipeline_pricing_v1"
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
            Decimal("0.00240000")
        ),
        total_cost_usd=(
            Decimal("0.00422000")
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
        "test-openai-pipeline-"
        f"{TEST_SUITE_ID}-"
        f"{scenario}"
    )


def decode_jsonb(
    value: Any,
) -> Any:
    """Декодирует jsonb, полученный от asyncpg."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(
                "Поле jsonb содержит "
                "некорректный JSON."
            ) from error

    return value


async def load_run_by_model(
    pool: asyncpg.Pool,
    *,
    model_name: str,
) -> asyncpg.Record:
    """Загружает тестовый ranking_run."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                ranking_run_id,
                request_key,
                run_status,
                formula_version,
                model_name,
                prompt_version,
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
            "Ожидался ровно один ranking_run "
            f"для модели {model_name!r}, "
            f"получено: {len(records)}"
        )

    return records[0]


async def load_score_count(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> int:
    """Возвращает число сохранённых оценок."""

    async with pool.acquire() as connection:
        score_count = await connection.fetchval(
            """
            SELECT count(*)
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            """,
            ranking_run_id,
        )

    return int(score_count)


async def test_successful_pipeline(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Проверяет успешный полный конвейер."""

    model_name = build_model_name(
        scenario="success"
    )

    test_model_names.add(
        model_name
    )

    fake_client = (
        FakeStructuredRankingClient()
    )

    evaluator = OpenAIRankingEvaluator(
        client=fake_client,
        model_name=model_name,
    )

    first_result = (
        await run_reserved_openai_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    )

    candidate_count = len(
        first_result
        .candidate_selection
        .candidates
    )

    assert candidate_count > 0

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
    assert first_result.completed is True

    assert (
        first_result
        .duplicate_request_blocked
        is False
    )

    assert first_result.evaluation is not None
    assert first_result.completion is not None

    assert (
        first_result.run_status
        == "completed"
    )

    assert len(fake_client.requests) == 1

    assert (
        first_result
        .completion
        .candidate_count
        == candidate_count
    )

    assert (
        first_result
        .completion
        .scored_count
        == candidate_count
    )

    assert (
        first_result
        .completion
        .eligible_count
        == candidate_count
    )

    assert len(
        first_result.completion.scores
    ) == candidate_count

    assert all(
        score.scores_match
        for score
        in first_result.completion.scores
    )

    print("Successful protected pipeline: OK")
    print(
        "ranking_run_id="
        f"{first_result.ranking_run_id}"
    )
    print(
        "request_key="
        f"{first_result.request_key.value}"
    )
    print(
        f"candidate_count={candidate_count}"
    )
    print("model_called=true")
    print("run_status=completed")
    print(
        "python_postgres_scores_match=true"
    )
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )

    record = await load_run_by_model(
        pool,
        model_name=model_name,
    )

    openai_usage = decode_jsonb(
        record["openai_usage"]
    )

    openai_cost = decode_jsonb(
        record["openai_cost"]
    )

    assert isinstance(openai_usage, dict)
    assert isinstance(openai_cost, dict)

    assert (
        record["ranking_run_id"]
        == first_result.ranking_run_id
    )

    assert (
        record["request_key"]
        == first_result.request_key.value
    )

    assert record["run_status"] == "completed"

    assert (
        record["candidate_count"]
        == candidate_count
    )

    assert (
        record["scored_count"]
        == candidate_count
    )

    assert (
        record["eligible_count"]
        == candidate_count
    )

    assert record["error_message"] is None
    assert record["finished_at"] is not None

    assert (
        openai_usage["input_tokens"]
        == 1000
    )

    assert (
        openai_usage[
            "regular_input_tokens"
        ]
        == 900
    )

    assert (
        openai_usage[
            "cached_input_tokens"
        ]
        == 100
    )

    assert (
        openai_usage["output_tokens"]
        == 200
    )

    assert (
        openai_usage["total_tokens"]
        == 1200
    )

    assert (
        openai_cost["model_name"]
        == model_name
    )

    assert (
        openai_cost["total_cost_usd"]
        == "0.00422000"
    )

    score_count = await load_score_count(
        pool,
        ranking_run_id=(
            first_result.ranking_run_id
        ),
    )

    assert score_count == candidate_count

    print()
    print("Persisted pipeline data: OK")
    print(
        f"stored_score_count={score_count}"
    )
    print(
        "input_tokens="
        f"{openai_usage['input_tokens']}"
    )
    print(
        "output_tokens="
        f"{openai_usage['output_tokens']}"
    )
    print(
        "total_cost_usd="
        f"{openai_cost['total_cost_usd']}"
    )

    second_result = (
        await run_reserved_openai_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    )

    assert (
        second_result.ranking_run_id
        == first_result.ranking_run_id
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

    assert second_result.evaluation is None
    assert second_result.completion is None

    assert (
        second_result.run_status
        == "completed"
    )

    assert len(fake_client.requests) == 1

    repeated_score_count = (
        await load_score_count(
            pool,
            ranking_run_id=(
                first_result.ranking_run_id
            ),
        )
    )

    assert (
        repeated_score_count
        == candidate_count
    )

    print()
    print("Repeated pipeline request: OK")
    print("model_called=false")
    print(
        "duplicate_request_blocked=true"
    )
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )
    print(
        "duplicate_score_insertion=false"
    )


async def test_failed_pipeline(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Проверяет ошибку модели и статус failed."""

    model_name = build_model_name(
        scenario="failure"
    )

    test_model_names.add(
        model_name
    )

    fake_client = FakeStructuredRankingClient(
        fail_request=True
    )

    evaluator = OpenAIRankingEvaluator(
        client=fake_client,
        model_name=model_name,
    )

    try:
        await run_reserved_openai_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    except SyntheticModelError as error:
        assert (
            "Synthetic model failure"
            in str(error)
        )

        print()
        print("Failed protected pipeline: OK")
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
            "Тестовая ошибка модели "
            "не была передана вызывающему коду."
        )

    assert len(fake_client.requests) == 1

    record = await load_run_by_model(
        pool,
        model_name=model_name,
    )

    ranking_run_id = int(
        record["ranking_run_id"]
    )

    assert record["run_status"] == "failed"
    assert record["scored_count"] == 0
    assert record["eligible_count"] == 0
    assert record["finished_at"] is not None

    assert (
        "Synthetic model failure"
        in record["error_message"]
    )

    failure_payload = decode_jsonb(
        record["failure"]
    )

    assert isinstance(
        failure_payload,
        dict,
    )

    assert (
        failure_payload["error_type"]
        == "SyntheticModelError"
    )

    assert (
        "Synthetic model failure"
        in failure_payload["error_message"]
    )

    score_count = await load_score_count(
        pool,
        ranking_run_id=ranking_run_id,
    )

    assert score_count == 0

    print()
    print("Persisted pipeline failure: OK")
    print(
        f"ranking_run_id={ranking_run_id}"
    )
    print("run_status=failed")
    print("stored_score_count=0")
    print(
        "failure_error_type="
        f"{failure_payload['error_type']}"
    )

    repeated_result = (
        await run_reserved_openai_ranking(
            pool,
            evaluator=evaluator,
            as_of=AS_OF,
            window_hours=WINDOW_HOURS,
            limit=CANDIDATE_LIMIT,
            source_codes=SOURCE_CODES,
        )
    )

    assert (
        repeated_result.ranking_run_id
        == ranking_run_id
    )

    assert (
        repeated_result.reservation.created_new
        is False
    )

    assert repeated_result.model_called is False

    assert (
        repeated_result
        .duplicate_request_blocked
        is True
    )

    assert (
        repeated_result.run_status
        == "failed"
    )

    assert len(fake_client.requests) == 1

    print()
    print(
        "Repeated failed pipeline request: OK"
    )
    print("model_called=false")
    print(
        "duplicate_request_blocked=true"
    )
    print(
        "fake_model_call_count="
        f"{len(fake_client.requests)}"
    )


async def cleanup_test_runs(
    pool: asyncpg.Pool,
    *,
    test_model_names: set[str],
) -> None:
    """Удаляет все временные данные теста."""

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

    deleted_run_ids = sorted(
        int(record["ranking_run_id"])
        for record in deleted_records
    )

    async with pool.acquire() as connection:
        remaining_run_count = (
            await connection.fetchval(
                """
                SELECT count(*)
                FROM top3_news.ranking_runs
                WHERE model_name = ANY($1::text[])
                """,
                sorted(test_model_names),
            )
        )

        remaining_score_count = (
            await connection.fetchval(
                """
                SELECT count(*)
                FROM top3_news.news_scores AS ns
                JOIN top3_news.ranking_runs AS rr
                  ON rr.ranking_run_id
                     = ns.ranking_run_id
                WHERE rr.model_name
                    = ANY($1::text[])
                """,
                sorted(test_model_names),
            )
        )

    assert remaining_run_count == 0
    assert remaining_score_count == 0

    print()
    print("Test data cleanup: OK")
    print(
        "deleted_ranking_run_ids="
        + (
            ",".join(
                str(ranking_run_id)
                for ranking_run_id
                in deleted_run_ids
            )
            if deleted_run_ids
            else "none"
        )
    )
    print(
        "temporary_runs_and_scores_deleted=true"
    )


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
        "Database changes: temporary runs "
        "and scores inserted and deleted"
    )
    print("Telegram publication: not performed")
    print("Protected OpenAI pipeline test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )