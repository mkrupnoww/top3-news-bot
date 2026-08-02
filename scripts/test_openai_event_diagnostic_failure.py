import asyncio
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
from app.ranking.event_evaluator import (
    EventRankingModelRequest,
    EventRankingModelResponse,
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

MODEL_NAME = (
    "test-openai-event-diagnostic-"
    f"{uuid4().hex}"
)

MACRO_TOPICS = (
    "creative_cast_production",
    "business_economy_law",
    "festivals_awards_criticism",
    "trailers_premieres_releases",
    "people_conflicts_legal",
)


class FakeInsufficientTop3Client:
    """Возвращает только один допустимый инфоповод."""

    def __init__(self) -> None:
        self.requests: list[
            EventRankingModelRequest
        ] = []

    async def create_response(
        self,
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Формирует детерминированный ответ без сети."""

        self.requests.append(request)

        payload = json.loads(
            request.input_text
        )

        candidates = payload.get(
            "candidates"
        )

        if not isinstance(candidates, list):
            raise AssertionError(
                "Запрос не содержит candidates."
            )

        events: list[
            dict[str, object]
        ] = []

        for index, candidate in enumerate(
            candidates
        ):
            if not isinstance(candidate, dict):
                raise AssertionError(
                    "Кандидат должен быть объектом."
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

            q_score = 1 if index == 0 else 0

            events.append(
                {
                    "representative_news_id": news_id,
                    "event_title": (
                        "Synthetic diagnostic event "
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
                    "q_score": q_score,
                    "impact_reason": (
                        "Synthetic diagnostic "
                        "industry impact."
                    ),
                    "hook_reason": (
                        "Synthetic diagnostic "
                        "editorial hook."
                    ),
                    "q_reason": (
                        "Synthetic verified event."
                        if q_score == 1
                        else (
                            "Synthetic event excluded "
                            "by quality zero."
                        )
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
                            "source_weight": 3,
                            "membership_reason": (
                                "Synthetic primary "
                                "representative."
                            ),
                        }
                    ],
                }
            )

        usage = OpenAITokenUsage(
            input_tokens=1000,
            cached_input_tokens=100,
            cache_write_tokens=0,
            output_tokens=300,
            reasoning_tokens=75,
            total_tokens=1300,
        )

        cost_estimate = OpenAICostEstimate(
            model_name=request.model,
            pricing_version=(
                "synthetic_diagnostic_pricing_v1"
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

        return EventRankingModelResponse(
            output_text=json.dumps(
                {
                    "events": events,
                },
                ensure_ascii=False,
            ),
            usage=usage,
            cost_estimate=cost_estimate,
        )


def decode_jsonb(
    value: Any,
) -> Any:
    """Декодирует jsonb из asyncpg."""

    if isinstance(value, str):
        return json.loads(value)

    return value


async def load_run(
    pool: asyncpg.Pool,
) -> asyncpg.Record:
    """Загружает тестовый ranking_run."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                ranking_run_id,
                request_key,
                run_status,
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
                    AS failure,
                parameters->>'failure_version'
                    AS failure_version,
                parameters->>'diagnostic_scores_persisted'
                    AS diagnostic_scores_persisted,
                parameters->>'event_count'
                    AS event_count,
                parameters->>'combination_count'
                    AS combination_count
            FROM top3_news.ranking_runs
            WHERE model_name = $1
            """,
            MODEL_NAME,
        )

    if len(records) != 1:
        raise AssertionError(
            "Ожидался один тестовый ranking_run, "
            f"получено: {len(records)}"
        )

    return records[0]


async def load_counts(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> asyncpg.Record:
    """Загружает число сохранённых v2-сущностей."""

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
                    FROM top3_news.ranking_audience_metrics
                    WHERE ranking_run_id = $1
                ) AS metric_count,
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
            "Не удалось загрузить v2 counts."
        )

    return record


async def load_scores(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> tuple[asyncpg.Record, ...]:
    """Загружает диагностические баллы."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                news_id,
                individual_score,
                is_eligible,
                exclusion_reason,
                selected_for_top3,
                top3_position,
                resonance_confidence
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            ORDER BY rank_position, news_id
            """,
            ranking_run_id,
        )

    return tuple(records)


async def delete_test_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> None:
    """Удаляет тестовый запуск с каскадными данными."""

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
            "Не удалось удалить тестовый запуск: "
            f"{result}"
        )


async def main() -> int:
    """Проверяет диагностическое сохранение сбоя."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    ranking_run_id: int | None = None

    try:
        fake_client = (
            FakeInsufficientTop3Client()
        )

        evaluator = OpenAIEventRankingEvaluator(
            client=fake_client,
            model_name=MODEL_NAME,
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
        except ValueError as error:
            assert "eligible_count=1" in str(
                error
            )

            print(
                "Insufficient TOP-3 error "
                "propagation: OK"
            )
            print(
                "raised_error="
                f"{error}"
            )
        else:
            raise AssertionError(
                "Недостаточный TOP-3 "
                "не был заблокирован."
            )

        assert len(fake_client.requests) == 1

        run_record = await load_run(
            pool
        )

        ranking_run_id = int(
            run_record["ranking_run_id"]
        )

        usage = decode_jsonb(
            run_record["openai_usage"]
        )

        cost = decode_jsonb(
            run_record["openai_cost"]
        )

        failure = decode_jsonb(
            run_record["failure"]
        )

        candidate_count = int(
            run_record["candidate_count"]
        )

        assert run_record["run_status"] == "failed"
        assert run_record["scored_count"] == (
            candidate_count
        )
        assert run_record["eligible_count"] == 1
        assert run_record["finished_at"] is not None
        assert "eligible_count=1" in (
            run_record["error_message"]
        )

        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 300
        assert usage["total_tokens"] == 1300

        assert cost["model_name"] == MODEL_NAME
        assert cost["total_cost_usd"] == (
            "0.00542000"
        )

        assert failure["error_type"] == (
            "ValueError"
        )
        assert failure["stage"] == (
            "top3_selection"
        )
        assert "eligible_count=1" in (
            failure["error_message"]
        )

        assert run_record["failure_version"] == (
            "reserved_event_ranking_"
            "diagnostic_failure_v1"
        )
        assert (
            run_record[
                "diagnostic_scores_persisted"
            ]
            == "true"
        )
        assert int(
            run_record["event_count"]
        ) == candidate_count
        assert int(
            run_record["combination_count"]
        ) == 0

        counts = await load_counts(
            pool,
            ranking_run_id=ranking_run_id,
        )

        assert int(counts["event_count"]) == (
            candidate_count
        )
        assert int(counts["member_count"]) == (
            candidate_count
        )
        assert int(counts["metric_count"]) == 0
        assert int(counts["score_count"]) == (
            candidate_count
        )
        assert int(
            counts["combination_count"]
        ) == 0
        assert int(
            counts["combination_item_count"]
        ) == 0

        scores = await load_scores(
            pool,
            ranking_run_id=ranking_run_id,
        )

        assert len(scores) == candidate_count

        eligible_scores = tuple(
            record
            for record in scores
            if record["is_eligible"] is True
        )

        assert len(eligible_scores) == 1

        assert all(
            record["selected_for_top3"] is False
            and record["top3_position"] is None
            and record[
                "resonance_confidence"
            ] == "unavailable"
            for record in scores
        )

        print()
        print(
            "Diagnostic failure persistence: OK"
        )
        print(
            "ranking_run_id="
            f"{ranking_run_id}"
        )
        print("run_status=failed")
        print(
            "candidate_count="
            f"{candidate_count}"
        )
        print(
            "scored_count="
            f"{run_record['scored_count']}"
        )
        print("eligible_count=1")
        print(
            "ranking_events="
            f"{counts['event_count']}"
        )
        print(
            "news_scores="
            f"{counts['score_count']}"
        )
        print("ranking_combinations=0")
        print(
            "failure_stage="
            f"{failure['stage']}"
        )
        print(
            "total_cost_usd="
            f"{cost['total_cost_usd']}"
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
        print(
            "Repeated failed request blocking: OK"
        )
        print("model_called=false")
        print("fake_model_call_count=1")

    finally:
        try:
            if ranking_run_id is not None:
                await delete_test_run(
                    pool,
                    ranking_run_id=(
                        ranking_run_id
                    ),
                )

                print()
                print("Test data cleanup: OK")
                print(
                    "temporary_ranking_run_id="
                    f"{ranking_run_id}"
                )
                print(
                    "temporary_v2_data_deleted=true"
                )
        finally:
            await close_database_pool(pool)

    print()
    print("OpenAI requests: not performed")
    print(
        "Database changes: temporary diagnostic "
        "data inserted and deleted"
    )
    print("Telegram publication: not performed")
    print(
        "OpenAI event diagnostic failure test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )