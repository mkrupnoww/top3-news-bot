import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.event_ranking_run_completion import (
    COMPLETION_VERSION,
    complete_reserved_event_ranking_run,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.ranking_run_completion import (
    fail_reserved_ranking_run,
)
from app.db.ranking_run_reservation import (
    RankingRunReservation,
    reserve_ranking_run,
)
from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.event_evaluator import (
    EVENT_EVALUATOR_VERSION,
    EVENT_PROMPT_VERSION,
    EventAssessment,
    EventMemberAssessment,
    EventRankingCoverageDiagnostics,
)
from app.ranking.event_formula_pipeline import (
    EventAudienceMetrics,
    EventFormulaCalculationResult,
    calculate_event_formula,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)
from app.ranking.openai_usage import (
    OpenAITokenUsage,
    calculate_openai_cost,
    get_model_pricing,
)
from app.ranking.request_key import (
    REQUEST_KEY_VERSION,
    RankingRequestKey,
)


TEST_NEWS_IDS = (
    7,
    8,
    9,
    10,
    11,
)

DEGRADED_PROCESSED_NEWS_IDS = (
    7,
    8,
    9,
    10,
)

DEGRADED_MISSING_NEWS_IDS = (
    11,
)

WINDOW_END = datetime(
    2026,
    7,
    31,
    11,
    21,
    tzinfo=timezone.utc,
)

WINDOW_START = (
    WINDOW_END
    - timedelta(hours=24)
)


def build_metadata() -> RankingEvaluatorMetadata:
    """Создаёт метаданные event-level оценщика."""

    return RankingEvaluatorMetadata(
        run_mode="openai_event_ranking",
        evaluator_name=(
            "OpenAIEventRankingEvaluator"
        ),
        evaluator_version=(
            EVENT_EVALUATOR_VERSION
        ),
        prompt_version=(
            EVENT_PROMPT_VERSION
        ),
        model_name="gpt-5.6-terra",
    )


def build_request_key(
    *,
    test_name: str,
) -> RankingRequestKey:
    """Создаёт уникальный тестовый request_key."""

    payload = {
        "test": test_name,
        "formula_version": (
            FULL_FORMULA_VERSION
        ),
        "nonce": uuid4().hex,
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


def build_candidate(
    *,
    news_id: int,
    age_hours: int,
) -> NewsCandidate:
    """Создаёт тестового кандидата."""

    return NewsCandidate(
        news_id=news_id,
        source_id=8,
        source_code="variety_film",
        source_name="Variety Film",
        collection_priority=100,
        processing_status="collected",
        title=f"Integration movie news {news_id}",
        summary=f"Integration summary {news_id}",
        author_name="Integration Test",
        source_published_at=(
            WINDOW_END
            - timedelta(hours=age_hours)
        ),
        age_hours=float(age_hours),
        source_url=(
            f"https://example.com/news/{news_id}"
        ),
        primary_image_url=None,
    )


def build_selection() -> CandidateSelectionResult:
    """Создаёт суточную выборку из пяти статей."""

    return CandidateSelectionResult(
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        window_hours=24.0,
        candidates=(
            build_candidate(
                news_id=7,
                age_hours=1,
            ),
            build_candidate(
                news_id=8,
                age_hours=2,
            ),
            build_candidate(
                news_id=9,
                age_hours=4,
            ),
            build_candidate(
                news_id=10,
                age_hours=3,
            ),
            build_candidate(
                news_id=11,
                age_hours=2,
            ),
        ),
    )


def build_member(
    *,
    news_id: int,
    source_relation: str,
    is_representative: bool,
    source_weight: int,
) -> EventMemberAssessment:
    """Создаёт участника инфоповода."""

    counts_toward_reach = (
        source_relation
        in {
            "primary",
            "independent",
        }
        and source_weight > 0
    )

    return EventMemberAssessment(
        news_id=news_id,
        source_relation=source_relation,
        is_representative=(
            is_representative
        ),
        is_independent_source=(
            counts_toward_reach
        ),
        counts_toward_reach=(
            counts_toward_reach
        ),
        source_weight=source_weight,
        membership_reason=(
            f"Integration relation for {news_id}."
        ),
    )


def build_events() -> tuple[
    EventAssessment,
    ...,
]:
    """Создаёт четыре инфоповода из пяти статей."""

    return (
        EventAssessment(
            representative_news_id=7,
            event_title=(
                "International production "
                "announcement"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=1)
            ),
            macro_topic=(
                "creative_cast_production"
            ),
            i_score="8.0",
            k_score="6.0",
            n_score="7.0",
            e_score="6.0",
            x_score="8.0",
            q_score="0.95",
            impact_reason=(
                "Large international production."
            ),
            hook_reason=(
                "Strong scale and novelty."
            ),
            q_reason=(
                "Primary and independent sources."
            ),
            members=(
                build_member(
                    news_id=7,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=3,
                ),
                build_member(
                    news_id=8,
                    source_relation=(
                        "independent"
                    ),
                    is_representative=False,
                    source_weight=2,
                ),
            ),
        ),
        EventAssessment(
            representative_news_id=9,
            event_title=(
                "Film company business decision"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=4)
            ),
            macro_topic=(
                "business_economy_law"
            ),
            i_score="8.0",
            k_score="4.0",
            n_score="6.0",
            e_score="5.0",
            x_score="6.0",
            q_score="0.95",
            impact_reason=(
                "Meaningful business impact."
            ),
            hook_reason=(
                "Noticeable market development."
            ),
            q_reason=(
                "Reliable primary publication."
            ),
            members=(
                build_member(
                    news_id=9,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=3,
                ),
            ),
        ),
        EventAssessment(
            representative_news_id=10,
            event_title=(
                "Festival programme expansion"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=3)
            ),
            macro_topic=(
                "festivals_awards_criticism"
            ),
            i_score="8.5",
            k_score="6.0",
            n_score="8.0",
            e_score="7.0",
            x_score="8.0",
            q_score="0.95",
            impact_reason=(
                "Important festival expansion."
            ),
            hook_reason=(
                "Distinctive festival development."
            ),
            q_reason=(
                "Confirmed by a primary source."
            ),
            members=(
                build_member(
                    news_id=10,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=3,
                ),
            ),
        ),
        EventAssessment(
            representative_news_id=11,
            event_title=(
                "Unverified film rumour"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=2)
            ),
            macro_topic="other",
            i_score="9.0",
            k_score="9.0",
            n_score="9.0",
            e_score="9.0",
            x_score="9.0",
            q_score="0.0",
            impact_reason=(
                "Potentially large impact."
            ),
            hook_reason=(
                "Highly unusual claim."
            ),
            q_reason=(
                "The claim cannot be verified."
            ),
            members=(
                build_member(
                    news_id=11,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=2,
                ),
            ),
        ),
    )


def build_audience_metrics() -> tuple[
    EventAudienceMetrics,
    ...,
]:
    """Создаёт полные и частичные метрики."""

    return (
        EventAudienceMetrics(
            news_id=7,
            view_count=1000,
            comment_count=100,
            share_count=50,
        ),
        EventAudienceMetrics(
            news_id=9,
            view_count=500,
            comment_count=None,
            share_count=25,
        ),
    )


def build_calculation(
) -> EventFormulaCalculationResult:
    """Выполняет тестовый расчёт v2."""

    return calculate_event_formula(
        selection=build_selection(),
        events=build_events(),
        audience_metrics=(
            build_audience_metrics()
        ),
    )


def build_degraded_calculation(
) -> EventFormulaCalculationResult:
    """Выполняет расчёт без пропущенной публикации 11."""

    full_selection = build_selection()

    processed_selection = replace(
        full_selection,
        candidates=tuple(
            candidate
            for candidate in full_selection.candidates
            if candidate.news_id
            in DEGRADED_PROCESSED_NEWS_IDS
        ),
    )

    processed_events = tuple(
        event
        for event in build_events()
        if event.representative_news_id != 11
    )

    return calculate_event_formula(
        selection=processed_selection,
        events=processed_events,
        audience_metrics=(
            build_audience_metrics()
        ),
    )


def build_degraded_diagnostics(
) -> EventRankingCoverageDiagnostics:
    """Создаёт coverage после неудачного repair."""

    return EventRankingCoverageDiagnostics(
        expected_news_ids=TEST_NEWS_IDS,
        processed_news_ids=(
            DEGRADED_PROCESSED_NEWS_IDS
        ),
        initial_missing_news_ids=(
            DEGRADED_MISSING_NEWS_IDS
        ),
        missing_news_ids=(
            DEGRADED_MISSING_NEWS_IDS
        ),
        repair_attempted=True,
        repair_succeeded=False,
        repair_error_type="ValueError",
        repair_error_message=(
            "Synthetic repair still omitted news_id=11."
        ),
        model_call_count=2,
    )


def build_usage() -> OpenAITokenUsage:
    """Создаёт тестовую телеметрию."""

    return OpenAITokenUsage(
        input_tokens=1570,
        cached_input_tokens=0,
        cache_write_tokens=1567,
        output_tokens=521,
        reasoning_tokens=49,
        total_tokens=2091,
    )


def decode_jsonb(
    value: Any,
) -> Any:
    """Преобразует jsonb из asyncpg."""

    if isinstance(value, str):
        return json.loads(value)

    return value


async def reserve_test_run(
    pool: asyncpg.Pool,
    *,
    request_key: RankingRequestKey,
    created_run_ids: set[int],
) -> RankingRunReservation:
    """Создаёт event-level reservation."""

    reservation = await reserve_ranking_run(
        pool,
        request_key=request_key,
        formula_version=(
            FULL_FORMULA_VERSION
        ),
        metadata=build_metadata(),
        window_started_at=WINDOW_START,
        window_finished_at=WINDOW_END,
        news_ids=TEST_NEWS_IDS,
    )

    created_run_ids.add(
        reservation.ranking_run_id
    )

    assert reservation.created_new is True
    assert reservation.should_call_model is True
    assert reservation.run_status == "running"
    assert reservation.formula_version == (
        FULL_FORMULA_VERSION
    )
    assert reservation.candidate_count == 5

    return reservation


async def delete_test_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> None:
    """Удаляет временный ranking_run."""

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
            "Не удалось удалить тестовый "
            f"ranking_run: {result}"
        )


async def assert_test_run_deleted(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> None:
    """Проверяет каскадное удаление всех v2 данных."""

    async with pool.acquire() as connection:
        counts = await connection.fetchrow(
            """
            SELECT
                (
                    SELECT count(*)
                    FROM top3_news.ranking_runs
                    WHERE ranking_run_id = $1
                ) AS run_count,
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

    assert counts is not None

    assert all(
        int(counts[field_name]) == 0
        for field_name in (
            "run_count",
            "event_count",
            "member_count",
            "metric_count",
            "score_count",
            "combination_count",
            "combination_item_count",
        )
    )


async def test_successful_completion(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Проверяет полное сохранение event-level v2."""

    metadata = build_metadata()

    request_key = build_request_key(
        test_name=(
            "event_ranking_successful_completion"
        )
    )

    reservation = await reserve_test_run(
        pool,
        request_key=request_key,
        created_run_ids=created_run_ids,
    )

    calculation = build_calculation()

    usage = build_usage()

    cost_estimate = calculate_openai_cost(
        usage,
        get_model_pricing(
            metadata.model_name
            or "gpt-5.6-terra"
        ),
    )

    result = (
        await complete_reserved_event_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=metadata,
            candidate_news_ids=TEST_NEWS_IDS,
            calculation=calculation,
            usage=usage,
            cost_estimate=cost_estimate,
        )
    )

    assert result.run_status == "completed"
    assert result.already_completed is False
    assert result.degraded is False
    assert result.processed_candidate_count == 5
    assert result.missing_news_ids == ()
    assert result.formula_version == (
        FULL_FORMULA_VERSION
    )
    assert result.candidate_count == 5
    assert result.scored_count == 4
    assert result.eligible_count == 3
    assert result.combination_count == 1
    assert result.winner_combination_id > 0
    assert len(result.persisted_events) == 4

    assert tuple(
        item.representative_news_id
        for item in result.persisted_events
    ) == (
        7,
        9,
        10,
        11,
    )

    print("Event ranking completion: OK")
    print(
        "ranking_run_id="
        f"{result.ranking_run_id}"
    )
    print("run_status=completed")
    print("candidate_count=5")
    print("scored_event_count=4")
    print("eligible_count=3")
    print("combination_count=1")
    print("already_completed=false")

    async with pool.acquire() as connection:
        run_record = await connection.fetchrow(
            """
            SELECT
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
                parameters->>'completion_version'
                    AS completion_version,
                parameters->>'event_count'
                    AS event_count,
                parameters->>'combination_count'
                    AS combination_count,
                parameters->'winner_news_ids'
                    AS winner_news_ids,
                parameters->>'degraded'
                    AS degraded,
                parameters->>'processed_candidate_count'
                    AS processed_candidate_count,
                parameters->'missing_news_ids'
                    AS missing_news_ids,
                parameters->'coverage'
                    AS coverage
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            reservation.ranking_run_id,
        )

        counts = await connection.fetchrow(
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
            reservation.ranking_run_id,
        )

        score_records = await connection.fetch(
            """
            SELECT
                news_id,
                individual_score,
                is_eligible,
                exclusion_reason,
                rank_position,
                resonance_confidence,
                selected_for_top3,
                top3_position
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            ORDER BY news_id
            """,
            reservation.ranking_run_id,
        )

        winner_record = await connection.fetchrow(
            """
            SELECT
                combination_id,
                combination_rank,
                final_top_score,
                diversity_score,
                distinct_macro_topic_count,
                is_winner
            FROM top3_news.ranking_combinations
            WHERE ranking_run_id = $1
              AND is_winner = true
            """,
            reservation.ranking_run_id,
        )

        winner_items = await connection.fetch(
            """
            SELECT
                ci.position,
                ns.news_id
            FROM top3_news.ranking_combination_items AS ci
            JOIN top3_news.news_scores AS ns
              ON ns.score_id = ci.score_id
             AND ns.ranking_run_id = ci.ranking_run_id
            JOIN top3_news.ranking_combinations AS rc
              ON rc.combination_id = ci.combination_id
             AND rc.ranking_run_id = ci.ranking_run_id
            WHERE ci.ranking_run_id = $1
              AND rc.is_winner = true
            ORDER BY ci.position
            """,
            reservation.ranking_run_id,
        )

    if run_record is None:
        raise AssertionError(
            "Завершённый ranking_run не найден."
        )

    if counts is None:
        raise AssertionError(
            "Не удалось получить v2 counts."
        )

    if winner_record is None:
        raise AssertionError(
            "Победившая комбинация не найдена."
        )

    openai_usage = decode_jsonb(
        run_record["openai_usage"]
    )

    openai_cost = decode_jsonb(
        run_record["openai_cost"]
    )

    winner_news_ids = decode_jsonb(
        run_record["winner_news_ids"]
    )

    missing_news_ids = decode_jsonb(
        run_record["missing_news_ids"]
    )

    coverage = decode_jsonb(
        run_record["coverage"]
    )

    assert run_record["run_status"] == (
        "completed"
    )
    assert run_record["formula_version"] == (
        FULL_FORMULA_VERSION
    )
    assert run_record["model_name"] == (
        metadata.model_name
    )
    assert run_record["prompt_version"] == (
        EVENT_PROMPT_VERSION
    )
    assert run_record["candidate_count"] == 5
    assert run_record["scored_count"] == 4
    assert run_record["eligible_count"] == 3
    assert run_record["error_message"] is None
    assert run_record["finished_at"] is not None
    assert run_record["completion_version"] == (
        COMPLETION_VERSION
    )
    assert run_record["event_count"] == "4"
    assert run_record["combination_count"] == "1"
    assert run_record["degraded"] == "false"
    assert run_record[
        "processed_candidate_count"
    ] == "5"
    assert missing_news_ids == []
    assert coverage["degraded"] is False
    assert coverage[
        "processed_news_ids"
    ] == list(TEST_NEWS_IDS)
    assert coverage["missing_news_ids"] == []

    assert openai_usage["input_tokens"] == 1570
    assert openai_usage["output_tokens"] == 521
    assert openai_usage["total_tokens"] == 2091
    assert openai_cost["model_name"] == (
        "gpt-5.6-terra"
    )

    assert winner_news_ids == list(
        calculation
        .top3_selection
        .winner
        .ordered_news_ids
    )

    assert int(counts["event_count"]) == 4
    assert int(counts["member_count"]) == 5
    assert int(counts["metric_count"]) == 2
    assert int(counts["score_count"]) == 4
    assert int(counts["combination_count"]) == 1
    assert int(
        counts["combination_item_count"]
    ) == 3

    assert len(score_records) == 4

    score_by_news_id = {
        int(record["news_id"]): record
        for record in score_records
    }

    assert score_by_news_id[11][
        "is_eligible"
    ] is False

    assert score_by_news_id[11][
        "exclusion_reason"
    ] == "quality_zero"

    selected_records = [
        record
        for record in score_records
        if record["selected_for_top3"] is True
    ]

    assert len(selected_records) == 3

    assert sorted(
        int(record["top3_position"])
        for record in selected_records
    ) == [
        1,
        2,
        3,
    ]

    assert winner_record["combination_rank"] == 1
    assert winner_record["is_winner"] is True
    assert winner_record["diversity_score"] == (
        Decimal("10.000000")
    )
    assert winner_record[
        "distinct_macro_topic_count"
    ] == 3
    assert winner_record["final_top_score"] == (
        calculation
        .top3_selection
        .winner
        .final_top_score
    )

    assert tuple(
        int(record["news_id"])
        for record in winner_items
    ) == (
        calculation
        .top3_selection
        .winner
        .ordered_news_ids
    )

    print()
    print("Persisted event model: OK")
    print("ranking_events=4")
    print("ranking_event_members=5")
    print("ranking_audience_metrics=2")
    print("news_scores=4")
    print("ranking_combinations=1")
    print("ranking_combination_items=3")
    print(
        "winner_news_ids="
        + ",".join(
            str(record["news_id"])
            for record in winner_items
        )
    )
    print(
        "python_postgres_scores_match=true"
    )
    print(
        "python_postgres_top_score_match=true"
    )

    repeated = (
        await complete_reserved_event_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=metadata,
            candidate_news_ids=TEST_NEWS_IDS,
            calculation=calculation,
            usage=usage,
            cost_estimate=cost_estimate,
        )
    )

    assert repeated.already_completed is True
    assert repeated.degraded is False
    assert repeated.processed_candidate_count == 5
    assert repeated.missing_news_ids == ()
    assert repeated.ranking_run_id == (
        result.ranking_run_id
    )
    assert repeated.combination_count == 1
    assert len(repeated.persisted_events) == 4

    print()
    print("Repeated event completion: OK")
    print("already_completed=true")
    print("duplicate_insertion=blocked")

    try:
        await fail_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            error_message=(
                "Completed event run must not fail."
            ),
            error_type="TestError",
        )
    except ValueError as error:
        assert "completed ranking_run" in str(
            error
        )

        print()
        print(
            "Completed-to-failed blocking: OK"
        )
    else:
        raise AssertionError(
            "Completed event ranking_run "
            "был ошибочно переведён в failed."
        )


async def test_degraded_completion(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Сохраняет неполное coverage как completed."""

    metadata = build_metadata()
    request_key = build_request_key(
        test_name=(
            "event_ranking_degraded_completion"
        )
    )

    reservation = await reserve_test_run(
        pool,
        request_key=request_key,
        created_run_ids=created_run_ids,
    )

    calculation = build_degraded_calculation()
    diagnostics = build_degraded_diagnostics()
    usage = build_usage()
    cost_estimate = calculate_openai_cost(
        usage,
        get_model_pricing(
            metadata.model_name
            or "gpt-5.6-terra"
        ),
    )

    result = (
        await complete_reserved_event_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=metadata,
            candidate_news_ids=TEST_NEWS_IDS,
            calculation=calculation,
            usage=usage,
            cost_estimate=cost_estimate,
            coverage_diagnostics=diagnostics,
        )
    )

    assert result.run_status == "completed"
    assert result.already_completed is False
    assert result.degraded is True
    assert result.candidate_count == 5
    assert result.processed_candidate_count == 4
    assert result.missing_news_ids == (11,)
    assert result.scored_count == 3
    assert result.eligible_count == 3
    assert result.combination_count == 1
    assert len(result.persisted_events) == 3

    async with pool.acquire() as connection:
        run_record = await connection.fetchrow(
            """
            SELECT
                run_status,
                candidate_count,
                scored_count,
                eligible_count,
                error_message,
                parameters->>'degraded'
                    AS degraded,
                parameters->>'degraded_reason'
                    AS degraded_reason,
                parameters->>'original_candidate_count'
                    AS original_candidate_count,
                parameters->>'processed_candidate_count'
                    AS processed_candidate_count,
                parameters->>'missing_candidate_count'
                    AS missing_candidate_count,
                parameters->>'repair_attempted'
                    AS repair_attempted,
                parameters->>'repair_succeeded'
                    AS repair_succeeded,
                parameters->'processed_news_ids'
                    AS processed_news_ids,
                parameters->'missing_news_ids'
                    AS missing_news_ids,
                parameters->'coverage'
                    AS coverage
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            reservation.ranking_run_id,
        )

        persisted_news_ids = await connection.fetch(
            """
            SELECT news_id
            FROM top3_news.ranking_event_members
            WHERE ranking_run_id = $1
            ORDER BY news_id
            """,
            reservation.ranking_run_id,
        )

        scored_news_ids = await connection.fetch(
            """
            SELECT news_id
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            ORDER BY news_id
            """,
            reservation.ranking_run_id,
        )

    if run_record is None:
        raise AssertionError(
            "Degraded ranking_run не найден."
        )

    processed_news_ids = decode_jsonb(
        run_record["processed_news_ids"]
    )
    missing_news_ids = decode_jsonb(
        run_record["missing_news_ids"]
    )
    coverage = decode_jsonb(
        run_record["coverage"]
    )

    assert run_record["run_status"] == "completed"
    assert run_record["candidate_count"] == 5
    assert run_record["scored_count"] == 3
    assert run_record["eligible_count"] == 3
    assert run_record["error_message"] is None
    assert run_record["degraded"] == "true"
    assert run_record["degraded_reason"] == (
        "incomplete_model_coverage_after_repair"
    )
    assert run_record[
        "original_candidate_count"
    ] == "5"
    assert run_record[
        "processed_candidate_count"
    ] == "4"
    assert run_record[
        "missing_candidate_count"
    ] == "1"
    assert run_record["repair_attempted"] == "true"
    assert run_record["repair_succeeded"] == "false"
    assert processed_news_ids == [7, 8, 9, 10]
    assert missing_news_ids == [11]
    assert coverage["model_call_count"] == 2
    assert coverage["repair_error_type"] == (
        "ValueError"
    )

    assert tuple(
        int(record["news_id"])
        for record in persisted_news_ids
    ) == (7, 8, 9, 10)
    assert tuple(
        int(record["news_id"])
        for record in scored_news_ids
    ) == (7, 9, 10)

    repeated = (
        await complete_reserved_event_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=metadata,
            candidate_news_ids=TEST_NEWS_IDS,
            calculation=calculation,
            usage=usage,
            cost_estimate=cost_estimate,
            coverage_diagnostics=diagnostics,
        )
    )

    assert repeated.already_completed is True
    assert repeated.degraded is True
    assert repeated.processed_candidate_count == 4
    assert repeated.missing_news_ids == (11,)

    print()
    print("Degraded event completion: OK")
    print("run_status=completed")
    print("degraded=true")
    print("candidate_count=5")
    print("processed_candidate_count=4")
    print("missing_news_ids=11")
    print("persisted_event_count=3")
    print("generation_status_compatible=true")


async def test_failed_run_blocking(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Проверяет запрет completion после failed."""

    request_key = build_request_key(
        test_name="event_ranking_failed_run"
    )

    reservation = await reserve_test_run(
        pool,
        request_key=request_key,
        created_run_ids=created_run_ids,
    )

    failure = await fail_reserved_ranking_run(
        pool,
        ranking_run_id=(
            reservation.ranking_run_id
        ),
        request_key=request_key.value,
        error_message=(
            "Synthetic event ranking failure."
        ),
        error_type="SyntheticEventError",
    )

    assert failure.run_status == "failed"

    usage = build_usage()

    cost_estimate = calculate_openai_cost(
        usage,
        get_model_pricing(
            "gpt-5.6-terra"
        ),
    )

    try:
        await complete_reserved_event_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=build_metadata(),
            candidate_news_ids=TEST_NEWS_IDS,
            calculation=build_calculation(),
            usage=usage,
            cost_estimate=cost_estimate,
        )
    except ValueError as error:
        assert "статусом failed" in str(error)

        print()
        print("Failed-to-completed blocking: OK")
        return

    raise AssertionError(
        "Failed event ranking_run "
        "был ошибочно завершён."
    )


async def test_transaction_rollback(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Проверяет откат всей транзакции при ошибке."""

    request_key = build_request_key(
        test_name=(
            "event_ranking_transaction_rollback"
        )
    )

    reservation = await reserve_test_run(
        pool,
        request_key=request_key,
        created_run_ids=created_run_ids,
    )

    calculation = build_calculation()

    first_item = calculation.calculated_events[0]

    tampered_score = replace(
        first_item.score,
        m_score=Decimal("0.000000"),
    )

    tampered_item = replace(
        first_item,
        score=tampered_score,
    )

    tampered_calculation = replace(
        calculation,
        calculated_events=(
            tampered_item,
            *calculation.calculated_events[1:],
        ),
    )

    usage = build_usage()

    cost_estimate = calculate_openai_cost(
        usage,
        get_model_pricing(
            "gpt-5.6-terra"
        ),
    )

    try:
        await complete_reserved_event_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=build_metadata(),
            candidate_news_ids=TEST_NEWS_IDS,
            calculation=tampered_calculation,
            usage=usage,
            cost_estimate=cost_estimate,
        )
    except RuntimeError as error:
        assert "PostgreSQL" in str(error)
    else:
        raise AssertionError(
            "Несогласованный Python-результат "
            "не был заблокирован."
        )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                run_status,
                scored_count,
                eligible_count
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            reservation.ranking_run_id,
        )

        counts = await connection.fetchrow(
            """
            SELECT
                (
                    SELECT count(*)
                    FROM top3_news.ranking_events
                    WHERE ranking_run_id = $1
                ) AS event_count,
                (
                    SELECT count(*)
                    FROM top3_news.news_scores
                    WHERE ranking_run_id = $1
                ) AS score_count,
                (
                    SELECT count(*)
                    FROM top3_news.ranking_combinations
                    WHERE ranking_run_id = $1
                ) AS combination_count
            """,
            reservation.ranking_run_id,
        )

    assert record is not None
    assert counts is not None

    assert record["run_status"] == "running"
    assert record["scored_count"] == 0
    assert record["eligible_count"] == 0

    assert int(counts["event_count"]) == 0
    assert int(counts["score_count"]) == 0
    assert int(counts["combination_count"]) == 0

    print()
    print("Transaction rollback: OK")
    print("run_status=running")
    print("persisted_event_count=0")
    print("persisted_score_count=0")
    print("persisted_combination_count=0")


async def cleanup_test_runs(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Удаляет временные тестовые запуски."""

    for ranking_run_id in sorted(
        created_run_ids
    ):
        await delete_test_run(
            pool,
            ranking_run_id=ranking_run_id,
        )

        await assert_test_run_deleted(
            pool,
            ranking_run_id=ranking_run_id,
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


async def main() -> int:
    """Запускает PostgreSQL-интеграционный тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    created_run_ids: set[int] = set()

    try:
        await test_successful_completion(
            pool,
            created_run_ids=created_run_ids,
        )

        await test_degraded_completion(
            pool,
            created_run_ids=created_run_ids,
        )

        await test_failed_run_blocking(
            pool,
            created_run_ids=created_run_ids,
        )

        await test_transaction_rollback(
            pool,
            created_run_ids=created_run_ids,
        )
    finally:
        try:
            await cleanup_test_runs(
                pool,
                created_run_ids=created_run_ids,
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
    print(
        "Event ranking run completion test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )