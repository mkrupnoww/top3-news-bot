import asyncio
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
    fail_reserved_ranking_run,
)
from app.db.ranking_run_reservation import (
    RankingRunReservation,
    reserve_ranking_run,
)
from app.db.ranking_scores import (
    ManualNewsAssessment,
)
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_evaluator import (
    OPENAI_EVALUATOR_VERSION,
    OPENAI_PROMPT_VERSION,
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
from app.ranking.score_formula import (
    FORMULA_VERSION,
)


TEST_NEWS_IDS = (
    7,
    8,
    9,
)


def build_metadata() -> RankingEvaluatorMetadata:
    """Создаёт рабочие метаданные оценщика."""

    return RankingEvaluatorMetadata(
        run_mode="openai_ranking",
        evaluator_name=(
            "OpenAIRankingEvaluator"
        ),
        evaluator_version=(
            OPENAI_EVALUATOR_VERSION
        ),
        prompt_version=(
            OPENAI_PROMPT_VERSION
        ),
        model_name="gpt-5.6-terra",
    )


def build_request_key(
    *,
    test_name: str,
) -> RankingRequestKey:
    """Создаёт уникальный ключ тестового запуска."""

    payload = {
        "test": test_name,
        "nonce": uuid4().hex,
    }

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    request_key = sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    return RankingRequestKey(
        value=request_key,
        version=REQUEST_KEY_VERSION,
        canonical_json=canonical_json,
    )


def build_assessments() -> tuple[
    ManualNewsAssessment,
    ...,
]:
    """Создаёт три тестовые оценки."""

    return (
        ManualNewsAssessment(
            news_id=7,
            f_score=Decimal("9.000000"),
            m_score=Decimal("4.000000"),
            r_score=Decimal("4.000000"),
            h_score=Decimal("6.000000"),
            q_score=Decimal("0.900000"),
            explanation=(
                "Свежая нишевая новость "
                "с заметным культурным крючком."
            ),
        ),
        ManualNewsAssessment(
            news_id=8,
            f_score=Decimal("8.500000"),
            m_score=Decimal("8.000000"),
            r_score=Decimal("7.000000"),
            h_score=Decimal("5.000000"),
            q_score=Decimal("0.950000"),
            explanation=(
                "Крупная индустриальная новость "
                "с высоким потенциальным охватом."
            ),
        ),
        ManualNewsAssessment(
            news_id=9,
            f_score=Decimal("9.500000"),
            m_score=Decimal("6.500000"),
            r_score=Decimal("6.500000"),
            h_score=Decimal("7.500000"),
            q_score=Decimal("0.900000"),
            explanation=(
                "Очень свежая новость с сильной "
                "темой искусственного интеллекта."
            ),
        ),
    )


def build_usage() -> OpenAITokenUsage:
    """Создаёт тестовую телеметрию токенов."""

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
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(
                "Поле jsonb содержит "
                "некорректный JSON."
            ) from error

    return value


async def reserve_test_run(
    pool: asyncpg.Pool,
    *,
    request_key: RankingRequestKey,
    created_run_ids: set[int],
) -> RankingRunReservation:
    """Создаёт тестовое резервирование."""

    metadata = build_metadata()

    from datetime import datetime, timezone

    reservation = await reserve_ranking_run(
        pool,
        request_key=request_key,
        formula_version=FORMULA_VERSION,
        metadata=metadata,
        window_started_at=datetime(
            2026,
            7,
            30,
            11,
            21,
            tzinfo=timezone.utc,
        ),
        window_finished_at=datetime(
            2026,
            7,
            31,
            11,
            21,
            tzinfo=timezone.utc,
        ),
        news_ids=TEST_NEWS_IDS,
    )

    created_run_ids.add(
        reservation.ranking_run_id
    )

    assert reservation.created_new is True
    assert reservation.should_call_model is True
    assert reservation.run_status == "running"

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
    """Проверяет каскадное удаление данных."""

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
            SELECT count(*)
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            """,
            ranking_run_id,
        )

    assert run_exists is False
    assert score_count == 0


async def test_successful_completion(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Проверяет успешное завершение запуска."""

    metadata = build_metadata()

    request_key = build_request_key(
        test_name=(
            "ranking_run_successful_completion"
        )
    )

    reservation = await reserve_test_run(
        pool,
        request_key=request_key,
        created_run_ids=created_run_ids,
    )

    usage = build_usage()

    cost_estimate = calculate_openai_cost(
        usage,
        get_model_pricing(
            "gpt-5.6-terra"
        ),
    )

    assessments = build_assessments()

    result = await complete_reserved_ranking_run(
        pool,
        ranking_run_id=(
            reservation.ranking_run_id
        ),
        request_key=request_key.value,
        metadata=metadata,
        assessments=assessments,
        usage=usage,
        cost_estimate=cost_estimate,
    )

    assert result.run_status == "completed"
    assert result.already_completed is False
    assert result.formula_version == FORMULA_VERSION
    assert result.candidate_count == 3
    assert result.scored_count == 3
    assert result.eligible_count == 3
    assert len(result.scores) == 3

    assert all(
        score.scores_match
        for score in result.scores
    )

    assert [
        score.news_id
        for score in result.scores
    ] == [
        8,
        9,
        7,
    ]

    assert [
        score.rank_position
        for score in result.scores
    ] == [
        1,
        2,
        3,
    ]

    print("Reserved run completion: OK")
    print(
        "ranking_run_id="
        f"{result.ranking_run_id}"
    )
    print("run_status=completed")
    print("already_completed=false")
    print(
        "score_news_ids="
        + ",".join(
            str(score.news_id)
            for score in result.scores
        )
    )
    print(
        "rank_positions="
        + ",".join(
            str(score.rank_position)
            for score in result.scores
        )
    )
    print("python_postgres_scores_match=true")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
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
                parameters->>'database_formula_check'
                    AS database_formula_check,
                parameters->>'completion_version'
                    AS completion_version
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            reservation.ranking_run_id,
        )

        score_records = await connection.fetch(
            """
            SELECT
                news_id,
                individual_score,
                rank_position,
                score_details->>'request_key'
                    AS score_request_key,
                score_details->>'formula_version'
                    AS score_formula_version,
                score_details->>'model_name'
                    AS score_model_name
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            ORDER BY rank_position
            """,
            reservation.ranking_run_id,
        )

    if record is None:
        raise AssertionError(
            "Завершённый ranking_run не найден."
        )

    openai_usage = decode_jsonb(
        record["openai_usage"]
    )

    openai_cost = decode_jsonb(
        record["openai_cost"]
    )

    assert isinstance(openai_usage, dict)
    assert isinstance(openai_cost, dict)

    assert record["run_status"] == "completed"
    assert (
        record["formula_version"]
        == FORMULA_VERSION
    )
    assert (
        record["model_name"]
        == metadata.model_name
    )
    assert (
        record["prompt_version"]
        == OPENAI_PROMPT_VERSION
    )
    assert record["candidate_count"] == 3
    assert record["scored_count"] == 3
    assert record["eligible_count"] == 3
    assert record["error_message"] is None
    assert record["finished_at"] is not None

    assert (
        openai_usage["input_tokens"]
        == 1570
    )
    assert (
        openai_usage[
            "regular_input_tokens"
        ]
        == 3
    )
    assert (
        openai_usage[
            "cache_write_tokens"
        ]
        == 1567
    )
    assert (
        openai_usage["output_tokens"]
        == 521
    )
    assert (
        openai_usage["total_tokens"]
        == 2091
    )

    assert (
        openai_cost["model_name"]
        == "gpt-5.6-terra"
    )
    assert (
        openai_cost["total_cost_usd"]
        == "0.01017550"
    )

    assert (
        record["database_formula_check"]
        == "true"
    )
    assert (
        record["completion_version"]
        == "reserved_ranking_completion_v1"
    )

    assert len(score_records) == 3

    for score_record in score_records:
        assert (
            score_record["score_request_key"]
            == request_key.value
        )
        assert (
            score_record[
                "score_formula_version"
            ]
            == FORMULA_VERSION
        )
        assert (
            score_record["score_model_name"]
            == metadata.model_name
        )

    print()
    print("Persisted completion data: OK")
    print(
        "input_tokens="
        f"{openai_usage['input_tokens']}"
    )
    print(
        "output_tokens="
        f"{openai_usage['output_tokens']}"
    )
    print(
        "total_tokens="
        f"{openai_usage['total_tokens']}"
    )
    print(
        "total_cost_usd="
        f"{openai_cost['total_cost_usd']}"
    )
    print(
        "stored_score_count="
        f"{len(score_records)}"
    )

    repeated_result = (
        await complete_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=metadata,
            assessments=assessments,
            usage=usage,
            cost_estimate=cost_estimate,
        )
    )

    assert (
        repeated_result.already_completed
        is True
    )
    assert (
        repeated_result.ranking_run_id
        == result.ranking_run_id
    )
    assert len(repeated_result.scores) == 3

    print()
    print("Repeated completion: OK")
    print("already_completed=true")
    print(
        "Duplicate score insertion: blocked"
    )

    try:
        await fail_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            error_message=(
                "Completed run must not fail."
            ),
            error_type="TestError",
        )
    except ValueError as error:
        assert (
            "completed ranking_run"
            in str(error)
        )

        print()
        print(
            "Completed-to-failed blocking: OK"
        )
    else:
        raise AssertionError(
            "Completed ranking_run был "
            "ошибочно переведён в failed."
        )


async def test_failed_completion(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Проверяет фиксацию ошибки запуска."""

    request_key = build_request_key(
        test_name=(
            "ranking_run_failed_completion"
        )
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
            "Synthetic OpenAI failure "
            "for integration test."
        ),
        error_type="SyntheticOpenAIError",
    )

    assert failure.run_status == "failed"
    assert failure.already_failed is False
    assert (
        failure.error_message
        == (
            "Synthetic OpenAI failure "
            "for integration test."
        )
    )

    print()
    print("Reserved run failure: OK")
    print(
        "ranking_run_id="
        f"{failure.ranking_run_id}"
    )
    print("run_status=failed")
    print("already_failed=false")

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                run_status,
                scored_count,
                eligible_count,
                error_message,
                finished_at,
                parameters->'failure'
                    AS failure,
                parameters->>'failure_version'
                    AS failure_version
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            reservation.ranking_run_id,
        )

        score_count = await connection.fetchval(
            """
            SELECT count(*)
            FROM top3_news.news_scores
            WHERE ranking_run_id = $1
            """,
            reservation.ranking_run_id,
        )

    if record is None:
        raise AssertionError(
            "Failed ranking_run не найден."
        )

    failure_payload = decode_jsonb(
        record["failure"]
    )

    assert isinstance(failure_payload, dict)

    assert record["run_status"] == "failed"
    assert record["scored_count"] == 0
    assert record["eligible_count"] == 0
    assert record["finished_at"] is not None
    assert score_count == 0

    assert (
        record["error_message"]
        == (
            "Synthetic OpenAI failure "
            "for integration test."
        )
    )

    assert (
        failure_payload["error_type"]
        == "SyntheticOpenAIError"
    )

    assert (
        failure_payload["error_message"]
        == (
            "Synthetic OpenAI failure "
            "for integration test."
        )
    )

    assert (
        record["failure_version"]
        == "reserved_ranking_failure_v1"
    )

    print()
    print("Persisted failure data: OK")
    print("stored_score_count=0")
    print(
        "error_type="
        f"{failure_payload['error_type']}"
    )

    repeated_failure = (
        await fail_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            error_message=(
                "A second error must not "
                "overwrite the first one."
            ),
            error_type="SecondError",
        )
    )

    assert (
        repeated_failure.already_failed
        is True
    )

    assert (
        repeated_failure.error_message
        == (
            "Synthetic OpenAI failure "
            "for integration test."
        )
    )

    print()
    print("Repeated failure: OK")
    print("already_failed=true")
    print(
        "Original error preserved=true"
    )

    usage = build_usage()

    cost_estimate = calculate_openai_cost(
        usage,
        get_model_pricing(
            "gpt-5.6-terra"
        ),
    )

    try:
        await complete_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=request_key.value,
            metadata=build_metadata(),
            assessments=build_assessments(),
            usage=usage,
            cost_estimate=cost_estimate,
        )
    except ValueError as error:
        assert (
            "статусом failed"
            in str(error)
        )

        print()
        print(
            "Failed-to-completed blocking: OK"
        )
    else:
        raise AssertionError(
            "Failed ranking_run был "
            "ошибочно завершён."
        )


async def cleanup_test_runs(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Удаляет все временные тестовые запуски."""

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
            "temporary_run_and_scores_deleted=true"
        )


async def main() -> int:
    """Запускает интеграционный тест."""

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

        await test_failed_completion(
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
        "Database changes: temporary runs "
        "and scores inserted and deleted"
    )
    print("Telegram publication: not performed")
    print(
        "Ranking run completion test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )