from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
from typing import Any, Mapping

import asyncpg

from app.ranking.score_formula import (
    FORMULA_VERSION,
    CalculatedScore,
    ScoreInput,
    calculate_individual_score,
    create_score_components,
)


@dataclass(frozen=True, slots=True)
class ManualNewsAssessment:
    """Ручные тестовые оценки одной новости."""

    news_id: int
    f_score: ScoreInput
    m_score: ScoreInput
    r_score: ScoreInput
    h_score: ScoreInput
    q_score: ScoreInput
    explanation: str


@dataclass(frozen=True, slots=True)
class PreparedNewsAssessment:
    """Проверенная оценка с рассчитанным баллом."""

    news_id: int
    explanation: str
    calculated_score: CalculatedScore
    rank_position: int


@dataclass(frozen=True, slots=True)
class PersistedNewsScore:
    """Сопоставление расчёта Python и PostgreSQL."""

    score_id: int
    news_id: int
    rank_position: int
    python_individual_score: Decimal
    postgres_individual_score: Decimal
    scores_match: bool


@dataclass(frozen=True, slots=True)
class RankingPersistenceResult:
    """Результат сохранения тестового ranking run."""

    ranking_run_id: int
    run_status: str
    formula_version: str
    candidate_count: int
    scored_count: int
    eligible_count: int
    already_persisted: bool
    scores: tuple[PersistedNewsScore, ...]


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Приводит дату с часовым поясом к UTC."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} должен содержать часовой пояс."
        )

    return value.astimezone(timezone.utc)


def _prepare_assessments(
    assessments: tuple[ManualNewsAssessment, ...],
) -> tuple[PreparedNewsAssessment, ...]:
    """Проверяет оценки и назначает позиции рейтинга."""

    if not assessments:
        raise ValueError(
            "Список оценок не может быть пустым."
        )

    news_ids = [
        assessment.news_id
        for assessment in assessments
    ]

    if any(news_id <= 0 for news_id in news_ids):
        raise ValueError(
            "Все news_id должны быть больше нуля."
        )

    if len(set(news_ids)) != len(news_ids):
        raise ValueError(
            "Каждый news_id должен встречаться один раз."
        )

    calculated_items: list[
        tuple[ManualNewsAssessment, CalculatedScore]
    ] = []

    for assessment in assessments:
        explanation = assessment.explanation.strip()

        if not explanation:
            raise ValueError(
                "explanation не может быть пустым: "
                f"news_id={assessment.news_id}"
            )

        components = create_score_components(
            f_score=assessment.f_score,
            m_score=assessment.m_score,
            r_score=assessment.r_score,
            h_score=assessment.h_score,
            q_score=assessment.q_score,
        )

        calculated_score = calculate_individual_score(
            components
        )

        calculated_items.append(
            (
                assessment,
                calculated_score,
            )
        )

    calculated_items.sort(
        key=lambda item: (
            -item[1].individual_score,
            item[0].news_id,
        )
    )

    return tuple(
        PreparedNewsAssessment(
            news_id=assessment.news_id,
            explanation=assessment.explanation.strip(),
            calculated_score=calculated_score,
            rank_position=rank_position,
        )
        for rank_position, (
            assessment,
            calculated_score,
        ) in enumerate(
            calculated_items,
            start=1,
        )
    )


async def _validate_news_items(
    connection: asyncpg.Connection,
    *,
    news_ids: tuple[int, ...],
) -> None:
    """Проверяет наличие новостей и допустимый статус."""

    records = await connection.fetch(
        """
        SELECT
            news_id,
            processing_status
        FROM news_items
        WHERE news_id = ANY($1::bigint[])
        ORDER BY news_id
        """,
        list(news_ids),
    )

    found_ids = {
        record["news_id"]
        for record in records
    }

    missing_ids = sorted(
        set(news_ids) - found_ids
    )

    if missing_ids:
        raise LookupError(
            "Не найдены новости: "
            + ",".join(
                str(news_id)
                for news_id in missing_ids
            )
        )

    invalid_records = [
        record
        for record in records
        if record["processing_status"]
        not in {"collected", "candidate"}
    ]

    if invalid_records:
        details = ", ".join(
            (
                f"{record['news_id']}:"
                f"{record['processing_status']}"
            )
            for record in invalid_records
        )

        raise ValueError(
            "Новости имеют неподходящий статус: "
            f"{details}"
        )


async def _load_persisted_scores(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    prepared_assessments: tuple[
        PreparedNewsAssessment,
        ...
    ],
) -> tuple[PersistedNewsScore, ...]:
    """Читает оценки и проверяет их против Python."""

    records = await connection.fetch(
        """
        SELECT
            score_id,
            news_id,
            f_score,
            m_score,
            r_score,
            h_score,
            q_score,
            individual_score,
            rank_position
        FROM news_scores
        WHERE ranking_run_id = $1
        ORDER BY rank_position, news_id
        """,
        ranking_run_id,
    )

    expected_by_news_id = {
        item.news_id: item
        for item in prepared_assessments
    }

    if len(records) != len(expected_by_news_id):
        raise ValueError(
            "Количество сохранённых оценок "
            "не совпадает с ожидаемым: "
            f"stored={len(records)}, "
            f"expected={len(expected_by_news_id)}"
        )

    persisted_scores: list[
        PersistedNewsScore
    ] = []

    for record in records:
        news_id = record["news_id"]

        expected = expected_by_news_id.get(
            news_id
        )

        if expected is None:
            raise ValueError(
                "В ranking run обнаружена "
                "неожиданная новость: "
                f"news_id={news_id}"
            )

        components = (
            expected.calculated_score.components
        )

        stored_components = (
            record["f_score"],
            record["m_score"],
            record["r_score"],
            record["h_score"],
            record["q_score"],
        )

        expected_components = (
            components.f_score,
            components.m_score,
            components.r_score,
            components.h_score,
            components.q_score,
        )

        if stored_components != expected_components:
            raise ValueError(
                "Компоненты ранее сохранённого "
                "теста не совпадают: "
                f"news_id={news_id}"
            )

        if (
            record["rank_position"]
            != expected.rank_position
        ):
            raise ValueError(
                "Позиция ранее сохранённого "
                "теста не совпадает: "
                f"news_id={news_id}"
            )

        python_score = (
            expected
            .calculated_score
            .individual_score
        )

        postgres_score = record[
            "individual_score"
        ]

        persisted_scores.append(
            PersistedNewsScore(
                score_id=record["score_id"],
                news_id=news_id,
                rank_position=(
                    record["rank_position"]
                ),
                python_individual_score=(
                    python_score
                ),
                postgres_individual_score=(
                    postgres_score
                ),
                scores_match=(
                    python_score
                    == postgres_score
                ),
            )
        )

    return tuple(persisted_scores)


async def persist_manual_ranking_test(
    pool: asyncpg.Pool,
    *,
    test_key: str,
    window_started_at: datetime,
    window_finished_at: datetime,
    assessments: tuple[ManualNewsAssessment, ...],
) -> RankingPersistenceResult:
    """
    Сохраняет тестовый ranking run и сверяет формулу.

    Если Python и PostgreSQL дают разные значения,
    транзакция откатывается целиком.
    """

    normalized_test_key = test_key.strip()

    if not normalized_test_key:
        raise ValueError(
            "test_key не может быть пустым."
        )

    normalized_window_start = _normalize_datetime(
        window_started_at,
        field_name="window_started_at",
    )

    normalized_window_end = _normalize_datetime(
        window_finished_at,
        field_name="window_finished_at",
    )

    if normalized_window_end <= normalized_window_start:
        raise ValueError(
            "window_finished_at должен быть "
            "позже window_started_at."
        )

    prepared_assessments = (
        _prepare_assessments(
            assessments
        )
    )

    news_ids = tuple(
        item.news_id
        for item in prepared_assessments
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtextextended($1, 0)
                )
                """,
                normalized_test_key,
            )

            existing_run = await connection.fetchrow(
                """
                SELECT
                    ranking_run_id,
                    run_status,
                    formula_version,
                    candidate_count,
                    scored_count,
                    eligible_count
                FROM ranking_runs
                WHERE formula_version = $1
                  AND parameters->>'test_key' = $2
                ORDER BY ranking_run_id DESC
                LIMIT 1
                FOR UPDATE
                """,
                FORMULA_VERSION,
                normalized_test_key,
            )

            if existing_run is not None:
                persisted_scores = (
                    await _load_persisted_scores(
                        connection,
                        ranking_run_id=(
                            existing_run[
                                "ranking_run_id"
                            ]
                        ),
                        prepared_assessments=(
                            prepared_assessments
                        ),
                    )
                )

                if not all(
                    score.scores_match
                    for score in persisted_scores
                ):
                    raise RuntimeError(
                        "Ранее сохранённые баллы "
                        "PostgreSQL не совпадают с Python."
                    )

                return RankingPersistenceResult(
                    ranking_run_id=(
                        existing_run[
                            "ranking_run_id"
                        ]
                    ),
                    run_status=(
                        existing_run["run_status"]
                    ),
                    formula_version=(
                        existing_run[
                            "formula_version"
                        ]
                    ),
                    candidate_count=(
                        existing_run[
                            "candidate_count"
                        ]
                    ),
                    scored_count=(
                        existing_run[
                            "scored_count"
                        ]
                    ),
                    eligible_count=(
                        existing_run[
                            "eligible_count"
                        ]
                    ),
                    already_persisted=True,
                    scores=persisted_scores,
                )

            await _validate_news_items(
                connection,
                news_ids=news_ids,
            )

            parameters = _encode_json(
                {
                    "mode": "manual_formula_test",
                    "test_key": normalized_test_key,
                    "news_ids": list(news_ids),
                    "scales": {
                        "f_score": "0..10",
                        "m_score": "0..10",
                        "r_score": "0..10",
                        "h_score": "0..10",
                        "q_score": "0..1",
                    },
                    "database_formula_check": True,
                }
            )

            ranking_run_id = await connection.fetchval(
                """
                INSERT INTO ranking_runs (
                    run_status,
                    formula_version,
                    model_name,
                    prompt_version,
                    window_started_at,
                    window_finished_at,
                    candidate_count,
                    scored_count,
                    eligible_count,
                    parameters
                )
                VALUES (
                    'running',
                    $1,
                    NULL,
                    'manual_formula_test_v1',
                    $2,
                    $3,
                    $4,
                    0,
                    0,
                    $5::jsonb
                )
                RETURNING ranking_run_id
                """,
                FORMULA_VERSION,
                normalized_window_start,
                normalized_window_end,
                len(prepared_assessments),
                parameters,
            )

            for item in prepared_assessments:
                calculated = item.calculated_score
                components = calculated.components

                score_details = _encode_json(
                    {
                        "mode": (
                            "manual_formula_test"
                        ),
                        "python_individual_score": (
                            str(
                                calculated
                                .individual_score
                            )
                        ),
                        "components": {
                            "freshness": str(
                                calculated
                                .freshness_component
                            ),
                            "magnitude": str(
                                calculated
                                .magnitude_component
                            ),
                            "resonance": str(
                                calculated
                                .resonance_component
                            ),
                            "hook_quality": str(
                                calculated
                                .hook_quality_component
                            ),
                        },
                    }
                )

                record = await connection.fetchrow(
                    """
                    INSERT INTO news_scores (
                        ranking_run_id,
                        news_id,
                        f_score,
                        m_score,
                        r_score,
                        h_score,
                        q_score,
                        is_eligible,
                        rank_position,
                        score_explanation,
                        score_details
                    )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        true,
                        $8,
                        $9,
                        $10::jsonb
                    )
                    RETURNING
                        score_id,
                        individual_score
                    """,
                    ranking_run_id,
                    item.news_id,
                    components.f_score,
                    components.m_score,
                    components.r_score,
                    components.h_score,
                    components.q_score,
                    item.rank_position,
                    item.explanation,
                    score_details,
                )

                if (
                    record["individual_score"]
                    != calculated.individual_score
                ):
                    raise RuntimeError(
                        "Расчёт PostgreSQL не совпал "
                        "с расчётом Python: "
                        f"news_id={item.news_id}, "
                        "python="
                        f"{calculated.individual_score}, "
                        "postgres="
                        f"{record['individual_score']}"
                    )

            await connection.execute(
                """
                UPDATE ranking_runs
                SET
                    run_status = 'completed',
                    scored_count = $2,
                    eligible_count = $2,
                    finished_at = now(),
                    error_message = NULL
                WHERE ranking_run_id = $1
                """,
                ranking_run_id,
                len(prepared_assessments),
            )

            persisted_scores = (
                await _load_persisted_scores(
                    connection,
                    ranking_run_id=ranking_run_id,
                    prepared_assessments=(
                        prepared_assessments
                    ),
                )
            )

    return RankingPersistenceResult(
        ranking_run_id=int(ranking_run_id),
        run_status="completed",
        formula_version=FORMULA_VERSION,
        candidate_count=len(
            prepared_assessments
        ),
        scored_count=len(
            prepared_assessments
        ),
        eligible_count=len(
            prepared_assessments
        ),
        already_persisted=False,
        scores=persisted_scores,
    )