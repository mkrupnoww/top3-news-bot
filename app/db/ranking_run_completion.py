from dataclasses import dataclass
from decimal import Decimal
import json
from typing import Any, Mapping

import asyncpg

from app.db.ranking_scores import (
    ManualNewsAssessment,
    PersistedNewsScore,
)
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)
from app.ranking.request_key import (
    REQUEST_KEY_PATTERN,
)
from app.ranking.score_formula import (
    FORMULA_VERSION,
    CalculatedScore,
    calculate_individual_score,
    create_score_components,
)


@dataclass(frozen=True, slots=True)
class PreparedReservedAssessment:
    """Проверенная оценка с итоговым местом."""

    news_id: int
    explanation: str
    calculated_score: CalculatedScore
    rank_position: int


@dataclass(frozen=True, slots=True)
class RankingRunCompletionResult:
    """Результат завершения ranking run."""

    ranking_run_id: int
    request_key: str
    run_status: str
    formula_version: str
    candidate_count: int
    scored_count: int
    eligible_count: int
    already_completed: bool
    scores: tuple[
        PersistedNewsScore,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class RankingRunFailureResult:
    """Результат фиксации ошибки запуска."""

    ranking_run_id: int
    request_key: str
    run_status: str
    already_failed: bool
    error_message: str


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательное текстовое поле."""

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_request_key(
    request_key: str,
) -> str:
    """Проверяет формат request_key."""

    normalized_request_key = (
        _normalize_required_text(
            request_key,
            field_name="request_key",
        )
    )

    if not REQUEST_KEY_PATTERN.fullmatch(
        normalized_request_key
    ):
        raise ValueError(
            "request_key должен быть SHA-256 "
            "в нижнем регистре."
        )

    return normalized_request_key


def _normalize_metadata(
    metadata: RankingEvaluatorMetadata,
) -> RankingEvaluatorMetadata:
    """Проверяет метаданные оценщика."""

    model_name = metadata.model_name

    if model_name is None:
        raise ValueError(
            "metadata.model_name обязателен "
            "для OpenAI-запуска."
        )

    return RankingEvaluatorMetadata(
        run_mode=_normalize_required_text(
            metadata.run_mode,
            field_name="metadata.run_mode",
        ),
        evaluator_name=(
            _normalize_required_text(
                metadata.evaluator_name,
                field_name=(
                    "metadata.evaluator_name"
                ),
            )
        ),
        evaluator_version=(
            _normalize_required_text(
                metadata.evaluator_version,
                field_name=(
                    "metadata.evaluator_version"
                ),
            )
        ),
        prompt_version=(
            _normalize_required_text(
                metadata.prompt_version,
                field_name=(
                    "metadata.prompt_version"
                ),
            )
        ),
        model_name=_normalize_required_text(
            model_name,
            field_name="metadata.model_name",
        ),
    )


def _validate_input_news_ids(
    assessments: tuple[
        ManualNewsAssessment,
        ...,
    ],
) -> tuple[int, ...]:
    """Проверяет порядок входных news_id."""

    if not assessments:
        raise ValueError(
            "Список оценок не может быть пустым."
        )

    news_ids = tuple(
        assessment.news_id
        for assessment in assessments
    )

    for news_id in news_ids:
        if isinstance(news_id, bool):
            raise TypeError(
                "news_id не может быть bool."
            )

        if not isinstance(news_id, int):
            raise TypeError(
                "Каждый news_id должен быть int."
            )

        if news_id <= 0:
            raise ValueError(
                "Каждый news_id должен быть "
                "больше нуля."
            )

    if len(set(news_ids)) != len(news_ids):
        raise ValueError(
            "Каждый news_id должен "
            "встречаться ровно один раз."
        )

    return news_ids


def _prepare_assessments(
    assessments: tuple[
        ManualNewsAssessment,
        ...,
    ],
) -> tuple[
    PreparedReservedAssessment,
    ...,
]:
    """Рассчитывает баллы и места рейтинга."""

    _validate_input_news_ids(
        assessments
    )

    calculated_items: list[
        tuple[
            ManualNewsAssessment,
            CalculatedScore,
        ]
    ] = []

    for assessment in assessments:
        explanation = (
            assessment.explanation.strip()
        )

        if not explanation:
            raise ValueError(
                "explanation не может быть "
                "пустым: "
                f"news_id={assessment.news_id}"
            )

        components = create_score_components(
            f_score=assessment.f_score,
            m_score=assessment.m_score,
            r_score=assessment.r_score,
            h_score=assessment.h_score,
            q_score=assessment.q_score,
        )

        calculated_score = (
            calculate_individual_score(
                components
            )
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
        PreparedReservedAssessment(
            news_id=assessment.news_id,
            explanation=(
                assessment.explanation.strip()
            ),
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


def _build_usage_payload(
    usage: OpenAITokenUsage,
) -> dict[str, int]:
    """Формирует JSON с токенами."""

    return {
        "input_tokens": (
            usage.input_tokens
        ),
        "regular_input_tokens": (
            usage.regular_input_tokens
        ),
        "cached_input_tokens": (
            usage.cached_input_tokens
        ),
        "cache_write_tokens": (
            usage.cache_write_tokens
        ),
        "output_tokens": (
            usage.output_tokens
        ),
        "reasoning_tokens": (
            usage.reasoning_tokens
        ),
        "total_tokens": (
            usage.total_tokens
        ),
    }


def _build_cost_payload(
    cost_estimate: OpenAICostEstimate,
) -> dict[str, str]:
    """Формирует JSON с расчётом стоимости."""

    return {
        "model_name": (
            cost_estimate.model_name
        ),
        "pricing_version": (
            cost_estimate.pricing_version
        ),
        "regular_input_cost_usd": str(
            cost_estimate
            .regular_input_cost_usd
        ),
        "cached_input_cost_usd": str(
            cost_estimate
            .cached_input_cost_usd
        ),
        "cache_write_cost_usd": str(
            cost_estimate
            .cache_write_cost_usd
        ),
        "output_cost_usd": str(
            cost_estimate.output_cost_usd
        ),
        "total_cost_usd": str(
            cost_estimate.total_cost_usd
        ),
    }


def _validate_telemetry(
    *,
    metadata: RankingEvaluatorMetadata,
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
) -> None:
    """Проверяет согласованность телеметрии."""

    if metadata.model_name is None:
        raise ValueError(
            "metadata.model_name отсутствует."
        )

    if (
        cost_estimate.model_name
        != metadata.model_name
    ):
        raise ValueError(
            "Модель расчёта стоимости "
            "не совпадает с моделью оценщика: "
            f"cost={cost_estimate.model_name!r}, "
            f"metadata={metadata.model_name!r}"
        )

    component_total = (
        cost_estimate
        .regular_input_cost_usd
        + cost_estimate
        .cached_input_cost_usd
        + cost_estimate
        .cache_write_cost_usd
        + cost_estimate
        .output_cost_usd
    )

    if (
        component_total
        != cost_estimate.total_cost_usd
    ):
        raise ValueError(
            "total_cost_usd не совпадает "
            "с суммой компонентов стоимости."
        )

    if (
        usage.total_tokens
        != (
            usage.input_tokens
            + usage.output_tokens
        )
    ):
        raise ValueError(
            "total_tokens не совпадает "
            "с input_tokens + output_tokens."
        )


async def _validate_news_items(
    connection: asyncpg.Connection,
    *,
    news_ids: tuple[int, ...],
) -> None:
    """Повторно проверяет новости."""

    records = await connection.fetch(
        """
        SELECT
            news_id,
            processing_status
        FROM top3_news.news_items
        WHERE news_id = ANY($1::bigint[])
        ORDER BY news_id
        """,
        list(news_ids),
    )

    found_ids = {
        int(record["news_id"])
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
        not in {
            "collected",
            "candidate",
        }
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


def _validate_reserved_run(
    record: asyncpg.Record,
    *,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    input_news_ids: tuple[int, ...],
) -> None:
    """Проверяет зарезервированный запуск."""

    differences: list[str] = []

    expected_values: dict[
        str,
        object,
    ] = {
        "request_key": request_key,
        "formula_version": FORMULA_VERSION,
        "model_name": metadata.model_name,
        "prompt_version": (
            metadata.prompt_version
        ),
        "candidate_count": (
            len(input_news_ids)
        ),
        "run_mode": metadata.run_mode,
        "evaluator_name": (
            metadata.evaluator_name
        ),
        "evaluator_version": (
            metadata.evaluator_version
        ),
    }

    for field_name, expected_value in (
        expected_values.items()
    ):
        actual_value = record[field_name]

        if actual_value != expected_value:
            differences.append(
                f"{field_name}: "
                f"expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )

    if record["news_ids_match"] is not True:
        differences.append(
            "news_ids не совпадают "
            "с зарезервированным порядком."
        )

    if differences:
        raise ValueError(
            "Зарезервированный ranking_run "
            "не соответствует результату: "
            + "; ".join(differences)
        )


async def _load_persisted_scores(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    prepared_assessments: tuple[
        PreparedReservedAssessment,
        ...,
    ],
) -> tuple[
    PersistedNewsScore,
    ...,
]:
    """Читает и проверяет сохранённые оценки."""

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
            rank_position,
            score_explanation
        FROM top3_news.news_scores
        WHERE ranking_run_id = $1
        ORDER BY rank_position, news_id
        """,
        ranking_run_id,
    )

    expected_by_news_id = {
        item.news_id: item
        for item in prepared_assessments
    }

    if (
        len(records)
        != len(expected_by_news_id)
    ):
        raise ValueError(
            "Количество сохранённых оценок "
            "не совпадает с ожидаемым: "
            f"stored={len(records)}, "
            f"expected="
            f"{len(expected_by_news_id)}"
        )

    persisted_scores: list[
        PersistedNewsScore
    ] = []

    for record in records:
        news_id = int(
            record["news_id"]
        )

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

        if (
            stored_components
            != expected_components
        ):
            raise ValueError(
                "Сохранённые компоненты "
                "не совпадают с оценкой: "
                f"news_id={news_id}"
            )

        if (
            record["rank_position"]
            != expected.rank_position
        ):
            raise ValueError(
                "Сохранённая позиция "
                "не совпадает с расчётом: "
                f"news_id={news_id}"
            )

        if (
            record["score_explanation"]
            != expected.explanation
        ):
            raise ValueError(
                "Сохранённое объяснение "
                "не совпадает: "
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
                score_id=int(
                    record["score_id"]
                ),
                news_id=news_id,
                rank_position=int(
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

    return tuple(
        persisted_scores
    )


def _build_score_details(
    *,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    item: PreparedReservedAssessment,
) -> str:
    """Формирует детализацию одной оценки."""

    calculated = item.calculated_score

    return _encode_json(
        {
            "request_key": request_key,
            "formula_version": (
                FORMULA_VERSION
            ),
            "run_mode": metadata.run_mode,
            "evaluator_name": (
                metadata.evaluator_name
            ),
            "evaluator_version": (
                metadata.evaluator_version
            ),
            "prompt_version": (
                metadata.prompt_version
            ),
            "model_name": (
                metadata.model_name
            ),
            "python_individual_score": str(
                calculated.individual_score
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


async def complete_reserved_ranking_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    assessments: tuple[
        ManualNewsAssessment,
        ...,
    ],
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
) -> RankingRunCompletionResult:
    """
    Завершает ранее зарезервированный запуск.

    В одной транзакции:
    - проверяет reservation;
    - записывает news_scores;
    - сверяет Python и PostgreSQL;
    - сохраняет usage и стоимость;
    - переводит запуск в completed.
    """

    if isinstance(ranking_run_id, bool):
        raise TypeError(
            "ranking_run_id не может быть bool."
        )

    if not isinstance(ranking_run_id, int):
        raise TypeError(
            "ranking_run_id должен быть int."
        )

    if ranking_run_id <= 0:
        raise ValueError(
            "ranking_run_id должен быть "
            "больше нуля."
        )

    normalized_request_key = (
        _normalize_request_key(
            request_key
        )
    )

    normalized_metadata = (
        _normalize_metadata(
            metadata
        )
    )

    input_news_ids = (
        _validate_input_news_ids(
            assessments
        )
    )

    prepared_assessments = (
        _prepare_assessments(
            assessments
        )
    )

    _validate_telemetry(
        metadata=normalized_metadata,
        usage=usage,
        cost_estimate=cost_estimate,
    )

    encoded_news_ids = _encode_json(
        {
            "news_ids": list(
                input_news_ids
            )
        }
    )

    telemetry_parameters = _encode_json(
        {
            "openai_usage": (
                _build_usage_payload(
                    usage
                )
            ),
            "openai_cost": (
                _build_cost_payload(
                    cost_estimate
                )
            ),
            "database_formula_check": True,
            "completion_version": (
                "reserved_ranking_completion_v1"
            ),
        }
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await connection.fetchrow(
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
                    parameters->>'run_mode'
                        AS run_mode,
                    parameters->>'evaluator_name'
                        AS evaluator_name,
                    parameters->>'evaluator_version'
                        AS evaluator_version,
                    (
                        parameters->'news_ids'
                        =
                        (
                            $3::jsonb
                            -> 'news_ids'
                        )
                    ) AS news_ids_match
                FROM top3_news.ranking_runs
                WHERE ranking_run_id = $1
                  AND request_key = $2
                FOR UPDATE
                """,
                ranking_run_id,
                normalized_request_key,
                encoded_news_ids,
            )

            if record is None:
                raise LookupError(
                    "Зарезервированный "
                    "ranking_run не найден: "
                    f"ranking_run_id="
                    f"{ranking_run_id}"
                )

            _validate_reserved_run(
                record,
                request_key=(
                    normalized_request_key
                ),
                metadata=normalized_metadata,
                input_news_ids=input_news_ids,
            )

            if record["run_status"] == "failed":
                raise ValueError(
                    "Нельзя завершить ranking_run "
                    "со статусом failed."
                )

            if (
                record["run_status"]
                == "completed"
            ):
                persisted_scores = (
                    await _load_persisted_scores(
                        connection,
                        ranking_run_id=(
                            ranking_run_id
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
                        "Сохранённые баллы "
                        "PostgreSQL не совпадают "
                        "с Python."
                    )

                return RankingRunCompletionResult(
                    ranking_run_id=(
                        ranking_run_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    run_status="completed",
                    formula_version=(
                        FORMULA_VERSION
                    ),
                    candidate_count=(
                        record["candidate_count"]
                    ),
                    scored_count=(
                        record["scored_count"]
                    ),
                    eligible_count=(
                        record["eligible_count"]
                    ),
                    already_completed=True,
                    scores=persisted_scores,
                )

            if record["run_status"] != "running":
                raise ValueError(
                    "Неподдерживаемый статус "
                    "ranking_run: "
                    f"{record['run_status']}"
                )

            existing_score_count = (
                await connection.fetchval(
                    """
                    SELECT count(*)
                    FROM top3_news.news_scores
                    WHERE ranking_run_id = $1
                    """,
                    ranking_run_id,
                )
            )

            if existing_score_count != 0:
                raise RuntimeError(
                    "У running ranking_run уже "
                    "есть news_scores: "
                    f"count={existing_score_count}"
                )

            await _validate_news_items(
                connection,
                news_ids=input_news_ids,
            )

            for item in prepared_assessments:
                calculated = (
                    item.calculated_score
                )

                components = (
                    calculated.components
                )

                score_details = (
                    _build_score_details(
                        request_key=(
                            normalized_request_key
                        ),
                        metadata=(
                            normalized_metadata
                        ),
                        item=item,
                    )
                )

                inserted_score = (
                    await connection.fetchrow(
                        """
                        INSERT INTO
                            top3_news.news_scores (
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
                )

                postgres_score: Decimal = (
                    inserted_score[
                        "individual_score"
                    ]
                )

                if (
                    postgres_score
                    != calculated.individual_score
                ):
                    raise RuntimeError(
                        "Расчёт PostgreSQL "
                        "не совпал с Python: "
                        f"news_id={item.news_id}, "
                        "python="
                        f"{calculated.individual_score}, "
                        "postgres="
                        f"{postgres_score}"
                    )

            update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.ranking_runs
                    SET
                        run_status = 'completed',
                        scored_count = $3,
                        eligible_count = $3,
                        parameters = (
                            parameters
                            || $4::jsonb
                        ),
                        error_message = NULL,
                        finished_at = now()
                    WHERE ranking_run_id = $1
                      AND request_key = $2
                      AND run_status = 'running'
                    """,
                    ranking_run_id,
                    normalized_request_key,
                    len(prepared_assessments),
                    telemetry_parameters,
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось завершить "
                    "зарезервированный ranking_run: "
                    f"{update_result}"
                )

            persisted_scores = (
                await _load_persisted_scores(
                    connection,
                    ranking_run_id=(
                        ranking_run_id
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
                    "Баллы PostgreSQL "
                    "не совпадают с Python."
                )

    return RankingRunCompletionResult(
        ranking_run_id=ranking_run_id,
        request_key=normalized_request_key,
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
        already_completed=False,
        scores=persisted_scores,
    )


async def fail_reserved_ranking_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    request_key: str,
    error_message: str,
    error_type: str | None = None,
) -> RankingRunFailureResult:
    """Переводит зарезервированный запуск в failed."""

    if isinstance(ranking_run_id, bool):
        raise TypeError(
            "ranking_run_id не может быть bool."
        )

    if not isinstance(ranking_run_id, int):
        raise TypeError(
            "ranking_run_id должен быть int."
        )

    if ranking_run_id <= 0:
        raise ValueError(
            "ranking_run_id должен быть "
            "больше нуля."
        )

    normalized_request_key = (
        _normalize_request_key(
            request_key
        )
    )

    normalized_error_message = (
        _normalize_required_text(
            error_message,
            field_name="error_message",
        )
    )

    # Не сохраняем бесконтрольно огромный
    # текст стороннего исключения.
    normalized_error_message = (
        normalized_error_message[:8000]
    )

    normalized_error_type: str | None

    if error_type is None:
        normalized_error_type = None
    else:
        normalized_error_type = (
            _normalize_required_text(
                error_type,
                field_name="error_type",
            )[:500]
        )

    failure_parameters = _encode_json(
        {
            "failure": {
                "error_type": (
                    normalized_error_type
                ),
                "error_message": (
                    normalized_error_message
                ),
            },
            "failure_version": (
                "reserved_ranking_failure_v1"
            ),
        }
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await connection.fetchrow(
                """
                SELECT
                    ranking_run_id,
                    request_key,
                    run_status,
                    error_message
                FROM top3_news.ranking_runs
                WHERE ranking_run_id = $1
                  AND request_key = $2
                FOR UPDATE
                """,
                ranking_run_id,
                normalized_request_key,
            )

            if record is None:
                raise LookupError(
                    "Зарезервированный "
                    "ranking_run не найден: "
                    f"ranking_run_id="
                    f"{ranking_run_id}"
                )

            if (
                record["run_status"]
                == "completed"
            ):
                raise ValueError(
                    "Нельзя перевести completed "
                    "ranking_run в failed."
                )

            if record["run_status"] == "failed":
                return RankingRunFailureResult(
                    ranking_run_id=(
                        ranking_run_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    run_status="failed",
                    already_failed=True,
                    error_message=(
                        record["error_message"]
                        or normalized_error_message
                    ),
                )

            if record["run_status"] != "running":
                raise ValueError(
                    "Неподдерживаемый статус "
                    "ranking_run: "
                    f"{record['run_status']}"
                )

            existing_score_count = (
                await connection.fetchval(
                    """
                    SELECT count(*)
                    FROM top3_news.news_scores
                    WHERE ranking_run_id = $1
                    """,
                    ranking_run_id,
                )
            )

            if existing_score_count != 0:
                raise RuntimeError(
                    "Нельзя пометить запуск "
                    "failed: уже существуют "
                    "news_scores."
                )

            update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.ranking_runs
                    SET
                        run_status = 'failed',
                        scored_count = 0,
                        eligible_count = 0,
                        parameters = (
                            parameters
                            || $4::jsonb
                        ),
                        error_message = $3,
                        finished_at = now()
                    WHERE ranking_run_id = $1
                      AND request_key = $2
                      AND run_status = 'running'
                    """,
                    ranking_run_id,
                    normalized_request_key,
                    normalized_error_message,
                    failure_parameters,
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось перевести "
                    "ranking_run в failed: "
                    f"{update_result}"
                )

    return RankingRunFailureResult(
        ranking_run_id=ranking_run_id,
        request_key=normalized_request_key,
        run_status="failed",
        already_failed=False,
        error_message=(
            normalized_error_message
        ),
    )