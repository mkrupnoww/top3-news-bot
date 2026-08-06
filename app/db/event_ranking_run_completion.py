from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
from typing import Any, Mapping

import asyncpg

from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.event_evaluator import (
    EventRankingCoverageDiagnostics,
)
from app.ranking.event_formula_pipeline import (
    CalculatedEventScore,
    EventFormulaCalculationResult,
    EventScoreCalculationResult,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
    Top3CombinationScore,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)
from app.ranking.request_key import (
    REQUEST_KEY_PATTERN,
)


COMPLETION_VERSION = (
    "reserved_event_ranking_completion_v4"
)

DIAGNOSTIC_FAILURE_VERSION = (
    "reserved_event_ranking_diagnostic_failure_v1"
)

AUDIENCE_PLATFORM_CODE = (
    "aggregated_event_metrics"
)

DECIMAL_QUANTUM = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class PersistedEventScore:
    """Идентификаторы сохранённого инфоповода."""

    ranking_event_id: int
    representative_news_id: int
    score_id: int


@dataclass(frozen=True, slots=True)
class EventRankingRunCompletionResult:
    """Результат завершения event-level ranking run."""

    ranking_run_id: int
    request_key: str
    run_status: str
    formula_version: str
    candidate_count: int
    scored_count: int
    eligible_count: int
    combination_count: int
    winner_combination_id: int
    already_completed: bool
    persisted_events: tuple[
        PersistedEventScore,
        ...,
    ]
    degraded: bool
    processed_candidate_count: int
    missing_news_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EventRankingRunDiagnosticFailureResult:
    """Результат сохранения event-level диагностического сбоя."""

    ranking_run_id: int
    request_key: str
    run_status: str
    formula_version: str
    candidate_count: int
    scored_count: int
    eligible_count: int
    failure_stage: str
    already_failed: bool
    error_message: str
    persisted_events: tuple[
        PersistedEventScore,
        ...,
    ]


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
    """Проверяет обязательный текст."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть str."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_request_key(
    request_key: str,
) -> str:
    """Проверяет SHA-256 request_key."""

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


def _positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительный int."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} должен быть больше нуля."
        )

    return value


def _normalize_metadata(
    metadata: RankingEvaluatorMetadata,
) -> RankingEvaluatorMetadata:
    """Проверяет метаданные event-оценщика."""

    model_name = metadata.model_name

    if model_name is None:
        raise ValueError(
            "metadata.model_name обязателен "
            "для OpenAI-запуска."
        )

    normalized = RankingEvaluatorMetadata(
        run_mode=_normalize_required_text(
            metadata.run_mode,
            field_name="metadata.run_mode",
        ),
        evaluator_name=_normalize_required_text(
            metadata.evaluator_name,
            field_name="metadata.evaluator_name",
        ),
        evaluator_version=(
            _normalize_required_text(
                metadata.evaluator_version,
                field_name=(
                    "metadata.evaluator_version"
                ),
            )
        ),
        prompt_version=_normalize_required_text(
            metadata.prompt_version,
            field_name="metadata.prompt_version",
        ),
        model_name=_normalize_required_text(
            model_name,
            field_name="metadata.model_name",
        ),
    )

    if normalized.run_mode != (
        "openai_event_ranking"
    ):
        raise ValueError(
            "Для event completion требуется "
            "run_mode='openai_event_ranking'."
        )

    return normalized


def _normalize_news_ids(
    news_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Проверяет исходный порядок кандидатов."""

    if not isinstance(news_ids, tuple):
        raise TypeError(
            "candidate_news_ids должен быть tuple."
        )

    if not news_ids:
        raise ValueError(
            "candidate_news_ids "
            "не может быть пустым."
        )

    normalized = tuple(
        _positive_integer(
            news_id,
            field_name="candidate_news_id",
        )
        for news_id in news_ids
    )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "candidate_news_ids содержит "
            "повторяющиеся news_id."
        )

    return normalized


def _normalize_completion_coverage(
    *,
    candidate_news_ids: tuple[int, ...],
    diagnostics: (
        EventRankingCoverageDiagnostics
        | None
    ),
) -> EventRankingCoverageDiagnostics:
    """Проверяет coverage относительно reservation."""

    if diagnostics is None:
        return EventRankingCoverageDiagnostics(
            expected_news_ids=candidate_news_ids,
            processed_news_ids=candidate_news_ids,
        )

    if not isinstance(
        diagnostics,
        EventRankingCoverageDiagnostics,
    ):
        raise TypeError(
            "coverage_diagnostics должен быть "
            "EventRankingCoverageDiagnostics."
        )

    if (
        diagnostics.expected_news_ids
        != candidate_news_ids
    ):
        raise ValueError(
            "coverage expected_news_ids не совпадает "
            "с candidate_news_ids reservation."
        )

    return diagnostics


def _build_coverage_payload(
    diagnostics: EventRankingCoverageDiagnostics,
) -> dict[str, Any]:
    """Формирует JSON-диагностику полного или degraded run."""

    degraded_reason = (
        "incomplete_model_coverage_after_repair"
        if diagnostics.degraded
        else None
    )

    repair_error_type = diagnostics.repair_error_type
    repair_error_message = (
        diagnostics.repair_error_message
    )

    return {
        "degraded": diagnostics.degraded,
        "degraded_reason": degraded_reason,
        "original_candidate_count": len(
            diagnostics.expected_news_ids
        ),
        "processed_candidate_count": len(
            diagnostics.processed_news_ids
        ),
        "missing_candidate_count": len(
            diagnostics.missing_news_ids
        ),
        "expected_news_ids": list(
            diagnostics.expected_news_ids
        ),
        "processed_news_ids": list(
            diagnostics.processed_news_ids
        ),
        "initial_missing_news_ids": list(
            diagnostics.initial_missing_news_ids
        ),
        "missing_news_ids": list(
            diagnostics.missing_news_ids
        ),
        "repair_attempted": (
            diagnostics.repair_attempted
        ),
        "repair_succeeded": (
            diagnostics.repair_succeeded
        ),
        "repair_error_type": (
            None
            if repair_error_type is None
            else repair_error_type[:500]
        ),
        "repair_error_message": (
            None
            if repair_error_message is None
            else repair_error_message[:8000]
        ),
        "model_call_count": (
            diagnostics.model_call_count
        ),
        "story_cluster_verification": {
            "attempted": (
                diagnostics
                .story_cluster_verification_attempted
            ),
            "succeeded": (
                diagnostics
                .story_cluster_verification_succeeded
            ),
            "degraded": (
                diagnostics
                .story_cluster_verification_degraded
            ),
            "skipped_reason": (
                diagnostics
                .story_cluster_verification_skipped_reason
            ),
            "error_type": (
                None
                if diagnostics
                .story_cluster_verification_error_type
                is None
                else diagnostics
                .story_cluster_verification_error_type[:500]
            ),
            "error_message": (
                None
                if diagnostics
                .story_cluster_verification_error_message
                is None
                else diagnostics
                .story_cluster_verification_error_message[:8000]
            ),
            "prompt_version": (
                diagnostics
                .story_cluster_verification_prompt_version
            ),
            "cluster_count_before": (
                diagnostics.story_cluster_count_before
            ),
            "cluster_count_after": (
                diagnostics.story_cluster_count_after
            ),
            "multi_event_cluster_count_before": (
                diagnostics
                .story_cluster_multi_event_count_before
            ),
            "multi_event_cluster_count_after": (
                diagnostics
                .story_cluster_multi_event_count_after
            ),
            "verifier_event_count": (
                diagnostics.story_cluster_verifier_event_count
            ),
            "changes": [
                {
                    "original_story_cluster_key": (
                        change.original_story_cluster_key
                    ),
                    "representative_news_ids": list(
                        change.representative_news_ids
                    ),
                    "resulting_story_cluster_keys": list(
                        change.resulting_story_cluster_keys
                    ),
                }
                for change in (
                    diagnostics
                    .story_cluster_verification_changes
                )
            ],
        },
    }


def _decode_json_object(
    value: object,
    *,
    field_name: str,
) -> dict[str, Any]:
    """Декодирует jsonb-объект asyncpg."""

    decoded = value

    if isinstance(decoded, str):
        decoded = json.loads(decoded)

    if not isinstance(decoded, dict):
        raise ValueError(
            f"{field_name} должен быть JSON-объектом."
        )

    return decoded


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Приводит дату с часовым поясом к UTC."""

    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} должен быть datetime."
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} должен содержать "
            "часовой пояс."
        )

    return value.astimezone(timezone.utc)


def _decimal6(
    value: Decimal | int | str,
) -> Decimal:
    """Приводит число к scale=6."""

    return Decimal(
        str(value)
    ).quantize(
        DECIMAL_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _build_usage_payload(
    usage: OpenAITokenUsage,
) -> dict[str, int]:
    """Формирует JSON с токенами."""

    return {
        "input_tokens": usage.input_tokens,
        "regular_input_tokens": (
            usage.regular_input_tokens
        ),
        "cached_input_tokens": (
            usage.cached_input_tokens
        ),
        "cache_write_tokens": (
            usage.cache_write_tokens
        ),
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": (
            usage.reasoning_tokens
        ),
        "total_tokens": usage.total_tokens,
    }


def _build_cost_payload(
    cost_estimate: OpenAICostEstimate,
) -> dict[str, str]:
    """Формирует JSON со стоимостью."""

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
    """Проверяет usage и стоимость."""

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
            "не совпадает с моделью оценщика."
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


def _validate_scored_calculation(
    *,
    calculation: (
        EventScoreCalculationResult
        | EventFormulaCalculationResult
    ),
    coverage_news_ids: tuple[int, ...],
) -> None:
    """Проверяет общий результат расчёта баллов."""

    if calculation.formula_version != (
        FULL_FORMULA_VERSION
    ):
        raise ValueError(
            "Неподдерживаемая formula_version "
            "в результате расчёта."
        )

    if not calculation.calculated_events:
        raise ValueError(
            "calculated_events "
            "не может быть пустым."
        )

    member_news_ids = [
        news_id
        for item in calculation.calculated_events
        for news_id in item.event.member_news_ids
    ]

    if (
        len(member_news_ids)
        != len(set(member_news_ids))
    ):
        raise ValueError(
            "Один news_id входит "
            "в несколько инфоповодов."
        )

    if set(member_news_ids) != set(
        coverage_news_ids
    ):
        missing = sorted(
            set(coverage_news_ids)
            - set(member_news_ids)
        )

        unexpected = sorted(
            set(member_news_ids)
            - set(coverage_news_ids)
        )

        raise ValueError(
            "Инфоповоды не покрывают "
            "обрабатываемых кандидатов: "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )

    representative_news_ids = tuple(
        item.event.representative_news_id
        for item in calculation.calculated_events
    )

    score_news_ids = tuple(
        item.score.news_id
        for item in calculation.calculated_events
    )

    if representative_news_ids != (
        score_news_ids
    ):
        raise ValueError(
            "news_id полного балла должен "
            "совпадать с representative_news_id."
        )


def _validate_calculation(
    *,
    calculation: EventFormulaCalculationResult,
    processed_news_ids: tuple[int, ...],
) -> None:
    """Проверяет согласованность полного расчёта."""

    _validate_scored_calculation(
        calculation=calculation,
        coverage_news_ids=processed_news_ids,
    )

    winner = calculation.top3_selection.winner

    if winner.is_winner is not True:
        raise ValueError(
            "Победившая комбинация "
            "не отмечена is_winner=true."
        )

    score_id_set = {
        item.score.news_id
        for item in calculation.calculated_events
    }

    if not set(
        winner.news_ids
    ).issubset(score_id_set):
        raise ValueError(
            "Победившая комбинация содержит "
            "неизвестный news_id."
        )

def _event_key(
    item: CalculatedEventScore,
) -> str:
    """Создаёт SHA-256 ключ инфоповода."""

    event = item.event

    payload = {
        "formula_version": (
            FULL_FORMULA_VERSION
        ),
        "representative_news_id": (
            event.representative_news_id
        ),
        "event_title": event.event_title,
        "event_time_utc": (
            event.event_time_utc
            .astimezone(timezone.utc)
            .isoformat()
        ),
        "macro_topic": event.macro_topic,
        "story_cluster_key": (
            event.story_cluster_key
        ),
        "expert_scores": {
            "i": str(event.i_score),
            "k": str(event.k_score),
            "n": str(event.n_score),
            "e": str(event.e_score),
            "x": str(event.x_score),
            "q": str(event.q_score),
        },
        "members": [
            {
                "news_id": member.news_id,
                "source_relation": (
                    member.source_relation
                ),
                "is_representative": (
                    member.is_representative
                ),
                "is_independent_source": (
                    member.is_independent_source
                ),
                "counts_toward_reach": (
                    member.counts_toward_reach
                ),
                "source_weight": (
                    member.source_weight
                ),
            }
            for member in sorted(
                event.members,
                key=lambda value: value.news_id,
            )
        ],
    }

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def _combination_key(
    item: Top3CombinationScore,
) -> str:
    """Создаёт SHA-256 ключ комбинации."""

    canonical_json = json.dumps(
        {
            "formula_version": (
                FULL_FORMULA_VERSION
            ),
            "news_ids": list(
                item.news_ids
            ),
            "selection_policy_version": (
                item.selection_policy_version
            ),
            "story_cluster_keys": list(
                item.story_cluster_keys
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


def _rank_positions(
    calculation: (
        EventScoreCalculationResult
        | EventFormulaCalculationResult
    ),
) -> dict[int, int]:
    """Рассчитывает общий порядок по B_i."""

    ordered = sorted(
        calculation.calculated_events,
        key=lambda item: (
            -item.score
            .individual
            .individual_score,
            item.score.news_id,
        ),
    )

    return {
        item.score.news_id: position
        for position, item
        in enumerate(
            ordered,
            start=1,
        )
    }


def _winner_positions(
    calculation: EventFormulaCalculationResult,
) -> dict[int, int]:
    """Возвращает позиции победившего TOP-3."""

    return {
        news_id: position
        for position, news_id
        in enumerate(
            calculation
            .top3_selection
            .winner
            .ordered_news_ids,
            start=1,
        )
    }


def _event_details(
    item: CalculatedEventScore,
) -> dict[str, Any]:
    """Формирует прозрачные детали инфоповода."""

    event = item.event

    return {
        "formula_version": (
            FULL_FORMULA_VERSION
        ),
        "member_news_ids": list(
            event.member_news_ids
        ),
        "story_cluster_key": (
            event.story_cluster_key
        ),
        "expert_scores": {
            "i_score": str(event.i_score),
            "k_score": str(event.k_score),
            "n_score": str(event.n_score),
            "e_score": str(event.e_score),
            "x_score": str(event.x_score),
            "q_score": str(event.q_score),
        },
        "source_weight_sum": (
            event.source_weight_sum
        ),
    }


def _score_details(
    *,
    item: CalculatedEventScore,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    event_key: str,
) -> dict[str, Any]:
    """Формирует полную трассировку балла."""

    score = item.score

    return {
        "formula_version": (
            FULL_FORMULA_VERSION
        ),
        "request_key": request_key,
        "event_key": event_key,
        "model_name": metadata.model_name,
        "prompt_version": (
            metadata.prompt_version
        ),
        "evaluator_version": (
            metadata.evaluator_version
        ),
        "source_weight_sum": str(
            score.source_weight_sum
        ),
        "max_source_weight_sum": str(
            score.max_source_weight_sum
        ),
        "resonance": {
            "confidence": (
                score.resonance.confidence
            ),
            "effective_v_weight": str(
                score
                .resonance
                .effective_v_weight
            ),
            "effective_c_weight": str(
                score
                .resonance
                .effective_c_weight
            ),
            "effective_s_weight": str(
                score
                .resonance
                .effective_s_weight
            ),
        },
        "python_individual_score": str(
            score
            .individual
            .individual_score
        ),
        "components": {
            "freshness": str(
                score
                .individual
                .freshness_component
            ),
            "magnitude": str(
                score
                .individual
                .magnitude_component
            ),
            "resonance": str(
                score
                .individual
                .resonance_component
            ),
            "hook_quality": str(
                score
                .individual
                .hook_quality_component
            ),
        },
    }


def _score_explanation(
    item: CalculatedEventScore,
) -> str:
    """Формирует человекочитаемое объяснение."""

    event = item.event

    return (
        f"{event.event_title}. "
        f"I: {event.impact_reason} "
        f"H: {event.hook_reason} "
        f"Q: {event.q_reason}"
    )


def _combination_details(
    item: Top3CombinationScore,
) -> dict[str, Any]:
    """Формирует детали одной тройки."""

    return {
        "formula_version": (
            FULL_FORMULA_VERSION
        ),
        "news_ids": list(
            item.news_ids
        ),
        "ordered_news_ids": list(
            item.ordered_news_ids
        ),
        "python_final_top_score": str(
            item.final_top_score
        ),
        "selection_policy_version": (
            item.selection_policy_version
        ),
        "story_cluster_keys": list(
            item.story_cluster_keys
        ),
        "distinct_story_cluster_count": (
            item.distinct_story_cluster_count
        ),
        "passes_story_cluster_filter": (
            item.passes_story_cluster_filter
        ),
        "story_cluster_filter_applied": (
            item.story_cluster_filter_applied
        ),
        "story_cluster_fallback_used": (
            item.story_cluster_fallback_used
        ),
        "tie_break_order": (
            ["passes_story_cluster_filter"]
            if item.story_cluster_filter_applied
            else []
        )
        + [
            "final_top_score",
            "mean_m_score",
            "mean_q_score",
            "mean_f_score",
            "news_ids",
        ],
    }


def _selection_reason(
    item: Top3CombinationScore,
) -> str:
    """Формирует объяснение ранга комбинации."""

    if item.is_winner:
        prefix = "Победившая комбинация"
    elif (
        item.story_cluster_filter_applied
        and not item.passes_story_cluster_filter
    ):
        prefix = (
            "Отфильтрованная комбинация "
            "одной сюжетной семьи"
        )
    else:
        prefix = "Допустимая комбинация"

    return (
        f"{prefix}: rank={item.combination_rank}; "
        f"TOP(S)={item.final_top_score}; "
        f"mean_M={item.mean_m_score}; "
        f"mean_Q={item.mean_q_score}; "
        f"mean_F={item.mean_f_score}; "
        "story_clusters="
        f"{item.distinct_story_cluster_count}; "
        "passes_story_cluster_filter="
        f"{str(item.passes_story_cluster_filter).lower()}; "
        "story_cluster_filter_applied="
        f"{str(item.story_cluster_filter_applied).lower()}; "
        "story_cluster_fallback_used="
        f"{str(item.story_cluster_fallback_used).lower()}."
    )


async def _validate_news_items(
    connection: asyncpg.Connection,
    *,
    news_ids: tuple[int, ...],
) -> None:
    """Повторно проверяет исходные публикации."""

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
    candidate_news_ids: tuple[int, ...],
    calculation: (
        EventScoreCalculationResult
        | EventFormulaCalculationResult
    ),
) -> None:
    """Проверяет reservation перед записью."""

    expected_values: dict[
        str,
        object,
    ] = {
        "request_key": request_key,
        "formula_version": (
            FULL_FORMULA_VERSION
        ),
        "model_name": metadata.model_name,
        "prompt_version": (
            metadata.prompt_version
        ),
        "candidate_count": (
            len(candidate_news_ids)
        ),
        "run_mode": metadata.run_mode,
        "evaluator_name": (
            metadata.evaluator_name
        ),
        "evaluator_version": (
            metadata.evaluator_version
        ),
        "window_started_at": (
            calculation.window_start
        ),
        "window_finished_at": (
            calculation.window_end
        ),
    }

    differences: list[str] = []

    for field_name, expected_value in (
        expected_values.items()
    ):
        if record[field_name] != expected_value:
            differences.append(
                f"{field_name}: "
                f"expected={expected_value!r}, "
                f"actual={record[field_name]!r}"
            )

    if record["news_ids_match"] is not True:
        differences.append(
            "news_ids не совпадают "
            "с reservation."
        )

    if differences:
        raise ValueError(
            "Зарезервированный ranking_run "
            "не соответствует event-результату: "
            + "; ".join(differences)
        )


async def _load_and_verify_completed(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    calculation: EventFormulaCalculationResult,
) -> tuple[
    tuple[PersistedEventScore, ...],
    int,
]:
    """Загружает и сверяет сохранённый event-level результат."""

    rows = await connection.fetch(
        """
        SELECT
            e.ranking_event_id,
            e.representative_news_id,
            e.event_title,
            e.event_time_utc,
            e.macro_topic,
            e.source_weight_sum,
            s.score_id,
            s.f_score,
            s.m_score,
            s.r_score,
            s.h_score,
            s.q_score,
            s.individual_score,
            s.is_eligible,
            s.exclusion_reason,
            s.rank_position,
            s.age_hours,
            s.u_score,
            s.i_score,
            s.v_score,
            s.c_score,
            s.s_score,
            s.k_score,
            s.n_score,
            s.e_score,
            s.x_score,
            s.resonance_confidence,
            s.selected_for_top3,
            s.top3_position
        FROM top3_news.ranking_events AS e
        JOIN top3_news.news_scores AS s
          ON s.ranking_event_id = e.ranking_event_id
         AND s.ranking_run_id = e.ranking_run_id
        WHERE e.ranking_run_id = $1
        ORDER BY e.representative_news_id
        """,
        ranking_run_id,
    )

    expected_by_news_id = {
        item.event.representative_news_id: item
        for item in calculation.calculated_events
    }

    if len(rows) != len(expected_by_news_id):
        raise RuntimeError(
            "Количество сохранённых "
            "event scores не совпадает "
            "с Python-результатом."
        )

    rank_positions = _rank_positions(
        calculation
    )

    winner_positions = _winner_positions(
        calculation
    )

    persisted_events: list[
        PersistedEventScore
    ] = []

    for row in rows:
        news_id = int(
            row["representative_news_id"]
        )

        item = expected_by_news_id.get(
            news_id
        )

        if item is None:
            raise RuntimeError(
                "В БД найден неизвестный "
                f"representative_news_id={news_id}."
            )

        event = item.event
        score = item.score

        expected_values = {
            "event_title": event.event_title,
            "event_time_utc": (
                event.event_time_utc
            ),
            "macro_topic": event.macro_topic,
            "source_weight_sum": _decimal6(
                event.source_weight_sum
            ),
            "f_score": score.f_score,
            "m_score": score.m_score,
            "r_score": (
                score.resonance.r_score
            ),
            "h_score": score.h_score,
            "q_score": score.q_score,
            "individual_score": (
                score
                .individual
                .individual_score
            ),
            "is_eligible": score.is_eligible,
            "exclusion_reason": (
                score.exclusion_reason
            ),
            "rank_position": (
                rank_positions[news_id]
            ),
            "age_hours": score.age_hours,
            "u_score": score.u_score,
            "i_score": score.i_score,
            "v_score": (
                score.resonance.v_score
            ),
            "c_score": (
                score.resonance.c_score
            ),
            "s_score": (
                score.resonance.s_score
            ),
            "k_score": score.k_score,
            "n_score": score.n_score,
            "e_score": score.e_score,
            "x_score": score.x_score,
            "resonance_confidence": (
                score.resonance.confidence
            ),
            "selected_for_top3": (
                news_id in winner_positions
            ),
            "top3_position": (
                winner_positions.get(news_id)
            ),
        }

        differences = [
            (
                f"{field_name}: "
                f"expected={expected!r}, "
                f"actual={row[field_name]!r}"
            )
            for field_name, expected
            in expected_values.items()
            if row[field_name] != expected
        ]

        if differences:
            raise RuntimeError(
                "Сохранённый event score "
                f"не совпадает: news_id={news_id}; "
                + "; ".join(differences)
            )

        persisted_events.append(
            PersistedEventScore(
                ranking_event_id=int(
                    row["ranking_event_id"]
                ),
                representative_news_id=(
                    news_id
                ),
                score_id=int(
                    row["score_id"]
                ),
            )
        )

    combination_rows = await connection.fetch(
        """
        SELECT
            combination_id,
            combination_rank,
            mean_individual_score,
            diversity_score,
            final_top_score,
            mean_m_score,
            mean_q_score,
            mean_f_score,
            distinct_macro_topic_count,
            is_winner
        FROM top3_news.ranking_combinations
        WHERE ranking_run_id = $1
        ORDER BY combination_rank
        """,
        ranking_run_id,
    )

    expected_combinations = (
        calculation
        .top3_selection
        .combinations
    )

    if (
        len(combination_rows)
        != len(expected_combinations)
    ):
        raise RuntimeError(
            "Количество сохранённых комбинаций "
            "не совпадает с Python."
        )

    winner_combination_id: int | None = None

    for row, expected in zip(
        combination_rows,
        expected_combinations,
        strict=True,
    ):
        expected_values = {
            "combination_rank": (
                expected.combination_rank
            ),
            "mean_individual_score": (
                expected.mean_individual_score
            ),
            "diversity_score": (
                expected.diversity_score
            ),
            "final_top_score": (
                expected.final_top_score
            ),
            "mean_m_score": (
                expected.mean_m_score
            ),
            "mean_q_score": (
                expected.mean_q_score
            ),
            "mean_f_score": (
                expected.mean_f_score
            ),
            "distinct_macro_topic_count": (
                expected
                .distinct_macro_topic_count
            ),
            "is_winner": expected.is_winner,
        }

        differences = [
            (
                f"{field_name}: "
                f"expected={value!r}, "
                f"actual={row[field_name]!r}"
            )
            for field_name, value
            in expected_values.items()
            if row[field_name] != value
        ]

        if differences:
            raise RuntimeError(
                "Сохранённая комбинация "
                "не совпадает: "
                + "; ".join(differences)
            )

        if row["is_winner"] is True:
            winner_combination_id = int(
                row["combination_id"]
            )

    if winner_combination_id is None:
        raise RuntimeError(
            "В БД отсутствует победившая "
            "комбинация."
        )

    return (
        tuple(persisted_events),
        winner_combination_id,
    )


async def _insert_event_scores(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    calculation: (
        EventScoreCalculationResult
        | EventFormulaCalculationResult
    ),
    winner_positions: Mapping[int, int],
) -> tuple[
    tuple[PersistedEventScore, ...],
    dict[int, int],
]:
    """Сохраняет события, участников, метрики и баллы."""

    rank_positions = _rank_positions(
        calculation
    )

    persisted_events: list[
        PersistedEventScore
    ] = []

    score_ids_by_news_id: dict[
        int,
        int,
    ] = {}

    for item in calculation.calculated_events:
        event = item.event
        score = item.score
        event_key = _event_key(item)

        event_row = await connection.fetchrow(
            """
            INSERT INTO top3_news.ranking_events (
                ranking_run_id,
                event_key,
                representative_news_id,
                event_title,
                event_time_utc,
                macro_topic,
                impact_reason,
                hook_reason,
                q_reason,
                source_weight_sum,
                event_details
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11::jsonb
            )
            RETURNING ranking_event_id
            """,
            ranking_run_id,
            event_key,
            event.representative_news_id,
            event.event_title,
            event.event_time_utc,
            event.macro_topic,
            event.impact_reason,
            event.hook_reason,
            event.q_reason,
            event.source_weight_sum,
            _encode_json(
                _event_details(item)
            ),
        )

        ranking_event_id = int(
            event_row["ranking_event_id"]
        )

        for member in event.members:
            await connection.execute(
                """
                INSERT INTO
                    top3_news.ranking_event_members (
                        ranking_event_id,
                        ranking_run_id,
                        news_id,
                        is_representative,
                        is_independent_source,
                        counts_toward_reach,
                        source_weight,
                        source_relation,
                        membership_reason
                    )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9
                )
                """,
                ranking_event_id,
                ranking_run_id,
                member.news_id,
                member.is_representative,
                member.is_independent_source,
                member.counts_toward_reach,
                member.source_weight,
                member.source_relation,
                member.membership_reason,
            )

        raw_metrics = item.audience_metrics

        if any(
            value is not None
            for value in (
                raw_metrics.view_count,
                raw_metrics.comment_count,
                raw_metrics.share_count,
            )
        ):
            await connection.execute(
                """
                INSERT INTO
                    top3_news.ranking_audience_metrics (
                        ranking_event_id,
                        ranking_run_id,
                        platform_code,
                        measured_at,
                        metric_window_hours,
                        view_count,
                        comment_count,
                        share_count,
                        is_trusted,
                        raw_payload
                    )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, true,
                    $9::jsonb
                )
                """,
                ranking_event_id,
                ranking_run_id,
                AUDIENCE_PLATFORM_CODE,
                calculation.window_end,
                Decimal("24.000000"),
                raw_metrics.view_count,
                raw_metrics.comment_count,
                raw_metrics.share_count,
                _encode_json(
                    {
                        "resonance_confidence": (
                            score
                            .resonance
                            .confidence
                        ),
                        "normalized_scores": {
                            "v_score": (
                                None
                                if score
                                .resonance
                                .v_score
                                is None
                                else str(
                                    score
                                    .resonance
                                    .v_score
                                )
                            ),
                            "c_score": (
                                None
                                if score
                                .resonance
                                .c_score
                                is None
                                else str(
                                    score
                                    .resonance
                                    .c_score
                                )
                            ),
                            "s_score": (
                                None
                                if score
                                .resonance
                                .s_score
                                is None
                                else str(
                                    score
                                    .resonance
                                    .s_score
                                )
                            ),
                        },
                    }
                ),
            )

        selected_for_top3 = (
            score.news_id
            in winner_positions
        )

        score_row = await connection.fetchrow(
            """
            INSERT INTO top3_news.news_scores (
                ranking_run_id,
                news_id,
                ranking_event_id,
                f_score,
                m_score,
                r_score,
                h_score,
                q_score,
                is_eligible,
                exclusion_reason,
                rank_position,
                score_explanation,
                score_details,
                age_hours,
                u_score,
                i_score,
                v_score,
                c_score,
                s_score,
                k_score,
                n_score,
                e_score,
                x_score,
                resonance_confidence,
                selected_for_top3,
                top3_position
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13::jsonb, $14,
                $15, $16, $17, $18, $19,
                $20, $21, $22, $23, $24,
                $25, $26
            )
            RETURNING
                score_id,
                individual_score
            """,
            ranking_run_id,
            score.news_id,
            ranking_event_id,
            score.f_score,
            score.m_score,
            score.resonance.r_score,
            score.h_score,
            score.q_score,
            score.is_eligible,
            score.exclusion_reason,
            rank_positions[
                score.news_id
            ],
            _score_explanation(item),
            _encode_json(
                _score_details(
                    item=item,
                    request_key=request_key,
                    metadata=metadata,
                    event_key=event_key,
                )
            ),
            score.age_hours,
            score.u_score,
            score.i_score,
            score.resonance.v_score,
            score.resonance.c_score,
            score.resonance.s_score,
            score.k_score,
            score.n_score,
            score.e_score,
            score.x_score,
            score.resonance.confidence,
            selected_for_top3,
            winner_positions.get(
                score.news_id
            ),
        )

        postgres_score = (
            score_row["individual_score"]
        )

        if (
            postgres_score
            != score
            .individual
            .individual_score
        ):
            raise RuntimeError(
                "Расчёт PostgreSQL "
                "не совпал с Python: "
                f"news_id={score.news_id}."
            )

        score_id = int(
            score_row["score_id"]
        )

        score_ids_by_news_id[
            score.news_id
        ] = score_id

        persisted_events.append(
            PersistedEventScore(
                ranking_event_id=(
                    ranking_event_id
                ),
                representative_news_id=(
                    score.news_id
                ),
                score_id=score_id,
            )
        )

    return (
        tuple(persisted_events),
        score_ids_by_news_id,
    )


async def complete_reserved_event_ranking_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    candidate_news_ids: tuple[int, ...],
    calculation: EventFormulaCalculationResult,
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
    coverage_diagnostics: (
        EventRankingCoverageDiagnostics
        | None
    ) = None,
) -> EventRankingRunCompletionResult:
    """
    Атомарно сохраняет event-level результат.

    Полный и degraded-результат получают статус
    completed. Reservation остаётся привязанным к
    исходному набору, а расчёт покрывает только
    processed_news_ids из coverage-диагностики.
    """

    normalized_ranking_run_id = (
        _positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
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

    normalized_news_ids = (
        _normalize_news_ids(
            candidate_news_ids
        )
    )

    normalized_coverage = (
        _normalize_completion_coverage(
            candidate_news_ids=(
                normalized_news_ids
            ),
            diagnostics=coverage_diagnostics,
        )
    )

    _validate_calculation(
        calculation=calculation,
        processed_news_ids=(
            normalized_coverage
            .processed_news_ids
        ),
    )

    _validate_telemetry(
        metadata=normalized_metadata,
        usage=usage,
        cost_estimate=cost_estimate,
    )

    encoded_news_ids = _encode_json(
        {
            "news_ids": list(
                normalized_news_ids
            )
        }
    )

    coverage_payload = _build_coverage_payload(
        normalized_coverage
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
            "completion_version": (
                COMPLETION_VERSION
            ),
            "formula_calculated_in_python": True,
            "event_count": len(
                calculation.calculated_events
            ),
            "combination_count": len(
                calculation
                .top3_selection
                .combinations
            ),
            "winner_news_ids": list(
                calculation
                .top3_selection
                .winner
                .ordered_news_ids
            ),
            "top3_selection": {
                "policy_version": (
                    calculation
                    .top3_selection
                    .selection_policy_version
                ),
                "story_cluster_filter_applied": (
                    calculation
                    .top3_selection
                    .story_cluster_filter_applied
                ),
                "story_cluster_fallback_used": (
                    calculation
                    .top3_selection
                    .story_cluster_fallback_used
                ),
                "diverse_combination_count": (
                    calculation
                    .top3_selection
                    .story_cluster_diverse_combination_count
                ),
                "winner_story_cluster_keys": list(
                    calculation
                    .top3_selection
                    .winner
                    .story_cluster_keys
                ),
            },
            "degraded": (
                normalized_coverage.degraded
            ),
            "degraded_reason": (
                coverage_payload[
                    "degraded_reason"
                ]
            ),
            "original_candidate_count": (
                coverage_payload[
                    "original_candidate_count"
                ]
            ),
            "processed_candidate_count": (
                coverage_payload[
                    "processed_candidate_count"
                ]
            ),
            "missing_candidate_count": (
                coverage_payload[
                    "missing_candidate_count"
                ]
            ),
            "processed_news_ids": (
                coverage_payload[
                    "processed_news_ids"
                ]
            ),
            "missing_news_ids": (
                coverage_payload[
                    "missing_news_ids"
                ]
            ),
            "repair_attempted": (
                normalized_coverage
                .repair_attempted
            ),
            "repair_succeeded": (
                normalized_coverage
                .repair_succeeded
            ),
            "coverage": coverage_payload,
            "story_cluster_verification": (
                coverage_payload[
                    "story_cluster_verification"
                ]
            ),
            "audience_maxima": {
                "view_count": (
                    calculation
                    .audience_maxima
                    .max_view_count
                ),
                "comment_count": (
                    calculation
                    .audience_maxima
                    .max_comment_count
                ),
                "share_count": (
                    calculation
                    .audience_maxima
                    .max_share_count
                ),
            },
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
                    window_started_at,
                    window_finished_at,
                    candidate_count,
                    scored_count,
                    eligible_count,
                    parameters->>'run_mode'
                        AS run_mode,
                    parameters->>'evaluator_name'
                        AS evaluator_name,
                    parameters->>'evaluator_version'
                        AS evaluator_version,
                    parameters->'coverage'
                        AS stored_coverage,
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
                normalized_ranking_run_id,
                normalized_request_key,
                encoded_news_ids,
            )

            if record is None:
                raise LookupError(
                    "Зарезервированный event "
                    "ranking_run не найден: "
                    f"ranking_run_id="
                    f"{normalized_ranking_run_id}"
                )

            _validate_reserved_run(
                record,
                request_key=(
                    normalized_request_key
                ),
                metadata=normalized_metadata,
                candidate_news_ids=(
                    normalized_news_ids
                ),
                calculation=calculation,
            )

            if record["run_status"] == "failed":
                raise ValueError(
                    "Нельзя завершить ranking_run "
                    "со статусом failed."
                )

            if record["run_status"] == (
                "completed"
            ):
                stored_coverage = (
                    _decode_json_object(
                        record[
                            "stored_coverage"
                        ],
                        field_name=(
                            "parameters.coverage"
                        ),
                    )
                )

                if stored_coverage != coverage_payload:
                    raise ValueError(
                        "Повторный completion содержит "
                        "другую coverage-диагностику."
                    )

                (
                    persisted_events,
                    winner_combination_id,
                ) = (
                    await _load_and_verify_completed(
                        connection,
                        ranking_run_id=(
                            normalized_ranking_run_id
                        ),
                        calculation=calculation,
                    )
                )

                return (
                    EventRankingRunCompletionResult(
                        ranking_run_id=(
                            normalized_ranking_run_id
                        ),
                        request_key=(
                            normalized_request_key
                        ),
                        run_status="completed",
                        formula_version=(
                            FULL_FORMULA_VERSION
                        ),
                        candidate_count=int(
                            record[
                                "candidate_count"
                            ]
                        ),
                        scored_count=int(
                            record[
                                "scored_count"
                            ]
                        ),
                        eligible_count=int(
                            record[
                                "eligible_count"
                            ]
                        ),
                        combination_count=len(
                            calculation
                            .top3_selection
                            .combinations
                        ),
                        winner_combination_id=(
                            winner_combination_id
                        ),
                        already_completed=True,
                        persisted_events=(
                            persisted_events
                        ),
                        degraded=(
                            normalized_coverage
                            .degraded
                        ),
                        processed_candidate_count=(
                            len(
                                normalized_coverage
                                .processed_news_ids
                            )
                        ),
                        missing_news_ids=(
                            normalized_coverage
                            .missing_news_ids
                        ),
                    )
                )

            if record["run_status"] != "running":
                raise ValueError(
                    "Неподдерживаемый статус "
                    "ranking_run: "
                    f"{record['run_status']}"
                )

            existing_counts = (
                await connection.fetchrow(
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
                    normalized_ranking_run_id,
                )
            )

            if any(
                int(existing_counts[field_name])
                != 0
                for field_name in (
                    "event_count",
                    "score_count",
                    "combination_count",
                )
            ):
                raise RuntimeError(
                    "У running event ranking_run "
                    "уже есть сохранённые данные."
                )

            await _validate_news_items(
                connection,
                news_ids=normalized_news_ids,
            )

            winner_positions = (
                _winner_positions(
                    calculation
                )
            )

            (
                _persisted_events,
                score_ids_by_news_id,
            ) = await _insert_event_scores(
                connection,
                ranking_run_id=(
                    normalized_ranking_run_id
                ),
                request_key=(
                    normalized_request_key
                ),
                metadata=normalized_metadata,
                calculation=calculation,
                winner_positions=winner_positions,
            )

            for combination in (
                calculation
                .top3_selection
                .combinations
            ):
                combination_row = (
                    await connection.fetchrow(
                        """
                        INSERT INTO
                            top3_news.ranking_combinations (
                                ranking_run_id,
                                combination_key,
                                combination_rank,
                                mean_individual_score,
                                diversity_score,
                                mean_m_score,
                                mean_q_score,
                                mean_f_score,
                                distinct_macro_topic_count,
                                is_winner,
                                selection_reason,
                                combination_details
                            )
                        VALUES (
                            $1, $2, $3, $4, $5,
                            $6, $7, $8, $9, $10,
                            $11, $12::jsonb
                        )
                        RETURNING
                            combination_id,
                            final_top_score
                        """,
                        normalized_ranking_run_id,
                        _combination_key(
                            combination
                        ),
                        combination.combination_rank,
                        (
                            combination
                            .mean_individual_score
                        ),
                        combination.diversity_score,
                        combination.mean_m_score,
                        combination.mean_q_score,
                        combination.mean_f_score,
                        (
                            combination
                            .distinct_macro_topic_count
                        ),
                        combination.is_winner,
                        _selection_reason(
                            combination
                        ),
                        _encode_json(
                            _combination_details(
                                combination
                            )
                        ),
                    )
                )

                if (
                    combination_row[
                        "final_top_score"
                    ]
                    != combination.final_top_score
                ):
                    raise RuntimeError(
                        "TOP(S) PostgreSQL "
                        "не совпал с Python: "
                        "combination_rank="
                        f"{combination.combination_rank}"
                    )

                combination_id = int(
                    combination_row[
                        "combination_id"
                    ]
                )

                for position, news_id in enumerate(
                    combination.ordered_news_ids,
                    start=1,
                ):
                    await connection.execute(
                        """
                        INSERT INTO
                            top3_news.ranking_combination_items (
                                combination_id,
                                ranking_run_id,
                                score_id,
                                position
                            )
                        VALUES ($1, $2, $3, $4)
                        """,
                        combination_id,
                        normalized_ranking_run_id,
                        score_ids_by_news_id[
                            news_id
                        ],
                        position,
                    )

            update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.ranking_runs
                    SET
                        run_status = 'completed',
                        scored_count = $3,
                        eligible_count = $4,
                        parameters = (
                            parameters
                            || $5::jsonb
                        ),
                        error_message = NULL,
                        finished_at = now(),
                        updated_at = now()
                    WHERE ranking_run_id = $1
                      AND request_key = $2
                      AND run_status = 'running'
                    """,
                    normalized_ranking_run_id,
                    normalized_request_key,
                    len(
                        calculation
                        .calculated_events
                    ),
                    (
                        calculation
                        .top3_selection
                        .eligible_count
                    ),
                    telemetry_parameters,
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось завершить "
                    "event ranking_run: "
                    f"{update_result}"
                )

            (
                persisted_events,
                winner_combination_id,
            ) = await _load_and_verify_completed(
                connection,
                ranking_run_id=(
                    normalized_ranking_run_id
                ),
                calculation=calculation,
            )

    return EventRankingRunCompletionResult(
        ranking_run_id=(
            normalized_ranking_run_id
        ),
        request_key=normalized_request_key,
        run_status="completed",
        formula_version=FULL_FORMULA_VERSION,
        candidate_count=len(
            normalized_news_ids
        ),
        scored_count=len(
            calculation.calculated_events
        ),
        eligible_count=(
            calculation
            .top3_selection
            .eligible_count
        ),
        combination_count=len(
            calculation
            .top3_selection
            .combinations
        ),
        winner_combination_id=(
            winner_combination_id
        ),
        already_completed=False,
        persisted_events=persisted_events,
        degraded=normalized_coverage.degraded,
        processed_candidate_count=len(
            normalized_coverage.processed_news_ids
        ),
        missing_news_ids=(
            normalized_coverage.missing_news_ids
        ),
    )

async def fail_reserved_event_ranking_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    request_key: str,
    metadata: RankingEvaluatorMetadata,
    candidate_news_ids: tuple[int, ...],
    calculation: EventScoreCalculationResult,
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
    failure_stage: str,
    error_message: str,
    error_type: str | None = None,
) -> EventRankingRunDiagnosticFailureResult:
    """
    Атомарно сохраняет event-level диагностический сбой.

    Сохраняет рассчитанные инфоповоды и news_scores,
    usage, стоимость и этап сбоя. Комбинации TOP-3
    не создаются, а запуск переводится в failed.
    """

    normalized_ranking_run_id = (
        _positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
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

    normalized_news_ids = (
        _normalize_news_ids(
            candidate_news_ids
        )
    )

    _validate_scored_calculation(
        calculation=calculation,
        coverage_news_ids=(
            normalized_news_ids
        ),
    )

    _validate_telemetry(
        metadata=normalized_metadata,
        usage=usage,
        cost_estimate=cost_estimate,
    )

    normalized_failure_stage = (
        _normalize_required_text(
            failure_stage,
            field_name="failure_stage",
        )[:500]
    )

    normalized_error_message = (
        _normalize_required_text(
            error_message,
            field_name="error_message",
        )[:8000]
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

    encoded_news_ids = _encode_json(
        {
            "news_ids": list(
                normalized_news_ids
            )
        }
    )

    failure_parameters = _encode_json(
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
            "failure": {
                "error_type": (
                    normalized_error_type
                ),
                "error_message": (
                    normalized_error_message
                ),
                "stage": (
                    normalized_failure_stage
                ),
            },
            "failure_version": (
                DIAGNOSTIC_FAILURE_VERSION
            ),
            "formula_calculated_in_python": True,
            "diagnostic_scores_persisted": True,
            "event_count": len(
                calculation.calculated_events
            ),
            "combination_count": 0,
            "winner_news_ids": [],
            "audience_maxima": {
                "view_count": (
                    calculation
                    .audience_maxima
                    .max_view_count
                ),
                "comment_count": (
                    calculation
                    .audience_maxima
                    .max_comment_count
                ),
                "share_count": (
                    calculation
                    .audience_maxima
                    .max_share_count
                ),
            },
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
                    window_started_at,
                    window_finished_at,
                    candidate_count,
                    scored_count,
                    eligible_count,
                    error_message,
                    parameters->>'run_mode'
                        AS run_mode,
                    parameters->>'evaluator_name'
                        AS evaluator_name,
                    parameters->>'evaluator_version'
                        AS evaluator_version,
                    parameters->>'failure_version'
                        AS failure_version,
                    parameters->'failure'
                        AS failure,
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
                normalized_ranking_run_id,
                normalized_request_key,
                encoded_news_ids,
            )

            if record is None:
                raise LookupError(
                    "Зарезервированный event "
                    "ranking_run не найден: "
                    f"ranking_run_id="
                    f"{normalized_ranking_run_id}"
                )

            _validate_reserved_run(
                record,
                request_key=(
                    normalized_request_key
                ),
                metadata=normalized_metadata,
                candidate_news_ids=(
                    normalized_news_ids
                ),
                calculation=calculation,
            )

            if record["run_status"] == "completed":
                raise ValueError(
                    "Нельзя сохранить сбой для "
                    "completed ranking_run."
                )

            if record["run_status"] == "failed":
                existing_rows = await connection.fetch(
                    """
                    SELECT
                        e.ranking_event_id,
                        e.representative_news_id,
                        s.score_id
                    FROM top3_news.ranking_events AS e
                    JOIN top3_news.news_scores AS s
                      ON s.ranking_event_id
                         = e.ranking_event_id
                     AND s.ranking_run_id
                         = e.ranking_run_id
                    WHERE e.ranking_run_id = $1
                    ORDER BY e.representative_news_id
                    """,
                    normalized_ranking_run_id,
                )

                persisted_events = tuple(
                    PersistedEventScore(
                        ranking_event_id=int(
                            row["ranking_event_id"]
                        ),
                        representative_news_id=int(
                            row[
                                "representative_news_id"
                            ]
                        ),
                        score_id=int(
                            row["score_id"]
                        ),
                    )
                    for row in existing_rows
                )

                return (
                    EventRankingRunDiagnosticFailureResult(
                        ranking_run_id=(
                            normalized_ranking_run_id
                        ),
                        request_key=(
                            normalized_request_key
                        ),
                        run_status="failed",
                        formula_version=(
                            FULL_FORMULA_VERSION
                        ),
                        candidate_count=int(
                            record[
                                "candidate_count"
                            ]
                        ),
                        scored_count=int(
                            record["scored_count"]
                        ),
                        eligible_count=int(
                            record["eligible_count"]
                        ),
                        failure_stage=(
                            normalized_failure_stage
                        ),
                        already_failed=True,
                        error_message=(
                            record["error_message"]
                            or normalized_error_message
                        ),
                        persisted_events=(
                            persisted_events
                        ),
                    )
                )

            if record["run_status"] != "running":
                raise ValueError(
                    "Неподдерживаемый статус "
                    "ranking_run: "
                    f"{record['run_status']}"
                )

            existing_counts = (
                await connection.fetchrow(
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
                    normalized_ranking_run_id,
                )
            )

            if any(
                int(existing_counts[field_name])
                != 0
                for field_name in (
                    "event_count",
                    "score_count",
                    "combination_count",
                )
            ):
                raise RuntimeError(
                    "У running event ranking_run "
                    "уже есть сохранённые данные."
                )

            await _validate_news_items(
                connection,
                news_ids=normalized_news_ids,
            )

            (
                persisted_events,
                _score_ids_by_news_id,
            ) = await _insert_event_scores(
                connection,
                ranking_run_id=(
                    normalized_ranking_run_id
                ),
                request_key=(
                    normalized_request_key
                ),
                metadata=normalized_metadata,
                calculation=calculation,
                winner_positions={},
            )

            combination_count = (
                await connection.fetchval(
                    """
                    SELECT count(*)
                    FROM top3_news.ranking_combinations
                    WHERE ranking_run_id = $1
                    """,
                    normalized_ranking_run_id,
                )
            )

            if int(combination_count) != 0:
                raise RuntimeError(
                    "Диагностический сбой "
                    "не должен сохранять комбинации."
                )

            update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.ranking_runs
                    SET
                        run_status = 'failed',
                        scored_count = $3,
                        eligible_count = $4,
                        parameters = (
                            parameters
                            || $5::jsonb
                        ),
                        error_message = $6,
                        finished_at = now(),
                        updated_at = now()
                    WHERE ranking_run_id = $1
                      AND request_key = $2
                      AND run_status = 'running'
                    """,
                    normalized_ranking_run_id,
                    normalized_request_key,
                    len(
                        calculation
                        .calculated_events
                    ),
                    calculation.eligible_count,
                    failure_parameters,
                    normalized_error_message,
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось сохранить "
                    "event-level диагностический сбой: "
                    f"{update_result}"
                )

    return EventRankingRunDiagnosticFailureResult(
        ranking_run_id=(
            normalized_ranking_run_id
        ),
        request_key=normalized_request_key,
        run_status="failed",
        formula_version=FULL_FORMULA_VERSION,
        candidate_count=len(
            normalized_news_ids
        ),
        scored_count=len(
            calculation.calculated_events
        ),
        eligible_count=(
            calculation.eligible_count
        ),
        failure_stage=(
            normalized_failure_stage
        ),
        already_failed=False,
        error_message=(
            normalized_error_message
        ),
        persisted_events=(
            persisted_events
        ),
    )
