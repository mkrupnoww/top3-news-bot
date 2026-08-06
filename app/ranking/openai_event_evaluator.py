from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from app.prompt_loader import load_prompt
from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)
from app.ranking.event_evaluator import (
    EVENT_EVALUATOR_VERSION,
    EVENT_PROMPT_VERSION,
    MACRO_TOPICS,
    SOURCE_RELATIONS,
    STORY_CLUSTER_KEY_MAX_LENGTH,
    STORY_CLUSTER_KEY_PATTERN,
    EventAssessment,
    EventMemberAssessment,
    EventRankingCoverageDiagnostics,
    EventRankingEvaluationResult,
    EventRankingModelRequest,
    EventRankingModelResponse,
    StructuredEventRankingClient,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


class OpenAIEventMemberPayload(BaseModel):
    """Роль одной публикации внутри инфоповода."""

    model_config = ConfigDict(
        extra="forbid",
    )

    news_id: int = Field(gt=0)
    source_relation: str = Field(
        min_length=1,
        max_length=50,
    )
    is_representative: bool
    is_independent_source: bool
    counts_toward_reach: bool
    membership_reason: str = Field(
        min_length=1,
        max_length=1000,
    )

    @field_validator(
        "source_relation",
        "membership_reason",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: str,
    ) -> str:
        """Удаляет пробелы по краям."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "Текстовое поле не может "
                "быть пустым."
            )

        return normalized_value

    @field_validator("source_relation")
    @classmethod
    def validate_source_relation(
        cls,
        value: str,
    ) -> str:
        """Проверяет справочник роли источника."""

        if value not in SOURCE_RELATIONS:
            raise ValueError(
                "Неподдерживаемый "
                "source_relation."
            )

        return value


class OpenAIEventPayload(BaseModel):
    """Структурированное описание инфоповода."""

    model_config = ConfigDict(
        extra="forbid",
    )

    representative_news_id: int = Field(gt=0)
    event_title: str = Field(
        min_length=1,
        max_length=500,
    )
    event_time_utc: datetime
    macro_topic: str = Field(
        min_length=1,
        max_length=100,
    )
    story_cluster_key: str = Field(
        min_length=1,
        max_length=(
            STORY_CLUSTER_KEY_MAX_LENGTH
        ),
        pattern=STORY_CLUSTER_KEY_PATTERN,
    )
    i_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    k_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    n_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    e_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    x_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )
    q_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
    )
    impact_reason: str = Field(
        min_length=1,
        max_length=2000,
    )
    hook_reason: str = Field(
        min_length=1,
        max_length=2000,
    )
    q_reason: str = Field(
        min_length=1,
        max_length=2000,
    )
    members: list[
        OpenAIEventMemberPayload
    ] = Field(
        min_length=1,
        max_length=500,
    )

    @field_validator(
        "event_title",
        "impact_reason",
        "hook_reason",
        "q_reason",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: str,
    ) -> str:
        """Удаляет пробелы по краям."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "Текстовое поле не может "
                "быть пустым."
            )

        return normalized_value

    @field_validator("event_time_utc")
    @classmethod
    def validate_event_time(
        cls,
        value: datetime,
    ) -> datetime:
        """Требует часовой пояс и нормализует UTC."""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "event_time_utc должен "
                "содержать часовой пояс."
            )

        return value.astimezone(timezone.utc)

    @field_validator("story_cluster_key")
    @classmethod
    def validate_story_cluster_key(
        cls,
        value: str,
    ) -> str:
        """Нормализует ключ сюжетной семьи."""

        return value.strip().lower()

    @field_validator("macro_topic")
    @classmethod
    def validate_macro_topic(
        cls,
        value: str,
    ) -> str:
        """Проверяет справочник макротем."""

        normalized_value = value.strip()

        if normalized_value not in MACRO_TOPICS:
            raise ValueError(
                "Неподдерживаемый macro_topic."
            )

        return normalized_value


class OpenAIEventRankingPayload(BaseModel):
    """Полный структурированный ответ модели."""

    model_config = ConfigDict(
        extra="forbid",
    )

    events: list[
        OpenAIEventPayload
    ] = Field(
        min_length=1,
        max_length=500,
    )


@dataclass(frozen=True, slots=True)
class _ValidatedPayload:
    """Проверенный полный или частичный payload."""

    payload: OpenAIEventRankingPayload
    events: tuple[EventAssessment, ...]
    processed_news_ids: tuple[int, ...]
    missing_news_ids: tuple[int, ...]


SYSTEM_INSTRUCTIONS = load_prompt(
    "ranking/movie_news_event_ranking_prompt_v5.txt"
)

REPAIR_INSTRUCTIONS = (
    SYSTEM_INSTRUCTIONS
    + "\n\n"
    + "Это единственный корректирующий запрос. "
    + "Верни полный исправленный JSON по исходной "
    + "схеме events. Не возвращай только добавленный "
    + "фрагмент. Каждый expected_news_id должен "
    + "встречаться ровно один раз. Пропущенную "
    + "публикацию присоедини к существующему "
    + "инфоповоду, если это тот же факт, иначе создай "
    + "отдельный инфоповод. Не добавляй посторонние ID."
)


def _validate_selection(
    selection: CandidateSelectionResult,
) -> tuple[int, ...]:
    """Проверяет входной набор кандидатов."""

    if (
        selection.window_start.tzinfo is None
        or selection.window_start.utcoffset() is None
    ):
        raise ValueError(
            "window_start должен содержать "
            "часовой пояс."
        )

    if (
        selection.window_end.tzinfo is None
        or selection.window_end.utcoffset() is None
    ):
        raise ValueError(
            "window_end должен содержать "
            "часовой пояс."
        )

    if selection.window_end <= selection.window_start:
        raise ValueError(
            "window_end должен быть позже "
            "window_start."
        )

    if selection.window_hours <= 0:
        raise ValueError(
            "window_hours должен быть "
            "больше нуля."
        )

    if not selection.candidates:
        raise ValueError(
            "Список кандидатов "
            "не может быть пустым."
        )

    news_ids = tuple(
        candidate.news_id
        for candidate in selection.candidates
    )

    if any(
        isinstance(news_id, bool)
        or not isinstance(news_id, int)
        or news_id <= 0
        for news_id in news_ids
    ):
        raise ValueError(
            "Все news_id должны быть "
            "положительными int."
        )

    if len(set(news_ids)) != len(news_ids):
        raise ValueError(
            "Во входном наборе обнаружены "
            "повторяющиеся news_id."
        )

    for candidate in selection.candidates:
        _validate_candidate(
            candidate,
            selection=selection,
        )

    return news_ids


def _validate_candidate(
    candidate: NewsCandidate,
    *,
    selection: CandidateSelectionResult,
) -> None:
    """Проверяет одну публикацию-кандидат."""

    if (
        candidate.source_published_at.tzinfo is None
        or candidate.source_published_at.utcoffset()
        is None
    ):
        raise ValueError(
            "source_published_at должен "
            "содержать часовой пояс: "
            f"news_id={candidate.news_id}"
        )

    published_at = (
        candidate.source_published_at
        .astimezone(timezone.utc)
    )
    window_start = (
        selection.window_start
        .astimezone(timezone.utc)
    )
    window_end = (
        selection.window_end
        .astimezone(timezone.utc)
    )

    if not window_start <= published_at <= window_end:
        raise ValueError(
            "Кандидат находится вне "
            "временного окна: "
            f"news_id={candidate.news_id}"
        )

    if (
        candidate.age_hours < 0
        or candidate.age_hours
        > selection.window_hours + 0.001
    ):
        raise ValueError(
            "Некорректный age_hours: "
            f"news_id={candidate.news_id}, "
            f"age_hours={candidate.age_hours}"
        )

    if not candidate.title.strip():
        raise ValueError(
            "title не может быть пустым: "
            f"news_id={candidate.news_id}"
        )

    if not candidate.source_url.strip():
        raise ValueError(
            "source_url не может быть пустым: "
            f"news_id={candidate.news_id}"
        )

    source_weight = candidate.source_weight

    if source_weight is None:
        raise ValueError(
            "Для event-level ranking не настроен "
            "вес источника: "
            f"news_id={candidate.news_id}, "
            f"source_code={candidate.source_code!r}"
        )

    if (
        isinstance(source_weight, bool)
        or not isinstance(source_weight, int)
    ):
        raise ValueError(
            "Вес источника должен быть int: "
            f"news_id={candidate.news_id}, "
            f"source_weight={source_weight!r}"
        )

    if not 1 <= source_weight <= 3:
        raise ValueError(
            "Настроенный вес источника должен "
            "находиться в диапазоне 1..3: "
            f"news_id={candidate.news_id}, "
            f"source_weight={source_weight}"
        )


def _candidate_payload(
    candidate: NewsCandidate,
) -> dict[str, object]:
    """Сериализует кандидата для основного и repair-запроса."""

    return {
        "news_id": candidate.news_id,
        "source_id": candidate.source_id,
        "source_code": candidate.source_code,
        "source_name": candidate.source_name,
        "configured_source_weight": (
            candidate.source_weight
        ),
        "collection_priority": (
            candidate.collection_priority
        ),
        "title": candidate.title,
        "summary": candidate.summary,
        "author_name": candidate.author_name,
        "published_at": (
            candidate.source_published_at
            .astimezone(timezone.utc)
            .isoformat()
        ),
        "age_hours": round(
            candidate.age_hours,
            4,
        ),
        "source_url": candidate.source_url,
    }


def _build_input_text(
    selection: CandidateSelectionResult,
) -> str:
    """Формирует JSON-запрос для модели."""

    payload = {
        "task": (
            "group_and_assess_movie_news_events"
        ),
        "formula_version": FULL_FORMULA_VERSION,
        "expected_news_count": len(
            selection.candidates
        ),
        "expected_news_ids": [
            candidate.news_id
            for candidate in selection.candidates
        ],
        "window": {
            "started_at": (
                selection.window_start
                .astimezone(timezone.utc)
                .isoformat()
            ),
            "finished_at": (
                selection.window_end
                .astimezone(timezone.utc)
                .isoformat()
            ),
            "hours": selection.window_hours,
        },
        "expert_scores": {
            "i_score": "0..10",
            "k_score": "0..10",
            "n_score": "0..10",
            "e_score": "0..10",
            "x_score": "0..10",
            "q_score": "0..1",
        },
        "macro_topics": sorted(MACRO_TOPICS),
        "story_cluster_policy": {
            "format": "lower_snake_case",
            "maximum_length": (
                STORY_CLUSTER_KEY_MAX_LENGTH
            ),
            "purpose": (
                "group_distinct_events_from_the_"
                "same_overarching_story"
            ),
        },
        "source_relations": sorted(
            SOURCE_RELATIONS
        ),
        "source_weight_policy": {
            "configured_by": (
                "sources.settings.ranking."
                "source_weight"
            ),
            "model_must_not_return": True,
            "effective_weight_rules": {
                "counts_toward_reach_true": (
                    "use configured_source_weight"
                ),
                "otherwise": 0,
            },
        },
        "candidates": [
            _candidate_payload(candidate)
            for candidate in selection.candidates
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _build_repair_request(
    *,
    model_name: str,
    selection: CandidateSelectionResult,
    expected_news_ids: tuple[int, ...],
    missing_news_ids: tuple[int, ...],
    original_payload: OpenAIEventRankingPayload,
) -> EventRankingModelRequest:
    """Формирует единственный запрос исправления покрытия."""

    missing_set = set(missing_news_ids)

    payload = {
        "task": (
            "repair_movie_news_event_coverage"
        ),
        "formula_version": FULL_FORMULA_VERSION,
        "expected_news_count": len(
            expected_news_ids
        ),
        "expected_news_ids": list(
            expected_news_ids
        ),
        "missing_news_ids": list(
            missing_news_ids
        ),
        "window": {
            "started_at": (
                selection.window_start
                .astimezone(timezone.utc)
                .isoformat()
            ),
            "finished_at": (
                selection.window_end
                .astimezone(timezone.utc)
                .isoformat()
            ),
            "hours": selection.window_hours,
        },
        "original_response": (
            original_payload.model_dump(
                mode="json"
            )
        ),
        "missing_candidates": [
            _candidate_payload(candidate)
            for candidate in selection.candidates
            if candidate.news_id in missing_set
        ],
        "requirements": {
            "return_full_corrected_payload": True,
            "every_expected_news_id_exactly_once": True,
            "unexpected_news_ids_forbidden": True,
            "preserve_or_improve_valid_grouping": True,
            "attach_to_existing_event_when_same_fact": True,
            "otherwise_create_singleton_event": True,
        },
    }

    return EventRankingModelRequest(
        model=model_name,
        instructions=REPAIR_INSTRUCTIONS,
        input_text=json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _parse_response(
    response_text: str,
) -> OpenAIEventRankingPayload:
    """Проверяет JSON-ответ модели."""

    normalized_response = response_text.strip()

    if not normalized_response:
        raise ValueError(
            "Модель вернула пустой ответ."
        )

    try:
        return (
            OpenAIEventRankingPayload
            .model_validate_json(
                normalized_response
            )
        )
    except ValidationError as error:
        raise ValueError(
            "Ответ модели не соответствует "
            "event-level схеме рейтинга."
        ) from error


def _payload_coverage(
    *,
    expected_news_ids: tuple[int, ...],
    payload: OpenAIEventRankingPayload,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
]:
    """Проверяет идентификаторы и возвращает coverage."""

    response_news_ids = [
        member.news_id
        for event in payload.events
        for member in event.members
    ]

    duplicate_news_ids = sorted(
        {
            news_id
            for news_id in response_news_ids
            if response_news_ids.count(news_id) > 1
        }
    )

    if duplicate_news_ids:
        raise ValueError(
            "Модель распределила news_id "
            "по нескольким инфоповодам: "
            + ",".join(
                str(news_id)
                for news_id in duplicate_news_ids
            )
        )

    expected_set = set(expected_news_ids)
    response_set = set(response_news_ids)

    missing_news_ids = tuple(
        news_id
        for news_id in expected_news_ids
        if news_id not in response_set
    )
    unexpected_news_ids = sorted(
        response_set - expected_set
    )

    if unexpected_news_ids:
        raise ValueError(
            "Модель вернула некорректное "
            "распределение news_id: "
            f"missing={list(missing_news_ids)}, "
            f"unexpected={unexpected_news_ids}"
        )

    representative_news_ids = [
        event.representative_news_id
        for event in payload.events
    ]

    if (
        len(set(representative_news_ids))
        != len(representative_news_ids)
    ):
        raise ValueError(
            "Несколько инфоповодов используют "
            "одинаковый representative_news_id."
        )

    processed_news_ids = tuple(
        news_id
        for news_id in expected_news_ids
        if news_id in response_set
    )

    return (
        processed_news_ids,
        missing_news_ids,
    )


def _effective_source_weight(
    *,
    member: OpenAIEventMemberPayload,
    candidates_by_news_id: dict[int, NewsCandidate],
) -> int:
    """Назначает расчётный вес из конфигурации источника."""

    if not member.counts_toward_reach:
        return 0

    candidate = candidates_by_news_id[
        member.news_id
    ]
    source_weight = candidate.source_weight

    if source_weight is None:
        raise RuntimeError(
            "Вес источника неожиданно "
            "отсутствует после проверки: "
            f"news_id={member.news_id}"
        )

    return source_weight


def _build_events(
    *,
    payload: OpenAIEventRankingPayload,
    selection: CandidateSelectionResult,
) -> tuple[EventAssessment, ...]:
    """Преобразует payload, подставляя веса из БД."""

    candidates_by_news_id = {
        candidate.news_id: candidate
        for candidate in selection.candidates
    }

    return tuple(
        EventAssessment(
            representative_news_id=(
                event.representative_news_id
            ),
            event_title=event.event_title,
            event_time_utc=event.event_time_utc,
            macro_topic=event.macro_topic,
            story_cluster_key=(
                event.story_cluster_key
            ),
            i_score=event.i_score,
            k_score=event.k_score,
            n_score=event.n_score,
            e_score=event.e_score,
            x_score=event.x_score,
            q_score=event.q_score,
            impact_reason=event.impact_reason,
            hook_reason=event.hook_reason,
            q_reason=event.q_reason,
            members=tuple(
                EventMemberAssessment(
                    news_id=member.news_id,
                    source_relation=(
                        member.source_relation
                    ),
                    is_representative=(
                        member.is_representative
                    ),
                    is_independent_source=(
                        member.is_independent_source
                    ),
                    counts_toward_reach=(
                        member.counts_toward_reach
                    ),
                    source_weight=(
                        _effective_source_weight(
                            member=member,
                            candidates_by_news_id=(
                                candidates_by_news_id
                            ),
                        )
                    ),
                    membership_reason=(
                        member.membership_reason
                    ),
                )
                for member in event.members
            ),
        )
        for event in payload.events
    )


def _validate_event_coverage(
    *,
    expected_news_ids: tuple[int, ...],
    events: tuple[EventAssessment, ...],
) -> None:
    """Требует покрытие переданного обработанного набора."""

    response_news_ids = [
        news_id
        for event in events
        for news_id in event.member_news_ids
    ]

    duplicate_news_ids = sorted(
        {
            news_id
            for news_id in response_news_ids
            if response_news_ids.count(news_id) > 1
        }
    )

    if duplicate_news_ids:
        raise ValueError(
            "Модель распределила news_id "
            "по нескольким инфоповодам: "
            + ",".join(
                str(news_id)
                for news_id in duplicate_news_ids
            )
        )

    expected_set = set(expected_news_ids)
    response_set = set(response_news_ids)
    missing_news_ids = sorted(
        expected_set - response_set
    )
    unexpected_news_ids = sorted(
        response_set - expected_set
    )

    if missing_news_ids or unexpected_news_ids:
        raise ValueError(
            "Модель вернула некорректное "
            "распределение news_id: "
            f"missing={missing_news_ids}, "
            f"unexpected={unexpected_news_ids}"
        )


def _validate_event_times(
    *,
    selection: CandidateSelectionResult,
    events: tuple[EventAssessment, ...],
) -> None:
    """Проверяет попадание времени события в окно."""

    window_start = (
        selection.window_start
        .astimezone(timezone.utc)
    )
    window_end = (
        selection.window_end
        .astimezone(timezone.utc)
    )

    invalid_events = [
        event
        for event in events
        if not (
            window_start
            <= event.event_time_utc
            <= window_end
        )
    ]

    if invalid_events:
        details = ",".join(
            str(event.representative_news_id)
            for event in invalid_events
        )
        raise ValueError(
            "event_time_utc находится вне "
            "окна для representative_news_id: "
            f"{details}"
        )


def _sort_events_by_input_order(
    *,
    selection: CandidateSelectionResult,
    events: tuple[EventAssessment, ...],
) -> tuple[EventAssessment, ...]:
    """Делает порядок инфоповодов детерминированным."""

    input_position = {
        candidate.news_id: position
        for position, candidate
        in enumerate(selection.candidates)
    }

    return tuple(
        sorted(
            events,
            key=lambda event: (
                min(
                    input_position[news_id]
                    for news_id
                    in event.member_news_ids
                ),
                event.representative_news_id,
            ),
        )
    )


def _validate_payload(
    *,
    response: EventRankingModelResponse,
    selection: CandidateSelectionResult,
    expected_news_ids: tuple[int, ...],
) -> _ValidatedPayload:
    """Разбирает и полностью проверяет один ответ."""

    payload = _parse_response(
        response.output_text
    )
    (
        processed_news_ids,
        missing_news_ids,
    ) = _payload_coverage(
        expected_news_ids=expected_news_ids,
        payload=payload,
    )
    events = _build_events(
        payload=payload,
        selection=selection,
    )
    _validate_event_coverage(
        expected_news_ids=processed_news_ids,
        events=events,
    )
    _validate_event_times(
        selection=selection,
        events=events,
    )

    return _ValidatedPayload(
        payload=payload,
        events=_sort_events_by_input_order(
            selection=selection,
            events=events,
        ),
        processed_news_ids=processed_news_ids,
        missing_news_ids=missing_news_ids,
    )


def _combine_usage(
    responses: tuple[
        EventRankingModelResponse,
        ...,
    ],
) -> OpenAITokenUsage | None:
    """Суммирует usage успешных OpenAI-ответов."""

    usages = tuple(
        response.usage
        for response in responses
    )

    if all(usage is None for usage in usages):
        return None

    if any(usage is None for usage in usages):
        raise ValueError(
            "Не все ответы модели содержат usage."
        )

    typed_usages = tuple(
        usage
        for usage in usages
        if usage is not None
    )

    return OpenAITokenUsage(
        input_tokens=sum(
            usage.input_tokens
            for usage in typed_usages
        ),
        cached_input_tokens=sum(
            usage.cached_input_tokens
            for usage in typed_usages
        ),
        cache_write_tokens=sum(
            usage.cache_write_tokens
            for usage in typed_usages
        ),
        output_tokens=sum(
            usage.output_tokens
            for usage in typed_usages
        ),
        reasoning_tokens=sum(
            usage.reasoning_tokens
            for usage in typed_usages
        ),
        total_tokens=sum(
            usage.total_tokens
            for usage in typed_usages
        ),
    )


def _combine_cost(
    responses: tuple[
        EventRankingModelResponse,
        ...,
    ],
) -> OpenAICostEstimate | None:
    """Суммирует стоимость успешных OpenAI-ответов."""

    costs = tuple(
        response.cost_estimate
        for response in responses
    )

    if all(cost is None for cost in costs):
        return None

    if any(cost is None for cost in costs):
        raise ValueError(
            "Не все ответы модели содержат "
            "расчёт стоимости."
        )

    typed_costs = tuple(
        cost
        for cost in costs
        if cost is not None
    )
    first = typed_costs[0]

    if any(
        cost.model_name != first.model_name
        or cost.pricing_version
        != first.pricing_version
        for cost in typed_costs[1:]
    ):
        raise ValueError(
            "Нельзя суммировать стоимость "
            "разных моделей или тарифов."
        )

    return OpenAICostEstimate(
        model_name=first.model_name,
        pricing_version=first.pricing_version,
        regular_input_cost_usd=sum(
            (
                cost.regular_input_cost_usd
                for cost in typed_costs
            ),
            Decimal("0"),
        ),
        cached_input_cost_usd=sum(
            (
                cost.cached_input_cost_usd
                for cost in typed_costs
            ),
            Decimal("0"),
        ),
        cache_write_cost_usd=sum(
            (
                cost.cache_write_cost_usd
                for cost in typed_costs
            ),
            Decimal("0"),
        ),
        output_cost_usd=sum(
            (
                cost.output_cost_usd
                for cost in typed_costs
            ),
            Decimal("0"),
        ),
        total_cost_usd=sum(
            (
                cost.total_cost_usd
                for cost in typed_costs
            ),
            Decimal("0"),
        ),
    )


def _aggregate_response(
    *,
    chosen_response: EventRankingModelResponse,
    successful_responses: tuple[
        EventRankingModelResponse,
        ...,
    ],
) -> EventRankingModelResponse:
    """Возвращает выбранный JSON с общей телеметрией."""

    return EventRankingModelResponse(
        output_text=chosen_response.output_text,
        usage=_combine_usage(
            successful_responses
        ),
        cost_estimate=_combine_cost(
            successful_responses
        ),
    )


class OpenAIEventRankingEvaluator:
    """Event-level оценщик полной формулы v3."""

    def __init__(
        self,
        *,
        client: StructuredEventRankingClient,
        model_name: str,
    ) -> None:
        normalized_model_name = model_name.strip()

        if not normalized_model_name:
            raise ValueError(
                "model_name не может быть пустым."
            )

        self._client = client
        self._metadata = RankingEvaluatorMetadata(
            run_mode="openai_event_ranking",
            evaluator_name=(
                "OpenAIEventRankingEvaluator"
            ),
            evaluator_version=(
                EVENT_EVALUATOR_VERSION
            ),
            prompt_version=EVENT_PROMPT_VERSION,
            model_name=normalized_model_name,
        )

    @property
    def metadata(self) -> RankingEvaluatorMetadata:
        """Возвращает метаданные оценщика."""

        return self._metadata

    def build_request(
        self,
        selection: CandidateSelectionResult,
    ) -> EventRankingModelRequest:
        """Формирует запрос без вызова модели."""

        _validate_selection(selection)
        model_name = self._metadata.model_name

        if model_name is None:
            raise RuntimeError(
                "В метаданных отсутствует "
                "model_name."
            )

        return EventRankingModelRequest(
            model=model_name,
            instructions=SYSTEM_INSTRUCTIONS,
            input_text=_build_input_text(
                selection
            ),
        )

    async def evaluate_prepared_request(
        self,
        selection: CandidateSelectionResult,
        request: EventRankingModelRequest,
    ) -> EventRankingEvaluationResult:
        """Выполняет основной и максимум один repair-запрос."""

        expected_news_ids = _validate_selection(
            selection
        )
        expected_request = self.build_request(
            selection
        )

        if request != expected_request:
            raise ValueError(
                "Подготовленный event-level "
                "запрос не соответствует "
                "текущей выборке, модели "
                "или промпту."
            )

        primary_response = (
            await self._client.create_response(
                request
            )
        )
        primary = _validate_payload(
            response=primary_response,
            selection=selection,
            expected_news_ids=(
                expected_news_ids
            ),
        )

        successful_responses: list[
            EventRankingModelResponse
        ] = [primary_response]
        chosen_response = primary_response
        chosen = primary
        repair_attempted = False
        repair_succeeded = False
        repair_error_type: str | None = None
        repair_error_message: str | None = None

        if primary.missing_news_ids:
            repair_attempted = True
            model_name = self._metadata.model_name

            if model_name is None:
                raise RuntimeError(
                    "В метаданных отсутствует "
                    "model_name."
                )

            repair_request = _build_repair_request(
                model_name=model_name,
                selection=selection,
                expected_news_ids=(
                    expected_news_ids
                ),
                missing_news_ids=(
                    primary.missing_news_ids
                ),
                original_payload=(
                    primary.payload
                ),
            )

            try:
                repair_response = (
                    await self._client.create_response(
                        repair_request
                    )
                )
                successful_responses.append(
                    repair_response
                )
                repaired = _validate_payload(
                    response=repair_response,
                    selection=selection,
                    expected_news_ids=(
                        expected_news_ids
                    ),
                )

                if (
                    len(repaired.missing_news_ids)
                    < len(primary.missing_news_ids)
                ):
                    chosen = repaired
                    chosen_response = (
                        repair_response
                    )

                if not repaired.missing_news_ids:
                    chosen = repaired
                    chosen_response = (
                        repair_response
                    )
                    repair_succeeded = True

            except Exception as error:
                repair_error_type = (
                    type(error).__name__
                )
                repair_error_message = str(error)

        final_responses = tuple(
            successful_responses
        )
        diagnostics = (
            EventRankingCoverageDiagnostics(
                expected_news_ids=(
                    expected_news_ids
                ),
                processed_news_ids=(
                    chosen.processed_news_ids
                ),
                initial_missing_news_ids=(
                    primary.missing_news_ids
                ),
                missing_news_ids=(
                    chosen.missing_news_ids
                ),
                repair_attempted=(
                    repair_attempted
                ),
                repair_succeeded=(
                    repair_succeeded
                ),
                repair_error_type=(
                    repair_error_type
                ),
                repair_error_message=(
                    repair_error_message
                ),
                model_call_count=(
                    2 if repair_attempted else 1
                ),
            )
        )

        return EventRankingEvaluationResult(
            events=chosen.events,
            model_response=_aggregate_response(
                chosen_response=chosen_response,
                successful_responses=(
                    final_responses
                ),
            ),
            diagnostics=diagnostics,
            model_responses=final_responses,
        )

    async def evaluate_detailed(
        self,
        selection: CandidateSelectionResult,
    ) -> EventRankingEvaluationResult:
        """Оценивает инфоповоды с телеметрией."""

        request = self.build_request(selection)

        return await self.evaluate_prepared_request(
            selection,
            request,
        )

    async def evaluate(
        self,
        selection: CandidateSelectionResult,
    ) -> tuple[EventAssessment, ...]:
        """Возвращает только инфоповоды."""

        result = await self.evaluate_detailed(
            selection
        )

        return result.events
