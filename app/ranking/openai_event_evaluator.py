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
from app.ranking.event_evaluator import (
    EVENT_EVALUATOR_VERSION,
    EVENT_PROMPT_VERSION,
    MACRO_TOPICS,
    SOURCE_RELATIONS,
    EventAssessment,
    EventMemberAssessment,
    EventRankingEvaluationResult,
    EventRankingModelRequest,
    StructuredEventRankingClient,
)


class OpenAIEventMemberPayload(BaseModel):
    """Роль одной публикации внутри инфоповода."""

    model_config = ConfigDict(
        extra="forbid",
    )

    news_id: int = Field(
        gt=0,
    )

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

    representative_news_id: int = Field(
        gt=0,
    )

    event_title: str = Field(
        min_length=1,
        max_length=500,
    )

    event_time_utc: datetime

    macro_topic: str = Field(
        min_length=1,
        max_length=100,
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

        return value.astimezone(
            timezone.utc
        )

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


SYSTEM_INSTRUCTIONS = load_prompt(
    "ranking/movie_news_event_ranking_prompt_v2.txt"
)


def _validate_selection(
    selection: CandidateSelectionResult,
) -> tuple[int, ...]:
    """Проверяет входной набор кандидатов."""

    if (
        selection.window_start.tzinfo is None
        or selection.window_start.utcoffset()
        is None
    ):
        raise ValueError(
            "window_start должен содержать "
            "часовой пояс."
        )

    if (
        selection.window_end.tzinfo is None
        or selection.window_end.utcoffset()
        is None
    ):
        raise ValueError(
            "window_end должен содержать "
            "часовой пояс."
        )

    if (
        selection.window_end
        <= selection.window_start
    ):
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
        candidate.source_published_at.tzinfo
        is None
        or candidate
        .source_published_at
        .utcoffset()
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

    if not (
        window_start
        <= published_at
        <= window_end
    ):
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
            "Для event-level v2 не настроен "
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


def _build_input_text(
    selection: CandidateSelectionResult,
) -> str:
    """Формирует JSON-запрос для модели."""

    payload = {
        "task": (
            "group_and_assess_movie_news_events"
        ),
        "formula_version": "top3_cinema_v2",
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
        "macro_topics": sorted(
            MACRO_TOPICS
        ),
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
            {
                "news_id": candidate.news_id,
                "source_id": candidate.source_id,
                "source_code": (
                    candidate.source_code
                ),
                "source_name": (
                    candidate.source_name
                ),
                "configured_source_weight": (
                    candidate.source_weight
                ),
                "collection_priority": (
                    candidate.collection_priority
                ),
                "title": candidate.title,
                "summary": candidate.summary,
                "author_name": (
                    candidate.author_name
                ),
                "published_at": (
                    candidate
                    .source_published_at
                    .astimezone(timezone.utc)
                    .isoformat()
                ),
                "age_hours": round(
                    candidate.age_hours,
                    4,
                ),
                "source_url": (
                    candidate.source_url
                ),
            }
            for candidate
            in selection.candidates
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_response(
    response_text: str,
) -> OpenAIEventRankingPayload:
    """Проверяет JSON-ответ модели."""

    normalized_response = (
        response_text.strip()
    )

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


def _validate_payload_coverage(
    *,
    expected_news_ids: tuple[int, ...],
    payload: OpenAIEventRankingPayload,
) -> None:
    """Проверяет raw payload до построения доменных типов."""

    response_news_ids = [
        member.news_id
        for event in payload.events
        for member in event.members
    ]

    duplicate_news_ids = sorted(
        {
            news_id
            for news_id in response_news_ids
            if response_news_ids.count(
                news_id
            ) > 1
        }
    )

    if duplicate_news_ids:
        raise ValueError(
            "Модель распределила news_id "
            "по нескольким инфоповодам: "
            + ",".join(
                str(news_id)
                for news_id
                in duplicate_news_ids
            )
        )

    expected_set = set(
        expected_news_ids
    )

    response_set = set(
        response_news_ids
    )

    missing_news_ids = sorted(
        expected_set - response_set
    )

    unexpected_news_ids = sorted(
        response_set - expected_set
    )

    if (
        missing_news_ids
        or unexpected_news_ids
    ):
        raise ValueError(
            "Модель вернула некорректное "
            "распределение news_id: "
            f"missing={missing_news_ids}, "
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


def _effective_source_weight(
    *,
    member: OpenAIEventMemberPayload,
    candidates_by_news_id: dict[
        int,
        NewsCandidate,
    ],
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
) -> tuple[
    EventAssessment,
    ...,
]:
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
            event_time_utc=(
                event.event_time_utc
            ),
            macro_topic=event.macro_topic,
            i_score=event.i_score,
            k_score=event.k_score,
            n_score=event.n_score,
            e_score=event.e_score,
            x_score=event.x_score,
            q_score=event.q_score,
            impact_reason=(
                event.impact_reason
            ),
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
                        member
                        .is_independent_source
                    ),
                    counts_toward_reach=(
                        member
                        .counts_toward_reach
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
                        member
                        .membership_reason
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
    events: tuple[
        EventAssessment,
        ...,
    ],
) -> None:
    """Требует распределить каждый news_id ровно раз."""

    response_news_ids = [
        news_id
        for event in events
        for news_id in event.member_news_ids
    ]

    duplicate_news_ids = sorted(
        {
            news_id
            for news_id in response_news_ids
            if response_news_ids.count(
                news_id
            ) > 1
        }
    )

    if duplicate_news_ids:
        raise ValueError(
            "Модель распределила news_id "
            "по нескольким инфоповодам: "
            + ",".join(
                str(news_id)
                for news_id
                in duplicate_news_ids
            )
        )

    expected_set = set(
        expected_news_ids
    )

    response_set = set(
        response_news_ids
    )

    missing_news_ids = sorted(
        expected_set - response_set
    )

    unexpected_news_ids = sorted(
        response_set - expected_set
    )

    if (
        missing_news_ids
        or unexpected_news_ids
    ):
        raise ValueError(
            "Модель вернула некорректное "
            "распределение news_id: "
            f"missing={missing_news_ids}, "
            f"unexpected={unexpected_news_ids}"
        )

    representative_news_ids = [
        event.representative_news_id
        for event in events
    ]

    if (
        len(set(representative_news_ids))
        != len(representative_news_ids)
    ):
        raise ValueError(
            "Несколько инфоповодов используют "
            "одинаковый representative_news_id."
        )


def _validate_event_times(
    *,
    selection: CandidateSelectionResult,
    events: tuple[
        EventAssessment,
        ...,
    ],
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
    events: tuple[
        EventAssessment,
        ...,
    ],
) -> tuple[
    EventAssessment,
        ...,
]:
    """Делает порядок инфоповодов детерминированным."""

    input_position = {
        candidate.news_id: position
        for position, candidate
        in enumerate(
            selection.candidates
        )
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


class OpenAIEventRankingEvaluator:
    """Event-level оценщик полной формулы v2."""

    def __init__(
        self,
        *,
        client: StructuredEventRankingClient,
        model_name: str,
    ) -> None:
        normalized_model_name = (
            model_name.strip()
        )

        if not normalized_model_name:
            raise ValueError(
                "model_name не может быть пустым."
            )

        self._client = client
        self._metadata = (
            RankingEvaluatorMetadata(
                run_mode=(
                    "openai_event_ranking"
                ),
                evaluator_name=(
                    "OpenAIEventRankingEvaluator"
                ),
                evaluator_version=(
                    EVENT_EVALUATOR_VERSION
                ),
                prompt_version=(
                    EVENT_PROMPT_VERSION
                ),
                model_name=(
                    normalized_model_name
                ),
            )
        )

    @property
    def metadata(
        self,
    ) -> RankingEvaluatorMetadata:
        """Возвращает метаданные оценщика."""

        return self._metadata

    def build_request(
        self,
        selection: CandidateSelectionResult,
    ) -> EventRankingModelRequest:
        """Формирует запрос без вызова модели."""

        _validate_selection(
            selection
        )

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
        """Выполняет заранее сформированный запрос."""

        expected_news_ids = (
            _validate_selection(
                selection
            )
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

        model_response = (
            await self._client.create_response(
                request
            )
        )

        payload = _parse_response(
            model_response.output_text
        )

        _validate_payload_coverage(
            expected_news_ids=(
                expected_news_ids
            ),
            payload=payload,
        )

        events = _build_events(
            payload=payload,
            selection=selection,
        )

        _validate_event_coverage(
            expected_news_ids=(
                expected_news_ids
            ),
            events=events,
        )

        _validate_event_times(
            selection=selection,
            events=events,
        )

        ordered_events = (
            _sort_events_by_input_order(
                selection=selection,
                events=events,
            )
        )

        return EventRankingEvaluationResult(
            events=ordered_events,
            model_response=model_response,
        )

    async def evaluate_detailed(
        self,
        selection: CandidateSelectionResult,
    ) -> EventRankingEvaluationResult:
        """Оценивает инфоповоды с телеметрией."""

        request = self.build_request(
            selection
        )

        return await self.evaluate_prepared_request(
            selection,
            request,
        )

    async def evaluate(
        self,
        selection: CandidateSelectionResult,
    ) -> tuple[
        EventAssessment,
        ...,
    ]:
        """Возвращает только инфоповоды."""

        result = await self.evaluate_detailed(
            selection
        )

        return result.events
