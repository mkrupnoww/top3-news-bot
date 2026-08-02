from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from app.db.news_candidates import CandidateSelectionResult
from app.ranking.evaluator import RankingEvaluatorMetadata
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


EVENT_EVALUATOR_VERSION = (
    "event_ranking_evaluator_v1"
)

EVENT_PROMPT_VERSION = (
    "movie_news_event_ranking_prompt_v1"
)

MACRO_TOPICS = frozenset(
    {
        "business_economy_law",
        "people_conflicts_legal",
        "creative_cast_production",
        "trailers_premieres_releases",
        "festivals_awards_criticism",
        "box_office_audience_distribution",
        "other",
    }
)

SOURCE_RELATIONS = frozenset(
    {
        "primary",
        "independent",
        "syndicated",
        "duplicate",
    }
)


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательное текстовое поле."""

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


def _normalize_positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительный целый идентификатор."""

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


def _normalize_datetime_utc(
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
            f"{field_name} должен содержать часовой пояс."
        )

    return value.astimezone(timezone.utc)


def _normalize_decimal(
    value: Decimal | int | float | str,
    *,
    field_name: str,
    minimum: Decimal,
    maximum: Decimal,
) -> Decimal:
    """Преобразует число в Decimal и проверяет диапазон."""

    try:
        normalized_value = Decimal(
            str(value)
        )
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{field_name} должен быть числом."
        ) from error

    if not normalized_value.is_finite():
        raise ValueError(
            f"{field_name} должен быть конечным числом."
        )

    if (
        normalized_value < minimum
        or normalized_value > maximum
    ):
        raise ValueError(
            f"{field_name} должен находиться "
            f"в диапазоне от {minimum} до {maximum}: "
            f"value={normalized_value}"
        )

    return normalized_value


@dataclass(frozen=True, slots=True)
class EventMemberAssessment:
    """Роль одной публикации внутри инфоповода."""

    news_id: int
    source_relation: str
    is_representative: bool
    is_independent_source: bool
    counts_toward_reach: bool
    source_weight: int
    membership_reason: str

    def __post_init__(self) -> None:
        """Проверяет согласованность роли источника."""

        normalized_news_id = (
            _normalize_positive_integer(
                self.news_id,
                field_name="news_id",
            )
        )

        normalized_relation = (
            _normalize_required_text(
                self.source_relation,
                field_name="source_relation",
            )
        )

        normalized_reason = (
            _normalize_required_text(
                self.membership_reason,
                field_name="membership_reason",
            )
        )

        if (
            normalized_relation
            not in SOURCE_RELATIONS
        ):
            raise ValueError(
                "source_relation имеет "
                "неподдерживаемое значение: "
                f"{normalized_relation!r}"
            )

        for field_name, value in (
            (
                "is_representative",
                self.is_representative,
            ),
            (
                "is_independent_source",
                self.is_independent_source,
            ),
            (
                "counts_toward_reach",
                self.counts_toward_reach,
            ),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} должен быть bool."
                )

        if isinstance(self.source_weight, bool):
            raise TypeError(
                "source_weight не может быть bool."
            )

        if not isinstance(
            self.source_weight,
            int,
        ):
            raise TypeError(
                "source_weight должен быть int."
            )

        if not 0 <= self.source_weight <= 3:
            raise ValueError(
                "source_weight должен находиться "
                "в диапазоне 0..3."
            )

        if self.counts_toward_reach:
            if not self.is_independent_source:
                raise ValueError(
                    "Публикация, участвующая "
                    "в расчёте охвата, должна быть "
                    "независимым источником."
                )

            if self.source_weight <= 0:
                raise ValueError(
                    "Публикация, участвующая "
                    "в расчёте охвата, должна иметь "
                    "source_weight больше нуля."
                )

            if normalized_relation not in {
                "primary",
                "independent",
            }:
                raise ValueError(
                    "В расчёте охвата могут участвовать "
                    "только primary или independent "
                    "источники."
                )

        if normalized_relation in {
            "syndicated",
            "duplicate",
        }:
            if self.counts_toward_reach:
                raise ValueError(
                    "Перепечатка или дубль не может "
                    "участвовать в расчёте охвата."
                )

            if self.source_weight != 0:
                raise ValueError(
                    "Перепечатка или дубль должны иметь "
                    "source_weight=0."
                )

        object.__setattr__(
            self,
            "news_id",
            normalized_news_id,
        )

        object.__setattr__(
            self,
            "source_relation",
            normalized_relation,
        )

        object.__setattr__(
            self,
            "membership_reason",
            normalized_reason,
        )


@dataclass(frozen=True, slots=True)
class EventAssessment:
    """Экспертная модель одного инфоповода."""

    representative_news_id: int
    event_title: str
    event_time_utc: datetime
    macro_topic: str

    i_score: Decimal
    k_score: Decimal
    n_score: Decimal
    e_score: Decimal
    x_score: Decimal
    q_score: Decimal

    impact_reason: str
    hook_reason: str
    q_reason: str

    members: tuple[
        EventMemberAssessment,
        ...,
    ]

    def __post_init__(self) -> None:
        """Проверяет инфоповод и его участников."""

        representative_news_id = (
            _normalize_positive_integer(
                self.representative_news_id,
                field_name=(
                    "representative_news_id"
                ),
            )
        )

        event_title = _normalize_required_text(
            self.event_title,
            field_name="event_title",
        )

        event_time_utc = (
            _normalize_datetime_utc(
                self.event_time_utc,
                field_name="event_time_utc",
            )
        )

        macro_topic = _normalize_required_text(
            self.macro_topic,
            field_name="macro_topic",
        )

        if macro_topic not in MACRO_TOPICS:
            raise ValueError(
                "macro_topic имеет "
                "неподдерживаемое значение: "
                f"{macro_topic!r}"
            )

        normalized_scores = {
            "i_score": _normalize_decimal(
                self.i_score,
                field_name="i_score",
                minimum=Decimal("0"),
                maximum=Decimal("10"),
            ),
            "k_score": _normalize_decimal(
                self.k_score,
                field_name="k_score",
                minimum=Decimal("0"),
                maximum=Decimal("10"),
            ),
            "n_score": _normalize_decimal(
                self.n_score,
                field_name="n_score",
                minimum=Decimal("0"),
                maximum=Decimal("10"),
            ),
            "e_score": _normalize_decimal(
                self.e_score,
                field_name="e_score",
                minimum=Decimal("0"),
                maximum=Decimal("10"),
            ),
            "x_score": _normalize_decimal(
                self.x_score,
                field_name="x_score",
                minimum=Decimal("0"),
                maximum=Decimal("10"),
            ),
            "q_score": _normalize_decimal(
                self.q_score,
                field_name="q_score",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
        }

        impact_reason = (
            _normalize_required_text(
                self.impact_reason,
                field_name="impact_reason",
            )
        )

        hook_reason = (
            _normalize_required_text(
                self.hook_reason,
                field_name="hook_reason",
            )
        )

        q_reason = _normalize_required_text(
            self.q_reason,
            field_name="q_reason",
        )

        if not isinstance(self.members, tuple):
            raise TypeError(
                "members должен быть tuple."
            )

        if not self.members:
            raise ValueError(
                "Инфоповод должен содержать "
                "хотя бы одну публикацию."
            )

        member_news_ids = tuple(
            member.news_id
            for member in self.members
        )

        if (
            len(set(member_news_ids))
            != len(member_news_ids)
        ):
            raise ValueError(
                "members содержит повторяющиеся "
                "news_id."
            )

        if (
            representative_news_id
            not in member_news_ids
        ):
            raise ValueError(
                "representative_news_id должен "
                "входить в members."
            )

        representative_members = tuple(
            member
            for member in self.members
            if member.is_representative
        )

        if len(representative_members) != 1:
            raise ValueError(
                "Инфоповод должен иметь ровно "
                "одного is_representative=true."
            )

        if (
            representative_members[0].news_id
            != representative_news_id
        ):
            raise ValueError(
                "is_representative=true должен "
                "соответствовать "
                "representative_news_id."
            )

        object.__setattr__(
            self,
            "representative_news_id",
            representative_news_id,
        )

        object.__setattr__(
            self,
            "event_title",
            event_title,
        )

        object.__setattr__(
            self,
            "event_time_utc",
            event_time_utc,
        )

        object.__setattr__(
            self,
            "macro_topic",
            macro_topic,
        )

        for field_name, value in (
            normalized_scores.items()
        ):
            object.__setattr__(
                self,
                field_name,
                value,
            )

        object.__setattr__(
            self,
            "impact_reason",
            impact_reason,
        )

        object.__setattr__(
            self,
            "hook_reason",
            hook_reason,
        )

        object.__setattr__(
            self,
            "q_reason",
            q_reason,
        )

    @property
    def member_news_ids(
        self,
    ) -> tuple[int, ...]:
        """Возвращает news_id публикаций инфоповода."""

        return tuple(
            member.news_id
            for member in self.members
        )

    @property
    def source_weight_sum(
        self,
    ) -> int:
        """Суммирует веса независимых источников."""

        return sum(
            member.source_weight
            for member in self.members
            if member.counts_toward_reach
        )


@dataclass(frozen=True, slots=True)
class EventRankingModelRequest:
    """Подготовленный запрос event-level оценщика."""

    model: str
    instructions: str
    input_text: str


@dataclass(frozen=True, slots=True)
class EventRankingModelResponse:
    """Ответ event-level модели и телеметрия."""

    output_text: str
    usage: OpenAITokenUsage | None = None
    cost_estimate: OpenAICostEstimate | None = None


@dataclass(frozen=True, slots=True)
class EventRankingEvaluationResult:
    """Проверенные инфоповоды и ответ модели."""

    events: tuple[
        EventAssessment,
        ...,
    ]
    model_response: EventRankingModelResponse


@runtime_checkable
class StructuredEventRankingClient(
    Protocol
):
    """Транспортный интерфейс event-level модели."""

    async def create_response(
        self,
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Возвращает структурированный ответ."""

        ...


@runtime_checkable
class EventRankingEvaluator(
    Protocol
):
    """Интерфейс оценщика полной формулы v2."""

    @property
    def metadata(
        self,
    ) -> RankingEvaluatorMetadata:
        """Возвращает метаданные оценщика."""

        ...

    def build_request(
        self,
        selection: CandidateSelectionResult,
    ) -> EventRankingModelRequest:
        """Формирует запрос без обращения к модели."""

        ...

    async def evaluate_prepared_request(
        self,
        selection: CandidateSelectionResult,
        request: EventRankingModelRequest,
    ) -> EventRankingEvaluationResult:
        """Выполняет заранее сформированный запрос."""

        ...

    async def evaluate_detailed(
        self,
        selection: CandidateSelectionResult,
    ) -> EventRankingEvaluationResult:
        """Оценивает кандидатов с телеметрией."""

        ...

    async def evaluate(
        self,
        selection: CandidateSelectionResult,
    ) -> tuple[
        EventAssessment,
        ...,
    ]:
        """Возвращает только проверенные инфоповоды."""

        ...