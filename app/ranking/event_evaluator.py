from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Protocol, runtime_checkable

from app.db.news_candidates import CandidateSelectionResult
from app.ranking.evaluator import RankingEvaluatorMetadata
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


EVENT_EVALUATOR_VERSION = (
    "event_ranking_evaluator_v7"
)

EVENT_PROMPT_VERSION = (
    "movie_news_event_ranking_prompt_v6"
)

STORY_CLUSTER_VERIFIER_PROMPT_VERSION = (
    "movie_news_story_cluster_verifier_v1"
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

STORY_CLUSTER_KEY_PATTERN = (
    r"[a-z0-9]+(?:_[a-z0-9]+)*"
)
STORY_CLUSTER_KEY_MAX_LENGTH = 120


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
        normalized_value = Decimal(str(value))
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


def _normalize_news_ids(
    value: tuple[int, ...],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    """Проверяет упорядоченный набор news_id."""

    if not isinstance(value, tuple):
        raise TypeError(
            f"{field_name} должен быть tuple."
        )

    if not allow_empty and not value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    normalized = tuple(
        _normalize_positive_integer(
            news_id,
            field_name=f"{field_name} item",
        )
        for news_id in value
    )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{field_name} содержит повторяющиеся news_id."
        )

    return normalized


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

        normalized_news_id = _normalize_positive_integer(
            self.news_id,
            field_name="news_id",
        )
        normalized_relation = _normalize_required_text(
            self.source_relation,
            field_name="source_relation",
        )
        normalized_reason = _normalize_required_text(
            self.membership_reason,
            field_name="membership_reason",
        )

        if normalized_relation not in SOURCE_RELATIONS:
            raise ValueError(
                "source_relation имеет "
                "неподдерживаемое значение: "
                f"{normalized_relation!r}"
            )

        for field_name, value in (
            ("is_representative", self.is_representative),
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

        if not isinstance(self.source_weight, int):
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
    story_cluster_key: str

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

        representative_news_id = _normalize_positive_integer(
            self.representative_news_id,
            field_name="representative_news_id",
        )
        event_title = _normalize_required_text(
            self.event_title,
            field_name="event_title",
        )
        event_time_utc = _normalize_datetime_utc(
            self.event_time_utc,
            field_name="event_time_utc",
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

        story_cluster_key = (
            _normalize_required_text(
                self.story_cluster_key,
                field_name="story_cluster_key",
            )
            .lower()
        )

        if len(story_cluster_key) > (
            STORY_CLUSTER_KEY_MAX_LENGTH
        ):
            raise ValueError(
                "story_cluster_key не может быть "
                f"длиннее {STORY_CLUSTER_KEY_MAX_LENGTH} "
                "символов."
            )

        if re.fullmatch(
            STORY_CLUSTER_KEY_PATTERN,
            story_cluster_key,
        ) is None:
            raise ValueError(
                "story_cluster_key должен быть "
                "lower_snake_case из латинских "
                "букв и цифр."
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

        impact_reason = _normalize_required_text(
            self.impact_reason,
            field_name="impact_reason",
        )
        hook_reason = _normalize_required_text(
            self.hook_reason,
            field_name="hook_reason",
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

        if len(set(member_news_ids)) != len(member_news_ids):
            raise ValueError(
                "members содержит повторяющиеся "
                "news_id."
            )

        if representative_news_id not in member_news_ids:
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
        object.__setattr__(
            self,
            "story_cluster_key",
            story_cluster_key,
        )

        for field_name, value in normalized_scores.items():
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
    def member_news_ids(self) -> tuple[int, ...]:
        """Возвращает news_id публикаций инфоповода."""

        return tuple(
            member.news_id
            for member in self.members
        )

    @property
    def source_weight_sum(self) -> int:
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
class StoryClusterVerificationChange:
    """Изменение одного многособытийного story cluster."""

    original_story_cluster_key: str
    representative_news_ids: tuple[int, ...]
    resulting_story_cluster_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        """Проверяет структурированное описание изменения."""

        original_key = _normalize_required_text(
            self.original_story_cluster_key,
            field_name="original_story_cluster_key",
        ).lower()
        representative_news_ids = _normalize_news_ids(
            self.representative_news_ids,
            field_name="representative_news_ids",
            allow_empty=False,
        )

        if not isinstance(
            self.resulting_story_cluster_keys,
            tuple,
        ):
            raise TypeError(
                "resulting_story_cluster_keys должен быть tuple."
            )

        resulting_keys = tuple(
            _normalize_required_text(
                key,
                field_name=(
                    "resulting_story_cluster_keys item"
                ),
            ).lower()
            for key in self.resulting_story_cluster_keys
        )

        if not resulting_keys:
            raise ValueError(
                "resulting_story_cluster_keys не может быть пустым."
            )

        for field_name, key in (
            ("original_story_cluster_key", original_key),
            *(
                (
                    "resulting_story_cluster_keys item",
                    key,
                )
                for key in resulting_keys
            ),
        ):
            if len(key) > STORY_CLUSTER_KEY_MAX_LENGTH:
                raise ValueError(
                    f"{field_name} не может быть длиннее "
                    f"{STORY_CLUSTER_KEY_MAX_LENGTH} символов."
                )

            if re.fullmatch(
                STORY_CLUSTER_KEY_PATTERN,
                key,
            ) is None:
                raise ValueError(
                    f"{field_name} должен быть lower_snake_case."
                )

        if len(set(resulting_keys)) != len(resulting_keys):
            raise ValueError(
                "resulting_story_cluster_keys содержит повторы."
            )

        object.__setattr__(
            self,
            "original_story_cluster_key",
            original_key,
        )
        object.__setattr__(
            self,
            "representative_news_ids",
            representative_news_ids,
        )
        object.__setattr__(
            self,
            "resulting_story_cluster_keys",
            resulting_keys,
        )


@dataclass(frozen=True, slots=True)
class EventRankingCoverageDiagnostics:
    """Диагностика coverage, repair и cluster verifier."""

    expected_news_ids: tuple[int, ...]
    processed_news_ids: tuple[int, ...]
    initial_missing_news_ids: tuple[int, ...] = ()
    missing_news_ids: tuple[int, ...] = ()
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_error_type: str | None = None
    repair_error_message: str | None = None
    story_cluster_verification_attempted: bool = False
    story_cluster_verification_succeeded: bool = False
    story_cluster_verification_skipped_reason: str | None = None
    story_cluster_verification_error_type: str | None = None
    story_cluster_verification_error_message: str | None = None
    story_cluster_verification_prompt_version: str | None = None
    story_cluster_count_before: int = 0
    story_cluster_count_after: int = 0
    story_cluster_multi_event_count_before: int = 0
    story_cluster_multi_event_count_after: int = 0
    story_cluster_verifier_event_count: int = 0
    story_cluster_verification_changes: tuple[
        StoryClusterVerificationChange,
        ...,
    ] = ()
    model_call_count: int = 1

    def __post_init__(self) -> None:
        """Проверяет диагностическую модель."""

        expected = _normalize_news_ids(
            self.expected_news_ids,
            field_name="expected_news_ids",
            allow_empty=False,
        )
        processed = _normalize_news_ids(
            self.processed_news_ids,
            field_name="processed_news_ids",
            allow_empty=False,
        )
        initial_missing = _normalize_news_ids(
            self.initial_missing_news_ids,
            field_name="initial_missing_news_ids",
            allow_empty=True,
        )
        missing = _normalize_news_ids(
            self.missing_news_ids,
            field_name="missing_news_ids",
            allow_empty=True,
        )

        expected_set = set(expected)
        processed_set = set(processed)
        missing_set = set(missing)

        if not processed_set.issubset(expected_set):
            raise ValueError(
                "processed_news_ids содержит неожиданные news_id."
            )

        if not set(initial_missing).issubset(expected_set):
            raise ValueError(
                "initial_missing_news_ids содержит неожиданные news_id."
            )

        if not missing_set.issubset(expected_set):
            raise ValueError(
                "missing_news_ids содержит неожиданные news_id."
            )

        if processed_set & missing_set:
            raise ValueError(
                "processed_news_ids и missing_news_ids "
                "не должны пересекаться."
            )

        if processed_set | missing_set != expected_set:
            raise ValueError(
                "processed_news_ids и missing_news_ids "
                "не образуют полный expected_news_ids."
            )

        for field_name, value in (
            ("repair_attempted", self.repair_attempted),
            ("repair_succeeded", self.repair_succeeded),
            (
                "story_cluster_verification_attempted",
                self.story_cluster_verification_attempted,
            ),
            (
                "story_cluster_verification_succeeded",
                self.story_cluster_verification_succeeded,
            ),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} должен быть bool."
                )

        if self.repair_attempted and (
            self.story_cluster_verification_attempted
        ):
            raise ValueError(
                "Repair и story-cluster verifier не могут "
                "использоваться в одном запуске: максимум два "
                "вызова модели."
            )

        if self.repair_succeeded and not self.repair_attempted:
            raise ValueError(
                "repair_succeeded требует repair_attempted=true."
            )

        if self.repair_succeeded and missing:
            raise ValueError(
                "Успешный repair не может оставлять missing_news_ids."
            )

        if initial_missing and not self.repair_attempted:
            raise ValueError(
                "Пропуск в первом ответе требует repair_attempted=true."
            )

        if (
            self.story_cluster_verification_succeeded
            and not self.story_cluster_verification_attempted
        ):
            raise ValueError(
                "Успешная cluster verification требует attempted=true."
            )

        if (
            self.story_cluster_verification_skipped_reason
            is not None
            and self.story_cluster_verification_attempted
        ):
            raise ValueError(
                "Запущенный verifier не может иметь skipped_reason."
            )

        if not isinstance(
            self.story_cluster_verification_changes,
            tuple,
        ):
            raise TypeError(
                "story_cluster_verification_changes должен быть tuple."
            )

        for change in self.story_cluster_verification_changes:
            if not isinstance(
                change,
                StoryClusterVerificationChange,
            ):
                raise TypeError(
                    "story_cluster_verification_changes содержит "
                    "неподдерживаемый объект."
                )

        for field_name, value in (
            ("story_cluster_count_before", self.story_cluster_count_before),
            ("story_cluster_count_after", self.story_cluster_count_after),
            (
                "story_cluster_multi_event_count_before",
                self.story_cluster_multi_event_count_before,
            ),
            (
                "story_cluster_multi_event_count_after",
                self.story_cluster_multi_event_count_after,
            ),
            (
                "story_cluster_verifier_event_count",
                self.story_cluster_verifier_event_count,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field_name} должен быть int."
                )
            if value < 0:
                raise ValueError(
                    f"{field_name} не может быть отрицательным."
                )

        if not isinstance(self.model_call_count, int):
            raise TypeError(
                "model_call_count должен быть int."
            )

        if self.model_call_count not in {1, 2}:
            raise ValueError(
                "model_call_count должен быть 1 или 2."
            )

        second_call_attempted = (
            self.repair_attempted
            or self.story_cluster_verification_attempted
        )
        expected_model_call_count = (
            2 if second_call_attempted else 1
        )

        if self.model_call_count != expected_model_call_count:
            raise ValueError(
                "model_call_count не соответствует repair/verifier."
            )

        normalized_optional_text: dict[str, str | None] = {}
        for field_name in (
            "repair_error_type",
            "repair_error_message",
            "story_cluster_verification_skipped_reason",
            "story_cluster_verification_error_type",
            "story_cluster_verification_error_message",
            "story_cluster_verification_prompt_version",
        ):
            value = getattr(self, field_name)
            normalized_optional_text[field_name] = (
                None
                if value is None
                else _normalize_required_text(
                    value,
                    field_name=field_name,
                )
            )

        object.__setattr__(self, "expected_news_ids", expected)
        object.__setattr__(self, "processed_news_ids", processed)
        object.__setattr__(self, "initial_missing_news_ids", initial_missing)
        object.__setattr__(self, "missing_news_ids", missing)
        for field_name, value in normalized_optional_text.items():
            object.__setattr__(self, field_name, value)

    @property
    def degraded(self) -> bool:
        """Показывает неполное итоговое покрытие."""

        return bool(self.missing_news_ids)

    @property
    def story_cluster_verification_degraded(self) -> bool:
        """Показывает неуспешный запущенный verifier."""

        return (
            self.story_cluster_verification_attempted
            and not self.story_cluster_verification_succeeded
        )


@dataclass(frozen=True, slots=True)
class EventRankingEvaluationResult:
    """Проверенные инфоповоды и ответы модели."""

    events: tuple[
        EventAssessment,
        ...,
    ]
    model_response: EventRankingModelResponse
    diagnostics: (
        EventRankingCoverageDiagnostics
        | None
    ) = None
    model_responses: tuple[
        EventRankingModelResponse,
        ...,
    ] = ()


@runtime_checkable
class StructuredEventRankingClient(Protocol):
    """Транспортный интерфейс event-level модели."""

    async def create_response(
        self,
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Возвращает структурированный ответ."""

        ...


@runtime_checkable
class EventRankingEvaluator(Protocol):
    """Интерфейс оценщика полной формулы v4."""

    @property
    def metadata(self) -> RankingEvaluatorMetadata:
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
