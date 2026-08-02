from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TypeAlias

from app.db.news_candidates import (
    CandidateSelectionResult,
)
from app.ranking.event_evaluator import (
    EventAssessment,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
    FullNewsScore,
    Top3SelectionResult,
    calculate_full_news_score,
    normalize_audience_metric,
    select_top3_combination,
)


AudienceCount: TypeAlias = int | None

FORMULA_WINDOW_HOURS = Decimal("24")


@dataclass(frozen=True, slots=True)
class EventAudienceMetrics:
    """Фактические audience-метрики инфоповода."""

    news_id: int
    view_count: AudienceCount = None
    comment_count: AudienceCount = None
    share_count: AudienceCount = None

    def __post_init__(self) -> None:
        """Проверяет идентификатор и счётчики."""

        normalized_news_id = _positive_integer(
            self.news_id,
            field_name="news_id",
        )

        for field_name, value in (
            ("view_count", self.view_count),
            ("comment_count", self.comment_count),
            ("share_count", self.share_count),
        ):
            _optional_nonnegative_integer(
                value,
                field_name=field_name,
            )

        object.__setattr__(
            self,
            "news_id",
            normalized_news_id,
        )


@dataclass(frozen=True, slots=True)
class AudienceMetricMaxima:
    """Максимумы сырых метрик в текущем окне."""

    max_view_count: AudienceCount
    max_comment_count: AudienceCount
    max_share_count: AudienceCount


@dataclass(frozen=True, slots=True)
class CalculatedEventScore:
    """Инфоповод, его сырые метрики и полный балл."""

    event: EventAssessment
    audience_metrics: EventAudienceMetrics
    score: FullNewsScore


@dataclass(frozen=True, slots=True)
class EventScoreCalculationResult:
    """Промежуточный расчёт всех event-level баллов."""

    formula_version: str
    window_start: datetime
    window_end: datetime
    audience_maxima: AudienceMetricMaxima
    calculated_events: tuple[
        CalculatedEventScore,
        ...,
    ]

    @property
    def scores(
        self,
    ) -> tuple[
        FullNewsScore,
        ...,
    ]:
        """Возвращает только полные оценки."""

        return tuple(
            item.score
            for item in self.calculated_events
        )

    @property
    def eligible_count(self) -> int:
        """Возвращает число допустимых инфоповодов."""

        return sum(
            1
            for item in self.calculated_events
            if item.score.is_eligible
        )


@dataclass(frozen=True, slots=True)
class EventFormulaCalculationResult:
    """Результат полного расчёта TOP-3 для окна."""

    formula_version: str
    window_start: datetime
    window_end: datetime
    audience_maxima: AudienceMetricMaxima
    calculated_events: tuple[
        CalculatedEventScore,
        ...,
    ]
    top3_selection: Top3SelectionResult

    @property
    def scores(
        self,
    ) -> tuple[
        FullNewsScore,
        ...,
    ]:
        """Возвращает только полные оценки."""

        return tuple(
            item.score
            for item in self.calculated_events
        )


def _positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительное целое число."""

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


def _optional_nonnegative_integer(
    value: AudienceCount,
    *,
    field_name: str,
) -> AudienceCount:
    """Проверяет необязательный неотрицательный счётчик."""

    if value is None:
        return None

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int или None."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} не может быть отрицательным."
        )

    return value


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


def _timedelta_hours(
    value: timedelta,
) -> Decimal:
    """Преобразует timedelta в часы без float."""

    total_microseconds = (
        value.days
        * 24
        * 60
        * 60
        * 1_000_000
        + value.seconds
        * 1_000_000
        + value.microseconds
    )

    return (
        Decimal(total_microseconds)
        / Decimal(3_600_000_000)
    )


def _validate_selection(
    selection: CandidateSelectionResult,
) -> tuple[
    datetime,
    datetime,
    tuple[int, ...],
]:
    """Проверяет окно и набор публикаций."""

    window_start = _normalize_datetime(
        selection.window_start,
        field_name="window_start",
    )

    window_end = _normalize_datetime(
        selection.window_end,
        field_name="window_end",
    )

    if window_end <= window_start:
        raise ValueError(
            "window_end должен быть позже "
            "window_start."
        )

    window_hours = Decimal(
        str(selection.window_hours)
    )

    if window_hours != FORMULA_WINDOW_HOURS:
        raise ValueError(
            "Полная формула top3_cinema_v2 "
            "требует окно ровно 24 часа: "
            f"window_hours={window_hours}"
        )

    actual_window_hours = (
        _timedelta_hours(
            window_end - window_start
        )
    )

    if actual_window_hours != (
        FORMULA_WINDOW_HOURS
    ):
        raise ValueError(
            "Фактическая длительность окна "
            "должна составлять ровно 24 часа: "
            f"actual_hours={actual_window_hours}"
        )

    if not selection.candidates:
        raise ValueError(
            "Список кандидатов "
            "не может быть пустым."
        )

    candidate_news_ids = tuple(
        _positive_integer(
            candidate.news_id,
            field_name="candidate.news_id",
        )
        for candidate in selection.candidates
    )

    if (
        len(set(candidate_news_ids))
        != len(candidate_news_ids)
    ):
        raise ValueError(
            "Кандидаты содержат повторяющиеся "
            "news_id."
        )

    return (
        window_start,
        window_end,
        candidate_news_ids,
    )


def _validate_events(
    *,
    events: tuple[
        EventAssessment,
        ...,
    ],
    candidate_news_ids: tuple[int, ...],
    window_start: datetime,
    window_end: datetime,
) -> None:
    """Проверяет покрытие кандидатов инфоповодами."""

    if not isinstance(events, tuple):
        raise TypeError(
            "events должен быть tuple."
        )

    if not events:
        raise ValueError(
            "Список инфоповодов "
            "не может быть пустым."
        )

    candidate_set = set(
        candidate_news_ids
    )

    all_member_news_ids = [
        news_id
        for event in events
        for news_id in event.member_news_ids
    ]

    duplicate_member_ids = sorted(
        {
            news_id
            for news_id in all_member_news_ids
            if all_member_news_ids.count(
                news_id
            ) > 1
        }
    )

    if duplicate_member_ids:
        raise ValueError(
            "Один news_id включён в несколько "
            "инфоповодов: "
            + ",".join(
                str(news_id)
                for news_id
                in duplicate_member_ids
            )
        )

    event_member_set = set(
        all_member_news_ids
    )

    missing_news_ids = sorted(
        candidate_set - event_member_set
    )

    unexpected_news_ids = sorted(
        event_member_set - candidate_set
    )

    if (
        missing_news_ids
        or unexpected_news_ids
    ):
        raise ValueError(
            "Некорректное покрытие кандидатов "
            "инфоповодами: "
            f"missing={missing_news_ids}, "
            f"unexpected={unexpected_news_ids}"
        )

    representative_news_ids = tuple(
        event.representative_news_id
        for event in events
    )

    if (
        len(set(representative_news_ids))
        != len(representative_news_ids)
    ):
        raise ValueError(
            "representative_news_id "
            "должны быть уникальны."
        )

    for event in events:
        event_time = _normalize_datetime(
            event.event_time_utc,
            field_name="event_time_utc",
        )

        if not (
            window_start
            <= event_time
            <= window_end
        ):
            raise ValueError(
                "event_time_utc находится "
                "вне суточного окна: "
                "representative_news_id="
                f"{event.representative_news_id}"
            )


def _build_metrics_by_news_id(
    *,
    events: tuple[
        EventAssessment,
        ...,
    ],
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ],
) -> dict[
    int,
    EventAudienceMetrics,
]:
    """Строит полную карту сырых метрик."""

    if not isinstance(
        audience_metrics,
        tuple,
    ):
        raise TypeError(
            "audience_metrics должен быть tuple."
        )

    provided_news_ids = tuple(
        item.news_id
        for item in audience_metrics
    )

    if (
        len(set(provided_news_ids))
        != len(provided_news_ids)
    ):
        raise ValueError(
            "audience_metrics содержит "
            "повторяющиеся news_id."
        )

    event_news_ids = {
        event.representative_news_id
        for event in events
    }

    unexpected_news_ids = sorted(
        set(provided_news_ids)
        - event_news_ids
    )

    if unexpected_news_ids:
        raise ValueError(
            "Audience-метрики переданы "
            "для неизвестных инфоповодов: "
            + ",".join(
                str(news_id)
                for news_id
                in unexpected_news_ids
            )
        )

    provided_by_id = {
        item.news_id: item
        for item in audience_metrics
    }

    return {
        event.representative_news_id: (
            provided_by_id.get(
                event.representative_news_id,
                EventAudienceMetrics(
                    news_id=(
                        event
                        .representative_news_id
                    )
                ),
            )
        )
        for event in events
    }


def _maximum_available(
    values: tuple[
        AudienceCount,
        ...,
    ],
) -> AudienceCount:
    """Возвращает максимум доступных счётчиков."""

    available = tuple(
        value
        for value in values
        if value is not None
    )

    if not available:
        return None

    return max(available)


def _calculate_maxima(
    metrics_by_news_id: dict[
        int,
        EventAudienceMetrics,
    ],
) -> AudienceMetricMaxima:
    """Определяет максимумы метрик в окне."""

    metrics = tuple(
        metrics_by_news_id.values()
    )

    return AudienceMetricMaxima(
        max_view_count=_maximum_available(
            tuple(
                item.view_count
                for item in metrics
            )
        ),
        max_comment_count=_maximum_available(
            tuple(
                item.comment_count
                for item in metrics
            )
        ),
        max_share_count=_maximum_available(
            tuple(
                item.share_count
                for item in metrics
            )
        ),
    )


def _normalize_metric(
    *,
    value: AudienceCount,
    maximum: AudienceCount,
    field_name: str,
) -> Decimal | None:
    """Нормализует доступный счётчик в 0..10."""

    if value is None:
        return None

    if maximum is None:
        raise RuntimeError(
            f"Для {field_name} отсутствует "
            "максимум при доступном значении."
        )

    return normalize_audience_metric(
        value=value,
        maximum_value=maximum,
        field_name=field_name,
    )


def calculate_event_scores(
    *,
    selection: CandidateSelectionResult,
    events: tuple[
        EventAssessment,
        ...,
    ],
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ] = (),
) -> EventScoreCalculationResult:
    """
    Рассчитывает баллы всех инфоповодов.

    Выбор комбинации TOP-3 не выполняется.
    Поэтому результат остаётся доступным даже
    при числе допустимых инфоповодов меньше трёх.

    OpenAI, PostgreSQL и Telegram не вызываются.
    """

    (
        window_start,
        window_end,
        candidate_news_ids,
    ) = _validate_selection(
        selection
    )

    _validate_events(
        events=events,
        candidate_news_ids=(
            candidate_news_ids
        ),
        window_start=window_start,
        window_end=window_end,
    )

    metrics_by_news_id = (
        _build_metrics_by_news_id(
            events=events,
            audience_metrics=(
                audience_metrics
            ),
        )
    )

    audience_maxima = _calculate_maxima(
        metrics_by_news_id
    )

    max_source_weight_sum = max(
        (
            event.source_weight_sum
            for event in events
        ),
        default=0,
    )

    calculated_events: list[
        CalculatedEventScore
    ] = []

    for event in events:
        raw_metrics = metrics_by_news_id[
            event.representative_news_id
        ]

        age_hours = _timedelta_hours(
            window_end
            - event.event_time_utc
        )

        v_score = _normalize_metric(
            value=raw_metrics.view_count,
            maximum=(
                audience_maxima.max_view_count
            ),
            field_name="view_count",
        )

        c_score = _normalize_metric(
            value=raw_metrics.comment_count,
            maximum=(
                audience_maxima
                .max_comment_count
            ),
            field_name="comment_count",
        )

        s_score = _normalize_metric(
            value=raw_metrics.share_count,
            maximum=(
                audience_maxima.max_share_count
            ),
            field_name="share_count",
        )

        score = calculate_full_news_score(
            news_id=(
                event.representative_news_id
            ),
            macro_topic=event.macro_topic,
            age_hours=age_hours,
            source_weight_sum=(
                event.source_weight_sum
            ),
            max_source_weight_sum=(
                max_source_weight_sum
            ),
            i_score=event.i_score,
            v_score=v_score,
            c_score=c_score,
            s_score=s_score,
            k_score=event.k_score,
            n_score=event.n_score,
            e_score=event.e_score,
            x_score=event.x_score,
            q_score=event.q_score,
        )

        calculated_events.append(
            CalculatedEventScore(
                event=event,
                audience_metrics=raw_metrics,
                score=score,
            )
        )

    return EventScoreCalculationResult(
        formula_version=FULL_FORMULA_VERSION,
        window_start=window_start,
        window_end=window_end,
        audience_maxima=audience_maxima,
        calculated_events=tuple(
            calculated_events
        ),
    )


def select_event_top3(
    calculation: EventScoreCalculationResult,
) -> EventFormulaCalculationResult:
    """
    Выбирает победившую комбинацию TOP-3.

    При eligible_count < 3 функция выбрасывает
    ValueError, но исходный промежуточный объект
    calculation остаётся доступным вызывающему коду.
    """

    if not isinstance(
        calculation,
        EventScoreCalculationResult,
    ):
        raise TypeError(
            "calculation должен быть "
            "EventScoreCalculationResult."
        )

    if calculation.formula_version != (
        FULL_FORMULA_VERSION
    ):
        raise ValueError(
            "Неподдерживаемая formula_version "
            "промежуточного расчёта."
        )

    top3_selection = (
        select_top3_combination(
            calculation.scores
        )
    )

    return EventFormulaCalculationResult(
        formula_version=(
            calculation.formula_version
        ),
        window_start=calculation.window_start,
        window_end=calculation.window_end,
        audience_maxima=(
            calculation.audience_maxima
        ),
        calculated_events=(
            calculation.calculated_events
        ),
        top3_selection=top3_selection,
    )


def calculate_event_formula(
    *,
    selection: CandidateSelectionResult,
    events: tuple[
        EventAssessment,
        ...,
    ],
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ] = (),
) -> EventFormulaCalculationResult:
    """
    Выполняет полный детерминированный расчёт v2.

    Сохраняет прежний публичный интерфейс:
    рассчитывает все баллы и сразу выбирает TOP-3.

    OpenAI, PostgreSQL и Telegram не вызываются.
    """

    score_calculation = calculate_event_scores(
        selection=selection,
        events=events,
        audience_metrics=audience_metrics,
    )

    return select_event_top3(
        score_calculation
    )