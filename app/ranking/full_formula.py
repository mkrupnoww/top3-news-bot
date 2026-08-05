from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from itertools import combinations
from typing import TypeAlias

from app.ranking.score_formula import (
    CalculatedScore,
    calculate_individual_score,
    create_score_components,
)


ScoreInput: TypeAlias = Decimal | int | float | str
OptionalScoreInput: TypeAlias = ScoreInput | None

FULL_FORMULA_VERSION = "top3_cinema_v3"
SCORE_QUANTUM = Decimal("0.000001")
SCORE_MIN = Decimal("0")
SCORE_MAX = Decimal("10")
QUALITY_MAX = Decimal("1")
MAX_AGE_HOURS = Decimal("24")
ELIGIBILITY_THRESHOLD = Decimal("4.500000")

U_WEIGHT = Decimal("0.60")
I_WEIGHT = Decimal("0.40")
V_WEIGHT = Decimal("0.40")
C_WEIGHT = Decimal("0.35")
S_WEIGHT = Decimal("0.25")
K_WEIGHT = Decimal("0.30")
N_WEIGHT = Decimal("0.25")
E_WEIGHT = Decimal("0.25")
X_WEIGHT = Decimal("0.20")
D_WEIGHT = Decimal("0.15")

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

RESONANCE_CONFIDENCE_FULL = "full"
RESONANCE_CONFIDENCE_PARTIAL = "partial"
RESONANCE_CONFIDENCE_UNAVAILABLE = "unavailable"

EXCLUSION_REASON_QUALITY_ZERO = "quality_zero"
EXCLUSION_REASON_SCORE_BELOW_THRESHOLD = "individual_score_below_4_5"


@dataclass(frozen=True, slots=True)
class ResonanceCalculation:
    v_score: Decimal | None
    c_score: Decimal | None
    s_score: Decimal | None
    effective_v_weight: Decimal
    effective_c_weight: Decimal
    effective_s_weight: Decimal
    confidence: str
    r_score: Decimal


@dataclass(frozen=True, slots=True)
class FullNewsScore:
    formula_version: str
    news_id: int
    macro_topic: str
    age_hours: Decimal
    source_weight_sum: Decimal
    max_source_weight_sum: Decimal
    f_score: Decimal
    u_score: Decimal
    i_score: Decimal
    m_score: Decimal
    resonance: ResonanceCalculation
    k_score: Decimal
    n_score: Decimal
    e_score: Decimal
    x_score: Decimal
    h_score: Decimal
    q_score: Decimal
    individual: CalculatedScore
    is_eligible: bool
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class Top3CombinationScore:
    combination_rank: int
    is_winner: bool
    news_ids: tuple[int, int, int]
    ordered_news_ids: tuple[int, int, int]
    mean_individual_score: Decimal
    diversity_score: Decimal
    final_top_score: Decimal
    mean_m_score: Decimal
    mean_q_score: Decimal
    mean_f_score: Decimal
    distinct_macro_topic_count: int


@dataclass(frozen=True, slots=True)
class Top3SelectionResult:
    formula_version: str
    eligible_count: int
    combinations: tuple[Top3CombinationScore, ...]
    winner: Top3CombinationScore


def _decimal(value: ScoreInput, field_name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} должен быть числом.") from error

    if not result.is_finite():
        raise ValueError(f"{field_name} должен быть конечным числом.")

    return result.quantize(SCORE_QUANTUM, rounding=ROUND_HALF_UP)


def _optional_decimal(
    value: OptionalScoreInput,
    field_name: str,
) -> Decimal | None:
    return None if value is None else _decimal(value, field_name)


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(SCORE_QUANTUM, rounding=ROUND_HALF_UP)


def _require_range(
    value: Decimal,
    field_name: str,
    minimum: Decimal,
    maximum: Decimal,
) -> None:
    if value < minimum or value > maximum:
        raise ValueError(
            f"{field_name} должен находиться в диапазоне "
            f"от {minimum} до {maximum}: value={value}"
        )


def _standard_score(value: ScoreInput, field_name: str) -> Decimal:
    result = _decimal(value, field_name)
    _require_range(result, field_name, SCORE_MIN, SCORE_MAX)
    return result


def _news_id(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("news_id не может быть bool.")
    if not isinstance(value, int):
        raise TypeError("news_id должен быть int.")
    if value <= 0:
        raise ValueError("news_id должен быть больше нуля.")
    return value


def _macro_topic(value: str) -> str:
    result = value.strip()
    if result not in MACRO_TOPICS:
        raise ValueError(f"Неподдерживаемая macro_topic: {result!r}")
    return result


def calculate_freshness_score(age_hours: ScoreInput) -> Decimal:
    """F = 10 × sqrt(1 - h / 24)."""

    age = _decimal(age_hours, "age_hours")
    _require_range(age, "age_hours", SCORE_MIN, MAX_AGE_HOURS)
    return _quantize(SCORE_MAX * (Decimal("1") - age / MAX_AGE_HOURS).sqrt())


def calculate_media_reach_score(
    *,
    source_weight_sum: ScoreInput,
    max_source_weight_sum: ScoreInput,
) -> Decimal:
    """U = 10 × ln(1 + A) / ln(1 + A_max)."""

    value = _decimal(source_weight_sum, "source_weight_sum")
    maximum = _decimal(max_source_weight_sum, "max_source_weight_sum")

    if value < 0 or maximum < 0:
        raise ValueError("Суммы весов источников не могут быть отрицательными.")
    if value > maximum:
        raise ValueError("source_weight_sum не может превышать max_source_weight_sum.")
    if maximum == 0:
        return Decimal("0.000000")

    return _quantize(
        SCORE_MAX
        * (Decimal("1") + value).ln()
        / (Decimal("1") + maximum).ln()
    )


def normalize_audience_metric(
    *,
    value: ScoreInput,
    maximum_value: ScoreInput,
    field_name: str,
) -> Decimal:
    """Пилотная логарифмическая нормализация счётчика в шкалу 0–10."""

    metric = _decimal(value, field_name)
    maximum = _decimal(maximum_value, f"max_{field_name}")

    if metric < 0 or maximum < 0:
        raise ValueError("Audience-метрики не могут быть отрицательными.")
    if metric > maximum:
        raise ValueError(f"{field_name} не может превышать max_{field_name}.")
    if maximum == 0:
        return Decimal("0.000000")

    return _quantize(
        SCORE_MAX
        * (Decimal("1") + metric).ln()
        / (Decimal("1") + maximum).ln()
    )


def calculate_magnitude_score(
    *,
    u_score: ScoreInput,
    i_score: ScoreInput,
) -> Decimal:
    """M = 0.60U + 0.40I."""

    u_value = _standard_score(u_score, "u_score")
    i_value = _standard_score(i_score, "i_score")
    return _quantize(U_WEIGHT * u_value + I_WEIGHT * i_value)


def calculate_resonance_score(
    *,
    v_score: OptionalScoreInput,
    c_score: OptionalScoreInput,
    s_score: OptionalScoreInput,
) -> ResonanceCalculation:
    """R с перенормировкой весов доступных V, C и S."""

    v_value = _optional_decimal(v_score, "v_score")
    c_value = _optional_decimal(c_score, "c_score")
    s_value = _optional_decimal(s_score, "s_score")

    values = (
        ("v_score", v_value, V_WEIGHT),
        ("c_score", c_value, C_WEIGHT),
        ("s_score", s_value, S_WEIGHT),
    )
    available: list[tuple[Decimal, Decimal]] = []

    for field_name, value, weight in values:
        if value is not None:
            _require_range(value, field_name, SCORE_MIN, SCORE_MAX)
            available.append((value, weight))

    if not available:
        zero = Decimal("0.000000")
        return ResonanceCalculation(
            None,
            None,
            None,
            zero,
            zero,
            zero,
            RESONANCE_CONFIDENCE_UNAVAILABLE,
            zero,
        )

    weight_sum = sum((weight for _, weight in available), Decimal("0"))
    weighted_total = sum((value * weight for value, weight in available), Decimal("0"))

    def effective(value: Decimal | None, weight: Decimal) -> Decimal:
        return _quantize(weight / weight_sum) if value is not None else Decimal("0.000000")

    confidence = (
        RESONANCE_CONFIDENCE_FULL
        if len(available) == 3
        else RESONANCE_CONFIDENCE_PARTIAL
    )

    return ResonanceCalculation(
        v_value,
        c_value,
        s_value,
        effective(v_value, V_WEIGHT),
        effective(c_value, C_WEIGHT),
        effective(s_value, S_WEIGHT),
        confidence,
        _quantize(weighted_total / weight_sum),
    )


def calculate_hook_score(
    *,
    k_score: ScoreInput,
    n_score: ScoreInput,
    e_score: ScoreInput,
    x_score: ScoreInput,
) -> Decimal:
    """H = 0.30K + 0.25N + 0.25E + 0.20X."""

    k_value = _standard_score(k_score, "k_score")
    n_value = _standard_score(n_score, "n_score")
    e_value = _standard_score(e_score, "e_score")
    x_value = _standard_score(x_score, "x_score")
    return _quantize(
        K_WEIGHT * k_value
        + N_WEIGHT * n_value
        + E_WEIGHT * e_value
        + X_WEIGHT * x_value
    )


def calculate_full_news_score(
    *,
    news_id: int,
    macro_topic: str,
    age_hours: ScoreInput,
    source_weight_sum: ScoreInput,
    max_source_weight_sum: ScoreInput,
    i_score: ScoreInput,
    v_score: OptionalScoreInput,
    c_score: OptionalScoreInput,
    s_score: OptionalScoreInput,
    k_score: ScoreInput,
    n_score: ScoreInput,
    e_score: ScoreInput,
    x_score: ScoreInput,
    q_score: ScoreInput,
) -> FullNewsScore:
    """Полный расчёт одного инфоповода."""

    normalized_news_id = _news_id(news_id)
    normalized_topic = _macro_topic(macro_topic)
    age = _decimal(age_hours, "age_hours")
    source_sum = _decimal(source_weight_sum, "source_weight_sum")
    max_source_sum = _decimal(max_source_weight_sum, "max_source_weight_sum")
    i_value = _standard_score(i_score, "i_score")
    k_value = _standard_score(k_score, "k_score")
    n_value = _standard_score(n_score, "n_score")
    e_value = _standard_score(e_score, "e_score")
    x_value = _standard_score(x_score, "x_score")
    q_value = _decimal(q_score, "q_score")
    _require_range(q_value, "q_score", SCORE_MIN, QUALITY_MAX)

    f_value = calculate_freshness_score(age)
    u_value = calculate_media_reach_score(
        source_weight_sum=source_sum,
        max_source_weight_sum=max_source_sum,
    )
    m_value = calculate_magnitude_score(u_score=u_value, i_score=i_value)
    resonance = calculate_resonance_score(
        v_score=v_score,
        c_score=c_score,
        s_score=s_score,
    )
    h_value = calculate_hook_score(
        k_score=k_value,
        n_score=n_value,
        e_score=e_value,
        x_score=x_value,
    )
    individual = calculate_individual_score(
        create_score_components(
            f_score=f_value,
            m_score=m_value,
            r_score=resonance.r_score,
            h_score=h_value,
            q_score=q_value,
        )
    )

    if q_value == 0:
        eligible = False
        exclusion_reason = EXCLUSION_REASON_QUALITY_ZERO
    elif individual.individual_score < ELIGIBILITY_THRESHOLD:
        eligible = False
        exclusion_reason = EXCLUSION_REASON_SCORE_BELOW_THRESHOLD
    else:
        eligible = True
        exclusion_reason = None

    return FullNewsScore(
        FULL_FORMULA_VERSION,
        normalized_news_id,
        normalized_topic,
        age,
        source_sum,
        max_source_sum,
        f_value,
        u_value,
        i_value,
        m_value,
        resonance,
        k_value,
        n_value,
        e_value,
        x_value,
        h_value,
        q_value,
        individual,
        eligible,
        exclusion_reason,
    )


def calculate_diversity_score(
    macro_topics: tuple[str, str, str],
) -> Decimal:
    """D = 10 для трёх тем, 6 для двух, 0 для одной."""

    if len(macro_topics) != 3:
        raise ValueError("Для расчёта D требуется ровно три макротемы.")

    distinct_count = len({_macro_topic(topic) for topic in macro_topics})
    return {
        3: Decimal("10.000000"),
        2: Decimal("6.000000"),
        1: Decimal("0.000000"),
    }[distinct_count]


def _mean(values: tuple[Decimal, Decimal, Decimal]) -> Decimal:
    return _quantize(sum(values, Decimal("0")) / Decimal("3"))


def _combination(
    items: tuple[FullNewsScore, FullNewsScore, FullNewsScore],
) -> Top3CombinationScore:
    ordered = tuple(
        sorted(
            items,
            key=lambda item: (-item.individual.individual_score, item.news_id),
        )
    )
    news_ids = tuple(sorted(item.news_id for item in items))
    ordered_news_ids = tuple(item.news_id for item in ordered)
    mean_b = _mean(tuple(item.individual.individual_score for item in items))
    diversity = calculate_diversity_score(tuple(item.macro_topic for item in items))

    return Top3CombinationScore(
        combination_rank=0,
        is_winner=False,
        news_ids=(news_ids[0], news_ids[1], news_ids[2]),
        ordered_news_ids=(ordered_news_ids[0], ordered_news_ids[1], ordered_news_ids[2]),
        mean_individual_score=mean_b,
        diversity_score=diversity,
        final_top_score=_quantize(mean_b + D_WEIGHT * diversity),
        mean_m_score=_mean(tuple(item.m_score for item in items)),
        mean_q_score=_mean(tuple(item.q_score for item in items)),
        mean_f_score=_mean(tuple(item.f_score for item in items)),
        distinct_macro_topic_count=len({item.macro_topic for item in items}),
    )


def select_top3_combination(
    scores: tuple[FullNewsScore, ...],
) -> Top3SelectionResult:
    """Перебирает все допустимые тройки и выбирает победителя."""

    if not scores:
        raise ValueError("Список полных оценок не может быть пустым.")

    news_ids = tuple(item.news_id for item in scores)
    if len(news_ids) != len(set(news_ids)):
        raise ValueError("Каждый news_id должен встречаться один раз.")

    eligible = tuple(item for item in scores if item.is_eligible)
    if len(eligible) < 3:
        raise ValueError(
            "Для выбора TOP-3 требуется минимум три допустимых инфоповода: "
            f"eligible_count={len(eligible)}"
        )

    calculated = [
        _combination((items[0], items[1], items[2]))
        for items in combinations(eligible, 3)
    ]
    calculated.sort(
        key=lambda item: (
            -item.final_top_score,
            -item.mean_m_score,
            -item.mean_q_score,
            -item.mean_f_score,
            item.news_ids,
        )
    )

    ranked = tuple(
        Top3CombinationScore(
            rank,
            rank == 1,
            item.news_ids,
            item.ordered_news_ids,
            item.mean_individual_score,
            item.diversity_score,
            item.final_top_score,
            item.mean_m_score,
            item.mean_q_score,
            item.mean_f_score,
            item.distinct_macro_topic_count,
        )
        for rank, item in enumerate(calculated, start=1)
    )

    return Top3SelectionResult(
        FULL_FORMULA_VERSION,
        len(eligible),
        ranked,
        ranked[0],
    )