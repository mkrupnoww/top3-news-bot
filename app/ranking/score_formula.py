from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_HALF_UP,
)
from typing import TypeAlias


ScoreInput: TypeAlias = (
    Decimal
    | int
    | float
    | str
)


FORMULA_VERSION = "individual_score_v2"

SCORE_QUANTUM = Decimal("0.000001")

STANDARD_SCORE_MIN = Decimal("0")
STANDARD_SCORE_MAX = Decimal("10")

QUALITY_SCORE_MIN = Decimal("0")
QUALITY_SCORE_MAX = Decimal("1")

F_WEIGHT = Decimal("0.20")
M_WEIGHT = Decimal("0.30")
R_WEIGHT = Decimal("0.20")
HQ_WEIGHT = Decimal("0.15")


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    """Пять агрегированных компонентов рейтинга новости."""

    f_score: Decimal
    m_score: Decimal
    r_score: Decimal
    h_score: Decimal
    q_score: Decimal


@dataclass(frozen=True, slots=True)
class CalculatedScore:
    """Результат локального расчёта индивидуального балла."""

    formula_version: str
    components: ScoreComponents
    freshness_component: Decimal
    magnitude_component: Decimal
    resonance_component: Decimal
    hook_quality_component: Decimal
    individual_score: Decimal


def _to_decimal(
    value: ScoreInput,
    *,
    field_name: str,
) -> Decimal:
    """Безопасно преобразует число в Decimal."""

    try:
        decimal_value = Decimal(
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

    if not decimal_value.is_finite():
        raise ValueError(
            f"{field_name} должен быть конечным числом."
        )

    return decimal_value.quantize(
        SCORE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _validate_range(
    value: Decimal,
    *,
    field_name: str,
    minimum: Decimal,
    maximum: Decimal,
) -> None:
    """Проверяет допустимый диапазон компонента."""

    if value < minimum or value > maximum:
        raise ValueError(
            f"{field_name} должен находиться "
            f"в диапазоне от {minimum} до {maximum}: "
            f"value={value}"
        )


def create_score_components(
    *,
    f_score: ScoreInput,
    m_score: ScoreInput,
    r_score: ScoreInput,
    h_score: ScoreInput,
    q_score: ScoreInput,
) -> ScoreComponents:
    """
    Создаёт и валидирует компоненты рейтинга.

    F, M, R и H оцениваются по шкале 0–10.
    Q оценивается по шкале 0–1.
    """

    normalized_f = _to_decimal(
        f_score,
        field_name="f_score",
    )

    normalized_m = _to_decimal(
        m_score,
        field_name="m_score",
    )

    normalized_r = _to_decimal(
        r_score,
        field_name="r_score",
    )

    normalized_h = _to_decimal(
        h_score,
        field_name="h_score",
    )

    normalized_q = _to_decimal(
        q_score,
        field_name="q_score",
    )

    for field_name, value in (
        ("f_score", normalized_f),
        ("m_score", normalized_m),
        ("r_score", normalized_r),
        ("h_score", normalized_h),
    ):
        _validate_range(
            value,
            field_name=field_name,
            minimum=STANDARD_SCORE_MIN,
            maximum=STANDARD_SCORE_MAX,
        )

    _validate_range(
        normalized_q,
        field_name="q_score",
        minimum=QUALITY_SCORE_MIN,
        maximum=QUALITY_SCORE_MAX,
    )

    return ScoreComponents(
        f_score=normalized_f,
        m_score=normalized_m,
        r_score=normalized_r,
        h_score=normalized_h,
        q_score=normalized_q,
    )


def _quantize_result(
    value: Decimal,
) -> Decimal:
    """Округляет результат до масштаба PostgreSQL."""

    return value.quantize(
        SCORE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def calculate_individual_score(
    components: ScoreComponents,
) -> CalculatedScore:
    """
    Рассчитывает индивидуальный рейтинг новости.

    B = 0.20F
        + 0.30M
        + 0.20R
        + 0.15(H × Q)

    Итог округляется один раз после вычисления
    всей формулы, как generated-колонка PostgreSQL.
    """

    freshness_component = _quantize_result(
        F_WEIGHT
        * components.f_score
    )

    magnitude_component = _quantize_result(
        M_WEIGHT
        * components.m_score
    )

    resonance_component = _quantize_result(
        R_WEIGHT
        * components.r_score
    )

    hook_quality_component = _quantize_result(
        HQ_WEIGHT
        * (
            components.h_score
            * components.q_score
        )
    )

    individual_score = _quantize_result(
        F_WEIGHT
        * components.f_score
        + M_WEIGHT
        * components.m_score
        + R_WEIGHT
        * components.r_score
        + HQ_WEIGHT
        * (
            components.h_score
            * components.q_score
        )
    )

    return CalculatedScore(
        formula_version=FORMULA_VERSION,
        components=components,
        freshness_component=freshness_component,
        magnitude_component=magnitude_component,
        resonance_component=resonance_component,
        hook_quality_component=hook_quality_component,
        individual_score=individual_score,
    )