from decimal import Decimal

from app.ranking.score_formula import (
    FORMULA_VERSION,
    calculate_individual_score,
    create_score_components,
)


def test_maximum_score() -> None:
    """Проверяет максимально возможный балл."""

    components = create_score_components(
        f_score=10,
        m_score=10,
        r_score=10,
        h_score=10,
        q_score=1,
    )

    result = calculate_individual_score(
        components
    )

    assert (
        result.individual_score
        == Decimal("8.500000")
    )

    print("Maximum score: OK")
    print(
        f"individual_score="
        f"{result.individual_score}"
    )


def test_quality_modifier() -> None:
    """Проверяет, что Q модифицирует только H."""

    without_quality = (
        calculate_individual_score(
            create_score_components(
                f_score=10,
                m_score=10,
                r_score=10,
                h_score=10,
                q_score=0,
            )
        )
    )

    assert (
        without_quality.individual_score
        == Decimal("7.500000")
    )

    assert (
        without_quality.hook_quality_component
        == Decimal("0.000000")
    )

    print()
    print("Quality modifier: OK")
    print(
        "individual_score="
        f"{without_quality.individual_score}"
    )


def test_regular_score() -> None:
    """Проверяет обычный расчёт новости."""

    result = calculate_individual_score(
        create_score_components(
            f_score=8,
            m_score=6,
            r_score=5,
            h_score=7,
            q_score="0.9",
        )
    )

    assert (
        result.freshness_component
        == Decimal("1.600000")
    )

    assert (
        result.magnitude_component
        == Decimal("2.100000")
    )

    assert (
        result.resonance_component
        == Decimal("1.000000")
    )

    assert (
        result.hook_quality_component
        == Decimal("0.630000")
    )

    assert (
        result.individual_score
        == Decimal("5.330000")
    )

    print()
    print("Regular score: OK")
    print(
        f"freshness_component="
        f"{result.freshness_component}"
    )
    print(
        f"magnitude_component="
        f"{result.magnitude_component}"
    )
    print(
        f"resonance_component="
        f"{result.resonance_component}"
    )
    print(
        f"hook_quality_component="
        f"{result.hook_quality_component}"
    )
    print(
        f"individual_score="
        f"{result.individual_score}"
    )


def test_magnitude_weight_exceeds_hook_weight() -> None:
    """Фиксирует сниженный приоритет H×Q относительно M."""

    baseline = calculate_individual_score(
        create_score_components(
            f_score=5,
            m_score=5,
            r_score=5,
            h_score=5,
            q_score=1,
        )
    )

    magnitude_plus_one = calculate_individual_score(
        create_score_components(
            f_score=5,
            m_score=6,
            r_score=5,
            h_score=5,
            q_score=1,
        )
    )

    hook_plus_one = calculate_individual_score(
        create_score_components(
            f_score=5,
            m_score=5,
            r_score=5,
            h_score=6,
            q_score=1,
        )
    )

    magnitude_delta = (
        magnitude_plus_one.individual_score
        - baseline.individual_score
    )

    hook_delta = (
        hook_plus_one.individual_score
        - baseline.individual_score
    )

    assert magnitude_delta == Decimal("0.350000")
    assert hook_delta == Decimal("0.100000")
    assert magnitude_delta > hook_delta

    print()
    print("Magnitude vs hook priority: OK")
    print(f"magnitude_delta={magnitude_delta}")
    print(f"hook_delta={hook_delta}")


def test_single_final_rounding() -> None:
    """
    Проверяет округление только итогового B.

    Сумма отдельно округлённых диагностических
    компонентов отличается от результата полной
    формулы на последнем знаке.
    """

    result = calculate_individual_score(
        create_score_components(
            f_score="0.310791",
            m_score="0.387246",
            r_score="4.723361",
            h_score="1.729807",
            q_score="0.678077",
        )
    )

    rounded_component_sum = (
        result.freshness_component
        + result.magnitude_component
        + result.resonance_component
        + result.hook_quality_component
    )

    assert rounded_component_sum == (
        Decimal("1.259660")
    )

    assert result.individual_score == (
        Decimal("1.259661")
    )

    assert (
        result.individual_score
        != rounded_component_sum
    )

    print()
    print("Single final rounding: OK")
    print(
        f"rounded_component_sum="
        f"{rounded_component_sum}"
    )
    print(
        f"individual_score="
        f"{result.individual_score}"
    )


def test_invalid_values() -> None:
    """Проверяет блокировку недопустимых значений."""

    invalid_cases = [
        {
            "f_score": -1,
            "m_score": 5,
            "r_score": 5,
            "h_score": 5,
            "q_score": 1,
        },
        {
            "f_score": 5,
            "m_score": 11,
            "r_score": 5,
            "h_score": 5,
            "q_score": 1,
        },
        {
            "f_score": 5,
            "m_score": 5,
            "r_score": 5,
            "h_score": 5,
            "q_score": "1.1",
        },
    ]

    blocked_count = 0

    for invalid_case in invalid_cases:
        try:
            create_score_components(
                **invalid_case
            )
        except ValueError:
            blocked_count += 1
            continue

        raise AssertionError(
            "Недопустимый набор баллов "
            "не был заблокирован."
        )

    assert blocked_count == 3

    print()
    print("Invalid score blocking: OK")
    print(
        f"blocked_count={blocked_count}"
    )


def main() -> int:
    """Запускает тесты математической формулы."""

    assert FORMULA_VERSION == (
        "individual_score_v3"
    )

    print(
        f"formula_version={FORMULA_VERSION}"
    )
    print(
        "formula="
        "0.20F + 0.35M + 0.20R + 0.10(H × Q)"
    )
    print(
        "scales="
        "F/M/R/H: 0..10, Q: 0..1"
    )
    print()

    test_maximum_score()
    test_quality_modifier()
    test_regular_score()
    test_magnitude_weight_exceeds_hook_weight()
    test_single_final_rounding()
    test_invalid_values()

    print()
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Score formula test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())