from decimal import Decimal

from app.ranking.full_formula import (
    EXCLUSION_REASON_QUALITY_ZERO,
    EXCLUSION_REASON_SCORE_BELOW_THRESHOLD,
    FULL_FORMULA_VERSION,
    TOP3_SELECTION_POLICY_VERSION,
    RESONANCE_CONFIDENCE_FULL,
    RESONANCE_CONFIDENCE_PARTIAL,
    RESONANCE_CONFIDENCE_UNAVAILABLE,
    FullNewsScore,
    ResonanceCalculation,
    calculate_diversity_score,
    calculate_freshness_score,
    calculate_full_news_score,
    calculate_hook_score,
    calculate_magnitude_score,
    calculate_media_reach_score,
    calculate_resonance_score,
    normalize_audience_metric,
    select_top3_combination,
)
from app.ranking.score_formula import (
    CalculatedScore,
    create_score_components,
)


ZERO = Decimal("0.000000")


def test_freshness() -> None:
    """Проверяет формулу свежести и границы окна."""

    assert calculate_freshness_score(0) == Decimal("10.000000")
    assert calculate_freshness_score(6) == Decimal("8.660254")
    assert calculate_freshness_score(12) == Decimal("7.071068")
    assert calculate_freshness_score(18) == Decimal("5.000000")
    assert calculate_freshness_score(23) == Decimal("2.041241")
    assert calculate_freshness_score(24) == Decimal("0.000000")

    for invalid_age in (-1, 25):
        try:
            calculate_freshness_score(invalid_age)
        except ValueError:
            continue

        raise AssertionError(
            "Недопустимый age_hours не был заблокирован: "
            f"{invalid_age}"
        )

    print("Freshness formula: OK")


def test_media_reach_and_audience_normalization() -> None:
    """Проверяет логарифмическую нормализацию U, V, C и S."""

    assert (
        calculate_media_reach_score(
            source_weight_sum=0,
            max_source_weight_sum=0,
        )
        == ZERO
    )

    assert (
        calculate_media_reach_score(
            source_weight_sum=9,
            max_source_weight_sum=9,
        )
        == Decimal("10.000000")
    )

    assert (
        calculate_media_reach_score(
            source_weight_sum=3,
            max_source_weight_sum=9,
        )
        == Decimal("6.020600")
    )

    assert (
        normalize_audience_metric(
            value=100,
            maximum_value=1000,
            field_name="view_count",
        )
        == Decimal("6.680105")
    )

    try:
        calculate_media_reach_score(
            source_weight_sum=10,
            max_source_weight_sum=9,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "A_i > A_max не был заблокирован."
        )

    print("Reach and audience normalization: OK")


def test_aggregated_components() -> None:
    """Проверяет M, R и H, включая неполные данные R."""

    assert (
        calculate_magnitude_score(
            u_score=8,
            i_score=6,
        )
        == Decimal("7.200000")
    )

    full_resonance = calculate_resonance_score(
        v_score=8,
        c_score=6,
        s_score=4,
    )

    assert full_resonance.confidence == RESONANCE_CONFIDENCE_FULL
    assert full_resonance.effective_v_weight == Decimal("0.400000")
    assert full_resonance.effective_c_weight == Decimal("0.350000")
    assert full_resonance.effective_s_weight == Decimal("0.250000")
    assert full_resonance.r_score == Decimal("6.300000")

    partial_resonance = calculate_resonance_score(
        v_score=8,
        c_score=None,
        s_score=4,
    )

    assert partial_resonance.confidence == RESONANCE_CONFIDENCE_PARTIAL
    assert partial_resonance.effective_v_weight == Decimal("0.615385")
    assert partial_resonance.effective_c_weight == ZERO
    assert partial_resonance.effective_s_weight == Decimal("0.384615")
    assert partial_resonance.r_score == Decimal("6.461538")

    unavailable_resonance = calculate_resonance_score(
        v_score=None,
        c_score=None,
        s_score=None,
    )

    assert (
        unavailable_resonance.confidence
        == RESONANCE_CONFIDENCE_UNAVAILABLE
    )
    assert unavailable_resonance.r_score == ZERO

    assert (
        calculate_hook_score(
            k_score=8,
            n_score=6,
            e_score=7,
            x_score=5,
        )
        == Decimal("6.650000")
    )

    print("Aggregated M/R/H components: OK")


def test_full_news_score_and_eligibility() -> None:
    """Проверяет полный B и правила допустимости."""

    eligible = calculate_full_news_score(
        news_id=1,
        macro_topic="business_economy_law",
        age_hours=2,
        source_weight_sum=9,
        max_source_weight_sum=9,
        i_score=8,
        v_score=8,
        c_score=6,
        s_score=4,
        k_score=6,
        n_score=5,
        e_score=4,
        x_score=7,
        q_score=1,
    )

    assert eligible.formula_version == FULL_FORMULA_VERSION
    assert eligible.f_score == Decimal("9.574271")
    assert eligible.u_score == Decimal("10.000000")
    assert eligible.m_score == Decimal("9.200000")
    assert eligible.resonance.r_score == Decimal("6.300000")
    assert eligible.h_score == Decimal("5.450000")
    assert eligible.individual.individual_score == Decimal("6.752354")
    assert eligible.is_eligible is True
    assert eligible.exclusion_reason is None

    quality_zero = calculate_full_news_score(
        news_id=2,
        macro_topic="people_conflicts_legal",
        age_hours=0,
        source_weight_sum=10,
        max_source_weight_sum=10,
        i_score=10,
        v_score=10,
        c_score=10,
        s_score=10,
        k_score=10,
        n_score=10,
        e_score=10,
        x_score=10,
        q_score=0,
    )

    assert quality_zero.is_eligible is False
    assert (
        quality_zero.exclusion_reason
        == EXCLUSION_REASON_QUALITY_ZERO
    )

    low_score = calculate_full_news_score(
        news_id=3,
        macro_topic="other",
        age_hours=24,
        source_weight_sum=0,
        max_source_weight_sum=10,
        i_score=0,
        v_score=None,
        c_score=None,
        s_score=None,
        k_score=0,
        n_score=0,
        e_score=0,
        x_score=0,
        q_score=1,
    )

    assert low_score.individual.individual_score == ZERO
    assert low_score.is_eligible is False
    assert (
        low_score.exclusion_reason
        == EXCLUSION_REASON_SCORE_BELOW_THRESHOLD
    )

    print("Full news score and eligibility: OK")


def _selection_fixture() -> tuple[FullNewsScore, ...]:
    """Возвращает четыре допустимых новости для проверки комбинаций."""

    return (
        calculate_full_news_score(
            news_id=1,
            macro_topic="business_economy_law",
            age_hours=2,
            source_weight_sum=9,
            max_source_weight_sum=9,
            i_score=8,
            v_score=8,
            c_score=6,
            s_score=4,
            k_score=6,
            n_score=5,
            e_score=4,
            x_score=7,
            q_score=1,
        ),
        calculate_full_news_score(
            news_id=2,
            macro_topic="people_conflicts_legal",
            age_hours=4,
            source_weight_sum=7,
            max_source_weight_sum=9,
            i_score=7,
            v_score=7,
            c_score=6,
            s_score=5,
            k_score=8,
            n_score=7,
            e_score=7,
            x_score=7,
            q_score="0.9",
        ),
        calculate_full_news_score(
            news_id=3,
            macro_topic="creative_cast_production",
            age_hours=6,
            source_weight_sum=6,
            max_source_weight_sum=9,
            i_score=6,
            v_score=6,
            c_score=5,
            s_score=4,
            k_score=5,
            n_score=6,
            e_score=6,
            x_score=6,
            q_score="0.9",
        ),
        calculate_full_news_score(
            news_id=4,
            macro_topic="business_economy_law",
            age_hours=1,
            source_weight_sum=8,
            max_source_weight_sum=9,
            i_score=9,
            v_score=8,
            c_score=7,
            s_score=6,
            k_score=7,
            n_score=6,
            e_score=5,
            x_score=6,
            q_score=1,
        ),
    )


def test_diversity_and_combination_selection() -> None:
    """Проверяет D, TOP(S), перебор комбинаций и порядок TOP-3."""

    assert (
        calculate_diversity_score(
            (
                "business_economy_law",
                "people_conflicts_legal",
                "creative_cast_production",
            )
        )
        == Decimal("10.000000")
    )

    assert (
        calculate_diversity_score(
            (
                "business_economy_law",
                "business_economy_law",
                "creative_cast_production",
            )
        )
        == Decimal("6.000000")
    )

    assert (
        calculate_diversity_score(
            (
                "business_economy_law",
                "business_economy_law",
                "business_economy_law",
            )
        )
        == ZERO
    )

    selection = select_top3_combination(
        _selection_fixture()
    )

    assert selection.formula_version == FULL_FORMULA_VERSION
    assert selection.eligible_count == 4
    assert len(selection.combinations) == 4
    assert selection.winner.combination_rank == 1
    assert selection.winner.is_winner is True
    assert selection.winner.news_ids == (2, 3, 4)
    assert selection.winner.ordered_news_ids == (4, 2, 3)
    assert selection.winner.mean_individual_score == Decimal("6.457519")
    assert selection.winner.diversity_score == Decimal("10.000000")
    assert selection.winner.final_top_score == Decimal("7.957519")
    assert selection.winner.distinct_macro_topic_count == 3

    assert sum(
        1
        for combination in selection.combinations
        if combination.is_winner
    ) == 1

    print("Diversity and TOP(S) selection: OK")


def _manual_full_score(
    *,
    news_id: int,
    individual_score: str,
    m_score: str,
    q_score: str,
    f_score: str,
) -> FullNewsScore:
    """Создаёт контролируемую оценку для проверки tie-break правил."""

    f_value = Decimal(f_score)
    m_value = Decimal(m_score)
    q_value = Decimal(q_score)
    individual_value = Decimal(individual_score)

    components = create_score_components(
        f_score=f_value,
        m_score=m_value,
        r_score=0,
        h_score=0,
        q_score=q_value,
    )

    individual = CalculatedScore(
        formula_version="tie_break_test",
        components=components,
        freshness_component=ZERO,
        magnitude_component=ZERO,
        resonance_component=ZERO,
        hook_quality_component=ZERO,
        individual_score=individual_value,
    )

    resonance = ResonanceCalculation(
        v_score=None,
        c_score=None,
        s_score=None,
        effective_v_weight=ZERO,
        effective_c_weight=ZERO,
        effective_s_weight=ZERO,
        confidence=RESONANCE_CONFIDENCE_UNAVAILABLE,
        r_score=ZERO,
    )

    return FullNewsScore(
        formula_version=FULL_FORMULA_VERSION,
        news_id=news_id,
        macro_topic="business_economy_law",
        age_hours=ZERO,
        source_weight_sum=ZERO,
        max_source_weight_sum=ZERO,
        f_score=f_value,
        u_score=ZERO,
        i_score=ZERO,
        m_score=m_value,
        resonance=resonance,
        k_score=ZERO,
        n_score=ZERO,
        e_score=ZERO,
        x_score=ZERO,
        h_score=ZERO,
        q_score=q_value,
        individual=individual,
        is_eligible=True,
        exclusion_reason=None,
    )


def test_tie_break_rules() -> None:
    """Проверяет M, Q, F и news_id при равном TOP(S)."""

    by_m = select_top3_combination(
        tuple(
            _manual_full_score(
                news_id=news_id,
                individual_score="5.000000",
                m_score=str(news_id),
                q_score="0.500000",
                f_score="5.000000",
            )
            for news_id in range(1, 5)
        )
    )
    assert by_m.winner.news_ids == (2, 3, 4)

    by_q = select_top3_combination(
        tuple(
            _manual_full_score(
                news_id=news_id,
                individual_score="5.000000",
                m_score="5.000000",
                q_score=f"0.{news_id}",
                f_score="5.000000",
            )
            for news_id in range(1, 5)
        )
    )
    assert by_q.winner.news_ids == (2, 3, 4)

    by_f = select_top3_combination(
        tuple(
            _manual_full_score(
                news_id=news_id,
                individual_score="5.000000",
                m_score="5.000000",
                q_score="0.500000",
                f_score=str(news_id),
            )
            for news_id in range(1, 5)
        )
    )
    assert by_f.winner.news_ids == (2, 3, 4)

    stable_news_ids = select_top3_combination(
        tuple(
            _manual_full_score(
                news_id=news_id,
                individual_score="5.000000",
                m_score="5.000000",
                q_score="0.500000",
                f_score="5.000000",
            )
            for news_id in range(1, 5)
        )
    )
    assert stable_news_ids.winner.news_ids == (1, 2, 3)

    print("Combination tie-break rules: OK")


def test_story_cluster_selection_policy() -> None:
    """Приоритизирует три разные сюжетные семьи."""

    fixture = _selection_fixture()
    legacy = select_top3_combination(fixture)

    filtered = select_top3_combination(
        fixture,
        story_cluster_keys_by_news_id={
            1: "independent_story",
            2: "paramount_warner_merger",
            3: "spider_man_box_office",
            4: "paramount_warner_merger",
        },
    )

    assert (
        filtered.selection_policy_version
        == TOP3_SELECTION_POLICY_VERSION
    )
    assert filtered.story_cluster_filter_applied is True
    assert filtered.story_cluster_fallback_used is False
    assert (
        filtered.winner.distinct_story_cluster_count
        == 3
    )
    assert filtered.winner.passes_story_cluster_filter is True
    assert not {2, 4}.issubset(
        set(filtered.winner.news_ids)
    )

    fallback = select_top3_combination(
        fixture,
        story_cluster_keys_by_news_id={
            news_id: "single_megastory"
            for news_id in range(1, 5)
        },
    )

    assert fallback.story_cluster_filter_applied is False
    assert fallback.story_cluster_fallback_used is True
    assert fallback.winner.news_ids == legacy.winner.news_ids
    assert (
        fallback.winner.distinct_story_cluster_count
        == 1
    )

    print("Story cluster selection policy: OK")


def test_invalid_combination_inputs() -> None:
    """Проверяет блокировку недостаточного числа и дублей."""

    fixture = _selection_fixture()

    try:
        select_top3_combination(fixture[:2])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Выбор TOP-3 из двух новостей не был заблокирован."
        )

    try:
        select_top3_combination(
            (
                fixture[0],
                fixture[0],
                fixture[1],
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Повторяющийся news_id не был заблокирован."
        )

    print("Invalid combination inputs: OK")


def main() -> int:
    """Запускает изолированные тесты полной формулы."""

    print(f"full_formula_version={FULL_FORMULA_VERSION}")
    print("database_changes=not_performed")
    print("openai_requests=not_performed")
    print("telegram_requests=not_performed")
    print()

    test_freshness()
    test_media_reach_and_audience_normalization()
    test_aggregated_components()
    test_full_news_score_and_eligibility()
    test_diversity_and_combination_selection()
    test_tie_break_rules()
    test_story_cluster_selection_policy()
    test_invalid_combination_inputs()

    print()
    print("Full TOP-3 formula test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
