from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)
from app.ranking.event_evaluator import (
    EventAssessment,
    EventMemberAssessment,
)
from app.ranking.event_formula_pipeline import (
    EventAudienceMetrics,
    calculate_event_formula,
    calculate_event_scores,
    select_event_top3,
)
from app.ranking.full_formula import (
    EXCLUSION_REASON_QUALITY_ZERO,
    EXCLUSION_REASON_SCORE_BELOW_THRESHOLD,
    FULL_FORMULA_VERSION,
    RESONANCE_CONFIDENCE_FULL,
    RESONANCE_CONFIDENCE_PARTIAL,
    RESONANCE_CONFIDENCE_UNAVAILABLE,
)


WINDOW_END = datetime(
    2026,
    8,
    2,
    12,
    0,
    tzinfo=timezone.utc,
)

WINDOW_START = WINDOW_END - timedelta(
    hours=24
)


def build_candidate(
    *,
    news_id: int,
    published_hours_ago: int,
) -> NewsCandidate:
    """Создаёт одну публикацию-кандидат."""

    return NewsCandidate(
        news_id=news_id,
        source_id=news_id,
        source_code=f"source_{news_id}",
        source_name=f"Source {news_id}",
        collection_priority=100,
        processing_status="collected",
        title=f"Test movie news {news_id}",
        summary=f"Summary for {news_id}",
        author_name="Test Author",
        source_published_at=(
            WINDOW_END
            - timedelta(
                hours=published_hours_ago
            )
        ),
        age_hours=float(
            published_hours_ago
        ),
        source_url=(
            f"https://example.com/{news_id}"
        ),
        primary_image_url=None,
    )


def build_selection() -> CandidateSelectionResult:
    """Создаёт суточную выборку из пяти статей."""

    return CandidateSelectionResult(
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        window_hours=24.0,
        candidates=tuple(
            build_candidate(
                news_id=news_id,
                published_hours_ago=age,
            )
            for news_id, age in (
                (101, 1),
                (102, 2),
                (103, 4),
                (104, 3),
                (105, 2),
            )
        ),
    )


def build_member(
    *,
    news_id: int,
    source_relation: str,
    is_representative: bool,
    source_weight: int,
) -> EventMemberAssessment:
    """Создаёт участника инфоповода."""

    counts_toward_reach = (
        source_relation
        in {
            "primary",
            "independent",
        }
        and source_weight > 0
    )

    return EventMemberAssessment(
        news_id=news_id,
        source_relation=source_relation,
        is_representative=(
            is_representative
        ),
        is_independent_source=(
            counts_toward_reach
        ),
        counts_toward_reach=(
            counts_toward_reach
        ),
        source_weight=source_weight,
        membership_reason=(
            f"Test relation for news {news_id}."
        ),
    )


def build_events() -> tuple[
    EventAssessment,
    ...,
]:
    """Создаёт четыре инфоповода из пяти статей."""

    return (
        EventAssessment(
            representative_news_id=101,
            event_title=(
                "International production "
                "announcement"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=1)
            ),
            macro_topic=(
                "creative_cast_production"
            ),
            story_cluster_key="international_production",
            i_score="8.0",
            k_score="6.0",
            n_score="7.0",
            e_score="6.0",
            x_score="8.0",
            q_score="0.95",
            impact_reason=(
                "Large international production."
            ),
            hook_reason=(
                "Strong scale and novelty."
            ),
            q_reason=(
                "Primary and independent sources."
            ),
            members=(
                build_member(
                    news_id=101,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=3,
                ),
                build_member(
                    news_id=102,
                    source_relation=(
                        "independent"
                    ),
                    is_representative=False,
                    source_weight=2,
                ),
            ),
        ),
        EventAssessment(
            representative_news_id=103,
            event_title=(
                "Film company business decision"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=4)
            ),
            macro_topic=(
                "business_economy_law"
            ),
            story_cluster_key="film_company_business_decision",
            i_score="7.5",
            k_score="4.0",
            n_score="6.0",
            e_score="5.0",
            x_score="6.0",
            q_score="0.90",
            impact_reason=(
                "Meaningful business impact."
            ),
            hook_reason=(
                "Noticeable market development."
            ),
            q_reason=(
                "Reliable primary publication."
            ),
            members=(
                build_member(
                    news_id=103,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=3,
                ),
            ),
        ),
        EventAssessment(
            representative_news_id=104,
            event_title=(
                "Festival programme expansion"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=3)
            ),
            macro_topic=(
                "festivals_awards_criticism"
            ),
            story_cluster_key="festival_programme_expansion",
            i_score="8.5",
            k_score="6.0",
            n_score="8.0",
            e_score="7.0",
            x_score="8.0",
            q_score="0.95",
            impact_reason=(
                "Important festival expansion."
            ),
            hook_reason=(
                "Distinctive festival development."
            ),
            q_reason=(
                "Confirmed by a primary source."
            ),
            members=(
                build_member(
                    news_id=104,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=3,
                ),
            ),
        ),
        EventAssessment(
            representative_news_id=105,
            event_title=(
                "Unverified film rumour"
            ),
            event_time_utc=(
                WINDOW_END
                - timedelta(hours=2)
            ),
            macro_topic="other",
            story_cluster_key="unverified_film_rumour",
            i_score="9.0",
            k_score="9.0",
            n_score="9.0",
            e_score="9.0",
            x_score="9.0",
            q_score="0.0",
            impact_reason=(
                "Potentially large impact."
            ),
            hook_reason=(
                "Highly unusual claim."
            ),
            q_reason=(
                "The claim cannot be verified."
            ),
            members=(
                build_member(
                    news_id=105,
                    source_relation="primary",
                    is_representative=True,
                    source_weight=2,
                ),
            ),
        ),
    )


def build_audience_metrics() -> tuple[
    EventAudienceMetrics,
    ...,
]:
    """Создаёт полные и частичные метрики."""

    return (
        EventAudienceMetrics(
            news_id=101,
            view_count=1000,
            comment_count=100,
            share_count=50,
        ),
        EventAudienceMetrics(
            news_id=103,
            view_count=500,
            comment_count=None,
            share_count=25,
        ),
    )


def test_complete_calculation() -> None:
    """Проверяет полный детерминированный расчёт."""

    result = calculate_event_formula(
        selection=build_selection(),
        events=build_events(),
        audience_metrics=(
            build_audience_metrics()
        ),
    )

    assert result.formula_version == (
        FULL_FORMULA_VERSION
    )

    assert result.window_start == WINDOW_START
    assert result.window_end == WINDOW_END

    assert (
        result
        .audience_maxima
        .max_view_count
        == 1000
    )

    assert (
        result
        .audience_maxima
        .max_comment_count
        == 100
    )

    assert (
        result
        .audience_maxima
        .max_share_count
        == 50
    )

    assert len(result.calculated_events) == 4
    assert len(result.scores) == 4

    by_news_id = {
        item.score.news_id: item
        for item in result.calculated_events
    }

    first = by_news_id[101]
    second = by_news_id[103]
    third = by_news_id[104]
    excluded = by_news_id[105]

    assert first.event.member_news_ids == (
        101,
        102,
    )

    assert (
        first.score.source_weight_sum
        == Decimal("5.000000")
    )

    assert (
        first.score.max_source_weight_sum
        == Decimal("5.000000")
    )

    assert first.score.age_hours == (
        Decimal("1.000000")
    )

    assert (
        first.score.resonance.confidence
        == RESONANCE_CONFIDENCE_FULL
    )

    assert (
        first.score.resonance.r_score
        == Decimal("10.000000")
    )

    assert (
        second.score.resonance.confidence
        == RESONANCE_CONFIDENCE_PARTIAL
    )

    assert (
        second.score.resonance.c_score
        is None
    )

    assert (
        second.score.resonance.r_score
        > Decimal("0")
    )

    assert (
        third.score.resonance.confidence
        == RESONANCE_CONFIDENCE_UNAVAILABLE
    )

    assert (
        third.score.resonance.r_score
        == Decimal("0.000000")
    )

    assert third.audience_metrics == (
        EventAudienceMetrics(news_id=104)
    )

    assert excluded.score.is_eligible is False

    assert excluded.score.exclusion_reason == (
        EXCLUSION_REASON_QUALITY_ZERO
    )

    assert (
        result.top3_selection.eligible_count
        == 3
    )
    assert result.strict_eligible_count == 3
    assert result.eligibility_fallback_used is False
    assert result.eligibility_fallback_threshold is None
    assert result.fallback_promoted_news_ids == ()

    assert len(
        result.top3_selection.combinations
    ) == 1

    winner = result.top3_selection.winner

    assert winner.news_ids == (
        101,
        103,
        104,
    )

    assert winner.is_winner is True

    assert (
        winner.distinct_macro_topic_count
        == 3
    )

    assert (
        winner.diversity_score
        == Decimal("10.000000")
    )
    assert (
        result.top3_selection
        .selection_policy_version
        == "story_cluster_diversity_v1"
    )
    assert (
        result.top3_selection
        .story_cluster_filter_applied
        is True
    )
    assert (
        winner.distinct_story_cluster_count
        == 3
    )
    assert winner.passes_story_cluster_filter is True

    assert set(
        winner.ordered_news_ids
    ) == {
        101,
        103,
        104,
    }

    print("Complete event formula calculation: OK")
    print(
        "representative_news_ids="
        + ",".join(
            str(item.score.news_id)
            for item
            in result.calculated_events
        )
    )
    print(
        "eligible_count="
        f"{result.top3_selection.eligible_count}"
    )
    print(
        "winner_news_ids="
        + ",".join(
            str(news_id)
            for news_id in winner.news_ids
        )
    )
    print(
        "diversity_score="
        f"{winner.diversity_score}"
    )


def test_story_cluster_diversity_policy() -> None:
    """Проверяет фильтр одной мегатемы и fallback."""

    events = list(build_events())
    events[1] = replace(
        events[1],
        story_cluster_key=(
            events[0].story_cluster_key
        ),
    )
    events[3] = replace(
        events[3],
        story_cluster_key="confirmed_rumour_event",
        q_score=Decimal("1.0"),
    )

    filtered = calculate_event_formula(
        selection=build_selection(),
        events=tuple(events),
    )

    assert (
        filtered.top3_selection
        .story_cluster_filter_applied
        is True
    )
    assert (
        filtered.top3_selection
        .story_cluster_fallback_used
        is False
    )
    assert (
        filtered.top3_selection.winner
        .distinct_story_cluster_count
        == 3
    )
    assert not {
        101,
        103,
    }.issubset(
        set(
            filtered.top3_selection
            .winner.news_ids
        )
    )

    fallback_events = tuple(
        replace(
            event,
            story_cluster_key="single_megastory",
        )
        for event in build_events()
    )
    fallback = calculate_event_formula(
        selection=build_selection(),
        events=fallback_events,
    )

    assert (
        fallback.top3_selection
        .story_cluster_filter_applied
        is False
    )
    assert (
        fallback.top3_selection
        .story_cluster_fallback_used
        is True
    )
    assert (
        fallback.top3_selection.winner
        .distinct_story_cluster_count
        == 1
    )

    print()
    print("Story cluster diversity policy: OK")
    print("fallback_preserves_top3=true")


def test_no_audience_metrics() -> None:
    """Проверяет прозрачный R=0 без метрик."""

    result = calculate_event_formula(
        selection=build_selection(),
        events=build_events(),
    )

    assert all(
        item.score.resonance.confidence
        == RESONANCE_CONFIDENCE_UNAVAILABLE
        for item in result.calculated_events
    )

    assert all(
        item.score.resonance.r_score
        == Decimal("0.000000")
        for item in result.calculated_events
    )

    print()
    print("Unavailable audience metrics: OK")
    print("all_resonance_scores=0")



def test_insufficient_top3_score_floor_fallback() -> None:
    """Проверяет promotion score 3.0..3.5 при strict eligible_count=2."""

    score_calculation = calculate_event_scores(
        selection=build_selection(),
        events=build_events(),
    )

    adjusted_events = tuple(
        replace(
            item,
            score=replace(
                item.score,
                individual=replace(
                    item.score.individual,
                    individual_score=(
                        Decimal("3.400000")
                    ),
                ),
                is_eligible=False,
                exclusion_reason=(
                    EXCLUSION_REASON_SCORE_BELOW_THRESHOLD
                ),
            ),
        )
        if item.score.news_id == 104
        else item
        for item in score_calculation.calculated_events
    )

    adjusted_calculation = replace(
        score_calculation,
        calculated_events=adjusted_events,
    )

    assert adjusted_calculation.eligible_count == 2

    result = select_event_top3(
        adjusted_calculation
    )

    assert result.strict_eligible_count == 2
    assert result.eligibility_fallback_used is True
    assert result.eligibility_fallback_threshold == (
        Decimal("3.000000")
    )
    assert result.fallback_promoted_news_ids == (104,)
    assert result.top3_selection.eligible_count == 3

    effective_by_news_id = {
        item.score.news_id: item.score
        for item in result.calculated_events
    }

    assert effective_by_news_id[104].is_eligible is True
    assert effective_by_news_id[104].exclusion_reason is None
    assert effective_by_news_id[105].is_eligible is False
    assert effective_by_news_id[105].exclusion_reason == (
        EXCLUSION_REASON_QUALITY_ZERO
    )

    assert set(
        result.top3_selection.winner.news_ids
    ) == {101, 103, 104}

    print()
    print("Insufficient TOP-3 score-floor fallback: OK")
    print("strict_eligible_count=2")
    print("effective_eligible_count=3")
    print("fallback_promoted_news_ids=104")



def test_intermediate_scores_survive_insufficient_top3(
) -> None:
    """Сохраняет баллы при eligible_count меньше трёх."""

    events = list(build_events())

    events[1] = replace(
        events[1],
        q_score=Decimal("0"),
        q_reason=(
            "Synthetic diagnostic exclusion."
        ),
    )

    events[2] = replace(
        events[2],
        q_score=Decimal("0"),
        q_reason=(
            "Synthetic diagnostic exclusion."
        ),
    )

    score_calculation = calculate_event_scores(
        selection=build_selection(),
        events=tuple(events),
    )

    assert score_calculation.formula_version == (
        FULL_FORMULA_VERSION
    )

    assert len(
        score_calculation.calculated_events
    ) == 4

    assert len(score_calculation.scores) == 4

    assert score_calculation.eligible_count == 1

    eligible_news_ids = tuple(
        item.score.news_id
        for item
        in score_calculation.calculated_events
        if item.score.is_eligible
    )

    excluded_news_ids = tuple(
        item.score.news_id
        for item
        in score_calculation.calculated_events
        if not item.score.is_eligible
    )

    assert eligible_news_ids == (
        101,
    )

    assert excluded_news_ids == (
        103,
        104,
        105,
    )

    assert all(
        item.score.exclusion_reason
        == EXCLUSION_REASON_QUALITY_ZERO
        for item
        in score_calculation.calculated_events
        if not item.score.is_eligible
    )

    try:
        select_event_top3(
            score_calculation
        )
    except ValueError as error:
        assert "eligible_count=1" in str(error)

        print()
        print(
            "Intermediate scores on insufficient "
            "TOP-3: OK"
        )
        print("calculated_event_count=4")
        print("eligible_count=1")
        print(
            "eligible_news_ids="
            + ",".join(
                str(news_id)
                for news_id
                in eligible_news_ids
            )
        )
        print(
            "excluded_news_ids="
            + ",".join(
                str(news_id)
                for news_id
                in excluded_news_ids
            )
        )
        return

    raise AssertionError(
        "TOP-3 был ошибочно выбран "
        "при eligible_count=1."
    )

def test_invalid_window() -> None:
    """Блокирует окно, отличное от 24 часов."""

    invalid_selection = replace(
        build_selection(),
        window_start=(
            WINDOW_END
            - timedelta(hours=23)
        ),
        window_hours=23.0,
    )

    try:
        calculate_event_formula(
            selection=invalid_selection,
            events=build_events(),
        )
    except ValueError as error:
        assert "ровно 24 часа" in str(error)

        print()
        print("Invalid window blocking: OK")
        return

    raise AssertionError(
        "Окно, отличное от 24 часов, "
        "не было заблокировано."
    )


def test_missing_candidate_coverage() -> None:
    """Блокирует кандидата без инфоповода."""

    incomplete_events = build_events()[:-1]

    try:
        calculate_event_formula(
            selection=build_selection(),
            events=incomplete_events,
        )
    except ValueError as error:
        assert "missing=[105]" in str(error)

        print()
        print(
            "Missing candidate coverage blocking: OK"
        )
        return

    raise AssertionError(
        "Неполное покрытие кандидатов "
        "не было заблокировано."
    )


def test_unknown_metrics_event() -> None:
    """Блокирует метрики неизвестного события."""

    invalid_metrics = (
        *build_audience_metrics(),
        EventAudienceMetrics(
            news_id=999,
            view_count=1,
        ),
    )

    try:
        calculate_event_formula(
            selection=build_selection(),
            events=build_events(),
            audience_metrics=invalid_metrics,
        )
    except ValueError as error:
        assert "999" in str(error)

        print()
        print(
            "Unknown audience event blocking: OK"
        )
        return

    raise AssertionError(
        "Метрики неизвестного инфоповода "
        "не были заблокированы."
    )


def test_duplicate_metrics() -> None:
    """Блокирует повторные метрики события."""

    duplicate_metrics = (
        EventAudienceMetrics(
            news_id=101,
            view_count=100,
        ),
        EventAudienceMetrics(
            news_id=101,
            view_count=200,
        ),
    )

    try:
        calculate_event_formula(
            selection=build_selection(),
            events=build_events(),
            audience_metrics=duplicate_metrics,
        )
    except ValueError as error:
        assert "повторяющиеся news_id" in str(
            error
        )

        print()
        print(
            "Duplicate audience metrics blocking: OK"
        )
        return

    raise AssertionError(
        "Повторные audience-метрики "
        "не были заблокированы."
    )


def test_event_outside_window() -> None:
    """Блокирует событие вне суточного окна."""

    events = list(build_events())

    events[0] = replace(
        events[0],
        event_time_utc=(
            WINDOW_START
            - timedelta(seconds=1)
        ),
    )

    try:
        calculate_event_formula(
            selection=build_selection(),
            events=tuple(events),
        )
    except ValueError as error:
        assert "вне суточного окна" in str(
            error
        )
        assert "101" in str(error)

        print()
        print(
            "Event outside window blocking: OK"
        )
        return

    raise AssertionError(
        "Событие вне окна не было "
        "заблокировано."
    )


def main() -> int:
    """Запускает тест вычислительного слоя."""

    print(
        "formula_version="
        f"{FULL_FORMULA_VERSION}"
    )
    print("openai_requests=not_performed")
    print("database_changes=not_performed")
    print("telegram_requests=not_performed")
    print()

    test_complete_calculation()
    test_story_cluster_diversity_policy()
    test_no_audience_metrics()
    test_insufficient_top3_score_floor_fallback()
    test_intermediate_scores_survive_insufficient_top3()
    test_invalid_window()
    test_missing_candidate_coverage()
    test_unknown_metrics_event()
    test_duplicate_metrics()
    test_event_outside_window()

    print()
    print("Event formula pipeline test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
