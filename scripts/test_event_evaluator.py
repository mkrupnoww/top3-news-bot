from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.ranking.event_evaluator import (
    EVENT_EVALUATOR_VERSION,
    MACRO_TOPICS,
    SOURCE_RELATIONS,
    EventAssessment,
    EventMemberAssessment,
)


def build_valid_members() -> tuple[
    EventMemberAssessment,
    ...,
]:
    """Создаёт корректный состав одного инфоповода."""

    return (
        EventMemberAssessment(
            news_id=101,
            source_relation="primary",
            is_representative=True,
            is_independent_source=True,
            counts_toward_reach=True,
            source_weight=3,
            membership_reason=(
                "Первичная публикация сильного "
                "профильного источника."
            ),
        ),
        EventMemberAssessment(
            news_id=102,
            source_relation="independent",
            is_representative=False,
            is_independent_source=True,
            counts_toward_reach=True,
            source_weight=2,
            membership_reason=(
                "Независимое подтверждение "
                "крупного национального СМИ."
            ),
        ),
        EventMemberAssessment(
            news_id=103,
            source_relation="syndicated",
            is_representative=False,
            is_independent_source=False,
            counts_toward_reach=False,
            source_weight=0,
            membership_reason=(
                "Синдицированная перепечатка, "
                "не увеличивающая охват."
            ),
        ),
    )


def build_valid_event() -> EventAssessment:
    """Создаёт корректную экспертную оценку инфоповода."""

    return EventAssessment(
        representative_news_id=101,
        event_title=(
            "Studio Announces International "
            "Film Project"
        ),
        event_time_utc=datetime(
            2026,
            8,
            2,
            10,
            30,
            tzinfo=timezone(
                timedelta(hours=2)
            ),
        ),
        macro_topic="creative_cast_production",
        story_cluster_key="test_story",
        i_score="7.5",
        k_score="2.0",
        n_score="6.5",
        e_score="5.0",
        x_score="7.0",
        q_score="0.95",
        impact_reason=(
            "Проект влияет на международное "
            "производство и дистрибуцию."
        ),
        hook_reason=(
            "Необычный масштаб и редкое "
            "международное партнёрство."
        ),
        q_reason=(
            "Есть первичная публикация и "
            "независимое подтверждение."
        ),
        members=build_valid_members(),
    )


def test_valid_event() -> None:
    """Проверяет нормализацию корректного инфоповода."""

    event = build_valid_event()

    assert event.representative_news_id == 101
    assert event.event_title == (
        "Studio Announces International "
        "Film Project"
    )

    assert event.event_time_utc == datetime(
        2026,
        8,
        2,
        8,
        30,
        tzinfo=timezone.utc,
    )

    assert event.macro_topic == (
        "creative_cast_production"
    )
    assert event.story_cluster_key == (
        "test_story"
    )

    assert event.i_score == Decimal("7.5")
    assert event.q_score == Decimal("0.95")

    assert event.member_news_ids == (
        101,
        102,
        103,
    )

    assert event.source_weight_sum == 5

    print("Valid event normalization: OK")
    print(
        "event_time_utc="
        f"{event.event_time_utc.isoformat()}"
    )
    print(
        f"source_weight_sum="
        f"{event.source_weight_sum}"
    )


def test_allowed_enums() -> None:
    """Проверяет зафиксированные справочники."""

    assert MACRO_TOPICS == frozenset(
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

    assert SOURCE_RELATIONS == frozenset(
        {
            "primary",
            "independent",
            "syndicated",
            "duplicate",
        }
    )

    print()
    print("Macro topics and relations: OK")


def test_invalid_reach_member() -> None:
    """Блокирует невалидный источник охвата."""

    try:
        EventMemberAssessment(
            news_id=201,
            source_relation="syndicated",
            is_representative=False,
            is_independent_source=False,
            counts_toward_reach=True,
            source_weight=2,
            membership_reason=(
                "Некорректная тестовая строка."
            ),
        )
    except ValueError:
        print()
        print("Invalid reach member blocking: OK")
        return

    raise AssertionError(
        "Синдицированный источник ошибочно "
        "допущен в расчёт охвата."
    )


def test_duplicate_member_ids() -> None:
    """Блокирует повторяющиеся news_id."""

    members = (
        EventMemberAssessment(
            news_id=301,
            source_relation="primary",
            is_representative=True,
            is_independent_source=True,
            counts_toward_reach=True,
            source_weight=3,
            membership_reason="Первый участник.",
        ),
        EventMemberAssessment(
            news_id=301,
            source_relation="independent",
            is_representative=False,
            is_independent_source=True,
            counts_toward_reach=True,
            source_weight=2,
            membership_reason="Повторяющийся участник.",
        ),
    )

    try:
        EventAssessment(
            representative_news_id=301,
            event_title="Duplicate member test",
            event_time_utc=datetime.now(
                timezone.utc
            ),
            macro_topic="other",
            story_cluster_key="test_story",
            i_score=5,
            k_score=5,
            n_score=5,
            e_score=5,
            x_score=5,
            q_score=1,
            impact_reason="Тест.",
            hook_reason="Тест.",
            q_reason="Тест.",
            members=members,
        )
    except ValueError as error:
        assert "повторяющиеся news_id" in str(
            error
        )

        print()
        print("Duplicate member blocking: OK")
        return

    raise AssertionError(
        "Повторяющиеся news_id не были "
        "заблокированы."
    )


def test_representative_rules() -> None:
    """Проверяет единственного представителя."""

    members = (
        EventMemberAssessment(
            news_id=401,
            source_relation="primary",
            is_representative=True,
            is_independent_source=True,
            counts_toward_reach=True,
            source_weight=3,
            membership_reason="Первый представитель.",
        ),
        EventMemberAssessment(
            news_id=402,
            source_relation="independent",
            is_representative=True,
            is_independent_source=True,
            counts_toward_reach=True,
            source_weight=2,
            membership_reason="Второй представитель.",
        ),
    )

    try:
        EventAssessment(
            representative_news_id=401,
            event_title="Representative test",
            event_time_utc=datetime.now(
                timezone.utc
            ),
            macro_topic="other",
            story_cluster_key="test_story",
            i_score=5,
            k_score=5,
            n_score=5,
            e_score=5,
            x_score=5,
            q_score=1,
            impact_reason="Тест.",
            hook_reason="Тест.",
            q_reason="Тест.",
            members=members,
        )
    except ValueError as error:
        assert "ровно одного" in str(error)

        print()
        print(
            "Representative uniqueness blocking: OK"
        )
        return

    raise AssertionError(
        "Несколько представителей не были "
        "заблокированы."
    )


def test_invalid_macro_topic() -> None:
    """Блокирует неизвестную макротему."""

    valid = build_valid_event()

    try:
        EventAssessment(
            representative_news_id=(
                valid.representative_news_id
            ),
            event_title=valid.event_title,
            event_time_utc=valid.event_time_utc,
            macro_topic="unknown_topic",
            story_cluster_key="test_story",
            i_score=valid.i_score,
            k_score=valid.k_score,
            n_score=valid.n_score,
            e_score=valid.e_score,
            x_score=valid.x_score,
            q_score=valid.q_score,
            impact_reason=valid.impact_reason,
            hook_reason=valid.hook_reason,
            q_reason=valid.q_reason,
            members=valid.members,
        )
    except ValueError as error:
        assert "macro_topic" in str(error)

        print()
        print("Invalid macro topic blocking: OK")
        return

    raise AssertionError(
        "Неизвестная макротема не была "
        "заблокирована."
    )


def test_invalid_story_cluster_key() -> None:
    """Блокирует некорректный ключ сюжетной семьи."""

    valid = build_valid_event()

    try:
        EventAssessment(
            representative_news_id=(
                valid.representative_news_id
            ),
            event_title=valid.event_title,
            event_time_utc=valid.event_time_utc,
            macro_topic=valid.macro_topic,
            story_cluster_key="Paramount Warner",
            i_score=valid.i_score,
            k_score=valid.k_score,
            n_score=valid.n_score,
            e_score=valid.e_score,
            x_score=valid.x_score,
            q_score=valid.q_score,
            impact_reason=valid.impact_reason,
            hook_reason=valid.hook_reason,
            q_reason=valid.q_reason,
            members=valid.members,
        )
    except ValueError as error:
        assert "story_cluster_key" in str(error)

        print()
        print(
            "Invalid story cluster key blocking: OK"
        )
        return

    raise AssertionError(
        "Некорректный story_cluster_key "
        "не был заблокирован."
    )


def test_invalid_score_range() -> None:
    """Блокирует экспертную оценку вне шкалы."""

    valid = build_valid_event()

    try:
        EventAssessment(
            representative_news_id=(
                valid.representative_news_id
            ),
            event_title=valid.event_title,
            event_time_utc=valid.event_time_utc,
            macro_topic=valid.macro_topic,
            story_cluster_key="test_story",
            i_score=11,
            k_score=valid.k_score,
            n_score=valid.n_score,
            e_score=valid.e_score,
            x_score=valid.x_score,
            q_score=valid.q_score,
            impact_reason=valid.impact_reason,
            hook_reason=valid.hook_reason,
            q_reason=valid.q_reason,
            members=valid.members,
        )
    except ValueError as error:
        assert "i_score" in str(error)

        print()
        print("Invalid expert score blocking: OK")
        return

    raise AssertionError(
        "Оценка вне диапазона не была "
        "заблокирована."
    )


def main() -> int:
    """Запускает тест event-level контракта."""

    print(
        "event_evaluator_version="
        f"{EVENT_EVALUATOR_VERSION}"
    )
    print(
        "openai_requests=not_performed"
    )
    print(
        "database_changes=not_performed"
    )
    print(
        "telegram_requests=not_performed"
    )
    print()

    test_valid_event()
    test_allowed_enums()
    test_invalid_reach_member()
    test_duplicate_member_ids()
    test_representative_rules()
    test_invalid_macro_topic()
    test_invalid_story_cluster_key()
    test_invalid_score_range()

    print()
    print("Event evaluator contract test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
