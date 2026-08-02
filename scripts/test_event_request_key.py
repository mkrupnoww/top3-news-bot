from datetime import datetime, timedelta, timezone
import json

from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.event_evaluator import (
    EventRankingModelRequest,
)
from app.ranking.event_formula_pipeline import (
    EventAudienceMetrics,
)
from app.ranking.event_request_key import (
    EVENT_REQUEST_KEY_VERSION,
    create_event_ranking_request_key,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)


WINDOW_START = datetime(
    2026,
    7,
    30,
    11,
    21,
    tzinfo=timezone.utc,
)

WINDOW_END = datetime(
    2026,
    7,
    31,
    11,
    21,
    tzinfo=timezone.utc,
)

NEWS_IDS = (
    7,
    8,
    9,
)


def build_metadata(
    *,
    model_name: str = "gpt-5.6-terra",
) -> RankingEvaluatorMetadata:
    """Создаёт метаданные event-оценщика."""

    return RankingEvaluatorMetadata(
        run_mode="openai_event_ranking",
        evaluator_name=(
            "OpenAIEventRankingEvaluator"
        ),
        evaluator_version=(
            "event_ranking_evaluator_v1"
        ),
        prompt_version=(
            "movie_news_event_ranking_prompt_v1"
        ),
        model_name=model_name,
    )


def build_model_request(
    *,
    model: str = "gpt-5.6-terra",
    input_text: str | None = None,
) -> EventRankingModelRequest:
    """Создаёт event-level запрос."""

    return EventRankingModelRequest(
        model=model,
        instructions=(
            "Сгруппируй публикации "
            "в киноинфоповоды."
        ),
        input_text=(
            input_text
            or (
                '{"candidates":['
                '{"news_id":7},'
                '{"news_id":8},'
                '{"news_id":9}'
                "]}"
            )
        ),
    )


def build_metrics(
) -> tuple[
    EventAudienceMetrics,
    ...,
]:
    """Создаёт снимок audience-метрик."""

    return (
        EventAudienceMetrics(
            news_id=9,
            view_count=300,
            comment_count=30,
            share_count=3,
        ),
        EventAudienceMetrics(
            news_id=7,
            view_count=100,
            comment_count=None,
            share_count=1,
        ),
    )


def build_request_key(
    *,
    metadata: RankingEvaluatorMetadata
    | None = None,
    model_request: EventRankingModelRequest
    | None = None,
    window_started_at: datetime = WINDOW_START,
    window_finished_at: datetime = WINDOW_END,
    news_ids: tuple[int, ...] = NEWS_IDS,
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ] | None = None,
):
    """Создаёт типовой event request_key."""

    return create_event_ranking_request_key(
        formula_version=FULL_FORMULA_VERSION,
        metadata=metadata or build_metadata(),
        model_request=(
            model_request
            or build_model_request()
        ),
        window_started_at=window_started_at,
        window_finished_at=(
            window_finished_at
        ),
        news_ids=news_ids,
        audience_metrics=(
            build_metrics()
            if audience_metrics is None
            else audience_metrics
        ),
    )


def test_deterministic_key() -> None:
    """Проверяет повторяемость SHA-256."""

    first = build_request_key()
    second = build_request_key()

    assert first.value == second.value
    assert (
        first.canonical_json
        == second.canonical_json
    )
    assert len(first.value) == 64
    assert all(
        character in "0123456789abcdef"
        for character in first.value
    )
    assert first.version == (
        EVENT_REQUEST_KEY_VERSION
    )

    print("Deterministic event request key: OK")
    print(f"request_key={first.value}")
    print(
        "request_key_version="
        f"{first.version}"
    )


def test_canonical_payload() -> None:
    """Проверяет состав канонического payload."""

    request_key = build_request_key()

    payload = json.loads(
        request_key.canonical_json
    )

    assert payload[
        "request_key_version"
    ] == EVENT_REQUEST_KEY_VERSION

    assert payload[
        "formula_version"
    ] == FULL_FORMULA_VERSION

    assert payload[
        "candidate_news_ids"
    ] == [
        7,
        8,
        9,
    ]

    assert payload[
        "audience_metrics"
    ] == [
        {
            "news_id": 7,
            "view_count": 100,
            "comment_count": None,
            "share_count": 1,
        },
        {
            "news_id": 9,
            "view_count": 300,
            "comment_count": 30,
            "share_count": 3,
        },
    ]

    assert payload[
        "window"
    ][
        "started_at"
    ] == "2026-07-30T11:21:00+00:00"

    assert payload[
        "window"
    ][
        "finished_at"
    ] == "2026-07-31T11:21:00+00:00"

    print()
    print("Canonical event payload: OK")
    print("audience_metrics_sorted=true")


def test_timezone_normalization() -> None:
    """Проверяет единый ключ для одного UTC-окна."""

    utc_key = build_request_key()

    plus_three = timezone(
        timedelta(hours=3)
    )

    shifted_key = build_request_key(
        window_started_at=datetime(
            2026,
            7,
            30,
            14,
            21,
            tzinfo=plus_three,
        ),
        window_finished_at=datetime(
            2026,
            7,
            31,
            14,
            21,
            tzinfo=plus_three,
        ),
    )

    assert utc_key.value == shifted_key.value

    print()
    print("Timezone normalization: OK")


def test_changed_metrics_change_key() -> None:
    """Проверяет влияние audience-снимка."""

    original = build_request_key()

    changed = build_request_key(
        audience_metrics=(
            EventAudienceMetrics(
                news_id=7,
                view_count=101,
                comment_count=None,
                share_count=1,
            ),
            EventAudienceMetrics(
                news_id=9,
                view_count=300,
                comment_count=30,
                share_count=3,
            ),
        )
    )

    assert original.value != changed.value

    print()
    print("Changed audience metrics change key: OK")


def test_changed_order_changes_key() -> None:
    """Проверяет значимость порядка кандидатов."""

    original = build_request_key()

    changed = build_request_key(
        news_ids=(
            9,
            8,
            7,
        )
    )

    assert original.value != changed.value

    print()
    print("Changed candidate order changes key: OK")


def test_duplicate_metric_blocking() -> None:
    """Проверяет повтор одного news_id в метриках."""

    try:
        build_request_key(
            audience_metrics=(
                EventAudienceMetrics(
                    news_id=7,
                    view_count=10,
                ),
                EventAudienceMetrics(
                    news_id=7,
                    view_count=20,
                ),
            )
        )
    except ValueError as error:
        assert "повторяющийся news_id" in str(
            error
        )

        print()
        print(
            "Duplicate audience metric blocking: OK"
        )
        return

    raise AssertionError(
        "Повторяющаяся audience-метрика "
        "не была заблокирована."
    )


def test_external_metric_blocking() -> None:
    """Проверяет метрику вне выборки."""

    try:
        build_request_key(
            audience_metrics=(
                EventAudienceMetrics(
                    news_id=10,
                    view_count=10,
                ),
            )
        )
    except ValueError as error:
        assert "вне текущей выборки" in str(
            error
        )

        print()
        print(
            "External audience metric blocking: OK"
        )
        return

    raise AssertionError(
        "Метрика вне выборки "
        "не была заблокирована."
    )


def test_model_mismatch_blocking() -> None:
    """Проверяет несовпадение моделей."""

    try:
        build_request_key(
            metadata=build_metadata(
                model_name="other-model"
            ),
        )
    except ValueError as error:
        assert (
            "Модель в metadata не совпадает"
            in str(error)
        )

        print()
        print("Model mismatch blocking: OK")
        return

    raise AssertionError(
        "Несовпадение моделей "
        "не было заблокировано."
    )


def test_naive_datetime_blocking() -> None:
    """Проверяет дату без часового пояса."""

    try:
        build_request_key(
            window_started_at=datetime(
                2026,
                7,
                30,
                11,
                21,
            )
        )
    except ValueError as error:
        assert "часовой пояс" in str(error)

        print()
        print("Naive datetime blocking: OK")
        return

    raise AssertionError(
        "Дата без часового пояса "
        "не была заблокирована."
    )


def main() -> int:
    """Запускает автономный тест."""

    test_deterministic_key()
    test_canonical_payload()
    test_timezone_normalization()
    test_changed_metrics_change_key()
    test_changed_order_changes_key()
    test_duplicate_metric_blocking()
    test_external_metric_blocking()
    test_model_mismatch_blocking()
    test_naive_datetime_blocking()

    print()
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Event request key test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())