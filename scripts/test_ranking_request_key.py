from datetime import datetime, timedelta, timezone
import json

from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_evaluator import (
    RankingModelRequest,
)
from app.ranking.request_key import (
    REQUEST_KEY_VERSION,
    create_ranking_request_key,
)


def build_metadata(
    *,
    model_name: str = "gpt-5.6-terra",
) -> RankingEvaluatorMetadata:
    """Создаёт метаданные тестового оценщика."""

    return RankingEvaluatorMetadata(
        run_mode="openai_ranking",
        evaluator_name="OpenAIRankingEvaluator",
        evaluator_version=(
            "openai_ranking_evaluator_v1"
        ),
        prompt_version=(
            "openai_ranking_prompt_v1"
        ),
        model_name=model_name,
    )


def build_model_request(
    *,
    model: str = "gpt-5.6-terra",
    input_text: str | None = None,
) -> RankingModelRequest:
    """Создаёт тестовый запрос модели."""

    return RankingModelRequest(
        model=model,
        instructions=(
            "Оцени кандидатов по формулам "
            "TOP 3 NEWS."
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


def build_request_key(
    *,
    metadata: RankingEvaluatorMetadata
    | None = None,
    model_request: RankingModelRequest
    | None = None,
    window_started_at: datetime
    | None = None,
    window_finished_at: datetime
    | None = None,
    news_ids: tuple[int, ...] = (
        7,
        8,
        9,
    ),
):
    """Создаёт ключ с типовыми параметрами."""

    return create_ranking_request_key(
        formula_version=(
            "individual_score_formula_v1"
        ),
        metadata=(
            metadata
            or build_metadata()
        ),
        model_request=(
            model_request
            or build_model_request()
        ),
        window_started_at=(
            window_started_at
            or datetime(
                2026,
                7,
                30,
                11,
                21,
                tzinfo=timezone.utc,
            )
        ),
        window_finished_at=(
            window_finished_at
            or datetime(
                2026,
                7,
                31,
                11,
                21,
                tzinfo=timezone.utc,
            )
        ),
        news_ids=news_ids,
    )


def test_deterministic_key() -> None:
    """Проверяет повторяемость SHA-256."""

    first_key = build_request_key()
    second_key = build_request_key()

    assert first_key.value == second_key.value
    assert (
        first_key.canonical_json
        == second_key.canonical_json
    )

    assert len(first_key.value) == 64

    assert all(
        character in "0123456789abcdef"
        for character in first_key.value
    )

    assert (
        first_key.version
        == REQUEST_KEY_VERSION
    )

    print("Deterministic request key: OK")
    print(f"request_key={first_key.value}")
    print(
        "request_key_version="
        f"{first_key.version}"
    )


def test_canonical_payload() -> None:
    """Проверяет каноническое содержимое."""

    request_key = build_request_key()

    payload = json.loads(
        request_key.canonical_json
    )

    assert payload[
        "request_key_version"
    ] == REQUEST_KEY_VERSION

    assert payload[
        "formula_version"
    ] == "individual_score_formula_v1"

    assert payload[
        "candidate_news_ids"
    ] == [
        7,
        8,
        9,
    ]

    assert payload[
        "evaluator"
    ][
        "model_name"
    ] == "gpt-5.6-terra"

    assert payload[
        "model_request"
    ][
        "model"
    ] == "gpt-5.6-terra"

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
    print("Canonical payload: OK")
    print(
        "candidate_news_ids="
        + ",".join(
            str(news_id)
            for news_id
            in payload[
                "candidate_news_ids"
            ]
        )
    )


def test_timezone_normalization() -> None:
    """
    Проверяет одинаковый ключ для одного момента.

    Даты передаются в разных часовых поясах,
    но обозначают тот же UTC-интервал.
    """

    utc_key = build_request_key()

    plus_three = timezone(
        timedelta(hours=3)
    )

    timezone_key = build_request_key(
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

    assert (
        utc_key.value
        == timezone_key.value
    )

    print()
    print("Timezone normalization: OK")


def test_changed_request_changes_key() -> None:
    """Проверяет изменение ключа."""

    original_key = build_request_key()

    changed_input_key = build_request_key(
        model_request=build_model_request(
            input_text=(
                '{"candidates":['
                '{"news_id":7},'
                '{"news_id":8},'
                '{"news_id":10}'
                "]}"
            )
        ),
        news_ids=(
            7,
            8,
            10,
        ),
    )

    changed_order_key = build_request_key(
        news_ids=(
            9,
            8,
            7,
        ),
    )

    changed_window_key = build_request_key(
        window_finished_at=datetime(
            2026,
            7,
            31,
            12,
            21,
            tzinfo=timezone.utc,
        )
    )

    assert (
        original_key.value
        != changed_input_key.value
    )

    assert (
        original_key.value
        != changed_order_key.value
    )

    assert (
        original_key.value
        != changed_window_key.value
    )

    print()
    print("Changed request changes key: OK")


def test_duplicate_news_ids_blocking() -> None:
    """Проверяет блокировку дубликатов."""

    try:
        build_request_key(
            news_ids=(
                7,
                8,
                8,
            )
        )
    except ValueError as error:
        assert "дубликаты" in str(error)

        print()
        print(
            "Duplicate news IDs blocking: OK"
        )
        return

    raise AssertionError(
        "Дублирующиеся news_id "
        "не были заблокированы."
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
        print(
            "Naive datetime blocking: OK"
        )
        return

    raise AssertionError(
        "Дата без часового пояса "
        "не была заблокирована."
    )


def test_model_mismatch_blocking() -> None:
    """Проверяет несовпадение моделей."""

    try:
        build_request_key(
            metadata=build_metadata(
                model_name="other-model"
            ),
            model_request=(
                build_model_request(
                    model="gpt-5.6-terra"
                )
            ),
        )
    except ValueError as error:
        assert (
            "Модель в metadata не совпадает"
            in str(error)
        )

        print()
        print(
            "Model mismatch blocking: OK"
        )
        return

    raise AssertionError(
        "Несовпадение моделей "
        "не было заблокировано."
    )


def main() -> int:
    """Запускает автономный тест."""

    test_deterministic_key()
    test_canonical_payload()
    test_timezone_normalization()
    test_changed_request_changes_key()
    test_duplicate_news_ids_blocking()
    test_naive_datetime_blocking()
    test_model_mismatch_blocking()

    print()
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Ranking request key test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())