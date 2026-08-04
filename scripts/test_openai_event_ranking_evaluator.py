import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json

from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)
from app.ranking.event_evaluator import (
    EVENT_EVALUATOR_VERSION,
    EVENT_PROMPT_VERSION,
    EventRankingEvaluator,
    EventRankingModelRequest,
    EventRankingModelResponse,
)
from app.ranking.openai_event_evaluator import (
    OpenAIEventRankingEvaluator,
    SYSTEM_INSTRUCTIONS,
)


class FakeStructuredEventRankingClient:
    """Тестовый клиент без сетевых запросов."""

    def __init__(
        self,
        response_text: str,
    ) -> None:
        self._response_text = response_text
        self.requests: list[
            EventRankingModelRequest
        ] = []

    async def create_response(
        self,
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Возвращает заранее заданный JSON."""

        self.requests.append(request)

        return EventRankingModelResponse(
            output_text=self._response_text,
        )


def build_selection() -> CandidateSelectionResult:
    """Создаёт три публикации в суточном окне."""

    window_end = datetime(
        2026,
        8,
        2,
        12,
        0,
        tzinfo=timezone.utc,
    )

    window_start = datetime(
        2026,
        8,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    candidates = (
        NewsCandidate(
            news_id=101,
            source_id=8,
            source_code="variety_film",
            source_name="Variety Film",
            collection_priority=100,
            processing_status="collected",
            title=(
                "Studio Announces International "
                "Film Project"
            ),
            summary=(
                "A major studio announced "
                "an international production."
            ),
            author_name="First Author",
            source_published_at=datetime(
                2026,
                8,
                2,
                10,
                0,
                tzinfo=timezone.utc,
            ),
            age_hours=2.0,
            source_url=(
                "https://example.com/news/101"
            ),
            primary_image_url=(
                "https://example.com/101.jpg"
            ),
            source_weight=3,
        ),
        NewsCandidate(
            news_id=102,
            source_id=9,
            source_code="example_wire",
            source_name="Example Wire",
            collection_priority=90,
            processing_status="candidate",
            title=(
                "International Film Project "
                "Confirmed"
            ),
            summary=(
                "A second publication covered "
                "the same studio announcement."
            ),
            author_name="Second Author",
            source_published_at=datetime(
                2026,
                8,
                2,
                9,
                0,
                tzinfo=timezone.utc,
            ),
            age_hours=3.0,
            source_url=(
                "https://example.com/news/102"
            ),
            primary_image_url=None,
            source_weight=2,
        ),
        NewsCandidate(
            news_id=103,
            source_id=10,
            source_code="festival_daily",
            source_name="Festival Daily",
            collection_priority=80,
            processing_status="collected",
            title=(
                "Festival Adds New Competition"
            ),
            summary=None,
            author_name=None,
            source_published_at=datetime(
                2026,
                8,
                2,
                8,
                0,
                tzinfo=timezone.utc,
            ),
            age_hours=4.0,
            source_url=(
                "https://example.com/news/103"
            ),
            primary_image_url=None,
            source_weight=1,
        ),
    )

    return CandidateSelectionResult(
        window_start=window_start,
        window_end=window_end,
        window_hours=24.0,
        candidates=candidates,
    )


def build_valid_response() -> str:
    """
    Возвращает два инфоповода.

    Модель не возвращает source_weight.
    Порядок событий специально обратный входному.
    """

    return json.dumps(
        {
            "events": [
                {
                    "representative_news_id": 103,
                    "event_title": (
                        "Festival Adds "
                        "New Competition"
                    ),
                    "event_time_utc": (
                        "2026-08-02T08:00:00Z"
                    ),
                    "macro_topic": (
                        "festivals_awards_criticism"
                    ),
                    "i_score": "4.5",
                    "k_score": "1.0",
                    "n_score": "5.5",
                    "e_score": "3.5",
                    "x_score": "6.0",
                    "q_score": "0.90",
                    "impact_reason": (
                        "Новая конкурсная программа "
                        "расширяет фестиваль."
                    ),
                    "hook_reason": (
                        "Изменение заметно внутри "
                        "фестивального потока."
                    ),
                    "q_reason": (
                        "Информация опубликована "
                        "профильным источником."
                    ),
                    "members": [
                        {
                            "news_id": 103,
                            "source_relation": (
                                "primary"
                            ),
                            "is_representative": True,
                            "is_independent_source": (
                                True
                            ),
                            "counts_toward_reach": (
                                True
                            ),
                            "membership_reason": (
                                "Первичная профильная "
                                "публикация."
                            ),
                        }
                    ],
                },
                {
                    "representative_news_id": 101,
                    "event_title": (
                        "Studio Announces "
                        "International Film Project"
                    ),
                    "event_time_utc": (
                        "2026-08-02T09:30:00+00:00"
                    ),
                    "macro_topic": (
                        "creative_cast_production"
                    ),
                    "i_score": "7.5",
                    "k_score": "2.0",
                    "n_score": "6.5",
                    "e_score": "5.0",
                    "x_score": "7.0",
                    "q_score": "0.95",
                    "impact_reason": (
                        "Проект влияет на "
                        "международное производство."
                    ),
                    "hook_reason": (
                        "Масштабное партнёрство "
                        "выделяется в потоке."
                    ),
                    "q_reason": (
                        "Есть первичная публикация "
                        "и повторное освещение."
                    ),
                    "members": [
                        {
                            "news_id": 101,
                            "source_relation": (
                                "primary"
                            ),
                            "is_representative": True,
                            "is_independent_source": (
                                True
                            ),
                            "counts_toward_reach": (
                                True
                            ),
                            "membership_reason": (
                                "Наиболее содержательная "
                                "первичная публикация."
                            ),
                        },
                        {
                            "news_id": 102,
                            "source_relation": (
                                "duplicate"
                            ),
                            "is_representative": False,
                            "is_independent_source": (
                                False
                            ),
                            "counts_toward_reach": (
                                False
                            ),
                            "membership_reason": (
                                "Повторяет тот же "
                                "инфоповод без новых фактов."
                            ),
                        },
                    ],
                },
            ]
        },
        ensure_ascii=False,
    )


def build_evaluator(
    response_text: str | None = None,
) -> tuple[
    OpenAIEventRankingEvaluator,
    FakeStructuredEventRankingClient,
]:
    """Создаёт оценщик и fake-клиент."""

    client = (
        FakeStructuredEventRankingClient(
            response_text
            if response_text is not None
            else build_valid_response()
        )
    )

    evaluator = OpenAIEventRankingEvaluator(
        client=client,
        model_name="test-model-no-network",
    )

    return evaluator, client


def test_build_request() -> None:
    """Проверяет запрос без обращения к модели."""

    evaluator, client = build_evaluator()
    selection = build_selection()

    request = evaluator.build_request(
        selection
    )

    assert len(client.requests) == 0
    assert request.model == (
        "test-model-no-network"
    )
    assert request.instructions == (
        SYSTEM_INSTRUCTIONS
    )
    assert request.instructions.startswith(
        "Ты выполняешь event-level анализ"
    )

    payload = json.loads(
        request.input_text
    )

    assert payload["task"] == (
        "group_and_assess_movie_news_events"
    )
    assert payload["formula_version"] == (
        "top3_cinema_v2"
    )
    assert payload["window"]["hours"] == 24.0
    assert payload["expected_news_count"] == 3
    assert payload["expected_news_ids"] == [
        101,
        102,
        103,
    ]

    assert [
        candidate["news_id"]
        for candidate
        in payload["candidates"]
    ] == [
        101,
        102,
        103,
    ]

    assert [
        candidate["configured_source_weight"]
        for candidate
        in payload["candidates"]
    ] == [
        3,
        2,
        1,
    ]

    assert (
        payload["source_weight_policy"]
        ["model_must_not_return"]
        is True
    )

    assert "source_weight_scale" not in payload

    forbidden_fields = {
        "f_score",
        "u_score",
        "m_score",
        "v_score",
        "c_score",
        "s_score",
        "r_score",
        "h_score",
        "individual_score",
        "diversity_score",
        "top_score",
    }

    assert forbidden_fields.isdisjoint(
        payload.keys()
    )

    assert evaluator.metadata.run_mode == (
        "openai_event_ranking"
    )
    assert (
        evaluator.metadata.evaluator_version
        == EVENT_EVALUATOR_VERSION
    )
    assert (
        evaluator.metadata.prompt_version
        == EVENT_PROMPT_VERSION
    )

    assert EVENT_EVALUATOR_VERSION == (
        "event_ranking_evaluator_v3"
    )
    assert EVENT_PROMPT_VERSION == (
        "movie_news_event_ranking_prompt_v3"
    )

    print("Request preparation: OK")
    print("client_call_count=0")
    print(
        "configured_source_weights=3,2,1"
    )
    print(
        f"prompt_chars="
        f"{len(request.instructions)}"
    )


async def test_prepared_request() -> None:
    """Проверяет подстановку веса из конфигурации."""

    evaluator, client = build_evaluator()
    selection = build_selection()

    request = evaluator.build_request(
        selection
    )

    result = (
        await evaluator
        .evaluate_prepared_request(
            selection,
            request,
        )
    )

    assert len(client.requests) == 1
    assert client.requests[0] == request
    assert result.model_response.usage is None
    assert (
        result.model_response.cost_estimate
        is None
    )

    assert tuple(
        event.representative_news_id
        for event in result.events
    ) == (
        101,
        103,
    )

    first_event = result.events[0]
    second_event = result.events[1]

    assert first_event.member_news_ids == (
        101,
        102,
    )
    assert first_event.source_weight_sum == 3

    first_weights = {
        member.news_id: member.source_weight
        for member in first_event.members
    }

    assert first_weights == {
        101: 3,
        102: 0,
    }

    assert second_event.source_weight_sum == 1
    assert second_event.members[0].source_weight == 1

    assert (
        first_event.event_time_utc
        == datetime(
            2026,
            8,
            2,
            9,
            30,
            tzinfo=timezone.utc,
        )
    )

    print()
    print(
        "Configured source weight application: OK"
    )
    print("client_call_count=1")
    print("first_event_weights=101:3,102:0")
    print("second_event_weights=103:1")


def test_missing_configured_source_weight() -> None:
    """Блокирует запрос до модели без веса в БД."""

    evaluator, client = build_evaluator()
    selection = build_selection()

    candidates = list(
        selection.candidates
    )

    candidates[1] = replace(
        candidates[1],
        source_weight=None,
    )

    invalid_selection = replace(
        selection,
        candidates=tuple(candidates),
    )

    try:
        evaluator.build_request(
            invalid_selection
        )
    except ValueError as error:
        assert "не настроен" in str(error)
        assert "example_wire" in str(error)
        assert len(client.requests) == 0

        print()
        print(
            "Missing source weight blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Отсутствующий source_weight "
        "не был заблокирован."
    )


async def test_model_source_weight_rejected() -> None:
    """Блокирует попытку модели вернуть вес."""

    payload = json.loads(
        build_valid_response()
    )

    payload["events"][0]["members"][0][
        "source_weight"
    ] = 3

    evaluator, client = build_evaluator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert (
            "не соответствует "
            "event-level схеме"
            in str(error)
        )
        assert len(client.requests) == 1

        print()
        print(
            "Model source_weight rejection: OK"
        )
        return

    raise AssertionError(
        "source_weight от модели "
        "не был заблокирован."
    )


async def test_modified_request_blocking() -> None:
    """Не отправляет изменённый запрос модели."""

    evaluator, client = build_evaluator()
    selection = build_selection()

    valid_request = evaluator.build_request(
        selection
    )

    modified_request = EventRankingModelRequest(
        model=valid_request.model,
        instructions=(
            valid_request.instructions + " "
        ),
        input_text=valid_request.input_text,
    )

    try:
        await evaluator.evaluate_prepared_request(
            selection,
            modified_request,
        )
    except ValueError as error:
        assert (
            "не соответствует"
            in str(error)
        )
        assert len(client.requests) == 0

        print()
        print(
            "Modified request blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Изменённый запрос не был "
        "заблокирован."
    )


async def test_common_interface() -> None:
    """Проверяет общий event-level интерфейс."""

    evaluator, client = build_evaluator()

    assert isinstance(
        evaluator,
        EventRankingEvaluator,
    )

    events = await evaluator.evaluate(
        build_selection()
    )

    assert len(client.requests) == 1
    assert len(events) == 2

    print()
    print(
        "Common event evaluator interface: OK"
    )
    print("event_count=2")


async def test_missing_candidate() -> None:
    """Блокирует пропущенную публикацию."""

    payload = json.loads(
        build_valid_response()
    )

    payload["events"][1]["members"] = [
        payload["events"][1]["members"][0]
    ]

    evaluator, _ = build_evaluator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "missing=[102]" in str(error)

        print()
        print(
            "Missing candidate blocking: OK"
        )
        return

    raise AssertionError(
        "Пропущенный news_id не был "
        "заблокирован."
    )


async def test_unexpected_candidate() -> None:
    """Блокирует посторонний news_id."""

    payload = json.loads(
        build_valid_response()
    )

    payload["events"][0]["members"].append(
        {
            "news_id": 999,
            "source_relation": "duplicate",
            "is_representative": False,
            "is_independent_source": False,
            "counts_toward_reach": False,
            "membership_reason": (
                "Посторонняя тестовая публикация."
            ),
        }
    )

    evaluator, _ = build_evaluator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "unexpected=[999]" in str(
            error
        )

        print()
        print(
            "Unexpected candidate blocking: OK"
        )
        return

    raise AssertionError(
        "Посторонний news_id не был "
        "заблокирован."
    )


async def test_cross_event_duplicate() -> None:
    """Блокирует news_id в двух инфоповодах."""

    payload = json.loads(
        build_valid_response()
    )

    payload["events"][0]["members"].append(
        {
            "news_id": 102,
            "source_relation": "duplicate",
            "is_representative": False,
            "is_independent_source": False,
            "counts_toward_reach": False,
            "membership_reason": (
                "Повторное ошибочное включение."
            ),
        }
    )

    evaluator, _ = build_evaluator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "нескольким инфоповодам" in str(
            error
        )

        print()
        print(
            "Cross-event duplicate blocking: OK"
        )
        return

    raise AssertionError(
        "news_id в двух инфоповодах "
        "не был заблокирован."
    )


async def test_event_time_outside_window() -> None:
    """Блокирует время события вне окна."""

    payload = json.loads(
        build_valid_response()
    )

    payload["events"][0][
        "event_time_utc"
    ] = "2026-08-01T11:59:59Z"

    evaluator, _ = build_evaluator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "вне окна" in str(error)
        assert "103" in str(error)

        print()
        print(
            "Event time window blocking: OK"
        )
        return

    raise AssertionError(
        "Время события вне окна "
        "не было заблокировано."
    )


async def test_invalid_schema() -> None:
    """Блокирует недопустимую оценку модели."""

    payload = json.loads(
        build_valid_response()
    )

    payload["events"][0]["q_score"] = 1.5

    evaluator, _ = build_evaluator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert (
            "не соответствует "
            "event-level схеме"
            in str(error)
        )

        print()
        print(
            "Invalid response schema blocking: OK"
        )
        return

    raise AssertionError(
        "Оценка вне схемы не была "
        "заблокирована."
    )


async def test_invalid_json() -> None:
    """Блокирует синтаксически неверный JSON."""

    evaluator, _ = build_evaluator(
        "{invalid-json"
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert (
            "не соответствует "
            "event-level схеме"
            in str(error)
        )

        print()
        print("Invalid JSON blocking: OK")
        return

    raise AssertionError(
        "Некорректный JSON не был "
        "заблокирован."
    )


async def test_empty_response() -> None:
    """Блокирует пустой ответ модели."""

    evaluator, _ = build_evaluator("   ")

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "пустой ответ" in str(error)

        print()
        print("Empty response blocking: OK")
        return

    raise AssertionError(
        "Пустой ответ не был "
        "заблокирован."
    )


def test_empty_selection() -> None:
    """Блокирует пустую выборку до модели."""

    evaluator, client = build_evaluator()

    selection = CandidateSelectionResult(
        window_start=datetime(
            2026,
            8,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        window_end=datetime(
            2026,
            8,
            2,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        window_hours=24.0,
        candidates=(),
    )

    try:
        evaluator.build_request(
            selection
        )
    except ValueError as error:
        assert "Список кандидатов" in str(
            error
        )
        assert len(client.requests) == 0

        print()
        print(
            "Empty selection blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Пустая выборка не была "
        "заблокирована."
    )


async def main() -> int:
    """Запускает fake-client тест v2."""

    test_build_request()
    await test_prepared_request()
    test_missing_configured_source_weight()
    await test_model_source_weight_rejected()
    await test_modified_request_blocking()
    await test_common_interface()
    await test_missing_candidate()
    await test_unexpected_candidate()
    await test_cross_event_duplicate()
    await test_event_time_outside_window()
    await test_invalid_schema()
    await test_invalid_json()
    await test_empty_response()
    test_empty_selection()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "OpenAI event ranking evaluator "
        "source-weight test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
