import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json

from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)
from app.ranking.event_evaluator import (
    EVENT_EVALUATOR_VERSION,
    EVENT_PROMPT_VERSION,
    STORY_CLUSTER_VERIFIER_PROMPT_VERSION,
    EventRankingEvaluator,
    EventRankingModelRequest,
    EventRankingModelResponse,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)
from app.ranking.openai_event_evaluator import (
    OpenAIEventRankingEvaluator,
    REPAIR_INSTRUCTIONS,
    STORY_CLUSTER_VERIFIER_INSTRUCTIONS,
    SYSTEM_INSTRUCTIONS,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


MODEL_NAME = "test-model-no-network"


class SyntheticRepairError(RuntimeError):
    """Тестовая ошибка корректирующего запроса."""


class FakeStructuredEventRankingClient:
    """Тестовый клиент с очередью ответов."""

    def __init__(
        self,
        responses: tuple[
            EventRankingModelResponse
            | Exception,
            ...,
        ],
    ) -> None:
        self._responses = list(responses)
        self.requests: list[
            EventRankingModelRequest
        ] = []

    async def create_response(
        self,
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Возвращает следующий ответ или ошибку."""

        self.requests.append(request)

        if not self._responses:
            raise AssertionError(
                "Fake-клиент не имеет "
                "подготовленного ответа."
            )

        response = self._responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


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


def build_valid_payload() -> dict[str, object]:
    """Возвращает два события и глобальный cluster-registry."""

    return {
        "events": [
            {
                "representative_news_id": 103,
                "event_title": (
                    "Festival Adds New Competition"
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
                        "source_relation": "primary",
                        "is_representative": True,
                        "is_independent_source": True,
                        "counts_toward_reach": True,
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
                        "source_relation": "primary",
                        "is_representative": True,
                        "is_independent_source": True,
                        "counts_toward_reach": True,
                        "membership_reason": (
                            "Наиболее содержательная "
                            "первичная публикация."
                        ),
                    },
                    {
                        "news_id": 102,
                        "source_relation": "duplicate",
                        "is_representative": False,
                        "is_independent_source": False,
                        "counts_toward_reach": False,
                        "membership_reason": (
                            "Повторяет тот же инфоповод "
                            "без новых фактов."
                        ),
                    },
                ],
            },
        ],
        "story_clusters": [
            {
                "story_cluster_key": (
                    "festival_new_competition"
                ),
                "representative_news_ids": [103],
                "cluster_reason": (
                    "Самостоятельное фестивальное "
                    "объявление."
                ),
            },
            {
                "story_cluster_key": (
                    "international_film_project"
                ),
                "representative_news_ids": [101],
                "cluster_reason": (
                    "Самостоятельный международный "
                    "кинопроект."
                ),
            },
        ],
    }

def encode_payload(
    payload: dict[str, object],
) -> str:
    """Кодирует тестовый JSON."""

    return json.dumps(
        payload,
        ensure_ascii=False,
    )


def build_partial_payload() -> dict[str, object]:
    """Возвращает валидный payload без news_id=102."""

    payload = build_valid_payload()
    events = payload["events"]

    if not isinstance(events, list):
        raise AssertionError(
            "events должен быть list."
        )

    second_event = events[1]

    if not isinstance(second_event, dict):
        raise AssertionError(
            "Второй event должен быть dict."
        )

    members = second_event["members"]

    if not isinstance(members, list):
        raise AssertionError(
            "members должен быть list."
        )

    second_event["members"] = [members[0]]

    return payload


def build_telemetry() -> tuple[
    OpenAITokenUsage,
    OpenAICostEstimate,
]:
    """Создаёт телеметрию одного запроса."""

    return (
        OpenAITokenUsage(
            input_tokens=1000,
            cached_input_tokens=100,
            cache_write_tokens=0,
            output_tokens=300,
            reasoning_tokens=75,
            total_tokens=1300,
        ),
        OpenAICostEstimate(
            model_name=MODEL_NAME,
            pricing_version=(
                "synthetic_event_pricing_v1"
            ),
            regular_input_cost_usd=(
                Decimal("0.00180000")
            ),
            cached_input_cost_usd=(
                Decimal("0.00002000")
            ),
            cache_write_cost_usd=(
                Decimal("0.00000000")
            ),
            output_cost_usd=(
                Decimal("0.00360000")
            ),
            total_cost_usd=(
                Decimal("0.00542000")
            ),
        ),
    )


def build_response(
    payload: dict[str, object] | str,
    *,
    with_telemetry: bool = False,
) -> EventRankingModelResponse:
    """Создаёт тестовый ответ модели."""

    output_text = (
        payload
        if isinstance(payload, str)
        else encode_payload(payload)
    )

    if not with_telemetry:
        return EventRankingModelResponse(
            output_text=output_text,
        )

    usage, cost = build_telemetry()

    return EventRankingModelResponse(
        output_text=output_text,
        usage=usage,
        cost_estimate=cost,
    )


def build_evaluator(
    *responses: (
        EventRankingModelResponse
        | Exception
    ),
) -> tuple[
    OpenAIEventRankingEvaluator,
    FakeStructuredEventRankingClient,
]:
    """Создаёт оценщик и fake-клиент."""

    prepared_responses = responses or (
        build_response(build_valid_payload()),
    )
    client = FakeStructuredEventRankingClient(
        tuple(prepared_responses)
    )
    evaluator = OpenAIEventRankingEvaluator(
        client=client,
        model_name=MODEL_NAME,
    )

    return evaluator, client


def test_build_request() -> None:
    """Проверяет основной запрос v6."""

    evaluator, client = build_evaluator()
    request = evaluator.build_request(
        build_selection()
    )

    assert len(client.requests) == 0
    assert request.model == MODEL_NAME
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
        "group_assess_and_cluster_movie_news_events"
    )
    assert payload["formula_version"] == (
        FULL_FORMULA_VERSION
    )
    assert payload["expected_news_count"] == 3
    assert payload["expected_news_ids"] == [
        101,
        102,
        103,
    ]
    assert payload["story_cluster_policy"][
        "format"
    ] == "lower_snake_case"
    assert payload["story_cluster_policy"][
        "output_location"
    ] == "top_level_story_clusters"
    assert payload["story_cluster_policy"][
        "global_comparison_required"
    ] is True
    assert payload["story_cluster_policy"][
        "coverage"
    ] == "exactly_once"
    assert [
        candidate["configured_source_weight"]
        for candidate in payload["candidates"]
    ] == [3, 2, 1]

    assert EVENT_EVALUATOR_VERSION == (
        "event_ranking_evaluator_v7"
    )
    assert EVENT_PROMPT_VERSION == (
        "movie_news_event_ranking_prompt_v6"
    )
    assert STORY_CLUSTER_VERIFIER_PROMPT_VERSION == (
        "movie_news_story_cluster_verifier_v1"
    )
    assert evaluator.metadata.evaluator_version == (
        EVENT_EVALUATOR_VERSION
    )
    assert evaluator.metadata.prompt_version == (
        EVENT_PROMPT_VERSION
    )

    print("Request preparation v7: OK")
    print("client_call_count=0")


async def test_complete_response() -> None:
    """Проверяет обычный путь без repair."""

    evaluator, client = build_evaluator()
    selection = build_selection()
    request = evaluator.build_request(selection)
    result = await evaluator.evaluate_prepared_request(
        selection,
        request,
    )

    assert len(client.requests) == 1
    assert result.diagnostics is not None
    assert result.diagnostics.degraded is False
    assert result.diagnostics.repair_attempted is False
    assert result.diagnostics.repair_succeeded is False
    assert result.diagnostics.model_call_count == 1
    assert (
        result.diagnostics
        .story_cluster_verification_attempted
        is False
    )
    assert (
        result.diagnostics
        .story_cluster_verification_skipped_reason
        == "no_multi_event_story_clusters"
    )
    assert result.diagnostics.processed_news_ids == (
        101,
        102,
        103,
    )
    assert result.diagnostics.missing_news_ids == ()
    assert len(result.model_responses) == 1
    assert tuple(
        event.representative_news_id
        for event in result.events
    ) == (101, 103)
    assert result.events[0].source_weight_sum == 3
    assert result.events[1].source_weight_sum == 1
    assert result.events[0].story_cluster_key == (
        "international_film_project"
    )
    assert result.events[1].story_cluster_key == (
        "festival_new_competition"
    )

    print()
    print("Complete response without repair: OK")
    print("client_call_count=1")


async def test_successful_repair() -> None:
    """Исправляет пропущенный ID одним запросом."""

    primary = build_response(
        build_partial_payload(),
        with_telemetry=True,
    )
    repair = build_response(
        build_valid_payload(),
        with_telemetry=True,
    )
    evaluator, client = build_evaluator(
        primary,
        repair,
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    assert client.requests[0].instructions == (
        SYSTEM_INSTRUCTIONS
    )
    assert client.requests[1].instructions == (
        REPAIR_INSTRUCTIONS
    )

    repair_payload = json.loads(
        client.requests[1].input_text
    )
    assert repair_payload["task"] == (
        "repair_movie_news_event_payload"
    )
    assert repair_payload["missing_news_ids"] == [
        102,
    ]
    assert repair_payload["expected_news_ids"] == [
        101,
        102,
        103,
    ]
    assert [
        item["news_id"]
        for item in repair_payload[
            "missing_candidates"
        ]
    ] == [102]
    assert (
        repair_payload["requirements"]
        ["return_full_corrected_payload"]
        is True
    )

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.degraded is False
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is True
    assert diagnostics.initial_missing_news_ids == (
        102,
    )
    assert diagnostics.missing_news_ids == ()
    assert diagnostics.processed_news_ids == (
        101,
        102,
        103,
    )
    assert diagnostics.model_call_count == 2
    assert len(result.model_responses) == 2

    usage = result.model_response.usage
    cost = result.model_response.cost_estimate
    assert usage is not None
    assert cost is not None
    assert usage.input_tokens == 2000
    assert usage.output_tokens == 600
    assert usage.reasoning_tokens == 150
    assert usage.total_tokens == 2600
    assert cost.total_cost_usd == (
        Decimal("0.01084000")
    )
    assert result.model_response.output_text == (
        repair.output_text
    )

    print()
    print("Single repair success: OK")
    print("client_call_count=2")
    print("missing_news_ids=none")
    print("aggregated_total_tokens=2600")


async def test_degraded_after_unsuccessful_repair() -> None:
    """Возвращает частичный результат после второго пропуска."""

    partial = build_partial_payload()
    evaluator, client = build_evaluator(
        build_response(
            partial,
            with_telemetry=True,
        ),
        build_response(
            partial,
            with_telemetry=True,
        ),
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.degraded is True
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is False
    assert diagnostics.initial_missing_news_ids == (
        102,
    )
    assert diagnostics.missing_news_ids == (
        102,
    )
    assert diagnostics.processed_news_ids == (
        101,
        103,
    )
    assert diagnostics.model_call_count == 2
    assert tuple(
        news_id
        for event in result.events
        for news_id in event.member_news_ids
    ) == (101, 103)

    usage = result.model_response.usage
    assert usage is not None
    assert usage.total_tokens == 2600

    print()
    print("Degraded result after repair: OK")
    print("client_call_count=2")
    print("processed_news_ids=101,103")
    print("missing_news_ids=102")


async def test_degraded_after_repair_error() -> None:
    """Не теряет первый валидный partial при ошибке repair."""

    evaluator, client = build_evaluator(
        build_response(
            build_partial_payload(),
            with_telemetry=True,
        ),
        SyntheticRepairError(
            "Synthetic repair timeout."
        ),
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.degraded is True
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is False
    assert diagnostics.repair_error_type == (
        "SyntheticRepairError"
    )
    assert diagnostics.repair_error_message == (
        "Synthetic repair timeout."
    )
    assert diagnostics.missing_news_ids == (
        102,
    )
    assert len(result.model_responses) == 1

    usage = result.model_response.usage
    assert usage is not None
    assert usage.total_tokens == 1300

    print()
    print("Repair transport failure degraded mode: OK")
    print("client_call_count=2")
    print("successful_response_count=1")


def build_shared_story_payload() -> dict[str, object]:
    """Объединяет два разных events в одну мегатему."""

    payload = build_valid_payload()
    payload["story_clusters"] = [
        {
            "story_cluster_key": (
                "paramount_warner_merger"
            ),
            "representative_news_ids": [
                101,
                103,
            ],
            "cluster_reason": (
                "Два отдельных развития одной "
                "сделки Paramount-Warner."
            ),
        }
    ]

    return payload


def build_invalid_story_cluster_payload(
) -> dict[str, object]:
    """Оставляет один event без cluster-assignment."""

    payload = build_valid_payload()
    payload["story_clusters"] = [
        {
            "story_cluster_key": (
                "international_film_project"
            ),
            "representative_news_ids": [101],
            "cluster_reason": (
                "Намеренно неполный реестр."
            ),
        }
    ]

    return payload


async def test_global_story_cluster_assignment() -> None:
    """Verifier разделяет чрезмерно широкий cluster."""

    evaluator, client = build_evaluator(
        build_response(
            build_shared_story_payload(),
            with_telemetry=True,
        ),
        build_response(
            build_valid_payload(),
            with_telemetry=True,
        ),
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    assert client.requests[1].instructions == (
        STORY_CLUSTER_VERIFIER_INSTRUCTIONS
    )
    verifier_payload = json.loads(
        client.requests[1].input_text
    )
    assert verifier_payload["task"] == (
        "verify_and_split_multi_event_story_clusters"
    )
    assert verifier_payload[
        "target_representative_news_ids"
    ] == [101, 103]
    assert verifier_payload["requirements"][
        "never_merge_different_original_clusters"
    ] is True

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.repair_attempted is False
    assert (
        diagnostics.story_cluster_verification_attempted
        is True
    )
    assert (
        diagnostics.story_cluster_verification_succeeded
        is True
    )
    assert (
        diagnostics.story_cluster_verification_degraded
        is False
    )
    assert diagnostics.model_call_count == 2
    assert diagnostics.story_cluster_count_before == 1
    assert diagnostics.story_cluster_count_after == 2
    assert (
        diagnostics.story_cluster_multi_event_count_before
        == 1
    )
    assert (
        diagnostics.story_cluster_multi_event_count_after
        == 0
    )
    assert diagnostics.story_cluster_verifier_event_count == 2
    assert len(
        diagnostics.story_cluster_verification_changes
    ) == 1
    assert {
        event.story_cluster_key
        for event in result.events
    } == {
        "festival_new_competition",
        "international_film_project",
    }

    usage = result.model_response.usage
    assert usage is not None
    assert usage.total_tokens == 2600

    print()
    print("Story cluster verifier split: OK")
    print("client_call_count=2")
    print("cluster_count=1->2")


async def test_story_cluster_verifier_failure() -> None:
    """Некорректный verifier сохраняет исходные clusters."""

    evaluator, client = build_evaluator(
        build_response(
            build_shared_story_payload(),
            with_telemetry=True,
        ),
        build_response(
            build_invalid_story_cluster_payload(),
            with_telemetry=True,
        ),
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.repair_attempted is False
    assert (
        diagnostics.story_cluster_verification_attempted
        is True
    )
    assert (
        diagnostics.story_cluster_verification_succeeded
        is False
    )
    assert (
        diagnostics.story_cluster_verification_degraded
        is True
    )
    assert diagnostics.model_call_count == 2
    assert diagnostics.story_cluster_count_before == 1
    assert diagnostics.story_cluster_count_after == 1
    assert diagnostics.story_cluster_verification_error_type == (
        "StoryClusterCoverageError"
    )
    assert {
        event.story_cluster_key
        for event in result.events
    } == {"paramount_warner_merger"}

    print()
    print("Story cluster verifier safe retention: OK")
    print("original_clusters_preserved=true")


async def test_story_cluster_repair_success() -> None:
    """Исправляет неполный глобальный cluster-registry."""

    evaluator, client = build_evaluator(
        build_response(
            build_invalid_story_cluster_payload(),
            with_telemetry=True,
        ),
        build_response(
            build_shared_story_payload(),
            with_telemetry=True,
        ),
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    repair_payload = json.loads(
        client.requests[1].input_text
    )
    assert repair_payload["missing_news_ids"] == []
    assert repair_payload[
        "story_cluster_validation"
    ]["valid"] is False
    assert repair_payload[
        "story_cluster_validation"
    ]["error_type"] == (
        "StoryClusterCoverageError"
    )
    assert repair_payload["requirements"][
        "rebuild_global_story_cluster_registry"
    ] is True

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.degraded is False
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is True
    assert diagnostics.missing_news_ids == ()
    assert diagnostics.model_call_count == 2
    assert (
        diagnostics.story_cluster_verification_attempted
        is False
    )
    assert (
        diagnostics.story_cluster_verification_skipped_reason
        == "repair_consumed_second_model_call"
    )
    assert {
        event.story_cluster_key
        for event in result.events
    } == {
        "paramount_warner_merger"
    }

    print()
    print("Story cluster repair success: OK")
    print("client_call_count=2")


async def test_story_cluster_fallback() -> None:
    """После неудачного repair применяет уникальные safe keys."""

    invalid_payload = (
        build_invalid_story_cluster_payload()
    )
    evaluator, client = build_evaluator(
        build_response(
            invalid_payload,
            with_telemetry=True,
        ),
        build_response(
            invalid_payload,
            with_telemetry=True,
        ),
    )
    result = await evaluator.evaluate_detailed(
        build_selection()
    )

    assert len(client.requests) == 2
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.degraded is False
    assert diagnostics.repair_attempted is True
    assert diagnostics.repair_succeeded is False
    assert diagnostics.repair_error_type == (
        "StoryClusterCoverageError"
    )
    assert diagnostics.missing_news_ids == ()
    assert {
        event.story_cluster_key
        for event in result.events
    } == {
        "event_101",
        "event_103",
    }

    usage = result.model_response.usage
    assert usage is not None
    assert usage.total_tokens == 2600

    print()
    print("Story cluster safe fallback: OK")
    print("fallback_preserves_events=true")


async def test_initial_unexpected_candidate() -> None:
    """Посторонний ID в первом ответе остаётся ошибкой."""

    payload = build_valid_payload()
    events = payload["events"]
    assert isinstance(events, list)
    first_event = events[0]
    assert isinstance(first_event, dict)
    members = first_event["members"]
    assert isinstance(members, list)
    members.append(
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

    evaluator, client = build_evaluator(
        build_response(payload)
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "unexpected=[999]" in str(error)
        assert len(client.requests) == 1
        print()
        print("Initial unexpected ID blocking: OK")
        return

    raise AssertionError(
        "Посторонний news_id не был "
        "заблокирован."
    )


async def test_initial_cross_event_duplicate() -> None:
    """Дубликат ID между событиями остаётся ошибкой."""

    payload = build_valid_payload()
    events = payload["events"]
    assert isinstance(events, list)
    first_event = events[0]
    assert isinstance(first_event, dict)
    members = first_event["members"]
    assert isinstance(members, list)
    members.append(
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

    evaluator, client = build_evaluator(
        build_response(payload)
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "нескольким инфоповодам" in str(
            error
        )
        assert len(client.requests) == 1
        print()
        print("Initial duplicate ID blocking: OK")
        return

    raise AssertionError(
        "news_id в двух инфоповодах "
        "не был заблокирован."
    )


async def test_model_source_weight_rejected() -> None:
    """Модель по-прежнему не может вернуть вес."""

    payload = build_valid_payload()
    events = payload["events"]
    assert isinstance(events, list)
    first_event = events[0]
    assert isinstance(first_event, dict)
    members = first_event["members"]
    assert isinstance(members, list)
    first_member = members[0]
    assert isinstance(first_member, dict)
    first_member["source_weight"] = 3

    evaluator, client = build_evaluator(
        build_response(payload)
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "event-level схеме" in str(error)
        assert len(client.requests) == 1
        print()
        print("Model source_weight rejection: OK")
        return

    raise AssertionError(
        "source_weight от модели "
        "не был заблокирован."
    )


def test_missing_configured_source_weight() -> None:
    """Блокирует запрос до модели без веса в БД."""

    evaluator, client = build_evaluator()
    selection = build_selection()
    candidates = list(selection.candidates)
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
        print("Missing source weight blocking: OK")
        return

    raise AssertionError(
        "Отсутствующий source_weight "
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
        assert "не соответствует" in str(error)
        assert len(client.requests) == 0
        print()
        print("Modified request blocking: OK")
        return

    raise AssertionError(
        "Изменённый запрос не был "
        "заблокирован."
    )


async def test_event_time_outside_window() -> None:
    """Блокирует время события вне окна."""

    payload = build_valid_payload()
    events = payload["events"]
    assert isinstance(events, list)
    first_event = events[0]
    assert isinstance(first_event, dict)
    first_event["event_time_utc"] = (
        "2026-08-01T11:59:59Z"
    )
    evaluator, client = build_evaluator(
        build_response(payload)
    )

    try:
        await evaluator.evaluate(
            build_selection()
        )
    except ValueError as error:
        assert "вне окна" in str(error)
        assert len(client.requests) == 1
        print()
        print("Event time window blocking: OK")
        return

    raise AssertionError(
        "Время события вне окна "
        "не было заблокировано."
    )


async def test_invalid_json_and_empty_response() -> None:
    """Проверяет синтаксический и пустой ответы."""

    for response_text, expected_fragment in (
        ("{invalid-json", "event-level схеме"),
        ("   ", "пустой ответ"),
    ):
        evaluator, client = build_evaluator(
            build_response(response_text)
        )

        try:
            await evaluator.evaluate(
                build_selection()
            )
        except ValueError as error:
            assert expected_fragment in str(error)
            assert len(client.requests) == 1
        else:
            raise AssertionError(
                "Некорректный ответ модели "
                "не был заблокирован."
            )

    print()
    print("Invalid and empty response blocking: OK")


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
        evaluator.build_request(selection)
    except ValueError as error:
        assert "Список кандидатов" in str(error)
        assert len(client.requests) == 0
        print()
        print("Empty selection blocking: OK")
        return

    raise AssertionError(
        "Пустая выборка не была "
        "заблокирована."
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
    print("Common event evaluator interface: OK")


async def main() -> int:
    """Запускает изолированный тест v7."""

    test_build_request()
    await test_complete_response()
    await test_successful_repair()
    await test_degraded_after_unsuccessful_repair()
    await test_degraded_after_repair_error()
    await test_global_story_cluster_assignment()
    await test_story_cluster_verifier_failure()
    await test_story_cluster_repair_success()
    await test_story_cluster_fallback()
    await test_initial_unexpected_candidate()
    await test_initial_cross_event_duplicate()
    await test_model_source_weight_rejected()
    test_missing_configured_source_weight()
    await test_modified_request_blocking()
    await test_event_time_outside_window()
    await test_invalid_json_and_empty_response()
    test_empty_selection()
    await test_common_interface()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "OpenAI event ranking repair/degraded/"
        "global-story-cluster-verifier test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
