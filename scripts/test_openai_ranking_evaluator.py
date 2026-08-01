import asyncio
from datetime import datetime, timezone
import json

from app.db.news_candidates import NewsCandidate
from app.ranking.openai_evaluator import (
    OpenAIRankingEvaluator,
    RankingModelRequest,
    RankingModelResponse,
)


class FakeStructuredRankingClient:
    """Тестовый клиент без сетевых запросов."""

    def __init__(
        self,
        response_text: str,
    ) -> None:
        self._response_text = response_text
        self.requests: list[
            RankingModelRequest
        ] = []

    async def create_response(
        self,
        request: RankingModelRequest,
    ) -> RankingModelResponse:
        """Возвращает заранее заданный JSON."""

        self.requests.append(request)

        return RankingModelResponse(
            output_text=self._response_text,
        )


def build_candidates() -> tuple[
    NewsCandidate,
    ...,
]:
    """Создаёт два тестовых кандидата."""

    return (
        NewsCandidate(
            news_id=101,
            source_id=8,
            source_code="variety_film",
            source_name="Variety Film",
            collection_priority=100,
            processing_status="collected",
            title=(
                "Major Film Studio Announces "
                "New International Project"
            ),
            summary=(
                "The studio announced a major "
                "international production."
            ),
            author_name="Test Author",
            source_published_at=datetime(
                2026,
                7,
                31,
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
        ),
        NewsCandidate(
            news_id=102,
            source_id=8,
            source_code="variety_film",
            source_name="Variety Film",
            collection_priority=100,
            processing_status="collected",
            title=(
                "Independent Festival Adds "
                "New Competition"
            ),
            summary=(
                "The festival introduced "
                "a new competition category."
            ),
            author_name="Second Author",
            source_published_at=datetime(
                2026,
                7,
                31,
                8,
                0,
                tzinfo=timezone.utc,
            ),
            age_hours=4.0,
            source_url=(
                "https://example.com/news/102"
            ),
            primary_image_url=None,
        ),
    )


def build_valid_response() -> str:
    """Создаёт корректный ответ модели."""

    return json.dumps(
        {
            "scores": [
                {
                    "news_id": 102,
                    "f_score": "8.0",
                    "m_score": "5.5",
                    "r_score": "4.5",
                    "h_score": "5.0",
                    "q_score": "0.85",
                    "explanation": (
                        "Свежая фестивальная "
                        "новость нишевого масштаба."
                    ),
                },
                {
                    "news_id": 101,
                    "f_score": "9.0",
                    "m_score": "8.0",
                    "r_score": "7.0",
                    "h_score": "6.5",
                    "q_score": "0.95",
                    "explanation": (
                        "Крупный международный "
                        "проект известной студии."
                    ),
                },
            ]
        },
        ensure_ascii=False,
    )


def build_evaluator(
    response_text: str | None = None,
) -> tuple[
    OpenAIRankingEvaluator,
    FakeStructuredRankingClient,
]:
    """Создаёт оценщик и fake-клиент."""

    client = FakeStructuredRankingClient(
        response_text or build_valid_response()
    )

    evaluator = OpenAIRankingEvaluator(
        client=client,
        model_name="test-model-no-network",
    )

    return evaluator, client


def test_build_request() -> None:
    """Проверяет подготовку запроса без модели."""

    evaluator, client = build_evaluator()

    candidates = build_candidates()

    request = evaluator.build_request(
        candidates
    )

    assert len(client.requests) == 0

    assert request.model == (
        "test-model-no-network"
    )

    assert request.instructions.strip()

    input_payload = json.loads(
        request.input_text
    )

    assert input_payload["task"] == (
        "score_movie_news_candidates"
    )

    assert input_payload["formula"] == (
        "0.20F + 0.30M + 0.20R "
        "+ 0.15(H × Q)"
    )

    assert [
        candidate["news_id"]
        for candidate
        in input_payload["candidates"]
    ] == [
        101,
        102,
    ]

    assert input_payload[
        "candidates"
    ][0]["age_hours"] == 2.0

    assert input_payload[
        "candidates"
    ][1]["age_hours"] == 4.0

    print("Request preparation: OK")
    print("client_call_count=0")
    print(
        f"model={request.model}"
    )
    print(
        "candidate_news_ids="
        + ",".join(
            str(candidate["news_id"])
            for candidate
            in input_payload["candidates"]
        )
    )


async def test_prepared_request() -> None:
    """Проверяет заранее подготовленный запрос."""

    evaluator, client = build_evaluator()

    candidates = build_candidates()

    request = evaluator.build_request(
        candidates
    )

    evaluation = (
        await evaluator
        .evaluate_prepared_request(
            candidates,
            request,
        )
    )

    assessments = evaluation.assessments

    assert len(client.requests) == 1
    assert client.requests[0] == request

    assert tuple(
        assessment.news_id
        for assessment in assessments
    ) == (
        101,
        102,
    )

    assert (
        evaluation.model_response.usage
        is None
    )

    assert (
        evaluation
        .model_response
        .cost_estimate
        is None
    )

    print()
    print("Prepared request evaluation: OK")
    print("client_call_count=1")
    print(
        "assessment_order="
        + ",".join(
            str(assessment.news_id)
            for assessment in assessments
        )
    )


async def test_modified_request_blocking() -> None:
    """
    Проверяет изменение подготовленного запроса.

    Изменённый запрос не должен быть отправлен
    клиенту модели.
    """

    evaluator, client = build_evaluator()

    candidates = build_candidates()

    valid_request = evaluator.build_request(
        candidates
    )

    modified_request = RankingModelRequest(
        model=valid_request.model,
        instructions=(
            valid_request.instructions
        ),
        input_text=(
            valid_request.input_text
            + " "
        ),
    )

    try:
        await evaluator.evaluate_prepared_request(
            candidates,
            modified_request,
        )
    except ValueError as error:
        assert (
            "Подготовленный запрос "
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
        "Изменённый подготовленный запрос "
        "не был заблокирован."
    )


async def test_valid_response() -> None:
    """Проверяет обычный детальный вызов."""

    evaluator, client = build_evaluator()

    candidates = build_candidates()

    evaluation = (
        await evaluator.evaluate_detailed(
            candidates
        )
    )

    assessments = evaluation.assessments

    assert (
        evaluation.model_response.usage
        is None
    )

    assert (
        evaluation
        .model_response
        .cost_estimate
        is None
    )

    assert len(client.requests) == 1
    assert len(assessments) == 2

    assert evaluator.metadata.run_mode == (
        "openai_ranking"
    )

    assert evaluator.metadata.model_name == (
        "test-model-no-network"
    )

    assert tuple(
        assessment.news_id
        for assessment in assessments
    ) == (
        101,
        102,
    )

    first_request = client.requests[0]

    assert first_request.model == (
        "test-model-no-network"
    )

    input_payload = json.loads(
        first_request.input_text
    )

    assert [
        candidate["news_id"]
        for candidate
        in input_payload["candidates"]
    ] == [
        101,
        102,
    ]

    first_assessment = assessments[0]

    assert str(
        first_assessment.f_score
    ) == "9.0"

    assert str(
        first_assessment.q_score
    ) == "0.95"

    print()
    print(
        "Valid structured response: OK"
    )
    print(
        f"client_call_count="
        f"{len(client.requests)}"
    )
    print(
        f"assessment_count="
        f"{len(assessments)}"
    )
    print(
        "assessment_order="
        + ",".join(
            str(assessment.news_id)
            for assessment in assessments
        )
    )
    print("usage_present=false")
    print("cost_estimate_present=false")


async def test_common_interface() -> None:
    """
    Проверяет совместимость с RankingEvaluator.

    Обычный метод evaluate() должен возвращать
    только набор оценок без телеметрии.
    """

    evaluator, client = build_evaluator()

    assessments = await evaluator.evaluate(
        build_candidates()
    )

    assert len(client.requests) == 1

    assert tuple(
        assessment.news_id
        for assessment in assessments
    ) == (
        101,
        102,
    )

    print()
    print(
        "Common evaluator interface: OK"
    )
    print(
        f"assessment_count="
        f"{len(assessments)}"
    )


async def test_missing_candidate() -> None:
    """Проверяет отсутствие одной новости."""

    response_text = json.dumps(
        {
            "scores": [
                {
                    "news_id": 101,
                    "f_score": 8,
                    "m_score": 7,
                    "r_score": 6,
                    "h_score": 5,
                    "q_score": 0.9,
                    "explanation": (
                        "Тестовая оценка."
                    ),
                }
            ]
        },
        ensure_ascii=False,
    )

    evaluator, _ = build_evaluator(
        response_text
    )

    try:
        await evaluator.evaluate(
            build_candidates()
        )
    except ValueError as error:
        assert "missing=[102]" in str(error)

        print()
        print(
            "Missing candidate blocking: OK"
        )
        return

    raise AssertionError(
        "Неполный ответ модели "
        "не был заблокирован."
    )


async def test_unexpected_candidate() -> None:
    """Проверяет посторонний news_id."""

    response_text = json.dumps(
        {
            "scores": [
                {
                    "news_id": 101,
                    "f_score": 8,
                    "m_score": 7,
                    "r_score": 6,
                    "h_score": 5,
                    "q_score": 0.9,
                    "explanation": (
                        "Первая оценка."
                    ),
                },
                {
                    "news_id": 102,
                    "f_score": 7,
                    "m_score": 6,
                    "r_score": 5,
                    "h_score": 4,
                    "q_score": 0.8,
                    "explanation": (
                        "Вторая оценка."
                    ),
                },
                {
                    "news_id": 999,
                    "f_score": 6,
                    "m_score": 5,
                    "r_score": 4,
                    "h_score": 3,
                    "q_score": 0.7,
                    "explanation": (
                        "Посторонняя оценка."
                    ),
                },
            ]
        },
        ensure_ascii=False,
    )

    evaluator, _ = build_evaluator(
        response_text
    )

    try:
        await evaluator.evaluate(
            build_candidates()
        )
    except ValueError as error:
        assert (
            "unexpected=[999]"
            in str(error)
        )

        print()
        print(
            "Unexpected candidate blocking: OK"
        )
        return

    raise AssertionError(
        "Посторонний news_id "
        "не был заблокирован."
    )


async def test_duplicate_candidate() -> None:
    """Проверяет повторяющийся news_id."""

    response_text = json.dumps(
        {
            "scores": [
                {
                    "news_id": 101,
                    "f_score": 8,
                    "m_score": 7,
                    "r_score": 6,
                    "h_score": 5,
                    "q_score": 0.9,
                    "explanation": (
                        "Первая оценка."
                    ),
                },
                {
                    "news_id": 101,
                    "f_score": 7,
                    "m_score": 6,
                    "r_score": 5,
                    "h_score": 4,
                    "q_score": 0.8,
                    "explanation": (
                        "Повторная оценка."
                    ),
                },
                {
                    "news_id": 102,
                    "f_score": 6,
                    "m_score": 5,
                    "r_score": 4,
                    "h_score": 3,
                    "q_score": 0.7,
                    "explanation": (
                        "Вторая новость."
                    ),
                },
            ]
        },
        ensure_ascii=False,
    )

    evaluator, _ = build_evaluator(
        response_text
    )

    try:
        await evaluator.evaluate(
            build_candidates()
        )
    except ValueError as error:
        assert (
            "повторяющиеся news_id"
            in str(error)
        )

        print()
        print(
            "Duplicate candidate blocking: OK"
        )
        return

    raise AssertionError(
        "Повторяющийся news_id "
        "не был заблокирован."
    )


async def test_invalid_score() -> None:
    """Проверяет недопустимую оценку."""

    response_text = json.dumps(
        {
            "scores": [
                {
                    "news_id": 101,
                    "f_score": 11,
                    "m_score": 7,
                    "r_score": 6,
                    "h_score": 5,
                    "q_score": 0.9,
                    "explanation": (
                        "Недопустимая оценка."
                    ),
                },
                {
                    "news_id": 102,
                    "f_score": 8,
                    "m_score": 6,
                    "r_score": 5,
                    "h_score": 4,
                    "q_score": 0.8,
                    "explanation": (
                        "Вторая оценка."
                    ),
                },
            ]
        },
        ensure_ascii=False,
    )

    evaluator, _ = build_evaluator(
        response_text
    )

    try:
        await evaluator.evaluate(
            build_candidates()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Invalid score blocking: OK"
        )
        return

    raise AssertionError(
        "Оценка вне диапазона "
        "не была заблокирована."
    )


async def test_invalid_json() -> None:
    """Проверяет синтаксически неверный JSON."""

    evaluator, _ = build_evaluator(
        "{invalid-json"
    )

    try:
        await evaluator.evaluate(
            build_candidates()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Invalid JSON blocking: OK"
        )
        return

    raise AssertionError(
        "Некорректный JSON "
        "не был заблокирован."
    )


async def test_empty_response() -> None:
    """Проверяет пустой ответ модели."""

    evaluator, _ = build_evaluator(
        "   "
    )

    try:
        await evaluator.evaluate(
            build_candidates()
        )
    except ValueError as error:
        assert "пустой ответ" in str(error)

        print()
        print(
            "Empty response blocking: OK"
        )
        return

    raise AssertionError(
        "Пустой ответ модели "
        "не был заблокирован."
    )


async def test_empty_candidates() -> None:
    """Проверяет пустой список кандидатов."""

    evaluator, client = build_evaluator()

    try:
        evaluator.build_request(())
    except ValueError as error:
        assert (
            "Список кандидатов"
            in str(error)
        )

        assert len(client.requests) == 0

        print()
        print(
            "Empty candidates blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Пустой список кандидатов "
        "не был заблокирован."
    )


async def main() -> int:
    """Запускает тест оценщика."""

    test_build_request()
    await test_prepared_request()
    await test_modified_request_blocking()
    await test_valid_response()
    await test_common_interface()
    await test_missing_candidate()
    await test_unexpected_candidate()
    await test_duplicate_candidate()
    await test_invalid_score()
    await test_invalid_json()
    await test_empty_response()
    await test_empty_candidates()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "OpenAI ranking evaluator "
        "fake-client test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )