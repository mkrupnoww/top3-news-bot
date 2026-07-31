import asyncio
from datetime import datetime, timezone
import json

from app.db.news_candidates import NewsCandidate
from app.ranking.openai_evaluator import (
    OpenAIRankingEvaluator,
    RankingModelRequest,
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
    ) -> str:
        """Возвращает заранее заданный JSON."""

        self.requests.append(request)

        return self._response_text


def build_candidates() -> tuple[
    NewsCandidate,
    ...
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


async def test_valid_response() -> None:
    """Проверяет корректный ответ."""

    client = FakeStructuredRankingClient(
        build_valid_response()
    )

    evaluator = OpenAIRankingEvaluator(
        client=client,
        model_name="test-model-no-network",
    )

    candidates = build_candidates()

    assessments = await evaluator.evaluate(
        candidates
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

    evaluator = OpenAIRankingEvaluator(
        client=(
            FakeStructuredRankingClient(
                response_text
            )
        ),
        model_name="test-model-no-network",
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

    evaluator = OpenAIRankingEvaluator(
        client=(
            FakeStructuredRankingClient(
                response_text
            )
        ),
        model_name="test-model-no-network",
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


async def main() -> int:
    """Запускает тест оценщика."""

    await test_valid_response()
    await test_missing_candidate()
    await test_invalid_score()

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