import asyncio
from dataclasses import dataclass
from typing import Any

from app.ranking.openai_client import (
    OPENAI_RANKING_RESPONSE_SCHEMA,
    OPENAI_RANKING_SCHEMA_NAME,
    OpenAIResponsesRankingClient,
)
from app.ranking.openai_evaluator import (
    RankingModelRequest,
)


@dataclass(frozen=True, slots=True)
class FakeResponse:
    """Поддельный ответ OpenAI SDK."""

    output_text: str


class FakeResponsesResource:
    """Поддельный ресурс client.responses."""

    def __init__(
        self,
        *,
        output_text: str,
    ) -> None:
        self._output_text = output_text
        self.calls: list[
            dict[str, Any]
        ] = []

    async def create(
        self,
        **kwargs: Any,
    ) -> FakeResponse:
        """Запоминает параметры без сети."""

        self.calls.append(kwargs)

        return FakeResponse(
            output_text=self._output_text
        )


class FakeAsyncOpenAIClient:
    """Поддельный AsyncOpenAI-клиент."""

    def __init__(
        self,
        *,
        output_text: str,
    ) -> None:
        self.responses = (
            FakeResponsesResource(
                output_text=output_text
            )
        )


def build_request() -> RankingModelRequest:
    """Создаёт тестовый запрос."""

    return RankingModelRequest(
        model="test-model-no-network",
        instructions=(
            "Верни структурированные оценки."
        ),
        input_text=(
            '{"candidates":[{"news_id":101}]}'
        ),
    )


async def test_successful_request() -> None:
    """Проверяет параметры вызова SDK."""

    expected_output = (
        '{"scores":[{'
        '"news_id":101,'
        '"f_score":9,'
        '"m_score":8,'
        '"r_score":7,'
        '"h_score":6,'
        '"q_score":0.9,'
        '"explanation":"Тест."'
        '}]}'
    )

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text=expected_output
    )

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
        )
    )

    actual_output = (
        await ranking_client.create_response(
            build_request()
        )
    )

    assert actual_output == expected_output

    assert len(
        fake_sdk_client.responses.calls
    ) == 1

    call = (
        fake_sdk_client.responses.calls[0]
    )

    assert call["model"] == (
        "test-model-no-network"
    )

    assert call["instructions"] == (
        "Верни структурированные оценки."
    )

    assert call["input"] == (
        '{"candidates":[{"news_id":101}]}'
    )

    assert call["store"] is False

    response_format = (
        call["text"]["format"]
    )

    assert response_format["type"] == (
        "json_schema"
    )

    assert response_format["name"] == (
        OPENAI_RANKING_SCHEMA_NAME
    )

    assert response_format["strict"] is True

    assert response_format["schema"] == (
        OPENAI_RANKING_RESPONSE_SCHEMA
    )

    schema = response_format["schema"]

    assert (
        schema["additionalProperties"]
        is False
    )

    score_item_schema = (
        schema[
            "properties"
        ][
            "scores"
        ][
            "items"
        ]
    )

    assert (
        score_item_schema[
            "additionalProperties"
        ]
        is False
    )

    assert set(
        score_item_schema["required"]
    ) == {
        "news_id",
        "f_score",
        "m_score",
        "r_score",
        "h_score",
        "q_score",
        "explanation",
    }

    print(
        "Responses API adapter call: OK"
    )
    print(
        "sdk_call_count="
        f"{len(fake_sdk_client.responses.calls)}"
    )
    print(
        "response_format="
        f"{response_format['type']}"
    )
    print(
        "schema_name="
        f"{response_format['name']}"
    )
    print(
        "strict_schema="
        f"{str(response_format['strict']).lower()}"
    )
    print(
        "store="
        f"{str(call['store']).lower()}"
    )


async def test_empty_output() -> None:
    """Проверяет блокировку пустого ответа."""

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=FakeAsyncOpenAIClient(
                output_text="   "
            )
        )
    )

    try:
        await ranking_client.create_response(
            build_request()
        )
    except ValueError as error:
        assert "пустой output_text" in str(
            error
        )

        print()
        print(
            "Empty output blocking: OK"
        )
        return

    raise AssertionError(
        "Пустой output_text "
        "не был заблокирован."
    )


async def test_invalid_request() -> None:
    """Проверяет пустую модель."""

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=FakeAsyncOpenAIClient(
                output_text='{"scores":[]}'
            )
        )
    )

    invalid_request = RankingModelRequest(
        model="   ",
        instructions="Тест.",
        input_text="{}",
    )

    try:
        await ranking_client.create_response(
            invalid_request
        )
    except ValueError as error:
        assert "request.model" in str(error)

        print()
        print(
            "Invalid request blocking: OK"
        )
        return

    raise AssertionError(
        "Пустая модель не была заблокирована."
    )


async def main() -> int:
    """Запускает тест адаптера SDK."""

    await test_successful_request()
    await test_empty_output()
    await test_invalid_request()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "OpenAI Responses client "
        "fake-SDK test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )