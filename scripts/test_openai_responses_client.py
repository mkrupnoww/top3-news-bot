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
class FakeInputTokenDetails:
    """Поддельная детализация входных токенов."""

    cached_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True, slots=True)
class FakeOutputTokenDetails:
    """Поддельная детализация выходных токенов."""

    reasoning_tokens: int


@dataclass(frozen=True, slots=True)
class FakeUsage:
    """Поддельный объект usage OpenAI SDK."""

    input_tokens: int
    input_tokens_details: (
        FakeInputTokenDetails
    )
    output_tokens: int
    output_tokens_details: (
        FakeOutputTokenDetails
    )
    total_tokens: int


@dataclass(frozen=True, slots=True)
class FakeResponse:
    """Поддельный ответ OpenAI SDK."""

    output_text: str
    usage: FakeUsage


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
            output_text=self._output_text,
            usage=FakeUsage(
                input_tokens=1500,
                input_tokens_details=(
                    FakeInputTokenDetails(
                        cached_tokens=400,
                        cache_write_tokens=100,
                    )
                ),
                output_tokens=300,
                output_tokens_details=(
                    FakeOutputTokenDetails(
                        reasoning_tokens=200,
                    )
                ),
                total_tokens=1800,
            ),
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
        model="gpt-5.6-terra",
        instructions=(
            "Верни структурированные оценки."
        ),
        input_text=(
            '{"candidates":[{"news_id":101}]}'
        ),
    )


async def test_successful_request() -> None:
    """
    Проверяет параметры вызова SDK.

    Также проверяет извлечение usage
    и расчёт примерной стоимости.
    """

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

    result = (
        await ranking_client.create_response(
            build_request()
        )
    )

    assert result.output_text == expected_output

    assert result.usage is not None
    assert result.cost_estimate is not None

    assert result.usage.input_tokens == 1500

    assert (
        result.usage.regular_input_tokens
        == 1000
    )

    assert (
        result.usage.cached_input_tokens
        == 400
    )

    assert (
        result.usage.cache_write_tokens
        == 100
    )

    assert result.usage.output_tokens == 300

    assert (
        result.usage.reasoning_tokens
        == 200
    )

    assert result.usage.total_tokens == 1800

    assert (
        result.cost_estimate.model_name
        == "gpt-5.6-terra"
    )

    assert str(
        result
        .cost_estimate
        .regular_input_cost_usd
    ) == "0.00200000"

    assert str(
        result
        .cost_estimate
        .cached_input_cost_usd
    ) == "0.00008000"

    assert str(
        result
        .cost_estimate
        .cache_write_cost_usd
    ) == "0.00025000"

    assert str(
        result
        .cost_estimate
        .output_cost_usd
    ) == "0.00360000"

    assert str(
        result
        .cost_estimate
        .total_cost_usd
    ) == "0.00593000"

    assert len(
        fake_sdk_client.responses.calls
    ) == 1

    call = (
        fake_sdk_client.responses.calls[0]
    )

    assert call["model"] == (
        "gpt-5.6-terra"
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
    print(
        "input_tokens="
        f"{result.usage.input_tokens}"
    )
    print(
        "regular_input_tokens="
        f"{result.usage.regular_input_tokens}"
    )
    print(
        "cached_input_tokens="
        f"{result.usage.cached_input_tokens}"
    )
    print(
        "cache_write_tokens="
        f"{result.usage.cache_write_tokens}"
    )
    print(
        "output_tokens="
        f"{result.usage.output_tokens}"
    )
    print(
        "reasoning_tokens="
        f"{result.usage.reasoning_tokens}"
    )
    print(
        "total_tokens="
        f"{result.usage.total_tokens}"
    )
    print(
        "estimated_cost_usd="
        f"{result.cost_estimate.total_cost_usd}"
    )


async def test_empty_output() -> None:
    """Проверяет блокировку пустого ответа."""

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text="   "
    )

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
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

        assert len(
            fake_sdk_client.responses.calls
        ) == 1

        print()
        print(
            "Empty output blocking: OK"
        )
        print("sdk_call_count=1")
        return

    raise AssertionError(
        "Пустой output_text "
        "не был заблокирован."
    )


async def test_missing_output_text() -> None:
    """Проверяет ответ без строкового output_text."""

    @dataclass(frozen=True, slots=True)
    class ResponseWithoutText:
        usage: FakeUsage

    class ResponsesWithoutText:
        def __init__(self) -> None:
            self.call_count = 0

        async def create(
            self,
            **kwargs: Any,
        ) -> ResponseWithoutText:
            self.call_count += 1

            return ResponseWithoutText(
                usage=FakeUsage(
                    input_tokens=10,
                    input_tokens_details=(
                        FakeInputTokenDetails(
                            cached_tokens=0,
                            cache_write_tokens=0,
                        )
                    ),
                    output_tokens=5,
                    output_tokens_details=(
                        FakeOutputTokenDetails(
                            reasoning_tokens=0,
                        )
                    ),
                    total_tokens=15,
                )
            )

    class ClientWithoutText:
        def __init__(self) -> None:
            self.responses = ResponsesWithoutText()

    fake_sdk_client = ClientWithoutText()

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
        )
    )

    try:
        await ranking_client.create_response(
            build_request()
        )
    except ValueError as error:
        assert (
            "не вернул текстовое поле output_text"
            in str(error)
        )

        assert (
            fake_sdk_client
            .responses
            .call_count
            == 1
        )

        print()
        print(
            "Missing output_text blocking: OK"
        )
        print("sdk_call_count=1")
        return

    raise AssertionError(
        "Ответ без output_text "
        "не был заблокирован."
    )


async def test_missing_usage() -> None:
    """Проверяет блокировку ответа без usage."""

    @dataclass(frozen=True, slots=True)
    class ResponseWithoutUsage:
        output_text: str

    class ResponsesWithoutUsage:
        def __init__(self) -> None:
            self.call_count = 0

        async def create(
            self,
            **kwargs: Any,
        ) -> ResponseWithoutUsage:
            self.call_count += 1

            return ResponseWithoutUsage(
                output_text=(
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
            )

    class ClientWithoutUsage:
        def __init__(self) -> None:
            self.responses = (
                ResponsesWithoutUsage()
            )

    fake_sdk_client = ClientWithoutUsage()

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
        )
    )

    try:
        await ranking_client.create_response(
            build_request()
        )
    except ValueError as error:
        assert "не содержит usage" in str(
            error
        )

        assert (
            fake_sdk_client
            .responses
            .call_count
            == 1
        )

        print()
        print(
            "Missing usage blocking: OK"
        )
        print("sdk_call_count=1")
        return

    raise AssertionError(
        "Ответ без usage "
        "не был заблокирован."
    )


async def test_invalid_model() -> None:
    """Проверяет пустую модель."""

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text='{"scores":[]}'
    )

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
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

        assert len(
            fake_sdk_client.responses.calls
        ) == 0

        print()
        print(
            "Invalid model blocking: OK"
        )
        print("sdk_call_count=0")
        return

    raise AssertionError(
        "Пустая модель не была заблокирована."
    )


async def test_invalid_instructions() -> None:
    """Проверяет пустые инструкции."""

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text='{"scores":[]}'
    )

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
        )
    )

    invalid_request = RankingModelRequest(
        model="gpt-5.6-terra",
        instructions="   ",
        input_text="{}",
    )

    try:
        await ranking_client.create_response(
            invalid_request
        )
    except ValueError as error:
        assert (
            "request.instructions"
            in str(error)
        )

        assert len(
            fake_sdk_client.responses.calls
        ) == 0

        print()
        print(
            "Invalid instructions blocking: OK"
        )
        print("sdk_call_count=0")
        return

    raise AssertionError(
        "Пустые инструкции "
        "не были заблокированы."
    )


async def test_invalid_input() -> None:
    """Проверяет пустой входной текст."""

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text='{"scores":[]}'
    )

    ranking_client = (
        OpenAIResponsesRankingClient(
            client=fake_sdk_client
        )
    )

    invalid_request = RankingModelRequest(
        model="gpt-5.6-terra",
        instructions="Тест.",
        input_text="   ",
    )

    try:
        await ranking_client.create_response(
            invalid_request
        )
    except ValueError as error:
        assert (
            "request.input_text"
            in str(error)
        )

        assert len(
            fake_sdk_client.responses.calls
        ) == 0

        print()
        print(
            "Invalid input blocking: OK"
        )
        print("sdk_call_count=0")
        return

    raise AssertionError(
        "Пустой input_text "
        "не был заблокирован."
    )


async def main() -> int:
    """Запускает тест адаптера SDK."""

    await test_successful_request()
    await test_empty_output()
    await test_missing_output_text()
    await test_missing_usage()
    await test_invalid_model()
    await test_invalid_instructions()
    await test_invalid_input()

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