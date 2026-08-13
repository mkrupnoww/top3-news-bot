import asyncio
from dataclasses import dataclass
from typing import Any

from app.generation.openai_client import (
    OPENAI_GENERATION_RESPONSE_SCHEMA,
    OPENAI_GENERATION_SCHEMA_NAME,
    OpenAIResponsesGenerationClient,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
)

from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
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
                input_tokens=1200,
                input_tokens_details=(
                    FakeInputTokenDetails(
                        cached_tokens=200,
                        cache_write_tokens=100,
                    )
                ),
                output_tokens=400,
                output_tokens_details=(
                    FakeOutputTokenDetails(
                        reasoning_tokens=150,
                    )
                ),
                total_tokens=1600,
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


def build_request() -> GenerationModelRequest:
    """Создаёт тестовый запрос генерации."""

    return GenerationModelRequest(
        model="gpt-5.6-terra",
        instructions=(
            "Сформируй структурированный "
            "Telegram-пост."
        ),
        input_text=(
            '{"news":['
            '{"position":1,"news_id":201},'
            '{"position":2,"news_id":202},'
            '{"position":3,"news_id":203}'
            ']}'
        ),
    )


def build_valid_output() -> str:
    """Создаёт корректный JSON-ответ."""

    return (
        "{"
        '"post_text":'
        '"**TOP-3 киноновости дня**",'
        '"items":['
        "{"
        '"position":1,'
        '"news_id":201,'
        '"headline":"Первая новость",'
        '"body":"Текст первой новости."'
        "},"
        "{"
        '"position":2,'
        '"news_id":202,'
        '"headline":"Вторая новость",'
        '"body":"Текст второй новости."'
        "},"
        "{"
        '"position":3,'
        '"news_id":203,'
        '"headline":"Третья новость",'
        '"body":"Текст третьей новости."'
        "}"
        "]"
        "}"
    )


async def test_successful_request() -> None:
    """
    Проверяет параметры вызова SDK.

    Также проверяет извлечение usage
    и расчёт примерной стоимости.
    """

    expected_output = build_valid_output()

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text=expected_output
    )

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    result = (
        await generation_client
        .create_response(
            build_request()
        )
    )

    assert (
        result.output_text
        == expected_output
    )

    assert result.usage is not None

    assert (
        result.cost_estimate
        is not None
    )

    assert (
        result.usage.input_tokens
        == 1200
    )

    assert (
        result
        .usage
        .regular_input_tokens
        == 900
    )

    assert (
        result
        .usage
        .cached_input_tokens
        == 200
    )

    assert (
        result
        .usage
        .cache_write_tokens
        == 100
    )

    assert (
        result.usage.output_tokens
        == 400
    )

    assert (
        result
        .usage
        .reasoning_tokens
        == 150
    )

    assert (
        result.usage.total_tokens
        == 1600
    )

    assert (
        result
        .cost_estimate
        .model_name
        == "gpt-5.6-terra"
    )

    assert str(
        result
        .cost_estimate
        .regular_input_cost_usd
    ) == "0.00180000"

    assert str(
        result
        .cost_estimate
        .cached_input_cost_usd
    ) == "0.00004000"

    assert str(
        result
        .cost_estimate
        .cache_write_cost_usd
    ) == "0.00025000"

    assert str(
        result
        .cost_estimate
        .output_cost_usd
    ) == "0.00480000"

    assert str(
        result
        .cost_estimate
        .total_cost_usd
    ) == "0.00689000"

    assert len(
        fake_sdk_client.responses.calls
    ) == 1

    call = (
        fake_sdk_client
        .responses
        .calls[0]
    )

    assert call["model"] == (
        "gpt-5.6-terra"
    )

    assert call["instructions"] == (
        "Сформируй структурированный "
        "Telegram-пост."
    )

    assert call["input"] == (
        '{"news":['
        '{"position":1,"news_id":201},'
        '{"position":2,"news_id":202},'
        '{"position":3,"news_id":203}'
        ']}'
    )

    assert call["store"] is False

    response_format = (
        call["text"]["format"]
    )

    assert (
        response_format["type"]
        == "json_schema"
    )

    assert (
        response_format["name"]
        == OPENAI_GENERATION_SCHEMA_NAME
    )

    assert (
        response_format["strict"]
        is True
    )

    assert (
        response_format["schema"]
        == OPENAI_GENERATION_RESPONSE_SCHEMA
    )

    schema = response_format["schema"]

    assert (
        schema["additionalProperties"]
        is False
    )

    assert set(
        schema["required"]
    ) == {
        "post_text",
        "items",
    }

    assert (
        schema[
            "properties"
        ][
            "post_text"
        ][
            "maxLength"
        ]
        == MAXIMUM_POST_LENGTH
    )

    items_schema = (
        schema[
            "properties"
        ][
            "items"
        ]
    )

    assert (
        items_schema["minItems"]
        == 3
    )

    assert (
        items_schema["maxItems"]
        == 3
    )

    item_schema = items_schema["items"]

    assert (
        item_schema[
            "additionalProperties"
        ]
        is False
    )

    assert set(
        item_schema["required"]
    ) == {
        "position",
        "news_id",
        "headline",
        "body",
    }

    assert (
        item_schema[
            "properties"
        ][
            "position"
        ][
            "minimum"
        ]
        == 1
    )

    assert (
        item_schema[
            "properties"
        ][
            "position"
        ][
            "maximum"
        ]
        == 3
    )

    print(
        "Generation Responses API "
        "adapter call: OK"
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

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    try:
        await generation_client.create_response(
            build_request()
        )
    except ValueError as error:
        assert (
            "пустой output_text"
            in str(error)
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
    """Проверяет ответ без output_text."""

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
            self.responses = (
                ResponsesWithoutText()
            )

    fake_sdk_client = ClientWithoutText()

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    try:
        await generation_client.create_response(
            build_request()
        )
    except ValueError as error:
        assert (
            "не вернул текстовое поле "
            "output_text"
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
            "Missing output_text "
            "blocking: OK"
        )
        print("sdk_call_count=1")
        return

    raise AssertionError(
        "Ответ без output_text "
        "не был заблокирован."
    )


async def test_missing_usage() -> None:
    """Проверяет ответ без usage."""

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
                    build_valid_output()
                )
            )

    class ClientWithoutUsage:
        def __init__(self) -> None:
            self.responses = (
                ResponsesWithoutUsage()
            )

    fake_sdk_client = ClientWithoutUsage()

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    try:
        await generation_client.create_response(
            build_request()
        )
    except ValueError as error:
        assert (
            "не содержит usage"
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
        output_text=build_valid_output()
    )

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    invalid_request = GenerationModelRequest(
        model="   ",
        instructions="Тест.",
        input_text="{}",
    )

    try:
        await generation_client.create_response(
            invalid_request
        )
    except ValueError as error:
        assert (
            "request.model"
            in str(error)
        )

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
        "Пустая модель "
        "не была заблокирована."
    )


async def test_invalid_instructions() -> None:
    """Проверяет пустые инструкции."""

    fake_sdk_client = FakeAsyncOpenAIClient(
        output_text=build_valid_output()
    )

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    invalid_request = GenerationModelRequest(
        model="gpt-5.6-terra",
        instructions="   ",
        input_text="{}",
    )

    try:
        await generation_client.create_response(
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
            "Invalid instructions "
            "blocking: OK"
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
        output_text=build_valid_output()
    )

    generation_client = (
        OpenAIResponsesGenerationClient(
            client=fake_sdk_client
        )
    )

    invalid_request = GenerationModelRequest(
        model="gpt-5.6-terra",
        instructions="Тест.",
        input_text="   ",
    )

    try:
        await generation_client.create_response(
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
    """Запускает тест адаптера генерации."""

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
        "OpenAI generation Responses client "
        "fake-SDK test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )