from typing import Any

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    StructuredGenerationClient,
)
from app.ranking.openai_client import (
    AsyncOpenAIClientProtocol,
)
from app.ranking.openai_usage import (
    calculate_openai_cost,
    extract_response_usage,
    get_model_pricing,
)

from app.generation.post_contract import (
    MAXIMUM_BODY_LENGTH,
    MAXIMUM_HEADLINE_LENGTH,
    MAXIMUM_POST_LENGTH,
)


OPENAI_GENERATION_SCHEMA_NAME = (
    "movie_news_telegram_post"
)


OPENAI_GENERATION_RESPONSE_SCHEMA: dict[
    str,
    Any,
] = {
    "type": "object",
    "properties": {
        "post_text": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAXIMUM_POST_LENGTH,
        },
        "items": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                    },
                    "news_id": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "headline": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAXIMUM_HEADLINE_LENGTH,
                    },
                    "body": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAXIMUM_BODY_LENGTH,
                    },
                    },
                "required": [
                    "position",
                    "news_id",
                    "headline",
                    "body",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "post_text",
        "items",
    ],
    "additionalProperties": False,
}


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Преобразует SDK Response в обычный dict."""

    model_dump = getattr(response, "model_dump", None)

    if callable(model_dump):
        dumped = model_dump(
            warnings=False
        )

        if isinstance(dumped, dict):
            return dumped

    if isinstance(response, dict):
        return response

    return {}


def _extract_web_search_telemetry(
    response: Any,
) -> tuple[int, tuple[str, ...]]:
    """Возвращает число web search и URL источников."""

    payload = _response_to_dict(response)
    output = payload.get("output")

    if not isinstance(output, list):
        return 0, ()

    web_search_call_count = 0
    urls: list[str] = []

    def collect_urls(value: Any) -> None:
        if isinstance(value, dict):
            url = value.get("url")

            if isinstance(url, str):
                normalized_url = url.strip()

                if (
                    normalized_url
                    and normalized_url not in urls
                ):
                    urls.append(normalized_url)

            for nested_value in value.values():
                collect_urls(nested_value)

        elif isinstance(value, list):
            for nested_value in value:
                collect_urls(nested_value)

    for item in output:
        if not isinstance(item, dict):
            continue

        if item.get("type") != "web_search_call":
            continue

        web_search_call_count += 1
        collect_urls(item.get("action"))

    return web_search_call_count, tuple(urls)


class OpenAIResponsesGenerationClient(
    StructuredGenerationClient
):
    """
    Адаптер OpenAI Responses API для постов.

    Получает внутренний GenerationModelRequest
    и возвращает структурированный ответ модели
    вместе с токенами и расчётом стоимости.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAIClientProtocol,
    ) -> None:
        if not isinstance(
            client,
            AsyncOpenAIClientProtocol,
        ):
            raise TypeError(
                "client не соответствует "
                "интерфейсу AsyncOpenAI."
            )

        self._client = client

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """
        Выполняет один запрос Responses API.

        Возвращает:
        - структурированный JSON-текст;
        - фактическое потребление токенов;
        - оценочную стоимость запроса.
        """

        normalized_model = (
            request.model.strip()
        )

        normalized_instructions = (
            request.instructions.strip()
        )

        normalized_input = (
            request.input_text.strip()
        )

        if not normalized_model:
            raise ValueError(
                "request.model не может "
                "быть пустым."
            )

        if not normalized_instructions:
            raise ValueError(
                "request.instructions не может "
                "быть пустым."
            )

        if not normalized_input:
            raise ValueError(
                "request.input_text не может "
                "быть пустым."
            )

        request_kwargs: dict[str, Any] = {
            "model": normalized_model,
            "instructions": (
                normalized_instructions
            ),
            "input": normalized_input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": (
                        OPENAI_GENERATION_SCHEMA_NAME
                    ),
                    "description": (
                        "Русскоязычный "
                        "Telegram-пост с TOP-3 "
                        "киноновостей."
                    ),
                    "schema": (
                        OPENAI_GENERATION_RESPONSE_SCHEMA
                    ),
                    "strict": True,
                }
            },
            "store": False,
        }

        if request.allow_web_search:
            request_kwargs.update(
                {
                    "tools": [
                        {
                            "type": "web_search",
                        }
                    ],
                    "tool_choice": "auto",
                    "include": [
                        (
                            "web_search_call."
                            "action.sources"
                        )
                    ],
                }
            )

        response = (
            await self._client.responses.create(
                **request_kwargs
            )
        )

        output_text = getattr(
            response,
            "output_text",
            None,
        )

        if not isinstance(
            output_text,
            str,
        ):
            raise ValueError(
                "OpenAI Responses API не вернул "
                "текстовое поле output_text."
            )

        normalized_output = (
            output_text.strip()
        )

        if not normalized_output:
            raise ValueError(
                "OpenAI Responses API вернул "
                "пустой output_text."
            )

        usage = extract_response_usage(
            response
        )

        pricing = get_model_pricing(
            normalized_model
        )

        cost_estimate = calculate_openai_cost(
            usage,
            pricing,
        )

        (
            web_search_call_count,
            web_source_urls,
        ) = _extract_web_search_telemetry(
            response
        )

        return GenerationModelResponse(
            output_text=normalized_output,
            usage=usage,
            cost_estimate=cost_estimate,
            web_search_used=(
                web_search_call_count > 0
            ),
            web_search_call_count=(
                web_search_call_count
            ),
            web_source_urls=web_source_urls,
        )