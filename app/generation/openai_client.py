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
            "maxLength": 4096,
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
                        "maxLength": 300,
                    },
                    "body": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1400,
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

        response = (
            await self._client.responses.create(
                model=normalized_model,
                instructions=(
                    normalized_instructions
                ),
                input=normalized_input,
                text={
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
                store=False,
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

        return GenerationModelResponse(
            output_text=normalized_output,
            usage=usage,
            cost_estimate=cost_estimate,
        )