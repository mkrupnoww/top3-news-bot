from typing import Any, Protocol, runtime_checkable

from app.ranking.openai_evaluator import (
    RankingModelRequest,
    RankingModelResponse,
    StructuredRankingClient,
)
from app.ranking.openai_usage import (
    calculate_openai_cost,
    extract_response_usage,
    get_model_pricing,
)


OPENAI_RANKING_SCHEMA_NAME = (
    "movie_news_ranking"
)


OPENAI_RANKING_RESPONSE_SCHEMA: dict[
    str,
    Any,
] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "minItems": 1,
            "maxItems": 500,
            "items": {
                "type": "object",
                "properties": {
                    "news_id": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "f_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "m_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "r_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "h_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "q_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "explanation": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                },
                "required": [
                    "news_id",
                    "f_score",
                    "m_score",
                    "r_score",
                    "h_score",
                    "q_score",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "scores",
    ],
    "additionalProperties": False,
}


@runtime_checkable
class ResponsesResourceProtocol(
    Protocol
):
    """Минимальный контракт Responses API."""

    async def create(
        self,
        **kwargs: Any,
    ) -> Any:
        """Создаёт ответ модели."""

        ...


@runtime_checkable
class AsyncOpenAIClientProtocol(
    Protocol
):
    """Минимальный контракт AsyncOpenAI."""

    responses: ResponsesResourceProtocol


class OpenAIResponsesRankingClient(
    StructuredRankingClient
):
    """
    Адаптер OpenAI Responses API.

    Получает внутренний RankingModelRequest
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
        request: RankingModelRequest,
    ) -> RankingModelResponse:
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
                            OPENAI_RANKING_SCHEMA_NAME
                        ),
                        "description": (
                            "Оценки кандидатов для "
                            "ежедневного TOP-3 "
                            "киноновостей."
                        ),
                        "schema": (
                            OPENAI_RANKING_RESPONSE_SCHEMA
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

        return RankingModelResponse(
            output_text=normalized_output,
            usage=usage,
            cost_estimate=cost_estimate,
        )