from typing import Any

from app.ranking.event_evaluator import (
    EventRankingModelRequest,
    EventRankingModelResponse,
    StructuredEventRankingClient,
)
from app.ranking.openai_client import (
    AsyncOpenAIClientProtocol,
)
from app.ranking.openai_usage import (
    calculate_openai_cost,
    extract_response_usage,
    get_model_pricing,
)


OPENAI_EVENT_RANKING_SCHEMA_NAME = (
    "movie_news_event_ranking"
)


OPENAI_EVENT_RANKING_RESPONSE_SCHEMA: dict[
    str,
    Any,
] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "minItems": 1,
            "maxItems": 500,
            "items": {
                "type": "object",
                "properties": {
                    "representative_news_id": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "event_title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                    "event_time_utc": {
                        "type": "string",
                        "format": "date-time",
                    },
                    "macro_topic": {
                        "type": "string",
                        "enum": [
                            "business_economy_law",
                            "people_conflicts_legal",
                            "creative_cast_production",
                            "trailers_premieres_releases",
                            "festivals_awards_criticism",
                            (
                                "box_office_audience_"
                                "distribution"
                            ),
                            "other",
                        ],
                    },
                    "i_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "k_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "n_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "e_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "x_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "q_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "impact_reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                    "hook_reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                    "q_reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                    "members": {
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
                                "source_relation": {
                                    "type": "string",
                                    "enum": [
                                        "primary",
                                        "independent",
                                        "syndicated",
                                        "duplicate",
                                    ],
                                },
                                "is_representative": {
                                    "type": "boolean",
                                },
                                "is_independent_source": {
                                    "type": "boolean",
                                },
                                "counts_toward_reach": {
                                    "type": "boolean",
                                },
                                "source_weight": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 3,
                                },
                                "membership_reason": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 1000,
                                },
                            },
                            "required": [
                                "news_id",
                                "source_relation",
                                "is_representative",
                                "is_independent_source",
                                "counts_toward_reach",
                                "source_weight",
                                "membership_reason",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "representative_news_id",
                    "event_title",
                    "event_time_utc",
                    "macro_topic",
                    "i_score",
                    "k_score",
                    "n_score",
                    "e_score",
                    "x_score",
                    "q_score",
                    "impact_reason",
                    "hook_reason",
                    "q_reason",
                    "members",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "events",
    ],
    "additionalProperties": False,
}


class OpenAIResponsesEventRankingClient(
    StructuredEventRankingClient
):
    """
    Адаптер OpenAI Responses API для event-level v2.

    Получает EventRankingModelRequest и возвращает
    структурированный JSON-ответ вместе с usage
    и оценочной стоимостью.
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
        request: EventRankingModelRequest,
    ) -> EventRankingModelResponse:
        """Выполняет один event-level запрос."""

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
                            OPENAI_EVENT_RANKING_SCHEMA_NAME
                        ),
                        "description": (
                            "Группировка публикаций "
                            "по инфоповодам и экспертные "
                            "компоненты полной формулы "
                            "TOP-3 киноновостей."
                        ),
                        "schema": (
                            OPENAI_EVENT_RANKING_RESPONSE_SCHEMA
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

        return EventRankingModelResponse(
            output_text=normalized_output,
            usage=usage,
            cost_estimate=cost_estimate,
        )