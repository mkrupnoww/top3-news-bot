from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import Settings
from app.ranking.openai_event_client import (
    OpenAIResponsesEventRankingClient,
)
from app.ranking.openai_event_evaluator import (
    OpenAIEventRankingEvaluator,
)
from app.ranking.openai_factory import (
    AsyncOpenAIClientFactory,
    create_openai_sdk_client,
)
from app.ranking.openai_client import (
    AsyncOpenAIClientProtocol,
)


@dataclass(frozen=True, slots=True)
class OpenAIEventRankingRuntime:
    """Собранные компоненты event-level v2."""

    sdk_client: AsyncOpenAIClientProtocol

    responses_client: (
        OpenAIResponsesEventRankingClient
    )

    evaluator: OpenAIEventRankingEvaluator


def create_openai_event_ranking_runtime(
    settings: Settings,
    *,
    client_factory: AsyncOpenAIClientFactory = (
        AsyncOpenAI
    ),
) -> OpenAIEventRankingRuntime:
    """
    Собирает event-level OpenAI-контур v2.

    Само создание runtime не выполняет
    запросов к OpenAI API.
    """

    sdk_client = create_openai_sdk_client(
        settings,
        client_factory=client_factory,
    )

    responses_client = (
        OpenAIResponsesEventRankingClient(
            client=sdk_client
        )
    )

    evaluator = OpenAIEventRankingEvaluator(
        client=responses_client,
        model_name=(
            settings.openai_ranking_model
        ),
    )

    return OpenAIEventRankingRuntime(
        sdk_client=sdk_client,
        responses_client=responses_client,
        evaluator=evaluator,
    )