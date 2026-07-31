from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

from app.config import Settings
from app.ranking.openai_client import (
    AsyncOpenAIClientProtocol,
    OpenAIResponsesRankingClient,
)
from app.ranking.openai_evaluator import (
    OpenAIRankingEvaluator,
)


class AsyncOpenAIClientFactory(Protocol):
    """Фабрика совместимого AsyncOpenAI-клиента."""

    def __call__(
        self,
        *,
        api_key: str,
        timeout: float,
        max_retries: int,
    ) -> AsyncOpenAIClientProtocol:
        """Создаёт клиент без выполнения запроса."""

        ...


@dataclass(frozen=True, slots=True)
class OpenAIRankingRuntime:
    """Собранные компоненты OpenAI-оценщика."""

    sdk_client: AsyncOpenAIClientProtocol
    responses_client: OpenAIResponsesRankingClient
    evaluator: OpenAIRankingEvaluator


def require_openai_api_key(
    settings: Settings,
) -> str:
    """Возвращает настроенный непустой API-ключ."""

    if settings.openai_api_key is None:
        raise ValueError(
            "OPENAI_API_KEY не настроен."
        )

    api_key = (
        settings
        .openai_api_key
        .get_secret_value()
        .strip()
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY не настроен."
        )

    return api_key


def create_openai_sdk_client(
    settings: Settings,
    *,
    client_factory: AsyncOpenAIClientFactory = (
        AsyncOpenAI
    ),
) -> AsyncOpenAIClientProtocol:
    """
    Создаёт AsyncOpenAI-клиент из настроек.

    Само создание клиента не выполняет
    запрос к OpenAI API.
    """

    api_key = require_openai_api_key(
        settings
    )

    return client_factory(
        api_key=api_key,
        timeout=(
            settings.openai_timeout_seconds
        ),
        max_retries=(
            settings.openai_max_retries
        ),
    )


def create_openai_ranking_runtime(
    settings: Settings,
    *,
    client_factory: AsyncOpenAIClientFactory = (
        AsyncOpenAI
    ),
) -> OpenAIRankingRuntime:
    """Собирает полный OpenAI-оценщик."""

    sdk_client = create_openai_sdk_client(
        settings,
        client_factory=client_factory,
    )

    responses_client = (
        OpenAIResponsesRankingClient(
            client=sdk_client
        )
    )

    evaluator = OpenAIRankingEvaluator(
        client=responses_client,
        model_name=(
            settings.openai_ranking_model
        ),
    )

    return OpenAIRankingRuntime(
        sdk_client=sdk_client,
        responses_client=responses_client,
        evaluator=evaluator,
    )