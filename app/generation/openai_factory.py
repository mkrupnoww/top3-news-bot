from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import Settings
from app.generation.openai_client import (
    OpenAIResponsesGenerationClient,
)
from app.generation.openai_generator import (
    OpenAITelegramPostGenerator,
)
from app.ranking.openai_client import (
    AsyncOpenAIClientProtocol,
)
from app.ranking.openai_factory import (
    AsyncOpenAIClientFactory,
    create_openai_sdk_client,
)


@dataclass(frozen=True, slots=True)
class OpenAIGenerationRuntime:
    """Собранные компоненты генератора поста."""

    sdk_client: AsyncOpenAIClientProtocol

    responses_client: (
        OpenAIResponsesGenerationClient
    )

    generator: OpenAITelegramPostGenerator


def create_openai_generation_runtime(
    settings: Settings,
    *,
    client_factory: AsyncOpenAIClientFactory = (
        AsyncOpenAI
    ),
) -> OpenAIGenerationRuntime:
    """
    Собирает генератор Telegram-поста.

    Само создание runtime не выполняет запрос
    к OpenAI API.
    """

    sdk_client = create_openai_sdk_client(
        settings,
        client_factory=client_factory,
    )

    responses_client = (
        OpenAIResponsesGenerationClient(
            client=sdk_client
        )
    )

    generator = OpenAITelegramPostGenerator(
        client=responses_client,
        model_name=(
            settings.openai_generation_model
        ),
    )

    return OpenAIGenerationRuntime(
        sdk_client=sdk_client,
        responses_client=responses_client,
        generator=generator,
    )