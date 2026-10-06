from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI

from app.config import Settings
from app.generation.image_generator import (
    OpenAIMovieNewsImageGenerator,
)
from app.generation.openai_image_client import (
    AsyncOpenAIImageClientProtocol,
    OpenAIImagesGenerationClient,
)
from app.ranking.openai_factory import (
    require_openai_api_key,
)


class AsyncOpenAIImageClientFactory(
    Protocol
):
    """Фабрика AsyncOpenAI для Image API."""

    def __call__(
        self,
        *,
        api_key: str,
        timeout: float,
        max_retries: int,
    ) -> AsyncOpenAIImageClientProtocol:
        """Создаёт клиент без API-запроса."""

        ...


@dataclass(frozen=True, slots=True)
class OpenAIImageGenerationRuntime:
    """Собранные компоненты Image API."""

    sdk_client: AsyncOpenAIImageClientProtocol
    images_client: OpenAIImagesGenerationClient
    generator: OpenAIMovieNewsImageGenerator


def create_openai_image_generation_runtime(
    settings: Settings,
    *,
    client_factory: (
        AsyncOpenAIImageClientFactory
    ) = AsyncOpenAI,
) -> OpenAIImageGenerationRuntime:
    """
    Собирает runtime генерации изображений.

    Само создание runtime не выполняет
    запрос к OpenAI API.
    """

    api_key = require_openai_api_key(
        settings
    )

    sdk_client = client_factory(
        api_key=api_key,
        timeout=(
            settings.openai_timeout_seconds
        ),
        max_retries=(
            settings.openai_max_retries
        ),
    )

    images_client = (
        OpenAIImagesGenerationClient(
            client=sdk_client
        )
    )

    generator = (
        OpenAIMovieNewsImageGenerator(
            client=images_client,
            model_name=(
                settings.openai_image_model
            ),
            size=(
                settings.openai_image_size
            ),
        )
    )

    return OpenAIImageGenerationRuntime(
        sdk_client=sdk_client,
        images_client=images_client,
        generator=generator,
    )