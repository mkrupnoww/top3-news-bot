import asyncio
import base64
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.generation.image_generator import (
    ImageGenerationNewsItem,
    ImageModelRequest,
)
from app.generation.openai_image_client import (
    OpenAIImagesGenerationClient,
)
from app.generation.openai_image_factory import (
    create_openai_image_generation_runtime,
)


SYNTHETIC_IMAGE_BYTES = (
    b"synthetic-image-api-bytes"
)


class FakeImagesResource:
    """Поддельный AsyncOpenAI.images."""

    def __init__(
        self,
        *,
        include_usage: bool = True,
    ) -> None:
        self.calls: list[
            dict[str, Any]
        ] = []
        self.include_usage = include_usage

    async def generate(
        self,
        **kwargs: Any,
    ) -> Any:
        """Возвращает синтетический ImagesResponse."""

        self.calls.append(
            dict(kwargs)
        )

        usage: Any

        if self.include_usage:
            usage = SimpleNamespace(
                input_tokens=120,
                input_tokens_details=(
                    SimpleNamespace(
                        text_tokens=120,
                        image_tokens=0,
                    )
                ),
                output_tokens=900,
                output_tokens_details=(
                    SimpleNamespace(
                        text_tokens=0,
                        image_tokens=900,
                    )
                ),
                total_tokens=1020,
            )
        else:
            usage = None

        return SimpleNamespace(
            created=1_800_000_000,
            background="opaque",
            data=[
                SimpleNamespace(
                    b64_json=(
                        base64.b64encode(
                            SYNTHETIC_IMAGE_BYTES
                        ).decode("ascii")
                    ),
                    revised_prompt=None,
                )
            ],
            output_format="png",
            quality="medium",
            size="1024x1536",
            usage=usage,
        )


class FakeAsyncOpenAIClient:
    """Поддельный AsyncOpenAI Image client."""

    def __init__(
        self,
        *,
        include_usage: bool = True,
    ) -> None:
        self.images = FakeImagesResource(
            include_usage=include_usage
        )


class RecordingClientFactory:
    """Запоминает создание SDK-клиента."""

    def __init__(self) -> None:
        self.client = FakeAsyncOpenAIClient()
        self.calls: list[
            dict[str, object]
        ] = []

    def __call__(
        self,
        *,
        api_key: str,
        timeout: float,
        max_retries: int,
    ) -> FakeAsyncOpenAIClient:
        """Возвращает fake client без сети."""

        self.calls.append(
            {
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries,
            }
        )

        return self.client


def build_settings(
    *,
    api_key: str | None,
) -> Settings:
    """Создаёт тестовые настройки."""

    return Settings(
        APP_ENV="testing",
        LOG_LEVEL="INFO",
        TELEGRAM_BOT_USERNAME=(
            "test_top3_news_bot"
        ),
        TELEGRAM_BOT_TOKEN=(
            "test-telegram-token"
        ),
        TELEGRAM_CHANNEL_ID=(
            -1001234567890
        ),
        DB_HOST="127.0.0.1",
        DB_PORT=5432,
        DB_NAME="test_top3_news_db",
        DB_USER="test_top3_news_app",
        DB_PASSWORD="test-db-password",
        DB_SCHEMA="top3_news",
        OPENAI_API_KEY=api_key,
        OPENAI_RANKING_MODEL=(
            "test-ranking-model"
        ),
        OPENAI_GENERATION_MODEL=(
            "test-generation-model"
        ),
        OPENAI_IMAGE_MODEL=(
            "test-image-model"
        ),
        OPENAI_IMAGE_SIZE="1024x1536",
        OPENAI_TIMEOUT_SECONDS=12.5,
        OPENAI_MAX_RETRIES=1,
    )


def build_news_item(
    position: int,
    news_id: int,
) -> ImageGenerationNewsItem:
    """Создаёт минимальную новость генератора."""

    return ImageGenerationNewsItem(
        position=position,
        news_id=news_id,
        title=(
            f"Тестовая киноновость {position}"
        ),
        summary=(
            f"Тестовое описание новости {position}."
        ),
    )


def test_factory_arguments() -> None:
    """Проверяет сборку Image runtime."""

    settings = build_settings(
        api_key="sk-local-image-factory-test"
    )

    factory = RecordingClientFactory()

    runtime = (
        create_openai_image_generation_runtime(
            settings,
            client_factory=factory,
        )
    )

    assert len(factory.calls) == 1

    call = factory.calls[0]

    assert call["api_key"] == (
        "sk-local-image-factory-test"
    )

    assert call["timeout"] == 12.5
    assert call["max_retries"] == 1

    assert runtime.sdk_client is (
        factory.client
    )

    assert (
        len(
            factory.client.images.calls
        )
        == 0
    )

    metadata = runtime.generator.metadata

    assert metadata.model_name == (
        "test-image-model"
    )

    request = runtime.generator.build_request(
        items=(
            build_news_item(1, 101),
            build_news_item(2, 102),
            build_news_item(3, 103),
        )
    )

    assert request.size == "1024x1536"
    assert request.quality == "medium"
    assert request.output_format == "png"
    assert request.background == "opaque"
    assert request.moderation == "auto"
    assert request.n == 1

    print("Image factory arguments: OK")
    print(
        "client_factory_calls="
        f"{len(factory.calls)}"
    )
    print(
        f"timeout_seconds={call['timeout']}"
    )
    print(
        f"max_retries={call['max_retries']}"
    )
    print("images_generate_calls=0")


async def test_image_adapter_request() -> None:
    """Проверяет images.generate без сети."""

    settings = build_settings(
        api_key="sk-local-image-adapter-test"
    )

    factory = RecordingClientFactory()

    runtime = (
        create_openai_image_generation_runtime(
            settings,
            client_factory=factory,
        )
    )

    request = ImageModelRequest(
        model="test-image-model",
        prompt="Синтетический image prompt.",
        size="1024x1536",
        quality="medium",
        output_format="png",
        background="opaque",
        moderation="auto",
        n=1,
    )

    response = (
        await runtime.images_client.create_image(
            request
        )
    )

    assert response.image_bytes == (
        SYNTHETIC_IMAGE_BYTES
    )

    assert response.created == 1_800_000_000
    assert response.output_format == "png"
    assert response.quality == "medium"
    assert response.size == "1024x1536"
    assert response.background == "opaque"
    assert response.revised_prompt is None

    usage = response.usage

    if usage is None:
        raise AssertionError(
            "Synthetic usage не извлечён."
        )

    assert usage.input_tokens == 120
    assert usage.input_text_tokens == 120
    assert usage.input_image_tokens == 0
    assert usage.output_tokens == 900
    assert usage.output_text_tokens == 0
    assert usage.output_image_tokens == 900
    assert usage.total_tokens == 1020

    assert (
        len(factory.client.images.calls)
        == 1
    )

    call = factory.client.images.calls[0]

    expected_call = {
        "model": "test-image-model",
        "prompt": (
            "Синтетический image prompt."
        ),
        "n": 1,
        "size": "1024x1536",
        "quality": "medium",
        "output_format": "png",
        "background": "opaque",
        "moderation": "auto",
    }

    assert call == expected_call
    assert "response_format" not in call
    assert "stream" not in call

    print()
    print("Image API adapter request: OK")
    print("images_generate_calls=1")
    print("response_format_sent=false")
    print("stream_sent=false")
    print("base64_decoded=true")
    print("usage_extracted=true")


async def test_optional_usage() -> None:
    """Проверяет допустимое отсутствие usage."""

    client = FakeAsyncOpenAIClient(
        include_usage=False
    )

    adapter = OpenAIImagesGenerationClient(
        client=client
    )

    response = await adapter.create_image(
        ImageModelRequest(
            model="test-image-model",
            prompt="No usage response.",
            size="1024x1536",
            quality="medium",
            output_format="png",
            background="opaque",
            moderation="auto",
            n=1,
        )
    )

    assert response.usage is None

    print()
    print("Optional Image usage: OK")
    print("usage_none_accepted=true")


def test_missing_api_key() -> None:
    """Проверяет обязательность API-ключа."""

    settings = build_settings(
        api_key=None
    )

    factory = RecordingClientFactory()

    try:
        create_openai_image_generation_runtime(
            settings,
            client_factory=factory,
        )
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(
            error
        )
        assert len(factory.calls) == 0

        print()
        print("Missing API key blocking: OK")
        print("client_factory_calls=0")
        return

    raise AssertionError(
        "Отсутствующий OPENAI_API_KEY "
        "не был заблокирован."
    )


async def test_real_sdk_construction() -> None:
    """
    Создаёт настоящий AsyncOpenAI без API-запроса.
    """

    settings = build_settings(
        api_key="sk-local-image-construction-test"
    )

    runtime = (
        create_openai_image_generation_runtime(
            settings
        )
    )

    assert isinstance(
        runtime.sdk_client,
        AsyncOpenAI,
    )

    assert (
        runtime.generator.metadata.model_name
        == "test-image-model"
    )

    close_method = getattr(
        runtime.sdk_client,
        "close",
        None,
    )

    if close_method is None:
        raise AssertionError(
            "AsyncOpenAI не содержит close()."
        )

    await close_method()

    print()
    print("Real AsyncOpenAI Image construction: OK")
    print(
        "sdk_client_type="
        f"{type(runtime.sdk_client).__name__}"
    )
    print("client_closed=true")


async def main() -> int:
    """Запускает тест Image API runtime."""

    test_factory_arguments()
    await test_image_adapter_request()
    await test_optional_usage()
    test_missing_api_key()
    await test_real_sdk_construction()

    print()
    print("Real API key used: no")
    print("OpenAI Image requests: not performed")
    print("Database changes: not performed")
    print("PNG files created: 0")
    print("Telegram publication: not performed")
    print("OpenAI Image client factory test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )