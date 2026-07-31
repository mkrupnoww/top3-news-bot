import asyncio
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.ranking.openai_factory import (
    create_openai_ranking_runtime,
)


class FakeResponsesResource:
    """Ресурс, запрещающий сетевые обращения."""

    def __init__(self) -> None:
        self.call_count = 0

    async def create(
        self,
        **kwargs: Any,
    ) -> Any:
        """Падает при неожиданном вызове API."""

        self.call_count += 1

        raise AssertionError(
            "responses.create() не должен "
            "вызываться при создании фабрики."
        )


class FakeAsyncOpenAIClient:
    """Поддельный клиент для проверки фабрики."""

    def __init__(self) -> None:
        self.responses = FakeResponsesResource()


class RecordingClientFactory:
    """Запоминает параметры создания клиента."""

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
        """Возвращает клиент без сети."""

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
    """Создаёт изолированные тестовые настройки."""

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
            "test-model-no-network"
        ),
        OPENAI_TIMEOUT_SECONDS=12.5,
        OPENAI_MAX_RETRIES=1,
    )


def test_factory_arguments() -> None:
    """Проверяет передачу настроек в SDK."""

    settings = build_settings(
        api_key="sk-local-factory-test"
    )

    factory = RecordingClientFactory()

    runtime = (
        create_openai_ranking_runtime(
            settings,
            client_factory=factory,
        )
    )

    assert len(factory.calls) == 1

    call = factory.calls[0]

    assert call["api_key"] == (
        "sk-local-factory-test"
    )

    assert call["timeout"] == 12.5
    assert call["max_retries"] == 1

    assert runtime.sdk_client is (
        factory.client
    )

    assert (
        factory.client
        .responses
        .call_count
        == 0
    )

    metadata = runtime.evaluator.metadata

    assert metadata.run_mode == (
        "openai_ranking"
    )

    assert metadata.evaluator_name == (
        "OpenAIRankingEvaluator"
    )

    assert metadata.model_name == (
        "test-model-no-network"
    )

    print("Factory arguments: OK")
    print(
        f"client_factory_calls="
        f"{len(factory.calls)}"
    )
    print(
        f"timeout_seconds="
        f"{call['timeout']}"
    )
    print(
        f"max_retries="
        f"{call['max_retries']}"
    )
    print(
        "responses_create_calls="
        f"{factory.client.responses.call_count}"
    )


def test_missing_api_key() -> None:
    """Проверяет обязательность API-ключа."""

    settings = build_settings(
        api_key=None
    )

    factory = RecordingClientFactory()

    try:
        create_openai_ranking_runtime(
            settings,
            client_factory=factory,
        )
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(
            error
        )

        assert len(factory.calls) == 0

        print()
        print(
            "Missing API key blocking: OK"
        )
        print(
            f"client_factory_calls="
            f"{len(factory.calls)}"
        )
        return

    raise AssertionError(
        "Отсутствующий OPENAI_API_KEY "
        "не был заблокирован."
    )


async def test_real_sdk_construction() -> None:
    """
    Создаёт настоящий AsyncOpenAI.

    API-методы не вызываются.
    """

    settings = build_settings(
        api_key="sk-local-construction-test"
    )

    runtime = (
        create_openai_ranking_runtime(
            settings
        )
    )

    assert isinstance(
        runtime.sdk_client,
        AsyncOpenAI,
    )

    assert runtime.evaluator.metadata.model_name == (
        "test-model-no-network"
    )

    close_method = getattr(
        runtime.sdk_client,
        "close",
        None,
    )

    if close_method is None:
        raise AssertionError(
            "AsyncOpenAI не содержит "
            "метод close()."
        )

    await close_method()

    print()
    print("Real AsyncOpenAI construction: OK")
    print(
        "sdk_client_type="
        f"{type(runtime.sdk_client).__name__}"
    )
    print("client_closed=true")


async def main() -> int:
    """Запускает проверки фабрики."""

    test_factory_arguments()
    test_missing_api_key()
    await test_real_sdk_construction()

    print()
    print(
        "Real API key used: no"
    )
    print(
        "OpenAI requests: not performed"
    )
    print(
        "Database changes: not performed"
    )
    print(
        "Telegram publication: not performed"
    )
    print(
        "OpenAI client factory test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )