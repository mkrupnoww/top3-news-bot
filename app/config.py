from functools import lru_cache
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    Настройки приложения.

    Значения загружаются из переменных окружения
    и локального файла .env.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_ignore_empty=True,
        extra="ignore",
    )

    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = Field(
        default="development",
        validation_alias="APP_ENV",
    )

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )

    telegram_bot_username: str = Field(
        min_length=1,
        validation_alias=(
            "TELEGRAM_BOT_USERNAME"
        ),
    )

    telegram_bot_token: SecretStr = Field(
        validation_alias="TELEGRAM_BOT_TOKEN",
    )

    telegram_channel_id: int = Field(
        validation_alias="TELEGRAM_CHANNEL_ID",
    )

    db_host: str = Field(
        default="127.0.0.1",
        min_length=1,
        validation_alias="DB_HOST",
    )

    db_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        validation_alias="DB_PORT",
    )

    db_name: str = Field(
        min_length=1,
        validation_alias="DB_NAME",
    )

    db_user: str = Field(
        min_length=1,
        validation_alias="DB_USER",
    )

    db_password: SecretStr = Field(
        validation_alias="DB_PASSWORD",
    )

    db_schema: str = Field(
        default="top3_news",
        pattern=r"^[a-z_][a-z0-9_]*$",
        validation_alias="DB_SCHEMA",
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )

    openai_ranking_model: str = Field(
        default="gpt-5.6-terra",
        min_length=1,
        max_length=128,
        validation_alias=(
            "OPENAI_RANKING_MODEL"
        ),
    )

    openai_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        le=300,
        validation_alias=(
            "OPENAI_TIMEOUT_SECONDS"
        ),
    )

    openai_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        validation_alias=(
            "OPENAI_MAX_RETRIES"
        ),
    )

    @field_validator(
        "telegram_channel_id"
    )
    @classmethod
    def validate_telegram_channel_id(
        cls,
        value: int,
    ) -> int:
        """Проверяет полный Bot API ID канала."""

        if not str(value).startswith("-100"):
            raise ValueError(
                "TELEGRAM_CHANNEL_ID "
                "должен начинаться с -100"
            )

        return value

    @field_validator(
        "openai_ranking_model"
    )
    @classmethod
    def validate_openai_ranking_model(
        cls,
        value: str,
    ) -> str:
        """Нормализует название OpenAI-модели."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "OPENAI_RANKING_MODEL "
                "не может быть пустым."
            )

        return normalized_value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Возвращает единый экземпляр настроек.
    """

    return Settings()