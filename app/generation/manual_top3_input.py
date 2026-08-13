from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
)

TextFormat = Literal[
    "markdown",
    "markdown_v2",
    "html",
    "plain_text",
]


_SOURCE_CODE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_-]{1,63}$"
)


def _validate_http_url(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет абсолютный HTTP или HTTPS URL."""

    normalized_value = value.strip()
    parsed = urlsplit(normalized_value)

    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
    ):
        raise ValueError(
            f"{field_name} должен быть абсолютным "
            "HTTP или HTTPS URL."
        )

    return normalized_value


class ManualTop3NewsItem(BaseModel):
    """Одна вручную выбранная новость выпуска TOP-3."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    position: int = Field(
        ge=1,
        le=3,
    )

    source_code: str = Field(
        min_length=2,
        max_length=64,
    )

    source_name: str = Field(
        min_length=1,
        max_length=250,
    )

    source_base_url: str | None = None

    source_url: str = Field(
        min_length=1,
        max_length=2000,
    )

    title: str = Field(
        min_length=1,
        max_length=1000,
    )

    summary: str = Field(
        min_length=1,
    )

    source_published_at: datetime

    primary_image_url: str | None = None

    image_credit: str | None = None

    selection_reason: str = Field(
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("source_code")
    @classmethod
    def validate_source_code(
        cls,
        value: str,
    ) -> str:
        """Проверяет технический код источника."""

        normalized_value = value.strip().lower()

        if not _SOURCE_CODE_PATTERN.fullmatch(
            normalized_value
        ):
            raise ValueError(
                "source_code должен содержать только "
                "латинские строчные буквы, цифры, "
                "дефис и нижнее подчёркивание."
            )

        return normalized_value

    @field_validator("source_url")
    @classmethod
    def validate_source_url(
        cls,
        value: str,
    ) -> str:
        """Проверяет ссылку на исходную публикацию."""

        return _validate_http_url(
            value,
            field_name="source_url",
        )

    @field_validator("source_base_url")
    @classmethod
    def validate_source_base_url(
        cls,
        value: str | None,
    ) -> str | None:
        """Проверяет базовый адрес источника."""

        if value is None:
            return None

        return _validate_http_url(
            value,
            field_name="source_base_url",
        )

    @field_validator("primary_image_url")
    @classmethod
    def validate_primary_image_url(
        cls,
        value: str | None,
    ) -> str | None:
        """Проверяет ссылку на основное изображение."""

        if value is None:
            return None

        return _validate_http_url(
            value,
            field_name="primary_image_url",
        )

    @field_validator("source_published_at")
    @classmethod
    def validate_source_published_at(
        cls,
        value: datetime,
    ) -> datetime:
        """Требует дату публикации с часовым поясом."""

        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "source_published_at должен содержать "
                "часовой пояс, например суффикс Z."
            )

        return value.astimezone(timezone.utc)


class ManualTop3Input(BaseModel):
    """Валидированный вход одного выпуска TOP-3."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    publication_date: date

    text_format: TextFormat = "markdown"

    post_text: str = Field(
        min_length=1,
        max_length=MAXIMUM_POST_LENGTH,
    )

    items: list[ManualTop3NewsItem] = Field(
        min_length=3,
        max_length=3,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("post_text")
    @classmethod
    def normalize_post_text(
        cls,
        value: str,
    ) -> str:
        """Удаляет внешние пустые строки из поста."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "post_text не может быть пустым."
            )

        return normalized_value

    @model_validator(mode="after")
    def validate_top3_integrity(
        self,
    ) -> "ManualTop3Input":
        """Проверяет целостность тройки новостей."""

        positions = {
            item.position
            for item in self.items
        }

        if positions != {1, 2, 3}:
            raise ValueError(
                "items должны содержать ровно "
                "позиции 1, 2 и 3."
            )

        normalized_urls = [
            item.source_url.rstrip("/")
            for item in self.items
        ]

        if len(set(normalized_urls)) != 3:
            raise ValueError(
                "Все три source_url должны быть уникальными."
            )

        return self


def load_manual_top3_input(
    file_path: Path,
) -> ManualTop3Input:
    """Загружает и валидирует JSON-файл выпуска."""

    try:
        raw_text = file_path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Файл не найден: {file_path}"
        ) from error

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Некорректный JSON в файле "
            f"{file_path}: строка {error.lineno}, "
            f"столбец {error.colno}: {error.msg}"
        ) from error

    return ManualTop3Input.model_validate(payload)