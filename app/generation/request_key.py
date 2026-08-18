from dataclasses import dataclass
from datetime import (
    date,
    datetime,
    timezone,
)
from hashlib import sha256
import json
import re
from typing import Any

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationNewsItem,
    OpenAIPostGeneratorMetadata,
)


GENERATION_REQUEST_KEY_VERSION = (
    "generation_request_key_v2"
)

GENERATION_REQUEST_KEY_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


@dataclass(frozen=True, slots=True)
class GenerationRequestKey:
    """Детерминированный ключ генерации поста."""

    value: str
    version: str
    canonical_json: str

    def __post_init__(self) -> None:
        """Проверяет формат ключа."""

        if not (
            GENERATION_REQUEST_KEY_PATTERN
            .fullmatch(self.value)
        ):
            raise ValueError(
                "value должен быть SHA-256 "
                "в нижнем регистре."
            )

        if not self.version.strip():
            raise ValueError(
                "version не может быть пустой."
            )

        if not self.canonical_json.strip():
            raise ValueError(
                "canonical_json "
                "не может быть пустым."
            )


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный текст."""

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительный идентификатор."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} должен быть "
            "больше нуля."
        )

    return value


def _normalize_publication_date(
    value: date,
) -> date:
    """Проверяет календарную дату выпуска."""

    if isinstance(value, datetime):
        raise TypeError(
            "publication_date должен быть date, "
            "а не datetime."
        )

    if not isinstance(value, date):
        raise TypeError(
            "publication_date должен быть date."
        )

    return value


def _normalize_telegram_chat_id(
    value: int,
) -> int:
    """Проверяет полный ID Telegram-канала."""

    if isinstance(value, bool):
        raise TypeError(
            "telegram_chat_id не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            "telegram_chat_id должен быть int."
        )

    if not str(value).startswith("-100"):
        raise ValueError(
            "telegram_chat_id должен начинаться "
            "с -100."
        )

    return value


def _normalize_news_items(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> tuple[int, ...]:
    """Проверяет состав и порядок TOP-3."""

    if len(items) != 3:
        raise ValueError(
            "Для ключа генерации требуется "
            "ровно три новости."
        )

    positions = tuple(
        item.position
        for item in items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Новости должны идти строго "
            "в порядке позиций 1, 2 и 3."
        )

    news_ids = tuple(
        _normalize_positive_integer(
            item.news_id,
            field_name=(
                f"news_id position={item.position}"
            ),
        )
        for item in items
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Все три news_id должны быть "
            "уникальными."
        )

    for item in items:
        _normalize_required_text(
            item.title,
            field_name=(
                f"title news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.summary,
            field_name=(
                f"summary news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.source_name,
            field_name=(
                "source_name "
                f"news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.source_url,
            field_name=(
                "source_url "
                f"news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.selection_reason,
            field_name=(
                "selection_reason "
                f"news_id={item.news_id}"
            ),
        )

        if item.official_trailer_url is not None:
            _normalize_required_text(
                item.official_trailer_url,
                field_name=(
                    "official_trailer_url "
                    f"news_id={item.news_id}"
                ),
            )

        published_at = (
            item.source_published_at
        )

        if (
            published_at.tzinfo is None
            or published_at.utcoffset() is None
        ):
            raise ValueError(
                "source_published_at должен "
                "содержать часовой пояс: "
                f"news_id={item.news_id}"
            )

        if (
            not item.individual_score.is_finite()
            or item.individual_score < 0
        ):
            raise ValueError(
                "individual_score должен быть "
                "конечным неотрицательным числом: "
                f"news_id={item.news_id}"
            )

    return news_ids


def _build_top3_payload(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> list[dict[str, Any]]:
    """Формирует прозрачное описание TOP-3."""

    return [
        {
            "position": item.position,
            "news_id": item.news_id,
            "title": item.title.strip(),
            "summary": item.summary.strip(),
            "source_name": (
                item.source_name.strip()
            ),
            "source_url": (
                item.source_url.strip()
            ),
            "source_published_at": (
                item
                .source_published_at
                .astimezone(timezone.utc)
                .isoformat()
            ),
            "individual_score": str(
                item.individual_score
            ),
            "selection_reason": (
                item.selection_reason.strip()
            ),
            **(
                {
                    "official_trailer_url": (
                        item
                        .official_trailer_url
                        .strip()
                    )
                }
                if (
                    item.official_trailer_url
                    is not None
                )
                else {}
            ),
        }
        for item in items
    ]


def _build_payload(
    *,
    ranking_run_id: int,
    publication_date: date,
    telegram_chat_id: int,
    metadata: OpenAIPostGeneratorMetadata,
    model_request: GenerationModelRequest,
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> dict[str, Any]:
    """Формирует каноническое содержимое ключа."""

    normalized_ranking_run_id = (
        _normalize_positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
    )

    normalized_publication_date = (
        _normalize_publication_date(
            publication_date
        )
    )

    normalized_telegram_chat_id = (
        _normalize_telegram_chat_id(
            telegram_chat_id
        )
    )

    news_ids = _normalize_news_items(
        items
    )

    normalized_model = (
        _normalize_required_text(
            model_request.model,
            field_name="model_request.model",
        )
    )

    normalized_instructions = (
        _normalize_required_text(
            model_request.instructions,
            field_name=(
                "model_request.instructions"
            ),
        )
    )

    normalized_input_text = (
        _normalize_required_text(
            model_request.input_text,
            field_name=(
                "model_request.input_text"
            ),
        )
    )

    normalized_metadata_model = (
        _normalize_required_text(
            metadata.model_name,
            field_name="metadata.model_name",
        )
    )

    if (
        normalized_metadata_model
        != normalized_model
    ):
        raise ValueError(
            "Модель в metadata не совпадает "
            "с model_request.model."
        )

    return {
        "generation_request_key_version": (
            GENERATION_REQUEST_KEY_VERSION
        ),
        "ranking_run_id": (
            normalized_ranking_run_id
        ),
        "publication_date": (
            normalized_publication_date.isoformat()
        ),
        "telegram_chat_id": (
            normalized_telegram_chat_id
        ),
        "generator": {
            "generator_name": (
                _normalize_required_text(
                    metadata.generator_name,
                    field_name=(
                        "metadata.generator_name"
                    ),
                )
            ),
            "generator_version": (
                _normalize_required_text(
                    metadata.generator_version,
                    field_name=(
                        "metadata.generator_version"
                    ),
                )
            ),
            "prompt_version": (
                _normalize_required_text(
                    metadata.prompt_version,
                    field_name=(
                        "metadata.prompt_version"
                    ),
                )
            ),
            "model_name": (
                normalized_metadata_model
            ),
            "text_format": (
                _normalize_required_text(
                    metadata.text_format,
                    field_name=(
                        "metadata.text_format"
                    ),
                )
            ),
        },
        "top3_news_ids": list(
            news_ids
        ),
        "top3": _build_top3_payload(
            items
        ),
        "model_request": {
            "model": normalized_model,
            "instructions": (
                normalized_instructions
            ),
            "input_text": (
                normalized_input_text
            ),
        },
    }


def create_generation_request_key(
    *,
    ranking_run_id: int,
    publication_date: date,
    telegram_chat_id: int,
    metadata: OpenAIPostGeneratorMetadata,
    model_request: GenerationModelRequest,
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> GenerationRequestKey:
    """Создаёт SHA-256 ключ генерации."""

    payload = _build_payload(
        ranking_run_id=ranking_run_id,
        publication_date=publication_date,
        telegram_chat_id=telegram_chat_id,
        metadata=metadata,
        model_request=model_request,
        items=items,
    )

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    request_key = sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    return GenerationRequestKey(
        value=request_key,
        version=(
            GENERATION_REQUEST_KEY_VERSION
        ),
        canonical_json=canonical_json,
    )
