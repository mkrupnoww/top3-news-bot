from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Literal

from app.generation.image_generator import (
    ImageGenerationNewsItem,
    ImageModelRequest,
    OpenAIImageGeneratorMetadata,
)


IMAGE_REQUEST_KEY_VERSION = (
    "image_request_key_v1"
)

IMAGE_REQUEST_KEY_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

_IMAGE_SIZE_PATTERN = re.compile(
    r"^(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)$"
)


ImageRequestKind = Literal[
    "initial",
    "regenerate",
]


@dataclass(frozen=True, slots=True)
class ImageRequestKey:
    """Детерминированный ключ генерации изображения."""

    value: str
    version: str
    canonical_json: str

    def __post_init__(self) -> None:
        """Проверяет формат ключа."""

        if not (
            IMAGE_REQUEST_KEY_PATTERN
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

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть строкой."
        )

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
    """Проверяет положительное целое число."""

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


def _normalize_request_kind(
    value: str,
) -> ImageRequestKind:
    """Проверяет тип image-generation."""

    if value not in {
        "initial",
        "regenerate",
    }:
        raise ValueError(
            "request_kind должен быть "
            "'initial' или 'regenerate'."
        )

    return value


def _normalize_review_action_id(
    *,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
) -> int | None:
    """Проверяет связь с ручной перегенерацией."""

    if request_kind == "initial":
        if review_action_id is not None:
            raise ValueError(
                "Для request_kind='initial' "
                "review_action_id должен быть None."
            )

        return None

    if review_action_id is None:
        raise ValueError(
            "Для request_kind='regenerate' "
            "требуется review_action_id."
        )

    return _normalize_positive_integer(
        review_action_id,
        field_name="review_action_id",
    )


def _normalize_image_size(
    value: str,
) -> str:
    """Проверяет размер и соотношение сторон 2:3."""

    normalized_value = _normalize_required_text(
        value,
        field_name="model_request.size",
    )

    match = _IMAGE_SIZE_PATTERN.fullmatch(
        normalized_value
    )

    if match is None:
        raise ValueError(
            "model_request.size должен иметь "
            "формат WIDTHxHEIGHT."
        )

    width = int(match.group("width"))
    height = int(match.group("height"))

    if width * 3 != height * 2:
        raise ValueError(
            "Для итоговой иллюстрации требуется "
            "соотношение сторон 2:3."
        )

    return normalized_value


def _normalize_news_items(
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> tuple[int, int, int]:
    """Проверяет состав и порядок TOP-3."""

    if not isinstance(items, tuple):
        raise TypeError(
            "items должен быть tuple."
        )

    if len(items) != 3:
        raise ValueError(
            "Для image request key требуется "
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

    return (
        int(news_ids[0]),
        int(news_ids[1]),
        int(news_ids[2]),
    )


def _build_model_request_payload(
    model_request: ImageModelRequest,
) -> dict[str, Any]:
    """Проверяет и сериализует точный Image API request."""

    normalized_model = (
        _normalize_required_text(
            model_request.model,
            field_name="model_request.model",
        )
    )

    normalized_prompt = (
        _normalize_required_text(
            model_request.prompt,
            field_name="model_request.prompt",
        )
    )

    normalized_size = _normalize_image_size(
        model_request.size
    )

    if model_request.quality not in {
        "low",
        "medium",
        "high",
    }:
        raise ValueError(
            "model_request.quality должен быть "
            "low, medium или high."
        )

    if model_request.output_format != "png":
        raise ValueError(
            "model_request.output_format "
            "должен быть 'png'."
        )

    if model_request.background != "opaque":
        raise ValueError(
            "model_request.background "
            "должен быть 'opaque'."
        )

    if model_request.moderation != "auto":
        raise ValueError(
            "model_request.moderation "
            "должен быть 'auto'."
        )

    normalized_count = (
        _normalize_positive_integer(
            model_request.n,
            field_name="model_request.n",
        )
    )

    if normalized_count != 1:
        raise ValueError(
            "Для текущего image pipeline "
            "model_request.n должен быть равен 1."
        )

    return {
        "model": normalized_model,
        "prompt": normalized_prompt,
        "size": normalized_size,
        "quality": model_request.quality,
        "output_format": (
            model_request.output_format
        ),
        "background": (
            model_request.background
        ),
        "moderation": (
            model_request.moderation
        ),
        "n": normalized_count,
    }


def _build_payload(
    *,
    batch_id: int,
    ranking_run_id: int,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
    metadata: OpenAIImageGeneratorMetadata,
    model_request: ImageModelRequest,
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> dict[str, Any]:
    """Формирует каноническое содержимое ключа."""

    normalized_batch_id = (
        _normalize_positive_integer(
            batch_id,
            field_name="batch_id",
        )
    )

    normalized_ranking_run_id = (
        _normalize_positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
    )

    normalized_request_kind = (
        _normalize_request_kind(
            request_kind
        )
    )

    normalized_review_action_id = (
        _normalize_review_action_id(
            request_kind=(
                normalized_request_kind
            ),
            review_action_id=review_action_id,
        )
    )

    news_ids = _normalize_news_items(
        items
    )

    model_request_payload = (
        _build_model_request_payload(
            model_request
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
        != model_request_payload["model"]
    ):
        raise ValueError(
            "Модель в metadata не совпадает "
            "с model_request.model."
        )

    return {
        "image_request_key_version": (
            IMAGE_REQUEST_KEY_VERSION
        ),
        "batch_id": normalized_batch_id,
        "ranking_run_id": (
            normalized_ranking_run_id
        ),
        "request_kind": (
            normalized_request_kind
        ),
        "review_action_id": (
            normalized_review_action_id
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
        },
        "top3_news_ids": list(
            news_ids
        ),
        "model_request": (
            model_request_payload
        ),
    }


def create_image_request_key(
    *,
    batch_id: int,
    ranking_run_id: int,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
    metadata: OpenAIImageGeneratorMetadata,
    model_request: ImageModelRequest,
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> ImageRequestKey:
    """Создаёт SHA-256 ключ image-generation."""

    payload = _build_payload(
        batch_id=batch_id,
        ranking_run_id=ranking_run_id,
        request_kind=request_kind,
        review_action_id=review_action_id,
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

    return ImageRequestKey(
        value=request_key,
        version=IMAGE_REQUEST_KEY_VERSION,
        canonical_json=canonical_json,
    )