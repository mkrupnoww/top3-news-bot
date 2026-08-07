from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationNewsItem,
    OPENAI_POST_REVISION_PROMPT_VERSION,
    OpenAIPostGeneratorMetadata,
)


GENERATION_REVISION_REQUEST_KEY_VERSION = (
    "generation_revision_request_key_v1"
)

GENERATION_REVISION_REQUEST_KEY_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

REVISION_REQUESTED_ACTION = "regenerate_text"


@dataclass(frozen=True, slots=True)
class GenerationRevisionRequestKey:
    """Детерминированный ключ редакционной ревизии."""

    value: str
    version: str
    canonical_json: str

    def __post_init__(self) -> None:
        """Проверяет формат ключа."""

        if not (
            GENERATION_REVISION_REQUEST_KEY_PATTERN
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


def _normalize_target_version_number(
    value: int,
) -> int:
    """Проверяет номер создаваемой версии."""

    normalized_value = (
        _normalize_positive_integer(
            value,
            field_name="target_version_number",
        )
    )

    if normalized_value <= 1:
        raise ValueError(
            "target_version_number должен "
            "быть больше 1."
        )

    return normalized_value


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


def _normalize_issues(
    issues: tuple[str, ...],
) -> tuple[str, ...]:
    """Проверяет редакционные замечания."""

    if not isinstance(issues, tuple):
        raise TypeError(
            "issues должен быть tuple."
        )

    if not issues:
        raise ValueError(
            "issues не может быть пустым."
        )

    normalized_issues: list[str] = []

    for index, issue in enumerate(
        issues,
        start=1,
    ):
        normalized_issues.append(
            _normalize_required_text(
                issue,
                field_name=(
                    f"issues[{index}]"
                ),
            )
        )

    return tuple(normalized_issues)


def _normalize_news_items(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> tuple[int, int, int]:
    """Проверяет состав и порядок TOP-3."""

    if len(items) != 3:
        raise ValueError(
            "Для revision request key требуется "
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


def _build_top3_payload(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> list[dict[str, Any]]:
    """
    Формирует фактический TOP-3.

    Для revision-ключа сохраняются только поля,
    которые разрешены модели как источник фактов.
    """

    return [
        {
            "position": item.position,
            "news_id": item.news_id,
            "title": item.title.strip(),
            "summary": item.summary.strip(),
        }
        for item in items
    ]


def _build_payload(
    *,
    batch_id: int,
    source_generated_post_id: int,
    review_action_id: int,
    target_version_number: int,
    source_post_text: str,
    editorial_comment: str,
    issues: tuple[str, ...],
    metadata: OpenAIPostGeneratorMetadata,
    model_request: GenerationModelRequest,
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
    revision_prompt_version: str,
) -> dict[str, Any]:
    """Формирует каноническое содержимое ключа."""

    normalized_batch_id = (
        _normalize_positive_integer(
            batch_id,
            field_name="batch_id",
        )
    )

    normalized_source_generated_post_id = (
        _normalize_positive_integer(
            source_generated_post_id,
            field_name=(
                "source_generated_post_id"
            ),
        )
    )

    normalized_review_action_id = (
        _normalize_positive_integer(
            review_action_id,
            field_name="review_action_id",
        )
    )

    normalized_target_version_number = (
        _normalize_target_version_number(
            target_version_number
        )
    )

    normalized_source_post_text = (
        _normalize_required_text(
            source_post_text,
            field_name="source_post_text",
        )
    )

    normalized_editorial_comment = (
        _normalize_required_text(
            editorial_comment,
            field_name="editorial_comment",
        )
    )

    normalized_issues = _normalize_issues(
        issues
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

    normalized_revision_prompt_version = (
        _normalize_required_text(
            revision_prompt_version,
            field_name=(
                "revision_prompt_version"
            ),
        )
    )

    return {
        "generation_revision_request_key_version": (
            GENERATION_REVISION_REQUEST_KEY_VERSION
        ),
        "batch_id": normalized_batch_id,
        "source_generated_post_id": (
            normalized_source_generated_post_id
        ),
        "review_action_id": (
            normalized_review_action_id
        ),
        "target_version_number": (
            normalized_target_version_number
        ),
        "requested_action": (
            REVISION_REQUESTED_ACTION
        ),
        "revision": {
            "source_post_text": (
                normalized_source_post_text
            ),
            "editorial_comment": (
                normalized_editorial_comment
            ),
            "issues": list(
                normalized_issues
            ),
        },
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
            "base_prompt_version": (
                _normalize_required_text(
                    metadata.prompt_version,
                    field_name=(
                        "metadata.prompt_version"
                    ),
                )
            ),
            "revision_prompt_version": (
                normalized_revision_prompt_version
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


def create_generation_revision_request_key(
    *,
    batch_id: int,
    source_generated_post_id: int,
    review_action_id: int,
    target_version_number: int,
    source_post_text: str,
    editorial_comment: str,
    issues: tuple[str, ...],
    metadata: OpenAIPostGeneratorMetadata,
    model_request: GenerationModelRequest,
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
    revision_prompt_version: str = (
        OPENAI_POST_REVISION_PROMPT_VERSION
    ),
) -> GenerationRevisionRequestKey:
    """Создаёт SHA-256 ключ редакционной ревизии."""

    payload = _build_payload(
        batch_id=batch_id,
        source_generated_post_id=(
            source_generated_post_id
        ),
        review_action_id=review_action_id,
        target_version_number=(
            target_version_number
        ),
        source_post_text=source_post_text,
        editorial_comment=editorial_comment,
        issues=issues,
        metadata=metadata,
        model_request=model_request,
        items=items,
        revision_prompt_version=(
            revision_prompt_version
        ),
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

    return GenerationRevisionRequestKey(
        value=request_key,
        version=(
            GENERATION_REVISION_REQUEST_KEY_VERSION
        ),
        canonical_json=canonical_json,
    )