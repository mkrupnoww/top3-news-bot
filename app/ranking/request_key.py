from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Any

from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_evaluator import (
    RankingModelRequest,
)


REQUEST_KEY_VERSION = (
    "ranking_request_key_v1"
)

REQUEST_KEY_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


@dataclass(frozen=True, slots=True)
class RankingRequestKey:
    """Детерминированный ключ запуска."""

    value: str
    version: str
    canonical_json: str

    def __post_init__(self) -> None:
        """Проверяет формат SHA-256."""

        if not REQUEST_KEY_PATTERN.fullmatch(
            self.value
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


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Приводит дату к UTC."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} должен содержать "
            "часовой пояс."
        )

    return value.astimezone(
        timezone.utc
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


def _normalize_news_ids(
    news_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Проверяет идентификаторы кандидатов."""

    if not news_ids:
        raise ValueError(
            "news_ids не может быть пустым."
        )

    for news_id in news_ids:
        if isinstance(news_id, bool):
            raise TypeError(
                "news_id не может быть bool."
            )

        if not isinstance(news_id, int):
            raise TypeError(
                "Каждый news_id должен быть int."
            )

        if news_id <= 0:
            raise ValueError(
                "Каждый news_id должен быть "
                "больше нуля."
            )

    if len(set(news_ids)) != len(news_ids):
        raise ValueError(
            "news_ids содержит дубликаты."
        )

    return news_ids


def _build_payload(
    *,
    formula_version: str,
    metadata: RankingEvaluatorMetadata,
    model_request: RankingModelRequest,
    window_started_at: datetime,
    window_finished_at: datetime,
    news_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Формирует каноническое содержимое ключа."""

    normalized_formula_version = (
        _normalize_required_text(
            formula_version,
            field_name="formula_version",
        )
    )

    normalized_start = _normalize_datetime(
        window_started_at,
        field_name="window_started_at",
    )

    normalized_finish = _normalize_datetime(
        window_finished_at,
        field_name="window_finished_at",
    )

    if normalized_finish <= normalized_start:
        raise ValueError(
            "window_finished_at должен быть "
            "позже window_started_at."
        )

    normalized_news_ids = (
        _normalize_news_ids(news_ids)
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

    metadata_model = metadata.model_name

    if metadata_model is None:
        raise ValueError(
            "metadata.model_name обязателен "
            "для OpenAI-запуска."
        )

    normalized_metadata_model = (
        _normalize_required_text(
            metadata_model,
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
        "request_key_version": (
            REQUEST_KEY_VERSION
        ),
        "formula_version": (
            normalized_formula_version
        ),
        "window": {
            "started_at": (
                normalized_start.isoformat()
            ),
            "finished_at": (
                normalized_finish.isoformat()
            ),
        },
        "evaluator": {
            "run_mode": (
                _normalize_required_text(
                    metadata.run_mode,
                    field_name=(
                        "metadata.run_mode"
                    ),
                )
            ),
            "evaluator_name": (
                _normalize_required_text(
                    metadata.evaluator_name,
                    field_name=(
                        "metadata.evaluator_name"
                    ),
                )
            ),
            "evaluator_version": (
                _normalize_required_text(
                    metadata.evaluator_version,
                    field_name=(
                        "metadata.evaluator_version"
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
        "candidate_news_ids": list(
            normalized_news_ids
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


def create_ranking_request_key(
    *,
    formula_version: str,
    metadata: RankingEvaluatorMetadata,
    model_request: RankingModelRequest,
    window_started_at: datetime,
    window_finished_at: datetime,
    news_ids: tuple[int, ...],
) -> RankingRequestKey:
    """Создаёт SHA-256 ключ запуска."""

    payload = _build_payload(
        formula_version=formula_version,
        metadata=metadata,
        model_request=model_request,
        window_started_at=window_started_at,
        window_finished_at=(
            window_finished_at
        ),
        news_ids=news_ids,
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

    return RankingRequestKey(
        value=request_key,
        version=REQUEST_KEY_VERSION,
        canonical_json=canonical_json,
    )