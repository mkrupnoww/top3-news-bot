from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.event_evaluator import (
    EventRankingModelRequest,
)
from app.ranking.event_formula_pipeline import (
    EventAudienceMetrics,
)
from app.ranking.request_key import (
    RankingRequestKey,
)


EVENT_REQUEST_KEY_VERSION = (
    "event_ranking_request_key_v1"
)


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный текст."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть str."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Приводит дату с часовым поясом к UTC."""

    if not isinstance(value, datetime):
        raise TypeError(
            f"{field_name} должен быть datetime."
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} должен содержать "
            "часовой пояс."
        )

    return value.astimezone(timezone.utc)


def _normalize_news_ids(
    news_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Проверяет порядок кандидатов."""

    if not isinstance(news_ids, tuple):
        raise TypeError(
            "news_ids должен быть tuple."
        )

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


def _build_audience_payload(
    *,
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ],
    news_ids: tuple[int, ...],
) -> list[dict[str, int | None]]:
    """Формирует канонический снимок метрик."""

    if not isinstance(audience_metrics, tuple):
        raise TypeError(
            "audience_metrics должен быть tuple."
        )

    candidate_id_set = set(news_ids)
    metric_news_ids: set[int] = set()
    payload: list[dict[str, int | None]] = []

    for item in audience_metrics:
        if not isinstance(
            item,
            EventAudienceMetrics,
        ):
            raise TypeError(
                "Каждый элемент audience_metrics "
                "должен быть EventAudienceMetrics."
            )

        if item.news_id not in candidate_id_set:
            raise ValueError(
                "Audience-метрика относится "
                "к новости вне текущей выборки: "
                f"news_id={item.news_id}"
            )

        if item.news_id in metric_news_ids:
            raise ValueError(
                "Audience-метрики содержат "
                "повторяющийся news_id: "
                f"{item.news_id}"
            )

        metric_news_ids.add(item.news_id)

        payload.append(
            {
                "news_id": item.news_id,
                "view_count": item.view_count,
                "comment_count": (
                    item.comment_count
                ),
                "share_count": item.share_count,
            }
        )

    payload.sort(
        key=lambda item: int(item["news_id"])
    )

    return payload


def _build_payload(
    *,
    formula_version: str,
    metadata: RankingEvaluatorMetadata,
    model_request: EventRankingModelRequest,
    window_started_at: datetime,
    window_finished_at: datetime,
    news_ids: tuple[int, ...],
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ],
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
            "metadata.model_name обязателен."
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
            EVENT_REQUEST_KEY_VERSION
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
        "audience_metrics": (
            _build_audience_payload(
                audience_metrics=audience_metrics,
                news_ids=normalized_news_ids,
            )
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


def create_event_ranking_request_key(
    *,
    formula_version: str,
    metadata: RankingEvaluatorMetadata,
    model_request: EventRankingModelRequest,
    window_started_at: datetime,
    window_finished_at: datetime,
    news_ids: tuple[int, ...],
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ] = (),
) -> RankingRequestKey:
    """Создаёт SHA-256 ключ event-level запуска."""

    payload = _build_payload(
        formula_version=formula_version,
        metadata=metadata,
        model_request=model_request,
        window_started_at=window_started_at,
        window_finished_at=(
            window_finished_at
        ),
        news_ids=news_ids,
        audience_metrics=audience_metrics,
    )

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    value = sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    return RankingRequestKey(
        value=value,
        version=EVENT_REQUEST_KEY_VERSION,
        canonical_json=canonical_json,
    )