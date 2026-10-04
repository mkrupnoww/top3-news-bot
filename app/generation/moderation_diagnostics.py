"""Сохраняет только фактически возвращённые API поля отказа."""

from collections.abc import Mapping
from typing import Any


def extract_image_error_diagnostics(error: Exception) -> dict[str, Any]:
    body = getattr(error, "body", None)
    body = body if isinstance(body, Mapping) else {}
    nested = body.get("error")
    detail = nested if isinstance(nested, Mapping) else body
    moderation = detail.get("moderation_details") or body.get("moderation_details")
    moderation = moderation if isinstance(moderation, Mapping) else {}
    code = getattr(error, "code", None) or detail.get("code")
    stage = moderation.get("moderation_stage")
    categories = moderation.get("categories")
    return {
        "error_code": code if isinstance(code, str) else None,
        "request_id": getattr(error, "request_id", None),
        "moderation_stage": stage if isinstance(stage, str) and stage in {"input", "output", "unknown"} else None,
        "moderation_categories": categories if isinstance(categories, (list, dict)) else None,
    }


def is_moderation_failure(record: Mapping[str, Any]) -> bool:
    """Доказанный отказ; неизвестные transport errors не считаются moderation."""

    metadata = record.get("response_metadata") or {}
    if isinstance(metadata, str):
        import json
        metadata = json.loads(metadata)
    diagnostic = metadata.get("openai_error", {})
    return (
        record["image_status"] == "failed"
        and record["error_type"] == "BadRequestError"
        and (
            diagnostic.get("error_code") == "moderation_blocked"
            or "moderation_blocked" in str(record["error_message"] or "").lower()
        )
    )
