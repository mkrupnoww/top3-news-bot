from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import asyncpg

from app.generation.image_generator import (
    OpenAIImageUsage,
)
from app.generation.image_request_key import (
    IMAGE_REQUEST_KEY_PATTERN,
)


IMAGE_GENERATION_COMPLETION_VERSION = (
    "image_generation_completion_v1"
)

IMAGE_GENERATION_FAILURE_VERSION = (
    "image_generation_failure_v1"
)

_IMAGE_SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


@dataclass(frozen=True, slots=True)
class ImageGenerationCompletionResult:
    """Результат завершения image-generation."""

    image_generation_id: int
    batch_id: int
    generated_post_id: int
    request_kind: str
    review_action_id: int | None
    request_key: str
    image_status: str
    batch_status: str
    post_status: str
    image_path: str
    image_sha256: str
    already_completed: bool


@dataclass(frozen=True, slots=True)
class ImageGenerationFailureResult:
    """Результат фиксации ошибки image-generation."""

    image_generation_id: int
    batch_id: int
    generated_post_id: int
    request_kind: str
    review_action_id: int | None
    request_key: str
    image_status: str
    batch_status: str
    post_status: str
    already_failed: bool
    error_type: str
    error_message: str


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
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


def _normalize_request_key(
    request_key: str,
) -> str:
    """Проверяет SHA-256 image request key."""

    normalized_request_key = (
        _normalize_required_text(
            request_key,
            field_name="request_key",
        )
    )

    if not (
        IMAGE_REQUEST_KEY_PATTERN
        .fullmatch(normalized_request_key)
    ):
        raise ValueError(
            "request_key должен быть SHA-256 "
            "в нижнем регистре."
        )

    return normalized_request_key


def _normalize_image_sha256(
    value: str,
) -> str:
    """Проверяет SHA-256 сохранённого PNG."""

    normalized_value = (
        _normalize_required_text(
            value,
            field_name="image_sha256",
        )
    )

    if not _IMAGE_SHA256_PATTERN.fullmatch(
        normalized_value
    ):
        raise ValueError(
            "image_sha256 должен быть SHA-256 "
            "в нижнем регистре."
        )

    return normalized_value


def _normalize_json_object(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, Any]:
    """Проверяет JSON-совместимый объект."""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} должен быть Mapping."
        )

    normalized_value = dict(value)

    try:
        encoded_value = json.dumps(
            normalized_value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        decoded_value = json.loads(
            encoded_value
        )
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"{field_name} должен быть "
            "JSON-совместимым объектом."
        ) from error

    if not isinstance(
        decoded_value,
        dict,
    ):
        raise ValueError(
            f"{field_name} должен быть "
            "JSON-объектом."
        )

    return decoded_value


def _normalize_optional_json_object(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any] | None:
    """Проверяет необязательный JSON-объект."""

    if value is None:
        return None

    return _normalize_json_object(
        value,
        field_name=field_name,
    )


def _build_usage_payload(
    usage: OpenAIImageUsage,
) -> dict[str, int]:
    """Формирует JSON с usage Image API."""

    return {
        "input_tokens": (
            usage.input_tokens
        ),
        "input_text_tokens": (
            usage.input_text_tokens
        ),
        "input_image_tokens": (
            usage.input_image_tokens
        ),
        "output_tokens": (
            usage.output_tokens
        ),
        "output_text_tokens": (
            usage.output_text_tokens
        ),
        "output_image_tokens": (
            usage.output_image_tokens
        ),
        "total_tokens": (
            usage.total_tokens
        ),
    }


def _normalize_telemetry(
    *,
    usage: OpenAIImageUsage | None,
    cost_payload: Mapping[str, Any] | None,
) -> tuple[
    dict[str, int] | None,
    dict[str, Any] | None,
]:
    """Проверяет парность usage и стоимости."""

    if (
        usage is None
        and cost_payload is None
    ):
        return None, None

    if (
        usage is None
        or cost_payload is None
    ):
        raise ValueError(
            "openai_usage и openai_cost "
            "должны присутствовать вместе "
            "или оба отсутствовать."
        )

    if not isinstance(
        usage,
        OpenAIImageUsage,
    ):
        raise TypeError(
            "usage должен быть OpenAIImageUsage."
        )

    normalized_cost_payload = (
        _normalize_json_object(
            cost_payload,
            field_name="cost_payload",
        )
    )

    return (
        _build_usage_payload(
            usage
        ),
        normalized_cost_payload,
    )


def _decode_json_object(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    """Извлекает JSON-объект из jsonb."""

    if isinstance(value, str):
        try:
            decoded_value = json.loads(
                value
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{field_name} содержит "
                "некорректный JSON."
            ) from error
    else:
        decoded_value = value

    if not isinstance(
        decoded_value,
        dict,
    ):
        raise ValueError(
            f"{field_name} должен быть "
            "JSON-объектом."
        )

    return decoded_value


def _request_payload_news_ids(
    payload: Mapping[str, Any],
) -> tuple[int, int, int]:
    """Читает TOP-3 news_id из request payload."""

    raw_news_ids = payload.get(
        "top3_news_ids"
    )

    if not isinstance(
        raw_news_ids,
        list,
    ) or len(raw_news_ids) != 3:
        raise ValueError(
            "request_payload.top3_news_ids "
            "должен содержать три значения."
        )

    news_ids = tuple(
        _normalize_positive_integer(
            value,
            field_name=(
                "request_payload.top3_news_ids"
            ),
        )
        for value in raw_news_ids
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "request_payload содержит "
            "дублирующиеся news_id."
        )

    return (
        int(news_ids[0]),
        int(news_ids[1]),
        int(news_ids[2]),
    )


def _request_model_and_prompt(
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Читает model и prompt из request payload."""

    model_request = payload.get(
        "model_request"
    )

    if not isinstance(
        model_request,
        dict,
    ):
        raise ValueError(
            "request_payload.model_request "
            "должен быть JSON-объектом."
        )

    model_name = _normalize_required_text(
        model_request.get("model"),
        field_name=(
            "request_payload.model_request.model"
        ),
    )

    prompt = _normalize_required_text(
        model_request.get("prompt"),
        field_name=(
            "request_payload.model_request.prompt"
        ),
    )

    return model_name, prompt


async def _load_image_generation_record(
    connection: asyncpg.Connection,
    *,
    image_generation_id: int,
    request_key: str,
) -> asyncpg.Record:
    """Блокирует image request и связанный контекст."""

    record = await connection.fetchrow(
        """
        SELECT
            igr.image_generation_id,
            igr.batch_id,
            igr.generated_post_id,
            igr.review_action_id,
            igr.image_request_key,
            igr.request_key_version,
            igr.image_status,
            igr.request_kind,
            igr.model_name,
            igr.generator_version,
            igr.prompt_version,
            igr.request_payload,
            igr.response_metadata,
            igr.openai_usage,
            igr.openai_cost,
            igr.image_path,
            igr.image_sha256,
            igr.error_type,
            igr.error_message,

            b.ranking_run_id,
            b.batch_status,

            gp.batch_id
                AS post_batch_id,
            gp.post_status,
            gp.image_path
                AS post_image_path,
            gp.image_sha256
                AS post_image_sha256,
            gp.image_prompt
                AS post_image_prompt,
            gp.image_model_name
                AS post_image_model_name,
            gp.image_prompt_version
                AS post_image_prompt_version,

            ra.generated_post_id
                AS review_generated_post_id,
            ra.reviewer_type,
            ra.decision,
            ra.requested_action
                AS review_requested_action,

            ARRAY(
                SELECT bi.position
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = igr.batch_id
                ORDER BY bi.position
            ) AS positions,

            ARRAY(
                SELECT bi.news_id
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = igr.batch_id
                ORDER BY bi.position
            ) AS batch_news_ids

        FROM top3_news.image_generation_requests AS igr
        JOIN top3_news.publication_batches AS b
          ON b.batch_id = igr.batch_id
        JOIN top3_news.generated_posts AS gp
          ON gp.generated_post_id =
             igr.generated_post_id
        LEFT JOIN top3_news.review_actions AS ra
          ON ra.review_action_id =
             igr.review_action_id
        WHERE igr.image_generation_id = $1
          AND igr.image_request_key = $2
        FOR UPDATE OF igr, b, gp
        """,
        image_generation_id,
        request_key,
    )

    if record is None:
        raise LookupError(
            "Зарезервированная image-generation "
            "не найдена: image_generation_id="
            f"{image_generation_id}"
        )

    return record


def _validate_image_generation_identity(
    record: asyncpg.Record,
) -> tuple[
    int,
    int,
    str,
    int | None,
    str,
    str,
]:
    """Проверяет неизменяемую часть reservation."""

    differences: list[str] = []

    batch_id = _normalize_positive_integer(
        int(record["batch_id"]),
        field_name="batch_id",
    )

    generated_post_id = (
        _normalize_positive_integer(
            int(record["generated_post_id"]),
            field_name="generated_post_id",
        )
    )

    if (
        int(record["post_batch_id"])
        != batch_id
    ):
        differences.append(
            "generated_post относится "
            "к другому batch"
        )

    request_kind = record["request_kind"]

    if request_kind not in {
        "initial",
        "regenerate",
    }:
        differences.append(
            "request_kind: "
            f"actual={request_kind!r}"
        )

    review_action_id_raw = (
        record["review_action_id"]
    )

    if request_kind == "initial":
        if review_action_id_raw is not None:
            differences.append(
                "initial содержит "
                "review_action_id"
            )

        review_action_id = None

        if (
            record["review_generated_post_id"]
            is not None
            or record["reviewer_type"] is not None
            or record["decision"] is not None
            or record["review_requested_action"]
            is not None
        ):
            differences.append(
                "initial неожиданно связан "
                "с review action"
            )
    else:
        if review_action_id_raw is None:
            differences.append(
                "regenerate не содержит "
                "review_action_id"
            )
            review_action_id = None
        else:
            review_action_id = (
                _normalize_positive_integer(
                    int(review_action_id_raw),
                    field_name="review_action_id",
                )
            )

        if (
            record["review_generated_post_id"]
            is None
            or int(
                record[
                    "review_generated_post_id"
                ]
            )
            != generated_post_id
        ):
            differences.append(
                "regenerate review action относится "
                "к другому generated_post"
            )

        expected_review_values = {
            "reviewer_type": "human",
            "decision": "changes_required",
            "review_requested_action": (
                "regenerate_image"
            ),
        }

        for field_name, expected_value in (
            expected_review_values.items()
        ):
            actual_value = record[field_name]

            if actual_value != expected_value:
                differences.append(
                    f"{field_name}: "
                    f"expected={expected_value!r}, "
                    f"actual={actual_value!r}"
                )

    request_key_version = (
        record["request_key_version"]
    )

    if not isinstance(
        request_key_version,
        str,
    ) or not request_key_version.strip():
        differences.append(
            "request_key_version отсутствует"
        )

    model_name = _normalize_required_text(
        record["model_name"],
        field_name="model_name",
    )

    prompt_version = (
        _normalize_required_text(
            record["prompt_version"],
            field_name="prompt_version",
        )
    )

    ranking_run_id = (
        _normalize_positive_integer(
            int(record["ranking_run_id"]),
            field_name="ranking_run_id",
        )
    )

    positions = tuple(
        int(value)
        for value in record["positions"]
    )

    if positions != (1, 2, 3):
        differences.append(
            "batch positions: "
            f"expected=(1, 2, 3), "
            f"actual={positions!r}"
        )

    batch_news_ids = tuple(
        int(value)
        for value in record[
            "batch_news_ids"
        ]
    )

    if len(batch_news_ids) != 3:
        differences.append(
            "publication batch должен содержать "
            "три news_id"
        )

    request_payload = (
        _decode_json_object(
            record["request_payload"],
            field_name="request_payload",
        )
    )

    payload_batch_id = (
        request_payload.get("batch_id")
    )

    if payload_batch_id != batch_id:
        differences.append(
            "request_payload.batch_id: "
            f"expected={batch_id!r}, "
            f"actual={payload_batch_id!r}"
        )

    payload_ranking_run_id = (
        request_payload.get(
            "ranking_run_id"
        )
    )

    if (
        payload_ranking_run_id
        != ranking_run_id
    ):
        differences.append(
            "request_payload.ranking_run_id: "
            f"expected={ranking_run_id!r}, "
            f"actual={payload_ranking_run_id!r}"
        )

    payload_request_kind = (
        request_payload.get("request_kind")
    )

    if payload_request_kind != request_kind:
        differences.append(
            "request_payload.request_kind: "
            f"expected={request_kind!r}, "
            f"actual={payload_request_kind!r}"
        )

    payload_review_action_id = (
        request_payload.get(
            "review_action_id"
        )
    )

    if (
        payload_review_action_id
        != review_action_id
    ):
        differences.append(
            "request_payload.review_action_id: "
            f"expected={review_action_id!r}, "
            f"actual={payload_review_action_id!r}"
        )

    payload_news_ids = (
        _request_payload_news_ids(
            request_payload
        )
    )

    if payload_news_ids != batch_news_ids:
        differences.append(
            "request_payload.top3_news_ids "
            "не совпадает с publication batch"
        )

    payload_model_name, prompt = (
        _request_model_and_prompt(
            request_payload
        )
    )

    if payload_model_name != model_name:
        differences.append(
            "request_payload model не совпадает "
            "с reservation model_name"
        )

    generator_payload = (
        request_payload.get("generator")
    )

    if not isinstance(
        generator_payload,
        dict,
    ):
        differences.append(
            "request_payload.generator "
            "должен быть JSON-объектом"
        )
    else:
        if (
            generator_payload.get(
                "prompt_version"
            )
            != prompt_version
        ):
            differences.append(
                "request_payload generator "
                "prompt_version не совпадает "
                "с reservation"
            )

    if differences:
        raise ValueError(
            "Image reservation содержит "
            "несогласованные данные: "
            + "; ".join(differences)
        )

    return (
        batch_id,
        generated_post_id,
        request_kind,
        review_action_id,
        model_name,
        prompt,
    )


def _validate_pre_completion_image_fields(
    record: asyncpg.Record,
    *,
    request_kind: str,
) -> None:
    """Проверяет image-поля до completion/failure."""

    fields = {
        "image_path": (
            record["post_image_path"]
        ),
        "image_sha256": (
            record["post_image_sha256"]
        ),
        "image_prompt": (
            record["post_image_prompt"]
        ),
        "image_model_name": (
            record["post_image_model_name"]
        ),
        "image_prompt_version": (
            record[
                "post_image_prompt_version"
            ]
        ),
    }

    if request_kind == "initial":
        present_fields = [
            field_name
            for field_name, value in fields.items()
            if value is not None
        ]

        if present_fields:
            raise ValueError(
                "Initial image-generation требует "
                "пустые image-поля generated_post: "
                + ", ".join(present_fields)
            )

        return

    missing_fields = [
        field_name
        for field_name, value in fields.items()
        if value is None
    ]

    if missing_fields:
        raise ValueError(
            "Regenerate image-generation требует "
            "полностью сохранённую текущую картинку. "
            "Отсутствуют поля: "
            + ", ".join(missing_fields)
        )


def _validate_reserved_runtime_context(
    record: asyncpg.Record,
    *,
    request_kind: str,
) -> None:
    """Проверяет состояние reserved image request."""

    if (
        record["batch_status"]
        != "awaiting_review"
    ):
        raise ValueError(
            "Reserved image-generation требует "
            "batch_status=awaiting_review."
        )

    if (
        record["post_status"]
        != "awaiting_review"
    ):
        raise ValueError(
            "Reserved image-generation требует "
            "post_status=awaiting_review."
        )

    _validate_pre_completion_image_fields(
        record,
        request_kind=request_kind,
    )


def _validate_existing_completion(
    record: asyncpg.Record,
    *,
    image_path: str,
    image_sha256: str,
    response_metadata: Mapping[str, Any],
    usage_payload: Mapping[str, Any] | None,
    cost_payload: Mapping[str, Any] | None,
    model_name: str,
    prompt: str,
) -> None:
    """Проверяет повторное completion."""

    differences: list[str] = []

    expected_values = {
        "image_path": image_path,
        "image_sha256": image_sha256,
        "post_image_path": image_path,
        "post_image_sha256": image_sha256,
        "post_image_prompt": prompt,
        "post_image_model_name": model_name,
        "post_image_prompt_version": (
            record["prompt_version"]
        ),
    }

    for field_name, expected_value in (
        expected_values.items()
    ):
        actual_value = record[field_name]

        if actual_value != expected_value:
            differences.append(
                f"{field_name}: "
                f"expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )

    actual_response_metadata = (
        _decode_json_object(
            record["response_metadata"],
            field_name="response_metadata",
        )
    )

    if (
        actual_response_metadata
        != dict(response_metadata)
    ):
        differences.append(
            "response_metadata differs"
        )

    if usage_payload is None:
        if record["openai_usage"] is not None:
            differences.append(
                "openai_usage должен быть NULL"
            )
    else:
        actual_usage = _decode_json_object(
            record["openai_usage"],
            field_name="openai_usage",
        )

        if actual_usage != dict(
            usage_payload
        ):
            differences.append(
                "openai_usage differs"
            )

    if cost_payload is None:
        if record["openai_cost"] is not None:
            differences.append(
                "openai_cost должен быть NULL"
            )
    else:
        actual_cost = _decode_json_object(
            record["openai_cost"],
            field_name="openai_cost",
        )

        if actual_cost != dict(
            cost_payload
        ):
            differences.append(
                "openai_cost differs"
            )

    if differences:
        raise ValueError(
            "Completed image-generation "
            "не соответствует повторному "
            "completion: "
            + "; ".join(differences)
        )


async def complete_reserved_image_generation(
    pool: asyncpg.Pool,
    *,
    image_generation_id: int,
    request_key: str,
    image_path: str,
    image_sha256: str,
    response_metadata: Mapping[str, Any],
    usage: OpenAIImageUsage | None = None,
    cost_payload: Mapping[str, Any] | None = None,
) -> ImageGenerationCompletionResult:
    """
    Завершает ранее зарезервированную image-generation.

    В одной транзакции:

    - проверяет reservation и сохранённый контекст;
    - сохраняет image-поля в generated_posts;
    - сохраняет response metadata и телеметрию;
    - переводит image request в completed;
    - publication_batch и generated_post
      остаются awaiting_review.

    Повторное completion с тем же результатом
    не изменяет данные повторно.
    """

    normalized_image_generation_id = (
        _normalize_positive_integer(
            image_generation_id,
            field_name="image_generation_id",
        )
    )

    normalized_request_key = (
        _normalize_request_key(
            request_key
        )
    )

    normalized_image_path = (
        _normalize_required_text(
            image_path,
            field_name="image_path",
        )
    )

    normalized_image_sha256 = (
        _normalize_image_sha256(
            image_sha256
        )
    )

    normalized_response_metadata = (
        _normalize_json_object(
            response_metadata,
            field_name="response_metadata",
        )
    )

    (
        usage_payload,
        normalized_cost_payload,
    ) = _normalize_telemetry(
        usage=usage,
        cost_payload=cost_payload,
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = (
                await _load_image_generation_record(
                    connection,
                    image_generation_id=(
                        normalized_image_generation_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                )
            )

            (
                batch_id,
                generated_post_id,
                request_kind,
                review_action_id,
                model_name,
                prompt,
            ) = _validate_image_generation_identity(
                record
            )

            image_status = (
                record["image_status"]
            )

            if image_status == "failed":
                raise ValueError(
                    "Нельзя завершить "
                    "image-generation со статусом "
                    "failed."
                )

            if image_status == "completed":
                _validate_existing_completion(
                    record,
                    image_path=(
                        normalized_image_path
                    ),
                    image_sha256=(
                        normalized_image_sha256
                    ),
                    response_metadata=(
                        normalized_response_metadata
                    ),
                    usage_payload=usage_payload,
                    cost_payload=(
                        normalized_cost_payload
                    ),
                    model_name=model_name,
                    prompt=prompt,
                )

                return ImageGenerationCompletionResult(
                    image_generation_id=(
                        normalized_image_generation_id
                    ),
                    batch_id=batch_id,
                    generated_post_id=(
                        generated_post_id
                    ),
                    request_kind=request_kind,
                    review_action_id=(
                        review_action_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    image_status="completed",
                    batch_status=(
                        record["batch_status"]
                    ),
                    post_status=(
                        record["post_status"]
                    ),
                    image_path=(
                        normalized_image_path
                    ),
                    image_sha256=(
                        normalized_image_sha256
                    ),
                    already_completed=True,
                )

            if image_status != "reserved":
                raise ValueError(
                    "Неподдерживаемый "
                    "image_status: "
                    f"{image_status!r}"
                )

            _validate_reserved_runtime_context(
                record,
                request_kind=request_kind,
            )

            post_update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.generated_posts
                    SET
                        image_path = $3,
                        image_sha256 = $4,
                        image_prompt = $5,
                        image_model_name = $6,
                        image_prompt_version = $7,
                        updated_at = now()
                    WHERE generated_post_id = $1
                      AND batch_id = $2
                      AND post_status =
                          'awaiting_review'
                    """,
                    generated_post_id,
                    batch_id,
                    normalized_image_path,
                    normalized_image_sha256,
                    prompt,
                    model_name,
                    record["prompt_version"],
                )
            )

            if post_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось сохранить image-поля "
                    "generated_post: "
                    f"{post_update_result}"
                )

            completion_update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.image_generation_requests
                    SET
                        image_status = 'completed',
                        response_metadata = $3::jsonb,
                        openai_usage = $4::jsonb,
                        openai_cost = $5::jsonb,
                        image_path = $6,
                        image_sha256 = $7,
                        error_type = NULL,
                        error_message = NULL,
                        completed_at = now(),
                        failed_at = NULL,
                        updated_at = now()
                    WHERE image_generation_id = $1
                      AND image_request_key = $2
                      AND image_status = 'reserved'
                    """,
                    normalized_image_generation_id,
                    normalized_request_key,
                    _encode_json(
                        normalized_response_metadata
                    ),
                    (
                        _encode_json(
                            usage_payload
                        )
                        if usage_payload is not None
                        else None
                    ),
                    (
                        _encode_json(
                            normalized_cost_payload
                        )
                        if (
                            normalized_cost_payload
                            is not None
                        )
                        else None
                    ),
                    normalized_image_path,
                    normalized_image_sha256,
                )
            )

            if (
                completion_update_result
                != "UPDATE 1"
            ):
                raise RuntimeError(
                    "Не удалось завершить "
                    "image-generation reservation: "
                    f"{completion_update_result}"
                )

            batch_update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.publication_batches
                    SET
                        batch_status =
                            'awaiting_review',
                        error_message = NULL,
                        updated_at = now()
                    WHERE batch_id = $1
                      AND batch_status =
                          'awaiting_review'
                    """,
                    batch_id,
                )
            )

            if batch_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось подтвердить "
                    "awaiting_review для batch: "
                    f"{batch_update_result}"
                )

    return ImageGenerationCompletionResult(
        image_generation_id=(
            normalized_image_generation_id
        ),
        batch_id=batch_id,
        generated_post_id=generated_post_id,
        request_kind=request_kind,
        review_action_id=review_action_id,
        request_key=normalized_request_key,
        image_status="completed",
        batch_status="awaiting_review",
        post_status="awaiting_review",
        image_path=normalized_image_path,
        image_sha256=normalized_image_sha256,
        already_completed=False,
    )


async def fail_reserved_image_generation(
    pool: asyncpg.Pool,
    *,
    image_generation_id: int,
    request_key: str,
    error_message: str,
    error_type: str,
    response_metadata: Mapping[str, Any] | None = None,
    usage: OpenAIImageUsage | None = None,
    cost_payload: Mapping[str, Any] | None = None,
) -> ImageGenerationFailureResult:
    """
    Фиксирует ошибку конкретной image reservation.

    publication_batch и generated_post
    не переводятся в failed.

    Если Image API уже успел вернуть ответ,
    response metadata и фактическую телеметрию
    можно сохранить вместе с failed.
    """

    normalized_image_generation_id = (
        _normalize_positive_integer(
            image_generation_id,
            field_name="image_generation_id",
        )
    )

    normalized_request_key = (
        _normalize_request_key(
            request_key
        )
    )

    normalized_error_type = (
        _normalize_required_text(
            error_type,
            field_name="error_type",
        )[:500]
    )

    normalized_error_message = (
        _normalize_required_text(
            error_message,
            field_name="error_message",
        )[:8000]
    )

    normalized_response_metadata = (
        _normalize_optional_json_object(
            response_metadata,
            field_name="response_metadata",
        )
    )

    (
        usage_payload,
        normalized_cost_payload,
    ) = _normalize_telemetry(
        usage=usage,
        cost_payload=cost_payload,
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = (
                await _load_image_generation_record(
                    connection,
                    image_generation_id=(
                        normalized_image_generation_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                )
            )

            (
                batch_id,
                generated_post_id,
                request_kind,
                review_action_id,
                _model_name,
                _prompt,
            ) = _validate_image_generation_identity(
                record
            )

            image_status = (
                record["image_status"]
            )

            if image_status == "failed":
                return ImageGenerationFailureResult(
                    image_generation_id=(
                        normalized_image_generation_id
                    ),
                    batch_id=batch_id,
                    generated_post_id=(
                        generated_post_id
                    ),
                    request_kind=request_kind,
                    review_action_id=(
                        review_action_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    image_status="failed",
                    batch_status=(
                        record["batch_status"]
                    ),
                    post_status=(
                        record["post_status"]
                    ),
                    already_failed=True,
                    error_type=(
                        record["error_type"]
                        or normalized_error_type
                    ),
                    error_message=(
                        record["error_message"]
                        or normalized_error_message
                    ),
                )

            if image_status == "completed":
                raise ValueError(
                    "Нельзя перевести completed "
                    "image-generation в failed."
                )

            if image_status != "reserved":
                raise ValueError(
                    "Неподдерживаемый "
                    "image_status: "
                    f"{image_status!r}"
                )

            _validate_reserved_runtime_context(
                record,
                request_kind=request_kind,
            )

            failure_update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.image_generation_requests
                    SET
                        image_status = 'failed',
                        response_metadata = $3::jsonb,
                        openai_usage = $4::jsonb,
                        openai_cost = $5::jsonb,
                        image_path = NULL,
                        image_sha256 = NULL,
                        error_type = $6,
                        error_message = $7,
                        completed_at = NULL,
                        failed_at = now(),
                        updated_at = now()
                    WHERE image_generation_id = $1
                      AND image_request_key = $2
                      AND image_status = 'reserved'
                    """,
                    normalized_image_generation_id,
                    normalized_request_key,
                    (
                        _encode_json(
                            normalized_response_metadata
                        )
                        if (
                            normalized_response_metadata
                            is not None
                        )
                        else None
                    ),
                    (
                        _encode_json(
                            usage_payload
                        )
                        if usage_payload is not None
                        else None
                    ),
                    (
                        _encode_json(
                            normalized_cost_payload
                        )
                        if (
                            normalized_cost_payload
                            is not None
                        )
                        else None
                    ),
                    normalized_error_type,
                    normalized_error_message,
                )
            )

            if (
                failure_update_result
                != "UPDATE 1"
            ):
                raise RuntimeError(
                    "Не удалось перевести "
                    "image-generation в failed: "
                    f"{failure_update_result}"
                )

    return ImageGenerationFailureResult(
        image_generation_id=(
            normalized_image_generation_id
        ),
        batch_id=batch_id,
        generated_post_id=generated_post_id,
        request_kind=request_kind,
        review_action_id=review_action_id,
        request_key=normalized_request_key,
        image_status="failed",
        batch_status="awaiting_review",
        post_status="awaiting_review",
        already_failed=False,
        error_type=normalized_error_type,
        error_message=normalized_error_message,
    )