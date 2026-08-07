from dataclasses import dataclass
import json
from typing import Any, Mapping

import asyncpg

from app.generation.openai_generator import (
    OPENAI_POST_REVISION_PROMPT_VERSION,
    OpenAIPostGenerationResult,
    OpenAIPostGeneratorMetadata,
)
from app.generation.revision_request_key import (
    GENERATION_REVISION_REQUEST_KEY_PATTERN,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


GENERATION_REVISION_COMPLETION_VERSION = (
    "generation_revision_completion_v1"
)

GENERATION_REVISION_FAILURE_VERSION = (
    "generation_revision_failure_v1"
)


@dataclass(frozen=True, slots=True)
class GenerationRevisionCompletionResult:
    """Результат сохранения новой версии поста."""

    generation_revision_id: int
    batch_id: int
    source_generated_post_id: int
    generated_post_id: int
    review_action_id: int
    request_key: str
    revision_status: str
    source_post_status: str
    post_status: str
    version_number: int
    text_format: str
    news_ids: tuple[int, int, int]
    already_completed: bool


@dataclass(frozen=True, slots=True)
class GenerationRevisionFailureResult:
    """Результат фиксации ошибки ревизии."""

    generation_revision_id: int
    batch_id: int
    source_generated_post_id: int
    review_action_id: int
    request_key: str
    revision_status: str
    batch_status: str
    source_post_status: str
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
    """Проверяет SHA-256 ключ ревизии."""

    normalized_request_key = (
        _normalize_required_text(
            request_key,
            field_name="request_key",
        )
    )

    if not (
        GENERATION_REVISION_REQUEST_KEY_PATTERN
        .fullmatch(normalized_request_key)
    ):
        raise ValueError(
            "request_key должен быть SHA-256 "
            "в нижнем регистре."
        )

    return normalized_request_key


def _normalize_metadata(
    metadata: OpenAIPostGeneratorMetadata,
) -> OpenAIPostGeneratorMetadata:
    """Проверяет метаданные генератора."""

    normalized_text_format = (
        _normalize_required_text(
            metadata.text_format,
            field_name="metadata.text_format",
        )
    )

    if normalized_text_format not in {
        "markdown",
        "markdown_v2",
        "html",
        "plain_text",
    }:
        raise ValueError(
            "metadata.text_format содержит "
            "неподдерживаемое значение."
        )

    return OpenAIPostGeneratorMetadata(
        generator_name=(
            _normalize_required_text(
                metadata.generator_name,
                field_name=(
                    "metadata.generator_name"
                ),
            )
        ),
        generator_version=(
            _normalize_required_text(
                metadata.generator_version,
                field_name=(
                    "metadata.generator_version"
                ),
            )
        ),
        prompt_version=(
            _normalize_required_text(
                metadata.prompt_version,
                field_name=(
                    "metadata.prompt_version"
                ),
            )
        ),
        model_name=(
            _normalize_required_text(
                metadata.model_name,
                field_name=(
                    "metadata.model_name"
                ),
            )
        ),
        text_format=normalized_text_format,
    )


def _build_usage_payload(
    usage: OpenAITokenUsage,
) -> dict[str, int]:
    """Формирует JSON с токенами OpenAI."""

    return {
        "input_tokens": (
            usage.input_tokens
        ),
        "regular_input_tokens": (
            usage.regular_input_tokens
        ),
        "cached_input_tokens": (
            usage.cached_input_tokens
        ),
        "cache_write_tokens": (
            usage.cache_write_tokens
        ),
        "output_tokens": (
            usage.output_tokens
        ),
        "reasoning_tokens": (
            usage.reasoning_tokens
        ),
        "total_tokens": (
            usage.total_tokens
        ),
    }


def _build_cost_payload(
    cost_estimate: OpenAICostEstimate,
) -> dict[str, str]:
    """Формирует JSON с оценкой стоимости."""

    return {
        "model_name": (
            cost_estimate.model_name
        ),
        "pricing_version": (
            cost_estimate.pricing_version
        ),
        "regular_input_cost_usd": str(
            cost_estimate
            .regular_input_cost_usd
        ),
        "cached_input_cost_usd": str(
            cost_estimate
            .cached_input_cost_usd
        ),
        "cache_write_cost_usd": str(
            cost_estimate
            .cache_write_cost_usd
        ),
        "output_cost_usd": str(
            cost_estimate.output_cost_usd
        ),
        "total_cost_usd": str(
            cost_estimate.total_cost_usd
        ),
    }


def _validate_telemetry(
    *,
    metadata: OpenAIPostGeneratorMetadata,
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
) -> None:
    """Проверяет согласованность телеметрии."""

    if (
        cost_estimate.model_name
        != metadata.model_name
    ):
        raise ValueError(
            "Модель расчёта стоимости "
            "не совпадает с моделью генератора: "
            f"cost={cost_estimate.model_name!r}, "
            f"metadata={metadata.model_name!r}"
        )

    component_total = (
        cost_estimate
        .regular_input_cost_usd
        + cost_estimate
        .cached_input_cost_usd
        + cost_estimate
        .cache_write_cost_usd
        + cost_estimate
        .output_cost_usd
    )

    if (
        component_total
        != cost_estimate.total_cost_usd
    ):
        raise ValueError(
            "total_cost_usd не совпадает "
            "с суммой компонентов стоимости."
        )

    if (
        usage.total_tokens
        != (
            usage.input_tokens
            + usage.output_tokens
        )
    ):
        raise ValueError(
            "total_tokens не совпадает "
            "с input_tokens + output_tokens."
        )


def _prepare_generation_result(
    result: OpenAIPostGenerationResult,
) -> tuple[
    str,
    tuple[int, int, int],
    list[dict[str, Any]],
    OpenAITokenUsage,
    OpenAICostEstimate,
]:
    """Проверяет результат модели."""

    post_text = (
        result.payload.post_text.strip()
    )

    if not post_text:
        raise ValueError(
            "post_text не может быть пустым."
        )

    if len(post_text) > 4096:
        raise ValueError(
            "post_text превышает ограничение "
            "Telegram в 4096 символов."
        )

    if len(result.payload.items) != 3:
        raise ValueError(
            "Результат ревизии должен "
            "содержать ровно три новости."
        )

    positions = tuple(
        item.position
        for item in result.payload.items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Позиции результата должны идти "
            "в порядке 1, 2 и 3."
        )

    news_ids = tuple(
        item.news_id
        for item in result.payload.items
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Результат содержит "
            "дублирующиеся news_id."
        )

    generated_items: list[
        dict[str, Any]
    ] = []

    for item in result.payload.items:
        news_id = _normalize_positive_integer(
            item.news_id,
            field_name="result.news_id",
        )

        generated_items.append(
            {
                "position": item.position,
                "news_id": news_id,
                "headline": (
                    _normalize_required_text(
                        item.headline,
                        field_name=(
                            "result.headline "
                            f"news_id={news_id}"
                        ),
                    )
                ),
                "body": (
                    _normalize_required_text(
                        item.body,
                        field_name=(
                            "result.body "
                            f"news_id={news_id}"
                        ),
                    )
                ),
            }
        )

    usage = result.model_response.usage

    if usage is None:
        raise ValueError(
            "В результате ревизии "
            "отсутствует OpenAI usage."
        )

    cost_estimate = (
        result.model_response.cost_estimate
    )

    if cost_estimate is None:
        raise ValueError(
            "В результате ревизии "
            "отсутствует оценка стоимости."
        )

    return (
        post_text,
        (
            int(news_ids[0]),
            int(news_ids[1]),
            int(news_ids[2]),
        ),
        generated_items,
        usage,
        cost_estimate,
    )


def _decode_request_payload(
    value: Any,
) -> dict[str, Any]:
    """Извлекает request_payload из jsonb."""

    if isinstance(value, str):
        try:
            decoded_value = json.loads(
                value
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "request_payload содержит "
                "некорректный JSON."
            ) from error
    else:
        decoded_value = value

    if not isinstance(
        decoded_value,
        dict,
    ):
        raise ValueError(
            "request_payload должен быть "
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


def _build_revision_generation_metadata(
    *,
    generation_revision_id: int,
    request_key: str,
    request_key_version: str,
    source_generated_post_id: int,
    review_action_id: int,
    target_version_number: int,
    metadata: OpenAIPostGeneratorMetadata,
    revision_prompt_version: str,
    news_ids: tuple[int, int, int],
    generated_items: list[
        dict[str, Any]
    ],
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
    post_text: str,
) -> dict[str, Any]:
    """Формирует metadata новой generated_posts."""

    return {
        "generation_mode": (
            "openai_revision"
        ),
        "generation_revision_id": (
            generation_revision_id
        ),
        "generation_revision_request_key": (
            request_key
        ),
        "generation_revision_request_key_version": (
            request_key_version
        ),
        "source_generated_post_id": (
            source_generated_post_id
        ),
        "review_action_id": (
            review_action_id
        ),
        "target_version_number": (
            target_version_number
        ),
        "requested_action": (
            "regenerate_text"
        ),
        "generator_name": (
            metadata.generator_name
        ),
        "generator_version": (
            metadata.generator_version
        ),
        "base_prompt_version": (
            metadata.prompt_version
        ),
        "revision_prompt_version": (
            revision_prompt_version
        ),
        "model_name": (
            metadata.model_name
        ),
        "text_format": (
            metadata.text_format
        ),
        "news_ids": list(news_ids),
        "news_count": 3,
        "generated_items": generated_items,
        "post_length": len(post_text),
        "openai_usage": (
            _build_usage_payload(usage)
        ),
        "openai_cost": (
            _build_cost_payload(
                cost_estimate
            )
        ),
        "completion_version": (
            GENERATION_REVISION_COMPLETION_VERSION
        ),
    }


async def _load_revision_record(
    connection: asyncpg.Connection,
    *,
    generation_revision_id: int,
    request_key: str,
) -> asyncpg.Record:
    """Блокирует revision request и связанный контекст."""

    record = await connection.fetchrow(
        """
        SELECT
            grr.generation_revision_id,
            grr.batch_id,
            grr.source_generated_post_id,
            grr.review_action_id,
            grr.target_version_number,
            grr.revision_request_key,
            grr.request_key_version,
            grr.revision_status,
            grr.requested_action,
            grr.editorial_comment,
            grr.issues,
            grr.model_name,
            grr.generator_version,
            grr.prompt_version,
            grr.text_format,
            grr.request_payload,
            grr.generated_post_id,
            grr.error_type,
            grr.error_message,

            b.batch_status,

            source_gp.version_number
                AS source_version_number,
            source_gp.post_status
                AS source_post_status,
            source_gp.post_text
                AS source_post_text,
            source_gp.text_format
                AS source_text_format,
            source_gp.image_path,
            source_gp.image_sha256,
            source_gp.image_prompt,
            source_gp.image_model_name,
            source_gp.image_prompt_version,

            ra.generated_post_id
                AS review_generated_post_id,
            ra.reviewer_type,
            ra.decision,
            ra.requested_action
                AS review_requested_action,

            ARRAY(
                SELECT bi.position
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = grr.batch_id
                ORDER BY bi.position
            ) AS positions,

            ARRAY(
                SELECT bi.news_id
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = grr.batch_id
                ORDER BY bi.position
            ) AS batch_news_ids

        FROM
            top3_news.generation_revision_requests
                AS grr
        JOIN top3_news.publication_batches AS b
          ON b.batch_id = grr.batch_id
        JOIN top3_news.generated_posts AS source_gp
          ON source_gp.generated_post_id =
             grr.source_generated_post_id
        JOIN top3_news.review_actions AS ra
          ON ra.review_action_id =
             grr.review_action_id
        WHERE grr.generation_revision_id = $1
          AND grr.revision_request_key = $2
        FOR UPDATE OF grr, b, source_gp, ra
        """,
        generation_revision_id,
        request_key,
    )

    if record is None:
        raise LookupError(
            "Зарезервированная revision "
            "не найдена: "
            "generation_revision_id="
            f"{generation_revision_id}"
        )

    return record


def _validate_revision_identity(
    record: asyncpg.Record,
    *,
    metadata: OpenAIPostGeneratorMetadata,
    revision_prompt_version: str,
    result_news_ids: tuple[int, int, int],
) -> tuple[
    int,
    int,
    int,
    int,
    str,
]:
    """Проверяет неизменяемую часть reservation."""

    differences: list[str] = []

    expected_values = {
        "requested_action": (
            "regenerate_text"
        ),
        "model_name": (
            metadata.model_name
        ),
        "generator_version": (
            metadata.generator_version
        ),
        "prompt_version": (
            revision_prompt_version
        ),
        "text_format": (
            metadata.text_format
        ),
        "source_text_format": (
            metadata.text_format
        ),
        "reviewer_type": "human",
        "decision": "changes_required",
        "review_requested_action": (
            "regenerate_text"
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

    batch_id = _normalize_positive_integer(
        int(record["batch_id"]),
        field_name="batch_id",
    )

    source_generated_post_id = (
        _normalize_positive_integer(
            int(
                record[
                    "source_generated_post_id"
                ]
            ),
            field_name=(
                "source_generated_post_id"
            ),
        )
    )

    review_action_id = (
        _normalize_positive_integer(
            int(record["review_action_id"]),
            field_name="review_action_id",
        )
    )

    target_version_number = (
        _normalize_positive_integer(
            int(
                record[
                    "target_version_number"
                ]
            ),
            field_name=(
                "target_version_number"
            ),
        )
    )

    source_version_number = int(
        record["source_version_number"]
    )

    if (
        target_version_number
        != source_version_number + 1
    ):
        differences.append(
            "target_version_number: "
            f"expected={source_version_number + 1}, "
            f"actual={target_version_number}"
        )

    if (
        int(
            record[
                "review_generated_post_id"
            ]
        )
        != source_generated_post_id
    ):
        differences.append(
            "review_action относится "
            "к другому generated_post"
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

    if (
        batch_news_ids
        != result_news_ids
    ):
        differences.append(
            "result news_ids не совпадают "
            "с publication batch: "
            f"expected={batch_news_ids!r}, "
            f"actual={result_news_ids!r}"
        )

    request_payload = (
        _decode_request_payload(
            record["request_payload"]
        )
    )

    payload_news_ids = (
        _request_payload_news_ids(
            request_payload
        )
    )

    if (
        payload_news_ids
        != result_news_ids
    ):
        differences.append(
            "result news_ids не совпадают "
            "с revision request payload: "
            f"expected={payload_news_ids!r}, "
            f"actual={result_news_ids!r}"
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
        normalized_request_key_version = ""
    else:
        normalized_request_key_version = (
            request_key_version.strip()
        )

    if differences:
        raise ValueError(
            "Зарезервированная revision "
            "не соответствует результату: "
            + "; ".join(differences)
        )

    return (
        batch_id,
        source_generated_post_id,
        review_action_id,
        target_version_number,
        normalized_request_key_version,
    )


async def _load_existing_target_post(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    generated_post_id: int,
    target_version_number: int,
    expected_post_text: str,
    expected_text_format: str,
    expected_model_name: str,
    expected_prompt_version: str,
    expected_metadata: Mapping[str, Any],
    source_record: asyncpg.Record,
) -> asyncpg.Record:
    """Читает и проверяет уже сохранённую target-версию."""

    record = await connection.fetchrow(
        """
        SELECT
            generated_post_id,
            batch_id,
            version_number,
            post_status,
            post_text,
            text_format,
            image_path,
            image_sha256,
            image_prompt,
            text_model_name,
            image_model_name,
            text_prompt_version,
            image_prompt_version,
            (
                generation_metadata
                = $3::jsonb
            ) AS metadata_match
        FROM top3_news.generated_posts
        WHERE generated_post_id = $1
          AND batch_id = $2
        """,
        generated_post_id,
        batch_id,
        _encode_json(expected_metadata),
    )

    if record is None:
        raise RuntimeError(
            "Revision уже completed, но "
            "generated_post не найден."
        )

    expected_values = {
        "version_number": (
            target_version_number
        ),
        "post_status": "awaiting_review",
        "post_text": expected_post_text,
        "text_format": (
            expected_text_format
        ),
        "image_path": (
            source_record["image_path"]
        ),
        "image_sha256": (
            source_record["image_sha256"]
        ),
        "image_prompt": (
            source_record["image_prompt"]
        ),
        "text_model_name": (
            expected_model_name
        ),
        "image_model_name": (
            source_record[
                "image_model_name"
            ]
        ),
        "text_prompt_version": (
            expected_prompt_version
        ),
        "image_prompt_version": (
            source_record[
                "image_prompt_version"
            ]
        ),
        "metadata_match": True,
    }

    differences: list[str] = []

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

    if differences:
        raise ValueError(
            "Существующая target-версия "
            "не соответствует результату "
            "revision: "
            + "; ".join(differences)
        )

    return record


async def complete_reserved_generation_revision(
    pool: asyncpg.Pool,
    *,
    generation_revision_id: int,
    request_key: str,
    metadata: OpenAIPostGeneratorMetadata,
    result: OpenAIPostGenerationResult,
    revision_prompt_version: str = (
        OPENAI_POST_REVISION_PROMPT_VERSION
    ),
) -> GenerationRevisionCompletionResult:
    """
    Завершает ранее зарезервированную текстовую ревизию.

    В одной транзакции:

    - проверяет reservation и результат модели;
    - создаёт новую generated_posts version N+1;
    - наследует image-поля исходной версии;
    - переводит исходную версию в superseded;
    - новую версию оставляет awaiting_review;
    - publication_batch оставляет awaiting_review;
    - revision request переводит в completed.

    Повторное завершение с тем же результатом
    не создаёт вторую версию.
    """

    normalized_generation_revision_id = (
        _normalize_positive_integer(
            generation_revision_id,
            field_name=(
                "generation_revision_id"
            ),
        )
    )

    normalized_request_key = (
        _normalize_request_key(
            request_key
        )
    )

    normalized_metadata = (
        _normalize_metadata(metadata)
    )

    normalized_revision_prompt_version = (
        _normalize_required_text(
            revision_prompt_version,
            field_name=(
                "revision_prompt_version"
            ),
        )
    )

    (
        post_text,
        result_news_ids,
        generated_items,
        usage,
        cost_estimate,
    ) = _prepare_generation_result(result)

    _validate_telemetry(
        metadata=normalized_metadata,
        usage=usage,
        cost_estimate=cost_estimate,
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await _load_revision_record(
                connection,
                generation_revision_id=(
                    normalized_generation_revision_id
                ),
                request_key=(
                    normalized_request_key
                ),
            )

            (
                batch_id,
                source_generated_post_id,
                review_action_id,
                target_version_number,
                request_key_version,
            ) = _validate_revision_identity(
                record,
                metadata=normalized_metadata,
                revision_prompt_version=(
                    normalized_revision_prompt_version
                ),
                result_news_ids=(
                    result_news_ids
                ),
            )

            generation_metadata = (
                _build_revision_generation_metadata(
                    generation_revision_id=(
                        normalized_generation_revision_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    request_key_version=(
                        request_key_version
                    ),
                    source_generated_post_id=(
                        source_generated_post_id
                    ),
                    review_action_id=(
                        review_action_id
                    ),
                    target_version_number=(
                        target_version_number
                    ),
                    metadata=(
                        normalized_metadata
                    ),
                    revision_prompt_version=(
                        normalized_revision_prompt_version
                    ),
                    news_ids=(
                        result_news_ids
                    ),
                    generated_items=(
                        generated_items
                    ),
                    usage=usage,
                    cost_estimate=(
                        cost_estimate
                    ),
                    post_text=post_text,
                )
            )

            revision_status = (
                record["revision_status"]
            )

            if revision_status == "failed":
                raise ValueError(
                    "Нельзя завершить revision "
                    "со статусом failed."
                )

            if revision_status == "completed":
                generated_post_id = (
                    record["generated_post_id"]
                )

                if generated_post_id is None:
                    raise RuntimeError(
                        "Completed revision "
                        "не содержит "
                        "generated_post_id."
                    )

                existing_post = (
                    await _load_existing_target_post(
                        connection,
                        batch_id=batch_id,
                        generated_post_id=int(
                            generated_post_id
                        ),
                        target_version_number=(
                            target_version_number
                        ),
                        expected_post_text=(
                            post_text
                        ),
                        expected_text_format=(
                            normalized_metadata
                            .text_format
                        ),
                        expected_model_name=(
                            normalized_metadata
                            .model_name
                        ),
                        expected_prompt_version=(
                            normalized_revision_prompt_version
                        ),
                        expected_metadata=(
                            generation_metadata
                        ),
                        source_record=record,
                    )
                )

                if (
                    record["batch_status"]
                    != "awaiting_review"
                ):
                    raise ValueError(
                        "Completed revision требует "
                        "batch_status=awaiting_review."
                    )

                if (
                    record[
                        "source_post_status"
                    ]
                    != "superseded"
                ):
                    raise ValueError(
                        "Completed revision требует "
                        "source_post_status="
                        "superseded."
                    )

                return (
                    GenerationRevisionCompletionResult(
                        generation_revision_id=(
                            normalized_generation_revision_id
                        ),
                        batch_id=batch_id,
                        source_generated_post_id=(
                            source_generated_post_id
                        ),
                        generated_post_id=int(
                            existing_post[
                                "generated_post_id"
                            ]
                        ),
                        review_action_id=(
                            review_action_id
                        ),
                        request_key=(
                            normalized_request_key
                        ),
                        revision_status="completed",
                        source_post_status=(
                            "superseded"
                        ),
                        post_status=(
                            existing_post[
                                "post_status"
                            ]
                        ),
                        version_number=(
                            target_version_number
                        ),
                        text_format=(
                            normalized_metadata
                            .text_format
                        ),
                        news_ids=(
                            result_news_ids
                        ),
                        already_completed=True,
                    )
                )

            if revision_status != "reserved":
                raise ValueError(
                    "Неподдерживаемый "
                    "revision_status: "
                    f"{revision_status!r}"
                )

            if (
                record["batch_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Reserved revision требует "
                    "batch_status=awaiting_review."
                )

            if (
                record["source_post_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Reserved revision требует "
                    "source_post_status="
                    "awaiting_review."
                )

            existing_target_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts
                    WHERE batch_id = $1
                      AND version_number = $2
                    """,
                    batch_id,
                    target_version_number,
                )
            )

            if existing_target_count != 0:
                raise RuntimeError(
                    "До completion уже существует "
                    "целевая версия generated_post: "
                    f"count={existing_target_count}"
                )

            generated_post_id = (
                await connection.fetchval(
                    """
                    INSERT INTO
                        top3_news.generated_posts (
                            batch_id,
                            version_number,
                            post_status,
                            post_text,
                            text_format,
                            image_path,
                            image_sha256,
                            image_prompt,
                            text_model_name,
                            image_model_name,
                            text_prompt_version,
                            image_prompt_version,
                            generation_metadata
                        )
                    VALUES (
                        $1,
                        $2,
                        'awaiting_review',
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10,
                        $11,
                        $12::jsonb
                    )
                    RETURNING generated_post_id
                    """,
                    batch_id,
                    target_version_number,
                    post_text,
                    normalized_metadata.text_format,
                    record["image_path"],
                    record["image_sha256"],
                    record["image_prompt"],
                    normalized_metadata.model_name,
                    record["image_model_name"],
                    normalized_revision_prompt_version,
                    record[
                        "image_prompt_version"
                    ],
                    _encode_json(
                        generation_metadata
                    ),
                )
            )

            source_update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.generated_posts
                    SET
                        post_status = 'superseded',
                        updated_at = now()
                    WHERE generated_post_id = $1
                      AND batch_id = $2
                      AND post_status =
                          'awaiting_review'
                    """,
                    source_generated_post_id,
                    batch_id,
                )
            )

            if source_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось перевести "
                    "исходную версию "
                    "в superseded: "
                    f"{source_update_result}"
                )

            revision_update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.generation_revision_requests
                    SET
                        revision_status =
                            'completed',
                        openai_usage = $3::jsonb,
                        openai_cost = $4::jsonb,
                        generated_post_id = $5,
                        error_type = NULL,
                        error_message = NULL,
                        completed_at = now(),
                        failed_at = NULL,
                        updated_at = now()
                    WHERE generation_revision_id = $1
                      AND revision_request_key = $2
                      AND revision_status =
                          'reserved'
                    """,
                    normalized_generation_revision_id,
                    normalized_request_key,
                    _encode_json(
                        _build_usage_payload(
                            usage
                        )
                    ),
                    _encode_json(
                        _build_cost_payload(
                            cost_estimate
                        )
                    ),
                    generated_post_id,
                )
            )

            if (
                revision_update_result
                != "UPDATE 1"
            ):
                raise RuntimeError(
                    "Не удалось завершить "
                    "зарезервированную revision: "
                    f"{revision_update_result}"
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

    return GenerationRevisionCompletionResult(
        generation_revision_id=(
            normalized_generation_revision_id
        ),
        batch_id=batch_id,
        source_generated_post_id=(
            source_generated_post_id
        ),
        generated_post_id=int(
            generated_post_id
        ),
        review_action_id=review_action_id,
        request_key=normalized_request_key,
        revision_status="completed",
        source_post_status="superseded",
        post_status="awaiting_review",
        version_number=target_version_number,
        text_format=(
            normalized_metadata.text_format
        ),
        news_ids=result_news_ids,
        already_completed=False,
    )


async def fail_reserved_generation_revision(
    pool: asyncpg.Pool,
    *,
    generation_revision_id: int,
    request_key: str,
    error_message: str,
    error_type: str,
) -> GenerationRevisionFailureResult:
    """
    Фиксирует ошибку конкретной revision reservation.

    publication_batch и исходный generated_post
    не переводятся в failed и остаются awaiting_review.
    """

    normalized_generation_revision_id = (
        _normalize_positive_integer(
            generation_revision_id,
            field_name=(
                "generation_revision_id"
            ),
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

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await _load_revision_record(
                connection,
                generation_revision_id=(
                    normalized_generation_revision_id
                ),
                request_key=(
                    normalized_request_key
                ),
            )

            batch_id = _normalize_positive_integer(
                int(record["batch_id"]),
                field_name="batch_id",
            )

            source_generated_post_id = (
                _normalize_positive_integer(
                    int(
                        record[
                            "source_generated_post_id"
                        ]
                    ),
                    field_name=(
                        "source_generated_post_id"
                    ),
                )
            )

            review_action_id = (
                _normalize_positive_integer(
                    int(
                        record[
                            "review_action_id"
                        ]
                    ),
                    field_name="review_action_id",
                )
            )

            revision_status = (
                record["revision_status"]
            )

            if revision_status == "failed":
                return (
                    GenerationRevisionFailureResult(
                        generation_revision_id=(
                            normalized_generation_revision_id
                        ),
                        batch_id=batch_id,
                        source_generated_post_id=(
                            source_generated_post_id
                        ),
                        review_action_id=(
                            review_action_id
                        ),
                        request_key=(
                            normalized_request_key
                        ),
                        revision_status="failed",
                        batch_status=(
                            record[
                                "batch_status"
                            ]
                        ),
                        source_post_status=(
                            record[
                                "source_post_status"
                            ]
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
                )

            if revision_status == "completed":
                raise ValueError(
                    "Нельзя перевести completed "
                    "revision в failed."
                )

            if revision_status != "reserved":
                raise ValueError(
                    "Неподдерживаемый "
                    "revision_status: "
                    f"{revision_status!r}"
                )

            if (
                record["batch_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Reserved revision требует "
                    "batch_status=awaiting_review."
                )

            if (
                record["source_post_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Reserved revision требует "
                    "source_post_status="
                    "awaiting_review."
                )

            target_generated_post_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts
                    WHERE batch_id = $1
                      AND version_number = $2
                    """,
                    batch_id,
                    int(
                        record[
                            "target_version_number"
                        ]
                    ),
                )
            )

            if target_generated_post_count != 0:
                raise RuntimeError(
                    "Нельзя пометить revision "
                    "failed: целевая версия "
                    "generated_post уже существует."
                )

            update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.generation_revision_requests
                    SET
                        revision_status = 'failed',
                        generated_post_id = NULL,
                        openai_usage = NULL,
                        openai_cost = NULL,
                        error_type = $3,
                        error_message = $4,
                        completed_at = NULL,
                        failed_at = now(),
                        updated_at = now()
                    WHERE generation_revision_id = $1
                      AND revision_request_key = $2
                      AND revision_status =
                          'reserved'
                    """,
                    normalized_generation_revision_id,
                    normalized_request_key,
                    normalized_error_type,
                    normalized_error_message,
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось перевести "
                    "revision в failed: "
                    f"{update_result}"
                )

    return GenerationRevisionFailureResult(
        generation_revision_id=(
            normalized_generation_revision_id
        ),
        batch_id=batch_id,
        source_generated_post_id=(
            source_generated_post_id
        ),
        review_action_id=review_action_id,
        request_key=normalized_request_key,
        revision_status="failed",
        batch_status="awaiting_review",
        source_post_status="awaiting_review",
        already_failed=False,
        error_type=normalized_error_type,
        error_message=(
            normalized_error_message
        ),
    )