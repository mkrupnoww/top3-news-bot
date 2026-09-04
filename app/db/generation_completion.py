from dataclasses import dataclass
from decimal import Decimal
import json
from typing import Any, Mapping

import asyncpg

from app.generation.openai_generator import (
    OpenAIPostGenerationResult,
    OpenAIPostGeneratorMetadata,
)
from app.generation.request_key import (
    GENERATION_REQUEST_KEY_PATTERN,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)

from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
)

GENERATION_COMPLETION_VERSION = (
    "reserved_generation_completion_v2"
)

GENERATION_FAILURE_VERSION = (
    "reserved_generation_failure_v1"
)

WEB_SEARCH_TOOL_PRICE_USD_PER_CALL = (
    Decimal("0.01")
)

WEB_SEARCH_TOOL_PRICING_VERSION = (
    "openai_web_search_2026_08_21"
)

GENERATION_COST_ACCOUNTING_VERSION = (
    "generation_cost_accounting_v1"
)


@dataclass(frozen=True, slots=True)
class GenerationCompletionResult:
    """Результат сохранения сгенерированного поста."""

    batch_id: int
    generated_post_id: int
    request_key: str
    batch_status: str
    post_status: str
    version_number: int
    text_format: str
    news_ids: tuple[int, int, int]
    already_completed: bool


@dataclass(frozen=True, slots=True)
class GenerationFailureResult:
    """Результат фиксации ошибки генерации."""

    batch_id: int
    request_key: str
    batch_status: str
    already_failed: bool
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

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_request_key(
    request_key: str,
) -> str:
    """Проверяет SHA-256 ключ генерации."""

    normalized_request_key = (
        _normalize_required_text(
            request_key,
            field_name="request_key",
        )
    )

    if not (
        GENERATION_REQUEST_KEY_PATTERN
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


def calculate_web_search_tool_cost(
    call_count: int,
) -> Decimal:
    """Рассчитывает стоимость Web Search tool calls."""

    if isinstance(call_count, bool):
        raise TypeError(
            "web_search_call_count не может "
            "быть bool."
        )

    if not isinstance(call_count, int):
        raise TypeError(
            "web_search_call_count должен быть int."
        )

    if call_count < 0:
        raise ValueError(
            "web_search_call_count не может "
            "быть отрицательным."
        )

    return (
        Decimal(call_count)
        * WEB_SEARCH_TOOL_PRICE_USD_PER_CALL
    )


def _build_web_search_payload(
    *,
    used: bool,
    call_count: int,
    source_urls: tuple[str, ...],
) -> tuple[dict[str, Any], Decimal]:
    """Формирует Web Search telemetry и её стоимость."""

    if not isinstance(used, bool):
        raise TypeError(
            "web_search_used должен быть bool."
        )

    tool_cost_usd = (
        calculate_web_search_tool_cost(
            call_count
        )
    )

    if used != (call_count > 0):
        raise ValueError(
            "web_search_used не согласован с "
            "web_search_call_count."
        )

    if not isinstance(source_urls, tuple):
        raise TypeError(
            "web_source_urls должен быть tuple."
        )

    normalized_urls: list[str] = []

    for url in source_urls:
        if not isinstance(url, str):
            raise TypeError(
                "Каждый web_source_url должен "
                "быть строкой."
            )

        normalized_url = url.strip()

        if not normalized_url:
            raise ValueError(
                "web_source_url не может быть "
                "пустым."
            )

        if normalized_url not in normalized_urls:
            normalized_urls.append(
                normalized_url
            )

    return (
        {
            "used": used,
            "call_count": call_count,
            "pricing_version": (
                WEB_SEARCH_TOOL_PRICING_VERSION
            ),
            "tool_price_usd_per_call": str(
                WEB_SEARCH_TOOL_PRICE_USD_PER_CALL
            ),
            "tool_cost_usd": str(
                tool_cost_usd
            ),
            "source_urls": normalized_urls,
        },
        tool_cost_usd,
    )


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

    if len(post_text) > MAXIMUM_POST_LENGTH:
        raise ValueError(
            "post_text превышает допустимую "
            "длину выпуска: "
            f"{MAXIMUM_POST_LENGTH} символов."
        )

    if len(result.payload.items) != 3:
        raise ValueError(
            "Результат генерации должен "
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

    for news_id in news_ids:
        _normalize_positive_integer(
            news_id,
            field_name="result.news_id",
        )

    generated_items: list[
        dict[str, Any]
    ] = []

    for item in result.payload.items:
        generated_items.append(
            {
                "position": item.position,
                "news_id": item.news_id,
                "headline": (
                    _normalize_required_text(
                        item.headline,
                        field_name=(
                            "result.headline "
                            f"news_id={item.news_id}"
                        ),
                    )
                ),
                "body": (
                    _normalize_required_text(
                        item.body,
                        field_name=(
                            "result.body "
                            f"news_id={item.news_id}"
                        ),
                    )
                ),
                **(
                    {
                        "official_trailer_url": item.official_trailer_url,
                        **(
                            {
                                "official_trailer_channel_name": (
                                    item.official_trailer_channel_name
                                )
                            }
                            if item.official_trailer_channel_name is not None
                            else {}
                        ),
                    }
                    if item.official_trailer_url is not None
                    else {}
                ),
            }
        )

    usage = result.model_response.usage

    if usage is None:
        raise ValueError(
            "В результате генерации "
            "отсутствует OpenAI usage."
        )

    cost_estimate = (
        result.model_response.cost_estimate
    )

    if cost_estimate is None:
        raise ValueError(
            "В результате генерации "
            "отсутствует оценка стоимости."
        )

    return (
        post_text,
        (
            news_ids[0],
            news_ids[1],
            news_ids[2],
        ),
        generated_items,
        usage,
        cost_estimate,
    )


def _build_generation_metadata(
    *,
    request_key: str,
    request_key_version: str,
    ranking_run_id: int,
    metadata: OpenAIPostGeneratorMetadata,
    news_ids: tuple[int, int, int],
    generated_items: list[
        dict[str, Any]
    ],
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
    post_text: str,
) -> dict[str, Any]:
    """Формирует метаданные generated_post."""

    return {
        "generation_mode": "openai",
        "generator_name": (
            metadata.generator_name
        ),
        "generator_version": (
            metadata.generator_version
        ),
        "prompt_version": (
            metadata.prompt_version
        ),
        "model_name": (
            metadata.model_name
        ),
        "text_format": (
            metadata.text_format
        ),
        "generation_request_key": (
            request_key
        ),
        "generation_request_key_version": (
            request_key_version
        ),
        "ranking_run_id": ranking_run_id,
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
            GENERATION_COMPLETION_VERSION
        ),
    }


def _build_batch_completion_metadata(
    *,
    generated_post_id: int,
    usage: OpenAITokenUsage,
    cost_estimate: OpenAICostEstimate,
    web_search_used: bool,
    web_search_call_count: int,
    web_source_urls: tuple[str, ...],
) -> dict[str, Any]:
    """Формирует дополнение metadata выпуска."""

    (
        web_search_payload,
        web_search_tool_cost_usd,
    ) = _build_web_search_payload(
        used=web_search_used,
        call_count=web_search_call_count,
        source_urls=web_source_urls,
    )

    generation_total_cost_usd = (
        cost_estimate.total_cost_usd
        + web_search_tool_cost_usd
    )

    return {
        "generation_completed": True,
        "generated_post_id": (
            generated_post_id
        ),
        "openai_usage": (
            _build_usage_payload(usage)
        ),
        "openai_cost": (
            _build_cost_payload(
                cost_estimate
            )
        ),
        "openai_web_search": (
            web_search_payload
        ),
        "generation_total_cost_usd": str(
            generation_total_cost_usd
        ),
        "generation_cost_accounting_version": (
            GENERATION_COST_ACCOUNTING_VERSION
        ),
        "generation_completion_version": (
            GENERATION_COMPLETION_VERSION
        ),
    }


async def _load_reserved_batch(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    request_key: str,
) -> asyncpg.Record:
    """Блокирует зарезервированный выпуск."""

    record = await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.batch_status,
            b.ranking_run_id,
            b.generation_request_key,

            b.metadata->>'generation_mode'
                AS generation_mode,

            b.metadata->>'generator_name'
                AS generator_name,

            b.metadata->>'generator_version'
                AS generator_version,

            b.metadata->>'prompt_version'
                AS prompt_version,

            b.metadata->>'model_name'
                AS model_name,

            b.metadata->>'text_format'
                AS text_format,

            b.metadata->>'generation_request_key'
                AS metadata_request_key,

            b.metadata->>'generation_request_key_version'
                AS request_key_version,

            (
                b.metadata->>'news_count'
            )::integer
                AS news_count,

            ARRAY(
                SELECT bi.position
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS positions,

            ARRAY(
                SELECT bi.news_id
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS news_ids,

            ARRAY(
                SELECT bi.score_id
                FROM top3_news.batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS score_ids

        FROM top3_news.publication_batches AS b
        WHERE b.batch_id = $1
          AND b.generation_request_key = $2
        FOR UPDATE
        """,
        batch_id,
        request_key,
    )

    if record is None:
        raise LookupError(
            "Зарезервированный выпуск "
            "не найден: "
            f"batch_id={batch_id}"
        )

    return record


def _validate_reserved_batch(
    record: asyncpg.Record,
    *,
    request_key: str,
    metadata: OpenAIPostGeneratorMetadata,
    result_news_ids: tuple[int, int, int],
) -> tuple[int, tuple[int, int, int]]:
    """Проверяет reservation и batch_items."""

    differences: list[str] = []

    expected_values = {
        "generation_request_key": (
            request_key
        ),
        "generation_mode": "openai",
        "generator_name": (
            metadata.generator_name
        ),
        "generator_version": (
            metadata.generator_version
        ),
        "prompt_version": (
            metadata.prompt_version
        ),
        "model_name": (
            metadata.model_name
        ),
        "text_format": (
            metadata.text_format
        ),
        "metadata_request_key": (
            request_key
        ),
        "news_count": 3,
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

    ranking_run_id = (
        record["ranking_run_id"]
    )

    if ranking_run_id is None:
        differences.append(
            "ranking_run_id отсутствует"
        )
        normalized_ranking_run_id = 0
    else:
        normalized_ranking_run_id = int(
            ranking_run_id
        )

        if normalized_ranking_run_id <= 0:
            differences.append(
                "ranking_run_id должен быть "
                "больше нуля"
            )

    positions = tuple(
        int(value)
        for value in record["positions"]
    )

    if positions != (1, 2, 3):
        differences.append(
            "positions: expected=(1, 2, 3), "
            f"actual={positions!r}"
        )

    persisted_news_ids = tuple(
        int(value)
        for value in record["news_ids"]
    )

    if persisted_news_ids != result_news_ids:
        differences.append(
            "news_ids: "
            f"expected={result_news_ids!r}, "
            f"actual={persisted_news_ids!r}"
        )

    score_ids = tuple(
        record["score_ids"]
    )

    if (
        len(score_ids) != 3
        or any(
            score_id is None
            for score_id in score_ids
        )
    ):
        differences.append(
            "score_ids должны содержать "
            "три ненулевых значения"
        )

    if differences:
        raise ValueError(
            "Зарезервированный выпуск "
            "не соответствует результату "
            "генерации: "
            + "; ".join(differences)
        )

    return (
        normalized_ranking_run_id,
        (
            persisted_news_ids[0],
            persisted_news_ids[1],
            persisted_news_ids[2],
        ),
    )


async def _load_existing_generated_post(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    expected_post_text: str,
    expected_text_format: str,
    expected_model_name: str,
    expected_prompt_version: str,
    expected_metadata: Mapping[str, Any],
) -> asyncpg.Record:
    """Читает и проверяет сохранённую версию 1."""

    record = await connection.fetchrow(
        """
        SELECT
            generated_post_id,
            version_number,
            post_status,
            post_text,
            text_format,
            text_model_name,
            text_prompt_version,
            (
                generation_metadata
                = $2::jsonb
            ) AS metadata_match
        FROM top3_news.generated_posts
        WHERE batch_id = $1
          AND version_number = 1
        """,
        batch_id,
        _encode_json(expected_metadata),
    )

    if record is None:
        raise RuntimeError(
            "Выпуск уже завершён, но "
            "generated_post версии 1 "
            "не найден."
        )

    differences: list[str] = []

    expected_values = {
        "version_number": 1,
        "post_text": expected_post_text,
        "text_format": (
            expected_text_format
        ),
        "text_model_name": (
            expected_model_name
        ),
        "text_prompt_version": (
            expected_prompt_version
        ),
        "metadata_match": True,
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

    if differences:
        raise ValueError(
            "Существующий generated_post "
            "не соответствует результату: "
            + "; ".join(differences)
        )

    return record


async def complete_reserved_generation(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
    request_key: str,
    metadata: OpenAIPostGeneratorMetadata,
    result: OpenAIPostGenerationResult,
) -> GenerationCompletionResult:
    """
    Сохраняет результат ранее зарезервированной генерации.

    В одной транзакции:

    - проверяет publication_batch и batch_items;
    - сохраняет generated_posts версии 1;
    - записывает usage и оценку стоимости;
    - переводит выпуск в awaiting_review.

    Повторное завершение с тем же результатом
    не создаёт вторую версию поста.
    """

    normalized_batch_id = (
        _normalize_positive_integer(
            batch_id,
            field_name="batch_id",
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
            batch_record = (
                await _load_reserved_batch(
                    connection,
                    batch_id=normalized_batch_id,
                    request_key=(
                        normalized_request_key
                    ),
                )
            )

            (
                ranking_run_id,
                persisted_news_ids,
            ) = _validate_reserved_batch(
                batch_record,
                request_key=(
                    normalized_request_key
                ),
                metadata=normalized_metadata,
                result_news_ids=(
                    result_news_ids
                ),
            )

            request_key_version = (
                batch_record[
                    "request_key_version"
                ].strip()
            )

            generation_metadata = (
                _build_generation_metadata(
                    request_key=(
                        normalized_request_key
                    ),
                    request_key_version=(
                        request_key_version
                    ),
                    ranking_run_id=(
                        ranking_run_id
                    ),
                    metadata=(
                        normalized_metadata
                    ),
                    news_ids=(
                        persisted_news_ids
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

            batch_status = (
                batch_record["batch_status"]
            )

            if batch_status == "failed":
                raise ValueError(
                    "Нельзя завершить выпуск "
                    "со статусом failed."
                )

            if batch_status != "ranked":
                existing_post = (
                    await _load_existing_generated_post(
                        connection,
                        batch_id=(
                            normalized_batch_id
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
                            normalized_metadata
                            .prompt_version
                        ),
                        expected_metadata=(
                            generation_metadata
                        ),
                    )
                )

                return GenerationCompletionResult(
                    batch_id=(
                        normalized_batch_id
                    ),
                    generated_post_id=int(
                        existing_post[
                            "generated_post_id"
                        ]
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    batch_status=(
                        batch_status
                    ),
                    post_status=(
                        existing_post[
                            "post_status"
                        ]
                    ),
                    version_number=1,
                    text_format=(
                        normalized_metadata
                        .text_format
                    ),
                    news_ids=(
                        persisted_news_ids
                    ),
                    already_completed=True,
                )

            existing_post_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts
                    WHERE batch_id = $1
                    """,
                    normalized_batch_id,
                )
            )

            if existing_post_count != 0:
                raise RuntimeError(
                    "У ranked-выпуска уже "
                    "существует generated_post: "
                    f"count={existing_post_count}"
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
                            text_model_name,
                            text_prompt_version,
                            generation_metadata
                        )
                    VALUES (
                        $1,
                        1,
                        'awaiting_review',
                        $2,
                        $3,
                        $4,
                        $5,
                        $6::jsonb
                    )
                    RETURNING generated_post_id
                    """,
                    normalized_batch_id,
                    post_text,
                    normalized_metadata.text_format,
                    normalized_metadata.model_name,
                    normalized_metadata.prompt_version,
                    _encode_json(
                        generation_metadata
                    ),
                )
            )

            batch_completion_metadata = (
                _build_batch_completion_metadata(
                    generated_post_id=int(
                        generated_post_id
                    ),
                    usage=usage,
                    cost_estimate=(
                        cost_estimate
                    ),
                    web_search_used=(
                        result.model_response
                        .web_search_used
                    ),
                    web_search_call_count=(
                        result.model_response
                        .web_search_call_count
                    ),
                    web_source_urls=(
                        result.model_response
                        .web_source_urls
                    ),
                )
            )

            update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.publication_batches
                    SET
                        batch_status =
                            'awaiting_review',
                        metadata = (
                            metadata
                            || $3::jsonb
                        ),
                        error_message = NULL
                    WHERE batch_id = $1
                      AND generation_request_key = $2
                      AND batch_status = 'ranked'
                    """,
                    normalized_batch_id,
                    normalized_request_key,
                    _encode_json(
                        batch_completion_metadata
                    ),
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось завершить "
                    "зарезервированную генерацию: "
                    f"{update_result}"
                )

    return GenerationCompletionResult(
        batch_id=normalized_batch_id,
        generated_post_id=int(
            generated_post_id
        ),
        request_key=normalized_request_key,
        batch_status="awaiting_review",
        post_status="awaiting_review",
        version_number=1,
        text_format=(
            normalized_metadata.text_format
        ),
        news_ids=result_news_ids,
        already_completed=False,
    )


async def fail_reserved_generation(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
    request_key: str,
    error_message: str,
    error_type: str | None = None,
) -> GenerationFailureResult:
    """Переводит зарезервированный выпуск в failed."""

    normalized_batch_id = (
        _normalize_positive_integer(
            batch_id,
            field_name="batch_id",
        )
    )

    normalized_request_key = (
        _normalize_request_key(
            request_key
        )
    )

    normalized_error_message = (
        _normalize_required_text(
            error_message,
            field_name="error_message",
        )[:8000]
    )

    normalized_error_type: str | None

    if error_type is None:
        normalized_error_type = None
    else:
        normalized_error_type = (
            _normalize_required_text(
                error_type,
                field_name="error_type",
            )[:500]
        )

    failure_metadata = {
        "generation_failed": True,
        "failure": {
            "error_type": (
                normalized_error_type
            ),
            "error_message": (
                normalized_error_message
            ),
        },
        "generation_failure_version": (
            GENERATION_FAILURE_VERSION
        ),
    }

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await connection.fetchrow(
                """
                SELECT
                    batch_id,
                    batch_status,
                    generation_request_key,
                    error_message
                FROM top3_news.publication_batches
                WHERE batch_id = $1
                  AND generation_request_key = $2
                FOR UPDATE
                """,
                normalized_batch_id,
                normalized_request_key,
            )

            if record is None:
                raise LookupError(
                    "Зарезервированный выпуск "
                    "не найден: "
                    f"batch_id={normalized_batch_id}"
                )

            if (
                record["batch_status"]
                == "failed"
            ):
                return GenerationFailureResult(
                    batch_id=(
                        normalized_batch_id
                    ),
                    request_key=(
                        normalized_request_key
                    ),
                    batch_status="failed",
                    already_failed=True,
                    error_message=(
                        record["error_message"]
                        or normalized_error_message
                    ),
                )

            if (
                record["batch_status"]
                != "ranked"
            ):
                raise ValueError(
                    "Нельзя перевести выпуск "
                    "в failed из статуса: "
                    f"{record['batch_status']}"
                )

            generated_post_count = (
                await connection.fetchval(
                    """
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts
                    WHERE batch_id = $1
                    """,
                    normalized_batch_id,
                )
            )

            if generated_post_count != 0:
                raise RuntimeError(
                    "Нельзя пометить выпуск "
                    "failed: уже существует "
                    "generated_post."
                )

            update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.publication_batches
                    SET
                        batch_status = 'failed',
                        metadata = (
                            metadata
                            || $3::jsonb
                        ),
                        error_message = $4
                    WHERE batch_id = $1
                      AND generation_request_key = $2
                      AND batch_status = 'ranked'
                    """,
                    normalized_batch_id,
                    normalized_request_key,
                    _encode_json(
                        failure_metadata
                    ),
                    normalized_error_message,
                )
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось перевести "
                    "зарезервированную генерацию "
                    "в failed: "
                    f"{update_result}"
                )

    return GenerationFailureResult(
        batch_id=normalized_batch_id,
        request_key=normalized_request_key,
        batch_status="failed",
        already_failed=False,
        error_message=(
            normalized_error_message
        ),
    )