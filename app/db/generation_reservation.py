from dataclasses import dataclass
from datetime import date, datetime
import json
from typing import Any, Mapping

import asyncpg

from app.db.generation_selection import (
    GenerationTop3Selection,
    _load_generation_combination,
    _load_generation_top3,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    OpenAIPostGeneratorMetadata,
)
from app.generation.request_key import (
    GenerationRequestKey,
    create_generation_request_key,
)


@dataclass(frozen=True, slots=True)
class GenerationReservation:
    """Результат резервирования генерации поста."""

    batch_id: int
    publication_date: date
    edition: int
    batch_status: str
    ranking_run_id: int
    request_key: str
    news_ids: tuple[int, int, int]
    score_ids: tuple[int, int, int]
    created_new: bool

    @property
    def should_call_model(self) -> bool:
        """Разрешает платный вызов только создателю нового выпуска."""

        return (
            self.created_new
            and self.batch_status == "ranked"
        )


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


def _normalize_publication_date(
    value: date,
) -> date:
    """Проверяет дату выпуска."""

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


def _normalize_metadata(
    metadata: OpenAIPostGeneratorMetadata,
) -> OpenAIPostGeneratorMetadata:
    """Проверяет метаданные генератора."""

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
        text_format=(
            _normalize_required_text(
                metadata.text_format,
                field_name=(
                    "metadata.text_format"
                ),
            )
        ),
    )


def _normalize_selection(
    selection: GenerationTop3Selection,
) -> GenerationTop3Selection:
    """Проверяет сохранённый TOP-3."""

    _normalize_positive_integer(
        selection.ranking_run_id,
        field_name=(
            "selection.ranking_run_id"
        ),
    )

    if selection.run_status != "completed":
        raise ValueError(
            "selection.run_status должен быть "
            "равен completed."
        )

    if selection.eligible_count < 3:
        raise ValueError(
            "selection.eligible_count должен "
            "быть не меньше трёх."
        )

    if len(selection.items) != 3:
        raise ValueError(
            "selection.items должен содержать "
            "ровно три новости."
        )

    if len(selection.score_ids) != 3:
        raise ValueError(
            "selection.score_ids должен "
            "содержать ровно три значения."
        )

    positions = tuple(
        item.position
        for item in selection.items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "selection.items должен идти "
            "в порядке позиций 1, 2 и 3."
        )

    news_ids = selection.news_ids

    if len(set(news_ids)) != 3:
        raise ValueError(
            "selection содержит "
            "дублирующиеся news_id."
        )

    for position, news_id in enumerate(
        news_ids,
        start=1,
    ):
        _normalize_positive_integer(
            news_id,
            field_name=(
                "selection.news_id "
                f"position={position}"
            ),
        )

    for position, score_id in enumerate(
        selection.score_ids,
        start=1,
    ):
        _normalize_positive_integer(
            score_id,
            field_name=(
                "selection.score_id "
                f"position={position}"
            ),
        )

    if len(set(selection.score_ids)) != 3:
        raise ValueError(
            "selection содержит "
            "дублирующиеся score_id."
        )

    return selection


def _validate_request_key(
    *,
    request_key: GenerationRequestKey,
    selection: GenerationTop3Selection,
    publication_date: date,
    telegram_chat_id: int,
    metadata: OpenAIPostGeneratorMetadata,
    model_request: GenerationModelRequest,
) -> None:
    """Сверяет ключ с текущими параметрами."""

    expected_key = (
        create_generation_request_key(
            ranking_run_id=(
                selection.ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    if request_key != expected_key:
        raise ValueError(
            "generation request_key не "
            "соответствует текущему TOP-3, "
            "дате, каналу, модели или промпту."
        )


def _build_batch_metadata(
    *,
    request_key: GenerationRequestKey,
    selection: GenerationTop3Selection,
    metadata: OpenAIPostGeneratorMetadata,
) -> dict[str, Any]:
    """Формирует метаданные резервирования."""

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
            request_key.value
        ),
        "generation_request_key_version": (
            request_key.version
        ),
        "ranking_run_id": (
            selection.ranking_run_id
        ),
        "news_ids": list(
            selection.news_ids
        ),
        "score_ids": list(
            selection.score_ids
        ),
        "news_count": 3,
        "idempotency_reserved": True,
    }


async def _find_existing_reservation(
    connection: asyncpg.Connection,
    *,
    request_key: str,
) -> asyncpg.Record | None:
    """Ищет ранее зарезервированный выпуск."""

    return await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.publication_date,
            b.edition,
            b.batch_status,
            b.ranking_run_id,
            b.target_telegram_chat_id,
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
                b.metadata->>'ranking_run_id'
            )::bigint
                AS metadata_ranking_run_id,

            (
                b.metadata->>'news_count'
            )::integer
                AS news_count,

            (
                b.metadata->>'idempotency_reserved'
            )::boolean
                AS idempotency_reserved,

            ARRAY(
                SELECT bi.position
                FROM batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS positions,

            ARRAY(
                SELECT bi.news_id
                FROM batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS news_ids,

            ARRAY(
                SELECT bi.score_id
                FROM batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS score_ids,

            ARRAY(
                SELECT COALESCE(
                    bi.selection_reason,
                    ''
                )
                FROM batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS selection_reasons

        FROM publication_batches AS b
        WHERE b.generation_request_key = $1
        FOR UPDATE
        """,
        request_key,
    )


def _validate_existing_reservation(
    record: asyncpg.Record,
    *,
    request_key: GenerationRequestKey,
    selection: GenerationTop3Selection,
    publication_date: date,
    telegram_chat_id: int,
    metadata: OpenAIPostGeneratorMetadata,
) -> None:
    """Проверяет найденное резервирование."""

    differences: list[str] = []

    expected_values = {
        "publication_date": (
            publication_date
        ),
        "ranking_run_id": (
            selection.ranking_run_id
        ),
        "target_telegram_chat_id": (
            telegram_chat_id
        ),
        "generation_request_key": (
            request_key.value
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
            request_key.value
        ),
        "request_key_version": (
            request_key.version
        ),
        "metadata_ranking_run_id": (
            selection.ranking_run_id
        ),
        "news_count": 3,
        "idempotency_reserved": True,
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

    allowed_statuses = {
        "ranked",
        "generated",
        "awaiting_review",
        "approved",
        "rejected",
        "publishing",
        "published",
        "failed",
    }

    if (
        record["batch_status"]
        not in allowed_statuses
    ):
        differences.append(
            "batch_status: "
            f"actual={record['batch_status']!r}"
        )

    actual_positions = tuple(
        int(value)
        for value in record["positions"]
    )

    if actual_positions != (1, 2, 3):
        differences.append(
            "positions: expected=(1, 2, 3), "
            f"actual={actual_positions!r}"
        )

    actual_news_ids = tuple(
        int(value)
        for value in record["news_ids"]
    )

    if actual_news_ids != selection.news_ids:
        differences.append(
            "news_ids: "
            f"expected={selection.news_ids!r}, "
            f"actual={actual_news_ids!r}"
        )

    actual_score_ids = tuple(
        (
            int(value)
            if value is not None
            else None
        )
        for value in record["score_ids"]
    )

    if (
        actual_score_ids
        != selection.score_ids
    ):
        differences.append(
            "score_ids: "
            f"expected={selection.score_ids!r}, "
            f"actual={actual_score_ids!r}"
        )

    expected_reasons = tuple(
        item.selection_reason
        for item in selection.items
    )

    actual_reasons = tuple(
        record["selection_reasons"]
    )

    if actual_reasons != expected_reasons:
        differences.append(
            "selection_reasons differ"
        )

    if differences:
        raise ValueError(
            "generation_request_key уже "
            "существует с другими параметрами: "
            + "; ".join(differences)
        )


def _build_existing_result(
    record: asyncpg.Record,
) -> GenerationReservation:
    """Создаёт результат существующего выпуска."""

    return GenerationReservation(
        batch_id=int(
            record["batch_id"]
        ),
        publication_date=(
            record["publication_date"]
        ),
        edition=int(
            record["edition"]
        ),
        batch_status=(
            record["batch_status"]
        ),
        ranking_run_id=int(
            record["ranking_run_id"]
        ),
        request_key=(
            record["generation_request_key"]
        ),
        news_ids=tuple(
            int(value)
            for value in record["news_ids"]
        ),
        score_ids=tuple(
            int(value)
            for value in record["score_ids"]
        ),
        created_new=False,
    )


async def reserve_generation(
    pool: asyncpg.Pool,
    *,
    request_key: GenerationRequestKey,
    selection: GenerationTop3Selection,
    combination_id: int | None = None,
    publication_date: date,
    telegram_chat_id: int,
    metadata: OpenAIPostGeneratorMetadata,
    model_request: GenerationModelRequest,
) -> GenerationReservation:
    """
    Резервирует выпуск до вызова OpenAI.

    Только процесс, создавший новый batch,
    получает should_call_model=True.

    Повторный процесс получает существующий
    выпуск и не выполняет платный API-запрос.
    """

    normalized_selection = (
        _normalize_selection(
            selection
        )
    )

    normalized_combination_id = (
        _normalize_positive_integer(
            combination_id,
            field_name="combination_id",
        )
        if combination_id is not None
        else None
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

    normalized_metadata = (
        _normalize_metadata(
            metadata
        )
    )

    _validate_request_key(
        request_key=request_key,
        selection=normalized_selection,
        publication_date=(
            normalized_publication_date
        ),
        telegram_chat_id=(
            normalized_telegram_chat_id
        ),
        metadata=normalized_metadata,
        model_request=model_request,
    )

    batch_metadata = (
        _build_batch_metadata(
            request_key=request_key,
            selection=normalized_selection,
            metadata=normalized_metadata,
        )
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    $1::bigint
                )
                """,
                (
                    normalized_publication_date
                    .toordinal()
                ),
            )

            if normalized_combination_id is None:
                current_selection = (
                    await _load_generation_top3(
                        connection,
                        ranking_run_id=(
                            normalized_selection
                            .ranking_run_id
                        ),
                    )
                )
            else:
                current_combination = (
                    await _load_generation_combination(
                        connection,
                        ranking_run_id=(
                            normalized_selection
                            .ranking_run_id
                        ),
                        combination_id=(
                            normalized_combination_id
                        ),
                    )
                )

                current_selection = (
                    current_combination.selection
                )

            if (
                current_selection
                != normalized_selection
            ):
                raise ValueError(
                    "Сохранённая ranking selection "
                    "изменилась после подготовки "
                    "запроса. Нужно сформировать "
                    "запрос и request_key заново."
                )

            existing_record = (
                await _find_existing_reservation(
                    connection,
                    request_key=(
                        request_key.value
                    ),
                )
            )

            if existing_record is not None:
                _validate_existing_reservation(
                    existing_record,
                    request_key=request_key,
                    selection=(
                        normalized_selection
                    ),
                    publication_date=(
                        normalized_publication_date
                    ),
                    telegram_chat_id=(
                        normalized_telegram_chat_id
                    ),
                    metadata=(
                        normalized_metadata
                    ),
                )

                return (
                    _build_existing_result(
                        existing_record
                    )
                )

            edition = (
                await connection.fetchval(
                    """
                    SELECT
                        COALESCE(
                            MAX(edition),
                            0
                        )::integer + 1
                    FROM publication_batches
                    WHERE publication_date = $1
                    """,
                    normalized_publication_date,
                )
            )

            batch_id = (
                await connection.fetchval(
                    """
                    INSERT INTO publication_batches (
                        publication_date,
                        edition,
                        ranking_run_id,
                        batch_status,
                        target_telegram_chat_id,
                        metadata,
                        generation_request_key
                    )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        'ranked',
                        $4,
                        $5::jsonb,
                        $6
                    )
                    RETURNING batch_id
                    """,
                    normalized_publication_date,
                    edition,
                    (
                        normalized_selection
                        .ranking_run_id
                    ),
                    normalized_telegram_chat_id,
                    _encode_json(
                        batch_metadata
                    ),
                    request_key.value,
                )
            )

            for item, score_id in zip(
                normalized_selection.items,
                normalized_selection.score_ids,
                strict=True,
            ):
                await connection.execute(
                    """
                    INSERT INTO batch_items (
                        batch_id,
                        news_id,
                        score_id,
                        position,
                        selection_reason
                    )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5
                    )
                    """,
                    batch_id,
                    item.news_id,
                    score_id,
                    item.position,
                    item.selection_reason,
                )

    return GenerationReservation(
        batch_id=int(batch_id),
        publication_date=(
            normalized_publication_date
        ),
        edition=int(edition),
        batch_status="ranked",
        ranking_run_id=(
            normalized_selection
            .ranking_run_id
        ),
        request_key=request_key.value,
        news_ids=(
            normalized_selection.news_ids
        ),
        score_ids=(
            normalized_selection.score_ids
        ),
        created_new=True,
    )