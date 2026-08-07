from dataclasses import dataclass
import json
from typing import Any, Mapping

import asyncpg

from app.db.generation_selection import (
    GenerationTop3Selection,
    _load_generation_top3,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationNewsItem,
    OPENAI_POST_REVISION_PROMPT_VERSION,
    OpenAIPostGeneratorMetadata,
)
from app.generation.revision_request_key import (
    GenerationRevisionRequestKey,
    create_generation_revision_request_key,
)


REVISION_RESERVATION_VERSION = (
    "generation_revision_reservation_v1"
)

REVISION_REQUESTED_ACTION = "regenerate_text"


@dataclass(frozen=True, slots=True)
class GenerationRevisionReservation:
    """Результат резервирования редакционной ревизии."""

    generation_revision_id: int
    batch_id: int
    source_generated_post_id: int
    review_action_id: int
    target_version_number: int
    revision_status: str
    request_key: str
    created_new: bool

    @property
    def should_call_model(self) -> bool:
        """Разрешает платный вызов только новой reservation."""

        return (
            self.created_new
            and self.revision_status == "reserved"
        )


def _encode_json(
    payload: Mapping[str, Any] | list[Any],
) -> str:
    """Преобразует JSON-совместимое значение для asyncpg."""

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

    return tuple(
        _normalize_required_text(
            issue,
            field_name=f"issues[{index}]",
        )
        for index, issue in enumerate(
            issues,
            start=1,
        )
    )


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


def _normalize_items(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> tuple[
    GenerationNewsItem,
    GenerationNewsItem,
    GenerationNewsItem,
]:
    """Проверяет factual-проекцию TOP-3."""

    if not isinstance(items, tuple):
        raise TypeError(
            "items должен быть tuple."
        )

    if len(items) != 3:
        raise ValueError(
            "Для ревизии требуется "
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

    news_ids: list[int] = []

    for item in items:
        news_ids.append(
            _normalize_positive_integer(
                item.news_id,
                field_name=(
                    f"news_id position={item.position}"
                ),
            )
        )

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

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Все три news_id должны быть "
            "уникальными."
        )

    return (
        items[0],
        items[1],
        items[2],
    )


def _factual_projection(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> tuple[
    tuple[int, int, str, str],
    tuple[int, int, str, str],
    tuple[int, int, str, str],
]:
    """Возвращает только разрешённые модели factual-поля."""

    normalized_items = _normalize_items(
        items
    )

    projection = tuple(
        (
            item.position,
            item.news_id,
            item.title.strip(),
            item.summary.strip(),
        )
        for item in normalized_items
    )

    return (
        projection[0],
        projection[1],
        projection[2],
    )


def _validate_request_key(
    *,
    request_key: GenerationRevisionRequestKey,
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
) -> None:
    """Сверяет request key со всеми текущими входами."""

    expected_key = (
        create_generation_revision_request_key(
            batch_id=batch_id,
            source_generated_post_id=(
                source_generated_post_id
            ),
            review_action_id=(
                review_action_id
            ),
            target_version_number=(
                target_version_number
            ),
            source_post_text=source_post_text,
            editorial_comment=(
                editorial_comment
            ),
            issues=issues,
            metadata=metadata,
            model_request=model_request,
            items=items,
            revision_prompt_version=(
                revision_prompt_version
            ),
        )
    )

    if request_key != expected_key:
        raise ValueError(
            "generation revision request_key "
            "не соответствует текущему посту, "
            "review action, TOP-3, модели "
            "или revision prompt."
        )


def _request_payload_from_key(
    request_key: GenerationRevisionRequestKey,
) -> dict[str, Any]:
    """Извлекает канонический payload из ключа."""

    try:
        payload = json.loads(
            request_key.canonical_json
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "request_key.canonical_json "
            "содержит некорректный JSON."
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "request_key.canonical_json должен "
            "содержать JSON-объект."
        )

    return payload


async def _find_existing_active_reservation(
    connection: asyncpg.Connection,
    *,
    request_key: str,
) -> asyncpg.Record | None:
    """Ищет активную или завершённую ревизию с этим ключом."""

    return await connection.fetchrow(
        """
        SELECT
            generation_revision_id,
            batch_id,
            source_generated_post_id,
            review_action_id,
            target_version_number,
            revision_request_key,
            request_key_version,
            revision_status,
            requested_action,
            editorial_comment,
            issues::text AS issues_json,
            model_name,
            generator_version,
            prompt_version,
            text_format,
            request_payload::text
                AS request_payload_json,
            generated_post_id
        FROM top3_news.generation_revision_requests
        WHERE revision_request_key = $1
          AND revision_status IN (
              'reserved',
              'completed'
          )
        FOR UPDATE
        """,
        request_key,
    )


def _validate_existing_reservation(
    record: asyncpg.Record,
    *,
    request_key: GenerationRevisionRequestKey,
    batch_id: int,
    source_generated_post_id: int,
    review_action_id: int,
    target_version_number: int,
    editorial_comment: str,
    issues: tuple[str, ...],
    metadata: OpenAIPostGeneratorMetadata,
    revision_prompt_version: str,
    request_payload: Mapping[str, Any],
) -> None:
    """Проверяет найденную reservation."""

    differences: list[str] = []

    expected_values = {
        "batch_id": batch_id,
        "source_generated_post_id": (
            source_generated_post_id
        ),
        "review_action_id": review_action_id,
        "target_version_number": (
            target_version_number
        ),
        "revision_request_key": (
            request_key.value
        ),
        "request_key_version": (
            request_key.version
        ),
        "requested_action": (
            REVISION_REQUESTED_ACTION
        ),
        "editorial_comment": (
            editorial_comment
        ),
        "model_name": metadata.model_name,
        "generator_version": (
            metadata.generator_version
        ),
        "prompt_version": (
            revision_prompt_version
        ),
        "text_format": (
            metadata.text_format
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

    try:
        actual_issues = tuple(
            json.loads(
                record["issues_json"]
            )
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ) as error:
        raise ValueError(
            "Существующая revision reservation "
            "содержит некорректный issues JSON."
        ) from error

    if actual_issues != issues:
        differences.append(
            "issues differ"
        )

    try:
        actual_payload = json.loads(
            record["request_payload_json"]
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ) as error:
        raise ValueError(
            "Существующая revision reservation "
            "содержит некорректный "
            "request_payload JSON."
        ) from error

    if actual_payload != request_payload:
        differences.append(
            "request_payload differs"
        )

    if (
        record["revision_status"]
        == "reserved"
        and record["generated_post_id"]
        is not None
    ):
        differences.append(
            "reserved revision содержит "
            "generated_post_id"
        )

    if (
        record["revision_status"]
        == "completed"
        and record["generated_post_id"]
        is None
    ):
        differences.append(
            "completed revision не содержит "
            "generated_post_id"
        )

    if differences:
        raise ValueError(
            "generation revision request_key "
            "уже существует с другими "
            "параметрами: "
            + "; ".join(differences)
        )


def _build_existing_result(
    record: asyncpg.Record,
) -> GenerationRevisionReservation:
    """Создаёт результат существующей reservation."""

    return GenerationRevisionReservation(
        generation_revision_id=int(
            record["generation_revision_id"]
        ),
        batch_id=int(
            record["batch_id"]
        ),
        source_generated_post_id=int(
            record["source_generated_post_id"]
        ),
        review_action_id=int(
            record["review_action_id"]
        ),
        target_version_number=int(
            record["target_version_number"]
        ),
        revision_status=(
            record["revision_status"]
        ),
        request_key=(
            record["revision_request_key"]
        ),
        created_new=False,
    )


async def _load_revision_context(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    source_generated_post_id: int,
    review_action_id: int,
) -> asyncpg.Record:
    """Блокирует batch, исходный пост и review action."""

    record = await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.batch_status,
            b.ranking_run_id,

            gp.generated_post_id,
            gp.batch_id AS source_batch_id,
            gp.version_number
                AS source_version_number,
            gp.post_status
                AS source_post_status,
            gp.post_text
                AS persisted_source_post_text,
            gp.text_format
                AS source_text_format,

            ra.review_action_id,
            ra.generated_post_id
                AS review_generated_post_id,
            ra.reviewer_type,
            ra.decision,
            ra.requested_action,
            ra.comment_text,
            ra.issues::text
                AS review_issues_json

        FROM top3_news.publication_batches AS b
        JOIN top3_news.generated_posts AS gp
          ON gp.generated_post_id = $2
        JOIN top3_news.review_actions AS ra
          ON ra.review_action_id = $3
        WHERE b.batch_id = $1
        FOR UPDATE OF b, gp, ra
        """,
        batch_id,
        source_generated_post_id,
        review_action_id,
    )

    if record is None:
        raise LookupError(
            "Не найден контекст редакционной "
            "ревизии: "
            f"batch_id={batch_id}, "
            "source_generated_post_id="
            f"{source_generated_post_id}, "
            f"review_action_id={review_action_id}"
        )

    return record


def _validate_revision_context(
    record: asyncpg.Record,
    *,
    batch_id: int,
    source_generated_post_id: int,
    review_action_id: int,
    target_version_number: int,
    source_post_text: str,
    editorial_comment: str,
    issues: tuple[str, ...],
    metadata: OpenAIPostGeneratorMetadata,
) -> int:
    """Проверяет допустимость новой reservation."""

    differences: list[str] = []

    expected_values = {
        "batch_id": batch_id,
        "batch_status": "awaiting_review",
        "generated_post_id": (
            source_generated_post_id
        ),
        "source_batch_id": batch_id,
        "source_post_status": (
            "awaiting_review"
        ),
        "review_action_id": (
            review_action_id
        ),
        "review_generated_post_id": (
            source_generated_post_id
        ),
        "reviewer_type": "human",
        "decision": "changes_required",
        "requested_action": (
            REVISION_REQUESTED_ACTION
        ),
        "persisted_source_post_text": (
            source_post_text
        ),
        "source_text_format": (
            metadata.text_format
        ),
        "comment_text": (
            editorial_comment
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

    source_version_number = int(
        record["source_version_number"]
    )

    if (
        target_version_number
        != source_version_number + 1
    ):
        differences.append(
            "target_version_number: "
            f"expected={source_version_number + 1!r}, "
            f"actual={target_version_number!r}"
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
        normalized_ranking_run_id = (
            int(ranking_run_id)
        )

        if normalized_ranking_run_id <= 0:
            differences.append(
                "ranking_run_id должен быть "
                "больше нуля"
            )

    try:
        review_issues = tuple(
            json.loads(
                record["review_issues_json"]
            )
        )
    except (
        json.JSONDecodeError,
        TypeError,
    ) as error:
        raise ValueError(
            "review_action содержит "
            "некорректный issues JSON."
        ) from error

    if review_issues != issues:
        differences.append(
            "review_action issues differ"
        )

    if differences:
        raise ValueError(
            "Текущий контекст не допускает "
            "резервирование ревизии: "
            + "; ".join(differences)
        )

    return normalized_ranking_run_id


async def _validate_current_top3(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> GenerationTop3Selection:
    """Сверяет factual-проекцию с текущим сохранённым TOP-3."""

    current_selection = (
        await _load_generation_top3(
            connection,
            ranking_run_id=ranking_run_id,
        )
    )

    expected_projection = (
        _factual_projection(items)
    )

    current_projection = (
        _factual_projection(
            current_selection.items
        )
    )

    if current_projection != expected_projection:
        raise ValueError(
            "Сохранённый TOP-3 изменился "
            "после подготовки revision request. "
            "Нужно сформировать запрос "
            "и request_key заново."
        )

    return current_selection


async def _validate_batch_items(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    expected_news_ids: tuple[int, int, int],
) -> None:
    """Сверяет состав publication batch с TOP-3."""

    records = await connection.fetch(
        """
        SELECT
            position,
            news_id
        FROM top3_news.batch_items
        WHERE batch_id = $1
        ORDER BY position
        """,
        batch_id,
    )

    positions = tuple(
        int(record["position"])
        for record in records
    )

    news_ids = tuple(
        int(record["news_id"])
        for record in records
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "publication batch должен "
            "содержать позиции 1, 2 и 3: "
            f"actual={positions!r}"
        )

    if news_ids != expected_news_ids:
        raise ValueError(
            "Состав publication batch "
            "не совпадает с сохранённым TOP-3: "
            f"expected={expected_news_ids!r}, "
            f"actual={news_ids!r}"
        )


async def _find_active_conflict(
    connection: asyncpg.Connection,
    *,
    review_action_id: int,
    batch_id: int,
    target_version_number: int,
) -> asyncpg.Record | None:
    """Ищет другой активный запрос для той же ревизии."""

    return await connection.fetchrow(
        """
        SELECT
            generation_revision_id,
            revision_request_key,
            revision_status,
            review_action_id,
            batch_id,
            target_version_number
        FROM top3_news.generation_revision_requests
        WHERE revision_status IN (
            'reserved',
            'completed'
        )
          AND (
              review_action_id = $1
              OR (
                  batch_id = $2
                  AND target_version_number = $3
              )
          )
        ORDER BY generation_revision_id DESC
        LIMIT 1
        FOR UPDATE
        """,
        review_action_id,
        batch_id,
        target_version_number,
    )


async def _validate_target_version_available(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    target_version_number: int,
) -> None:
    """Проверяет отсутствие уже созданной target-версии."""

    existing_generated_post_id = (
        await connection.fetchval(
            """
            SELECT generated_post_id
            FROM top3_news.generated_posts
            WHERE batch_id = $1
              AND version_number = $2
            """,
            batch_id,
            target_version_number,
        )
    )

    if existing_generated_post_id is not None:
        raise ValueError(
            "Целевая версия generated_posts "
            "уже существует: "
            f"batch_id={batch_id}, "
            f"version_number="
            f"{target_version_number}, "
            f"generated_post_id="
            f"{existing_generated_post_id}"
        )


async def reserve_generation_revision(
    pool: asyncpg.Pool,
    *,
    request_key: GenerationRevisionRequestKey,
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
) -> GenerationRevisionReservation:
    """
    Резервирует редакционную ревизию до вызова OpenAI.

    Reservation не меняет статусы publication_batch
    и generated_posts.

    Только процесс, создавший новую запись со статусом
    reserved, получает should_call_model=True.

    Активный или завершённый повтор с тем же ключом
    не выполняет платный API-запрос.

    После failed тот же детерминированный ключ может
    быть зарезервирован повторно новой строкой.
    """

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

    normalized_metadata = (
        _normalize_metadata(metadata)
    )

    normalized_items = _normalize_items(
        items
    )

    normalized_revision_prompt_version = (
        _normalize_required_text(
            revision_prompt_version,
            field_name=(
                "revision_prompt_version"
            ),
        )
    )

    _validate_request_key(
        request_key=request_key,
        batch_id=normalized_batch_id,
        source_generated_post_id=(
            normalized_source_generated_post_id
        ),
        review_action_id=(
            normalized_review_action_id
        ),
        target_version_number=(
            normalized_target_version_number
        ),
        source_post_text=(
            normalized_source_post_text
        ),
        editorial_comment=(
            normalized_editorial_comment
        ),
        issues=normalized_issues,
        metadata=normalized_metadata,
        model_request=model_request,
        items=normalized_items,
        revision_prompt_version=(
            normalized_revision_prompt_version
        ),
    )

    request_payload = (
        _request_payload_from_key(
            request_key
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
                normalized_review_action_id,
            )

            existing_record = (
                await _find_existing_active_reservation(
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
                    batch_id=normalized_batch_id,
                    source_generated_post_id=(
                        normalized_source_generated_post_id
                    ),
                    review_action_id=(
                        normalized_review_action_id
                    ),
                    target_version_number=(
                        normalized_target_version_number
                    ),
                    editorial_comment=(
                        normalized_editorial_comment
                    ),
                    issues=normalized_issues,
                    metadata=normalized_metadata,
                    revision_prompt_version=(
                        normalized_revision_prompt_version
                    ),
                    request_payload=(
                        request_payload
                    ),
                )

                if (
                    existing_record[
                        "revision_status"
                    ]
                    == "reserved"
                ):
                    context_record = (
                        await _load_revision_context(
                            connection,
                            batch_id=(
                                normalized_batch_id
                            ),
                            source_generated_post_id=(
                                normalized_source_generated_post_id
                            ),
                            review_action_id=(
                                normalized_review_action_id
                            ),
                        )
                    )

                    ranking_run_id = (
                        _validate_revision_context(
                            context_record,
                            batch_id=(
                                normalized_batch_id
                            ),
                            source_generated_post_id=(
                                normalized_source_generated_post_id
                            ),
                            review_action_id=(
                                normalized_review_action_id
                            ),
                            target_version_number=(
                                normalized_target_version_number
                            ),
                            source_post_text=(
                                normalized_source_post_text
                            ),
                            editorial_comment=(
                                normalized_editorial_comment
                            ),
                            issues=normalized_issues,
                            metadata=(
                                normalized_metadata
                            ),
                        )
                    )

                    selection = (
                        await _validate_current_top3(
                            connection,
                            ranking_run_id=(
                                ranking_run_id
                            ),
                            items=normalized_items,
                        )
                    )

                    await _validate_batch_items(
                        connection,
                        batch_id=(
                            normalized_batch_id
                        ),
                        expected_news_ids=(
                            selection.news_ids
                        ),
                    )

                return _build_existing_result(
                    existing_record
                )

            context_record = (
                await _load_revision_context(
                    connection,
                    batch_id=(
                        normalized_batch_id
                    ),
                    source_generated_post_id=(
                        normalized_source_generated_post_id
                    ),
                    review_action_id=(
                        normalized_review_action_id
                    ),
                )
            )

            ranking_run_id = (
                _validate_revision_context(
                    context_record,
                    batch_id=(
                        normalized_batch_id
                    ),
                    source_generated_post_id=(
                        normalized_source_generated_post_id
                    ),
                    review_action_id=(
                        normalized_review_action_id
                    ),
                    target_version_number=(
                        normalized_target_version_number
                    ),
                    source_post_text=(
                        normalized_source_post_text
                    ),
                    editorial_comment=(
                        normalized_editorial_comment
                    ),
                    issues=normalized_issues,
                    metadata=normalized_metadata,
                )
            )

            selection = (
                await _validate_current_top3(
                    connection,
                    ranking_run_id=ranking_run_id,
                    items=normalized_items,
                )
            )

            await _validate_batch_items(
                connection,
                batch_id=normalized_batch_id,
                expected_news_ids=(
                    selection.news_ids
                ),
            )

            conflict_record = (
                await _find_active_conflict(
                    connection,
                    review_action_id=(
                        normalized_review_action_id
                    ),
                    batch_id=(
                        normalized_batch_id
                    ),
                    target_version_number=(
                        normalized_target_version_number
                    ),
                )
            )

            if conflict_record is not None:
                raise ValueError(
                    "Для review_action или "
                    "целевой версии уже существует "
                    "другая активная revision "
                    "reservation: "
                    "generation_revision_id="
                    f"{conflict_record['generation_revision_id']}, "
                    "revision_status="
                    f"{conflict_record['revision_status']!r}"
                )

            await _validate_target_version_available(
                connection,
                batch_id=normalized_batch_id,
                target_version_number=(
                    normalized_target_version_number
                ),
            )

            generation_revision_id = (
                await connection.fetchval(
                    """
                    INSERT INTO
                        top3_news.generation_revision_requests (
                            batch_id,
                            source_generated_post_id,
                            review_action_id,
                            target_version_number,
                            revision_request_key,
                            request_key_version,
                            revision_status,
                            requested_action,
                            editorial_comment,
                            issues,
                            model_name,
                            generator_version,
                            prompt_version,
                            text_format,
                            request_payload
                        )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        'reserved',
                        'regenerate_text',
                        $7,
                        $8::jsonb,
                        $9,
                        $10,
                        $11,
                        $12,
                        $13::jsonb
                    )
                    RETURNING generation_revision_id
                    """,
                    normalized_batch_id,
                    normalized_source_generated_post_id,
                    normalized_review_action_id,
                    normalized_target_version_number,
                    request_key.value,
                    request_key.version,
                    normalized_editorial_comment,
                    _encode_json(
                        list(normalized_issues)
                    ),
                    normalized_metadata.model_name,
                    normalized_metadata.generator_version,
                    normalized_revision_prompt_version,
                    normalized_metadata.text_format,
                    _encode_json(
                        request_payload
                    ),
                )
            )

    return GenerationRevisionReservation(
        generation_revision_id=int(
            generation_revision_id
        ),
        batch_id=normalized_batch_id,
        source_generated_post_id=(
            normalized_source_generated_post_id
        ),
        review_action_id=(
            normalized_review_action_id
        ),
        target_version_number=(
            normalized_target_version_number
        ),
        revision_status="reserved",
        request_key=request_key.value,
        created_new=True,
    )