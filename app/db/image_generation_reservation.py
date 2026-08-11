from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import asyncpg

from app.db.generation_selection import (
    _load_generation_top3,
)
from app.generation.image_generator import (
    ImageGenerationNewsItem,
    ImageModelRequest,
    ImageQuality,
    OpenAIImageGeneratorMetadata,
)
from app.generation.image_request_key import (
    ImageRequestKey,
    ImageRequestKind,
    create_image_request_key,
)


IMAGE_GENERATION_RESERVATION_VERSION = (
    "image_generation_reservation_v1"
)

_IMAGE_SIZE_PATTERN = re.compile(
    r"^(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)$"
)


@dataclass(frozen=True, slots=True)
class ImageGenerationReservation:
    """Результат резервирования image-generation."""

    image_generation_id: int
    batch_id: int
    generated_post_id: int
    ranking_run_id: int
    request_kind: ImageRequestKind
    review_action_id: int | None
    image_status: str
    request_key: str
    created_new: bool

    @property
    def should_call_model(self) -> bool:
        """Разрешает платный вызов только новой reservation."""

        return (
            self.created_new
            and self.image_status == "reserved"
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


def _normalize_issues(
    issues: tuple[str, ...],
) -> tuple[str, ...]:
    """Проверяет редакционные замечания."""

    if not isinstance(issues, tuple):
        raise TypeError(
            "issues должен быть tuple."
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


def _normalize_request_context(
    *,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
    editorial_comment: str | None,
    issues: tuple[str, ...],
) -> tuple[
    int | None,
    str | None,
    tuple[str, ...],
]:
    """Проверяет initial/regenerate контекст."""

    normalized_issues = _normalize_issues(
        issues
    )

    if request_kind == "initial":
        if review_action_id is not None:
            raise ValueError(
                "Для initial review_action_id "
                "должен быть None."
            )

        if editorial_comment is not None:
            raise ValueError(
                "Для initial editorial_comment "
                "должен быть None."
            )

        if normalized_issues:
            raise ValueError(
                "Для initial issues должен "
                "быть пустым."
            )

        return None, None, ()

    if review_action_id is None:
        raise ValueError(
            "Для regenerate требуется "
            "review_action_id."
        )

    normalized_review_action_id = (
        _normalize_positive_integer(
            review_action_id,
            field_name="review_action_id",
        )
    )

    if editorial_comment is None:
        raise ValueError(
            "Для regenerate требуется "
            "editorial_comment."
        )

    normalized_editorial_comment = (
        _normalize_required_text(
            editorial_comment,
            field_name="editorial_comment",
        )
    )

    if not normalized_issues:
        raise ValueError(
            "Для regenerate issues не может "
            "быть пустым."
        )

    return (
        normalized_review_action_id,
        normalized_editorial_comment,
        normalized_issues,
    )


def _normalize_metadata(
    metadata: OpenAIImageGeneratorMetadata,
) -> OpenAIImageGeneratorMetadata:
    """Проверяет метаданные image-generator."""

    return OpenAIImageGeneratorMetadata(
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
    )


def _normalize_image_size(
    value: str,
) -> str:
    """Проверяет размер и проектное соотношение 2:3."""

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


def _normalize_model_request(
    request: ImageModelRequest,
) -> ImageModelRequest:
    """Проверяет точные параметры Image API."""

    normalized_model = (
        _normalize_required_text(
            request.model,
            field_name="model_request.model",
        )
    )

    normalized_prompt = (
        _normalize_required_text(
            request.prompt,
            field_name="model_request.prompt",
        )
    )

    normalized_size = _normalize_image_size(
        request.size
    )

    if request.quality not in {
        "low",
        "medium",
        "high",
    }:
        raise ValueError(
            "model_request.quality должен быть "
            "low, medium или high."
        )

    if request.output_format != "png":
        raise ValueError(
            "model_request.output_format "
            "должен быть 'png'."
        )

    if request.background != "opaque":
        raise ValueError(
            "model_request.background "
            "должен быть 'opaque'."
        )

    if request.moderation != "auto":
        raise ValueError(
            "model_request.moderation "
            "должен быть 'auto'."
        )

    if isinstance(request.n, bool):
        raise TypeError(
            "model_request.n не может быть bool."
        )

    if not isinstance(request.n, int):
        raise TypeError(
            "model_request.n должен быть int."
        )

    if request.n != 1:
        raise ValueError(
            "Для текущего image pipeline "
            "model_request.n должен быть равен 1."
        )

    return ImageModelRequest(
        model=normalized_model,
        prompt=normalized_prompt,
        size=normalized_size,
        quality=request.quality,
        output_format="png",
        background="opaque",
        moderation="auto",
        n=1,
    )


def _normalize_items(
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """Проверяет factual-проекцию TOP-3."""

    if not isinstance(items, tuple):
        raise TypeError(
            "items должен быть tuple."
        )

    if len(items) != 3:
        raise ValueError(
            "Для image-generation требуется "
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

    normalized_items: list[
        ImageGenerationNewsItem
    ] = []

    for item in items:
        news_id = _normalize_positive_integer(
            item.news_id,
            field_name=(
                f"news_id position={item.position}"
            ),
        )

        title = _normalize_required_text(
            item.title,
            field_name=(
                f"title news_id={news_id}"
            ),
        )

        summary = _normalize_required_text(
            item.summary,
            field_name=(
                f"summary news_id={news_id}"
            ),
        )

        news_ids.append(news_id)

        normalized_items.append(
            ImageGenerationNewsItem(
                position=item.position,
                news_id=news_id,
                title=title,
                summary=summary,
            )
        )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Все три news_id должны быть "
            "уникальными."
        )

    return (
        normalized_items[0],
        normalized_items[1],
        normalized_items[2],
    )


def _factual_projection(
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> tuple[
    tuple[int, int, str, str],
    tuple[int, int, str, str],
    tuple[int, int, str, str],
]:
    """Возвращает factual-поля image input."""

    normalized_items = _normalize_items(
        items
    )

    projection = tuple(
        (
            item.position,
            item.news_id,
            item.title,
            item.summary,
        )
        for item in normalized_items
    )

    return (
        projection[0],
        projection[1],
        projection[2],
    )


def _selection_factual_projection(
    selection: Any,
) -> tuple[
    tuple[int, int, str, str],
    tuple[int, int, str, str],
    tuple[int, int, str, str],
]:
    """Возвращает image factual-проекцию сохранённого TOP-3."""

    if len(selection.items) != 3:
        raise ValueError(
            "Сохранённый TOP-3 должен "
            "содержать ровно три новости."
        )

    items = tuple(
        ImageGenerationNewsItem(
            position=item.position,
            news_id=item.news_id,
            title=item.title,
            summary=item.summary,
        )
        for item in selection.items
    )

    return _factual_projection(
        items
    )


def _validate_request_key(
    *,
    request_key: ImageRequestKey,
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
) -> None:
    """Сверяет request key со всеми текущими входами."""

    expected_key = create_image_request_key(
        batch_id=batch_id,
        ranking_run_id=ranking_run_id,
        request_kind=request_kind,
        review_action_id=review_action_id,
        metadata=metadata,
        model_request=model_request,
        items=items,
    )

    if request_key != expected_key:
        raise ValueError(
            "image request_key не соответствует "
            "текущему batch, TOP-3, модели, "
            "промпту или параметрам изображения."
        )


def _request_payload_from_key(
    request_key: ImageRequestKey,
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
    """Ищет active/completed image request с этим ключом."""

    return await connection.fetchrow(
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
            igr.editorial_comment,
            igr.issues::text AS issues_json,
            igr.model_name,
            igr.generator_version,
            igr.prompt_version,
            igr.image_size,
            igr.image_quality,
            igr.output_format,
            igr.background,
            igr.moderation,
            igr.image_count,
            igr.request_payload::text
                AS request_payload_json,
            b.ranking_run_id
        FROM top3_news.image_generation_requests AS igr
        JOIN top3_news.publication_batches AS b
          ON b.batch_id = igr.batch_id
        WHERE igr.image_request_key = $1
          AND igr.image_status IN (
              'reserved',
              'completed'
          )
        FOR UPDATE OF igr, b
        """,
        request_key,
    )


def _validate_existing_reservation(
    record: asyncpg.Record,
    *,
    request_key: ImageRequestKey,
    batch_id: int,
    generated_post_id: int,
    ranking_run_id: int,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
    editorial_comment: str | None,
    issues: tuple[str, ...],
    metadata: OpenAIImageGeneratorMetadata,
    model_request: ImageModelRequest,
    request_payload: Mapping[str, Any],
) -> None:
    """Проверяет найденную image reservation."""

    differences: list[str] = []

    expected_values = {
        "batch_id": batch_id,
        "ranking_run_id": ranking_run_id,
        "review_action_id": review_action_id,
        "image_request_key": request_key.value,
        "request_key_version": request_key.version,
        "request_kind": request_kind,
        "editorial_comment": editorial_comment,
        "model_name": metadata.model_name,
        "generator_version": (
            metadata.generator_version
        ),
        "prompt_version": (
            metadata.prompt_version
        ),
        "image_size": model_request.size,
        "image_quality": (
            model_request.quality
        ),
        "output_format": (
            model_request.output_format
        ),
        "background": (
            model_request.background
        ),
        "moderation": (
            model_request.moderation
        ),
        "image_count": model_request.n,
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

    if (
        request_kind == "regenerate"
        or record["image_status"] == "reserved"
    ):
        if (
            int(record["generated_post_id"])
            != generated_post_id
        ):
            differences.append(
                "generated_post_id: "
                f"expected={generated_post_id!r}, "
                "actual="
                f"{int(record['generated_post_id'])!r}"
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
            "Существующая image reservation "
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
            "Существующая image reservation "
            "содержит некорректный "
            "request_payload JSON."
        ) from error

    if actual_payload != request_payload:
        differences.append(
            "request_payload differs"
        )

    if differences:
        raise ValueError(
            "image request_key уже существует "
            "с другими параметрами: "
            + "; ".join(differences)
        )


def _build_existing_result(
    record: asyncpg.Record,
) -> ImageGenerationReservation:
    """Создаёт результат существующей reservation."""

    return ImageGenerationReservation(
        image_generation_id=int(
            record["image_generation_id"]
        ),
        batch_id=int(
            record["batch_id"]
        ),
        generated_post_id=int(
            record["generated_post_id"]
        ),
        ranking_run_id=int(
            record["ranking_run_id"]
        ),
        request_kind=record["request_kind"],
        review_action_id=(
            int(record["review_action_id"])
            if record["review_action_id"] is not None
            else None
        ),
        image_status=record["image_status"],
        request_key=record[
            "image_request_key"
        ],
        created_new=False,
    )


def _validate_image_fields_empty(
    record: asyncpg.Record,
) -> None:
    """Проверяет отсутствие первичной картинки."""

    image_fields = {
        "image_path": record["image_path"],
        "image_sha256": (
            record["image_sha256"]
        ),
        "image_prompt": (
            record["image_prompt"]
        ),
        "image_model_name": (
            record["image_model_name"]
        ),
        "image_prompt_version": (
            record["image_prompt_version"]
        ),
    }

    present_fields = [
        field_name
        for field_name, value in image_fields.items()
        if value is not None
    ]

    if present_fields:
        raise ValueError(
            "Для initial image-поля "
            "generated_posts должны быть NULL: "
            + ", ".join(present_fields)
        )


def _validate_image_fields_complete(
    record: asyncpg.Record,
) -> None:
    """Проверяет наличие текущей картинки."""

    image_fields = {
        "image_path": record["image_path"],
        "image_sha256": (
            record["image_sha256"]
        ),
        "image_prompt": (
            record["image_prompt"]
        ),
        "image_model_name": (
            record["image_model_name"]
        ),
        "image_prompt_version": (
            record["image_prompt_version"]
        ),
    }

    missing_fields = [
        field_name
        for field_name, value in image_fields.items()
        if value is None
    ]

    if missing_fields:
        raise ValueError(
            "Для regenerate текущая картинка "
            "должна быть полностью сохранена. "
            "Отсутствуют поля: "
            + ", ".join(missing_fields)
        )

    for field_name in (
        "image_path",
        "image_prompt",
        "image_model_name",
        "image_prompt_version",
    ):
        value = record[field_name]

        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} должен быть строкой."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name} не может быть пустым."
            )

    image_sha256 = record["image_sha256"]

    if not isinstance(image_sha256, str):
        raise TypeError(
            "image_sha256 должен быть строкой."
        )

    if not re.fullmatch(
        r"^[0-9a-f]{64}$",
        image_sha256,
    ):
        raise ValueError(
            "image_sha256 должен быть SHA-256 "
            "в нижнем регистре."
        )


async def _load_initial_context(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    generated_post_id: int,
) -> asyncpg.Record:
    """Блокирует контекст первичной картинки."""

    record = await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.batch_status,
            b.ranking_run_id,

            gp.generated_post_id,
            gp.batch_id AS post_batch_id,
            gp.post_status,
            gp.image_path,
            gp.image_sha256,
            gp.image_prompt,
            gp.image_model_name,
            gp.image_prompt_version

        FROM top3_news.publication_batches AS b
        JOIN top3_news.generated_posts AS gp
          ON gp.batch_id = b.batch_id
        WHERE b.batch_id = $1
          AND gp.generated_post_id = $2
        FOR UPDATE OF b, gp
        """,
        batch_id,
        generated_post_id,
    )

    if record is None:
        raise LookupError(
            "Не найден контекст initial "
            "image-generation: "
            f"batch_id={batch_id}, "
            f"generated_post_id="
            f"{generated_post_id}"
        )

    return record


def _validate_initial_context(
    record: asyncpg.Record,
    *,
    batch_id: int,
    generated_post_id: int,
    ranking_run_id: int,
) -> None:
    """Проверяет допустимость initial reservation."""

    differences: list[str] = []

    expected_values = {
        "batch_id": batch_id,
        "batch_status": "awaiting_review",
        "generated_post_id": (
            generated_post_id
        ),
        "post_batch_id": batch_id,
        "post_status": "awaiting_review",
        "ranking_run_id": ranking_run_id,
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
            "Текущий контекст не допускает "
            "initial image-generation: "
            + "; ".join(differences)
        )

    _validate_image_fields_empty(
        record
    )


async def _load_regenerate_context(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    generated_post_id: int,
    review_action_id: int,
) -> asyncpg.Record:
    """Блокирует контекст редакционной перегенерации."""

    record = await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.batch_status,
            b.ranking_run_id,

            gp.generated_post_id,
            gp.batch_id AS post_batch_id,
            gp.post_status,
            gp.image_path,
            gp.image_sha256,
            gp.image_prompt,
            gp.image_model_name,
            gp.image_prompt_version,

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
          ON gp.batch_id = b.batch_id
        JOIN top3_news.review_actions AS ra
          ON ra.review_action_id = $3
        WHERE b.batch_id = $1
          AND gp.generated_post_id = $2
        FOR UPDATE OF b, gp, ra
        """,
        batch_id,
        generated_post_id,
        review_action_id,
    )

    if record is None:
        raise LookupError(
            "Не найден контекст regenerate "
            "image-generation: "
            f"batch_id={batch_id}, "
            f"generated_post_id="
            f"{generated_post_id}, "
            f"review_action_id="
            f"{review_action_id}"
        )

    return record


def _validate_regenerate_context(
    record: asyncpg.Record,
    *,
    batch_id: int,
    generated_post_id: int,
    ranking_run_id: int,
    review_action_id: int,
    editorial_comment: str,
    issues: tuple[str, ...],
) -> None:
    """Проверяет regenerate_image review action."""

    differences: list[str] = []

    expected_values = {
        "batch_id": batch_id,
        "batch_status": "awaiting_review",
        "ranking_run_id": ranking_run_id,
        "generated_post_id": (
            generated_post_id
        ),
        "post_batch_id": batch_id,
        "post_status": "awaiting_review",
        "review_action_id": (
            review_action_id
        ),
        "review_generated_post_id": (
            generated_post_id
        ),
        "reviewer_type": "human",
        "decision": "changes_required",
        "requested_action": (
            "regenerate_image"
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
            "regenerate image-generation: "
            + "; ".join(differences)
        )

    _validate_image_fields_complete(
        record
    )


async def _validate_current_top3(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> tuple[int, int, int]:
    """Сверяет image input с сохранённым TOP-3."""

    current_selection = (
        await _load_generation_top3(
            connection,
            ranking_run_id=ranking_run_id,
        )
    )

    if current_selection.run_status != "completed":
        raise ValueError(
            "ranking run должен иметь "
            "статус completed."
        )

    expected_projection = (
        _factual_projection(
            items
        )
    )

    current_projection = (
        _selection_factual_projection(
            current_selection
        )
    )

    if current_projection != expected_projection:
        raise ValueError(
            "Сохранённый TOP-3 изменился "
            "после подготовки image request. "
            "Нужно сформировать prompt "
            "и request_key заново."
        )

    news_ids = tuple(
        item.news_id
        for item in current_selection.items
    )

    if len(news_ids) != 3:
        raise ValueError(
            "Сохранённый TOP-3 должен "
            "содержать три news_id."
        )

    return (
        int(news_ids[0]),
        int(news_ids[1]),
        int(news_ids[2]),
    )


async def _validate_batch_items(
    connection: asyncpg.Connection,
    *,
    batch_id: int,
    expected_news_ids: tuple[int, int, int],
) -> None:
    """Сверяет publication batch с TOP-3."""

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
    request_key: str,
    batch_id: int,
    generated_post_id: int,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
) -> asyncpg.Record | None:
    """Ищет другую конфликтующую image reservation."""

    return await connection.fetchrow(
        """
        SELECT
            image_generation_id,
            image_request_key,
            image_status,
            request_kind,
            batch_id,
            generated_post_id,
            review_action_id
        FROM top3_news.image_generation_requests
        WHERE image_request_key <> $1
          AND image_status IN (
              'reserved',
              'completed'
          )
          AND (
              (
                  $4 = 'initial'
                  AND request_kind = 'initial'
                  AND batch_id = $2
              )
              OR
              (
                  $4 = 'regenerate'
                  AND request_kind = 'regenerate'
                  AND review_action_id = $5
              )
              OR
              (
                  image_status = 'reserved'
                  AND generated_post_id = $3
              )
          )
        ORDER BY image_generation_id DESC
        LIMIT 1
        FOR UPDATE
        """,
        request_key,
        batch_id,
        generated_post_id,
        request_kind,
        review_action_id,
    )


async def reserve_image_generation(
    pool: asyncpg.Pool,
    *,
    request_key: ImageRequestKey,
    batch_id: int,
    generated_post_id: int,
    ranking_run_id: int,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
    editorial_comment: str | None,
    issues: tuple[str, ...],
    metadata: OpenAIImageGeneratorMetadata,
    model_request: ImageModelRequest,
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> ImageGenerationReservation:
    """
    Резервирует image-generation до платного Image API.

    Reservation не меняет publication_batch
    и generated_posts.

    Только новая строка со статусом reserved
    получает should_call_model=True.

    reserved/completed с тем же ключом блокирует
    повторный платный вызов.

    После failed тот же детерминированный ключ
    может быть зарезервирован повторно новой строкой.
    """

    normalized_batch_id = (
        _normalize_positive_integer(
            batch_id,
            field_name="batch_id",
        )
    )

    normalized_generated_post_id = (
        _normalize_positive_integer(
            generated_post_id,
            field_name="generated_post_id",
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

    (
        normalized_review_action_id,
        normalized_editorial_comment,
        normalized_issues,
    ) = _normalize_request_context(
        request_kind=(
            normalized_request_kind
        ),
        review_action_id=review_action_id,
        editorial_comment=editorial_comment,
        issues=issues,
    )

    normalized_metadata = (
        _normalize_metadata(
            metadata
        )
    )

    normalized_model_request = (
        _normalize_model_request(
            model_request
        )
    )

    if (
        normalized_metadata.model_name
        != normalized_model_request.model
    ):
        raise ValueError(
            "metadata.model_name не совпадает "
            "с model_request.model."
        )

    normalized_items = _normalize_items(
        items
    )

    _validate_request_key(
        request_key=request_key,
        batch_id=normalized_batch_id,
        ranking_run_id=(
            normalized_ranking_run_id
        ),
        request_kind=(
            normalized_request_kind
        ),
        review_action_id=(
            normalized_review_action_id
        ),
        metadata=normalized_metadata,
        model_request=(
            normalized_model_request
        ),
        items=normalized_items,
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
                normalized_generated_post_id,
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
                    batch_id=(
                        normalized_batch_id
                    ),
                    generated_post_id=(
                        normalized_generated_post_id
                    ),
                    ranking_run_id=(
                        normalized_ranking_run_id
                    ),
                    request_kind=(
                        normalized_request_kind
                    ),
                    review_action_id=(
                        normalized_review_action_id
                    ),
                    editorial_comment=(
                        normalized_editorial_comment
                    ),
                    issues=normalized_issues,
                    metadata=normalized_metadata,
                    model_request=(
                        normalized_model_request
                    ),
                    request_payload=(
                        request_payload
                    ),
                )

                if (
                    existing_record[
                        "image_status"
                    ]
                    == "reserved"
                ):
                    if (
                        normalized_request_kind
                        == "initial"
                    ):
                        context_record = (
                            await _load_initial_context(
                                connection,
                                batch_id=(
                                    normalized_batch_id
                                ),
                                generated_post_id=(
                                    normalized_generated_post_id
                                ),
                            )
                        )

                        _validate_initial_context(
                            context_record,
                            batch_id=(
                                normalized_batch_id
                            ),
                            generated_post_id=(
                                normalized_generated_post_id
                            ),
                            ranking_run_id=(
                                normalized_ranking_run_id
                            ),
                        )

                    else:
                        if (
                            normalized_review_action_id
                            is None
                            or normalized_editorial_comment
                            is None
                        ):
                            raise RuntimeError(
                                "Нормализованный regenerate "
                                "контекст неполон."
                            )

                        context_record = (
                            await _load_regenerate_context(
                                connection,
                                batch_id=(
                                    normalized_batch_id
                                ),
                                generated_post_id=(
                                    normalized_generated_post_id
                                ),
                                review_action_id=(
                                    normalized_review_action_id
                                ),
                            )
                        )

                        _validate_regenerate_context(
                            context_record,
                            batch_id=(
                                normalized_batch_id
                            ),
                            generated_post_id=(
                                normalized_generated_post_id
                            ),
                            ranking_run_id=(
                                normalized_ranking_run_id
                            ),
                            review_action_id=(
                                normalized_review_action_id
                            ),
                            editorial_comment=(
                                normalized_editorial_comment
                            ),
                            issues=(
                                normalized_issues
                            ),
                        )

                    expected_news_ids = (
                        await _validate_current_top3(
                            connection,
                            ranking_run_id=(
                                normalized_ranking_run_id
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
                            expected_news_ids
                        ),
                    )

                return _build_existing_result(
                    existing_record
                )

            if (
                normalized_request_kind
                == "initial"
            ):
                context_record = (
                    await _load_initial_context(
                        connection,
                        batch_id=(
                            normalized_batch_id
                        ),
                        generated_post_id=(
                            normalized_generated_post_id
                        ),
                    )
                )

                _validate_initial_context(
                    context_record,
                    batch_id=normalized_batch_id,
                    generated_post_id=(
                        normalized_generated_post_id
                    ),
                    ranking_run_id=(
                        normalized_ranking_run_id
                    ),
                )

            else:
                if (
                    normalized_review_action_id
                    is None
                    or normalized_editorial_comment
                    is None
                ):
                    raise RuntimeError(
                        "Нормализованный regenerate "
                        "контекст неполон."
                    )

                context_record = (
                    await _load_regenerate_context(
                        connection,
                        batch_id=(
                            normalized_batch_id
                        ),
                        generated_post_id=(
                            normalized_generated_post_id
                        ),
                        review_action_id=(
                            normalized_review_action_id
                        ),
                    )
                )

                _validate_regenerate_context(
                    context_record,
                    batch_id=normalized_batch_id,
                    generated_post_id=(
                        normalized_generated_post_id
                    ),
                    ranking_run_id=(
                        normalized_ranking_run_id
                    ),
                    review_action_id=(
                        normalized_review_action_id
                    ),
                    editorial_comment=(
                        normalized_editorial_comment
                    ),
                    issues=normalized_issues,
                )

            expected_news_ids = (
                await _validate_current_top3(
                    connection,
                    ranking_run_id=(
                        normalized_ranking_run_id
                    ),
                    items=normalized_items,
                )
            )

            await _validate_batch_items(
                connection,
                batch_id=normalized_batch_id,
                expected_news_ids=(
                    expected_news_ids
                ),
            )

            conflict_record = (
                await _find_active_conflict(
                    connection,
                    request_key=(
                        request_key.value
                    ),
                    batch_id=normalized_batch_id,
                    generated_post_id=(
                        normalized_generated_post_id
                    ),
                    request_kind=(
                        normalized_request_kind
                    ),
                    review_action_id=(
                        normalized_review_action_id
                    ),
                )
            )

            if conflict_record is not None:
                raise ValueError(
                    "Для текущего image-контекста "
                    "уже существует другая "
                    "active reservation: "
                    "image_generation_id="
                    f"{conflict_record['image_generation_id']}, "
                    "image_status="
                    f"{conflict_record['image_status']!r}, "
                    "request_kind="
                    f"{conflict_record['request_kind']!r}"
                )

            image_generation_id = (
                await connection.fetchval(
                    """
                    INSERT INTO
                        top3_news.image_generation_requests (
                            batch_id,
                            generated_post_id,
                            review_action_id,
                            image_request_key,
                            request_key_version,
                            image_status,
                            request_kind,
                            editorial_comment,
                            issues,
                            model_name,
                            generator_version,
                            prompt_version,
                            image_size,
                            image_quality,
                            output_format,
                            background,
                            moderation,
                            image_count,
                            request_payload
                        )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        'reserved',
                        $6,
                        $7,
                        $8::jsonb,
                        $9,
                        $10,
                        $11,
                        $12,
                        $13,
                        $14,
                        $15,
                        $16,
                        $17,
                        $18::jsonb
                    )
                    RETURNING image_generation_id
                    """,
                    normalized_batch_id,
                    normalized_generated_post_id,
                    normalized_review_action_id,
                    request_key.value,
                    request_key.version,
                    normalized_request_kind,
                    normalized_editorial_comment,
                    _encode_json(
                        list(normalized_issues)
                    ),
                    normalized_metadata.model_name,
                    normalized_metadata.generator_version,
                    normalized_metadata.prompt_version,
                    normalized_model_request.size,
                    normalized_model_request.quality,
                    normalized_model_request.output_format,
                    normalized_model_request.background,
                    normalized_model_request.moderation,
                    normalized_model_request.n,
                    _encode_json(
                        request_payload
                    ),
                )
            )

    return ImageGenerationReservation(
        image_generation_id=int(
            image_generation_id
        ),
        batch_id=normalized_batch_id,
        generated_post_id=(
            normalized_generated_post_id
        ),
        ranking_run_id=(
            normalized_ranking_run_id
        ),
        request_kind=(
            normalized_request_kind
        ),
        review_action_id=(
            normalized_review_action_id
        ),
        image_status="reserved",
        request_key=request_key.value,
        created_new=True,
    )