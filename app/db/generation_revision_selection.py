from dataclasses import dataclass, replace
import json
from urllib.parse import urlsplit

import asyncpg

from app.db.generation_selection import (
    GenerationTop3Selection,
    _load_generation_batch_selection,
)
from app.generation.openai_generator import (
    GenerationNewsItem,
)


REVISION_REQUESTED_ACTION = "regenerate_text"


@dataclass(frozen=True, slots=True)
class GenerationRevisionSelection:
    """Контекст редакционной ревизии поста."""

    review_action_id: int
    batch_id: int
    source_generated_post_id: int
    source_version_number: int
    target_version_number: int
    source_post_text: str
    source_text_format: str
    editorial_comment: str
    issues: tuple[str, ...]
    selection: GenerationTop3Selection

    @property
    def ranking_run_id(self) -> int:
        """Возвращает ranking_run исходного выпуска."""

        return self.selection.ranking_run_id

    @property
    def items(
        self,
    ) -> tuple[
        GenerationNewsItem,
        GenerationNewsItem,
        GenerationNewsItem,
    ]:
        """Возвращает сохранённый TOP-3."""

        return self.selection.items

    @property
    def news_ids(self) -> tuple[int, int, int]:
        """Возвращает news_id в порядке TOP-3."""

        return self.selection.news_ids


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


def _decode_issues(
    value: str,
) -> tuple[str, ...]:
    """Проверяет issues из review_action."""

    if not isinstance(value, str):
        raise ValueError(
            "review_action issues отсутствует."
        )

    try:
        decoded_value = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            "review_action содержит "
            "некорректный issues JSON."
        ) from error

    if not isinstance(decoded_value, list):
        raise ValueError(
            "review_action issues должен "
            "быть JSON-массивом."
        )

    if not decoded_value:
        raise ValueError(
            "review_action issues не может "
            "быть пустым."
        )

    normalized_issues: list[str] = []

    for index, issue in enumerate(
        decoded_value,
        start=1,
    ):
        normalized_issues.append(
            _normalize_required_text(
                issue,
                field_name=(
                    f"review_action issues[{index}]"
                ),
            )
        )

    return tuple(normalized_issues)


async def _load_revision_record(
    connection: asyncpg.Connection,
    *,
    review_action_id: int,
) -> asyncpg.Record:
    """Читает исходный пост и review_action."""

    record = await connection.fetchrow(
        """
        SELECT
            ra.review_action_id,
            ra.generated_post_id
                AS review_generated_post_id,
            ra.reviewer_type,
            ra.decision,
            ra.requested_action,
            ra.comment_text,
            ra.issues::text
                AS review_issues_json,

            gp.generated_post_id
                AS source_generated_post_id,
            gp.batch_id,
            gp.version_number
                AS source_version_number,
            gp.post_status
                AS source_post_status,
            gp.post_text
                AS source_post_text,
            gp.text_format
                AS source_text_format,
            gp.generation_metadata::text
                AS source_generation_metadata_json,

            b.batch_status,
            b.ranking_run_id

        FROM top3_news.review_actions AS ra
        JOIN top3_news.generated_posts AS gp
          ON gp.generated_post_id =
             ra.generated_post_id
        JOIN top3_news.publication_batches AS b
          ON b.batch_id = gp.batch_id
        WHERE ra.review_action_id = $1
        """,
        review_action_id,
    )

    if record is None:
        raise LookupError(
            "Не найден review_action: "
            f"{review_action_id}"
        )

    return record


def _validate_revision_record(
    record: asyncpg.Record,
    *,
    review_action_id: int,
) -> tuple[
    int,
    int,
    int,
    str,
    str,
    str,
    tuple[str, ...],
    int,
]:
    """Проверяет пригодность review_action для ревизии."""

    differences: list[str] = []

    actual_review_action_id = (
        int(record["review_action_id"])
    )

    if (
        actual_review_action_id
        != review_action_id
    ):
        differences.append(
            "review_action_id: "
            f"expected={review_action_id!r}, "
            f"actual={actual_review_action_id!r}"
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

    review_generated_post_id = (
        _normalize_positive_integer(
            int(
                record[
                    "review_generated_post_id"
                ]
            ),
            field_name=(
                "review_generated_post_id"
            ),
        )
    )

    if (
        review_generated_post_id
        != source_generated_post_id
    ):
        differences.append(
            "review_generated_post_id: "
            f"expected="
            f"{source_generated_post_id!r}, "
            f"actual="
            f"{review_generated_post_id!r}"
        )

    batch_id = _normalize_positive_integer(
        int(record["batch_id"]),
        field_name="batch_id",
    )

    source_version_number = (
        _normalize_positive_integer(
            int(
                record[
                    "source_version_number"
                ]
            ),
            field_name=(
                "source_version_number"
            ),
        )
    )

    if record["reviewer_type"] != "human":
        differences.append(
            "reviewer_type: "
            "expected='human', "
            f"actual={record['reviewer_type']!r}"
        )

    if (
        record["decision"]
        != "changes_required"
    ):
        differences.append(
            "decision: "
            "expected='changes_required', "
            f"actual={record['decision']!r}"
        )

    if (
        record["requested_action"]
        != REVISION_REQUESTED_ACTION
    ):
        differences.append(
            "requested_action: "
            f"expected="
            f"{REVISION_REQUESTED_ACTION!r}, "
            f"actual="
            f"{record['requested_action']!r}"
        )

    source_post_status = (
        record["source_post_status"]
    )

    if source_post_status not in {
        "awaiting_review",
        "superseded",
    }:
        differences.append(
            "source_post_status: "
            "expected one of "
            "{'awaiting_review', 'superseded'}, "
            f"actual={source_post_status!r}"
        )

    if (
        record["batch_status"]
        != "awaiting_review"
    ):
        differences.append(
            "batch_status: "
            "expected='awaiting_review', "
            f"actual={record['batch_status']!r}"
        )

    ranking_run_id = record[
        "ranking_run_id"
    ]

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

    source_post_text = (
        _normalize_required_text(
            record["source_post_text"],
            field_name="source_post_text",
        )
    )

    source_text_format = (
        _normalize_required_text(
            record["source_text_format"],
            field_name="source_text_format",
        )
    )

    editorial_comment = (
        _normalize_required_text(
            record["comment_text"],
            field_name="editorial_comment",
        )
    )

    issues = _decode_issues(
        record["review_issues_json"]
    )

    if differences:
        raise ValueError(
            "review_action не допускает "
            "текстовую ревизию: "
            + "; ".join(differences)
        )

    return (
        batch_id,
        source_generated_post_id,
        source_version_number,
        source_post_text,
        source_text_format,
        editorial_comment,
        issues,
        normalized_ranking_run_id,
    )


def _hydrate_trailer_metadata(
    selection: GenerationTop3Selection,
    generation_metadata_json: str | None,
) -> GenerationTop3Selection:
    """Восстанавливает verified trailer metadata предыдущей версии."""

    if not generation_metadata_json:
        return selection

    try:
        metadata = json.loads(generation_metadata_json)
    except (TypeError, json.JSONDecodeError):
        return selection

    if not isinstance(metadata, dict):
        return selection

    raw_items = metadata.get("generated_items")
    if not isinstance(raw_items, list):
        return selection

    trailer_by_news_id: dict[int, tuple[str, str | None]] = {}

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue

        raw_news_id = raw_item.get("news_id")
        raw_url = raw_item.get("official_trailer_url")

        if isinstance(raw_news_id, bool) or not isinstance(raw_news_id, int):
            continue
        if not isinstance(raw_url, str):
            continue

        url = raw_url.strip()
        if not url:
            continue

        try:
            parsed = urlsplit(url)
        except ValueError:
            continue

        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            continue

        raw_channel = raw_item.get("official_trailer_channel_name")
        channel: str | None = None
        if isinstance(raw_channel, str) and raw_channel.strip():
            channel = raw_channel.strip()

        trailer_by_news_id[raw_news_id] = (url, channel)

    if not trailer_by_news_id:
        return selection

    hydrated_items: list[GenerationNewsItem] = []

    for item in selection.items:
        trailer = trailer_by_news_id.get(item.news_id)
        if trailer is None:
            hydrated_items.append(item)
            continue

        url, channel = trailer
        hydrated_items.append(
            replace(
                item,
                official_trailer_url=url,
                official_trailer_channel_name=channel,
            )
        )

    return replace(
        selection,
        items=(
            hydrated_items[0],
            hydrated_items[1],
            hydrated_items[2],
        ),
    )


async def load_generation_revision_selection(
    pool: asyncpg.Pool,
    *,
    review_action_id: int,
) -> GenerationRevisionSelection:
    """
    Загружает контекст ревизии по review_action_id.

    Функция выполняет только чтение.

    Атомарная повторная проверка batch,
    source generated_post, review_action
    и фактического TOP-3 этого batch выполняется
    позднее в reserve_generation_revision().
    """

    normalized_review_action_id = (
        _normalize_positive_integer(
            review_action_id,
            field_name="review_action_id",
        )
    )

    async with pool.acquire() as connection:
        record = await _load_revision_record(
            connection,
            review_action_id=(
                normalized_review_action_id
            ),
        )

        (
            batch_id,
            source_generated_post_id,
            source_version_number,
            source_post_text,
            source_text_format,
            editorial_comment,
            issues,
            ranking_run_id,
        ) = _validate_revision_record(
            record,
            review_action_id=(
                normalized_review_action_id
            ),
        )

        selection = (
            await _load_generation_batch_selection(
                connection,
                ranking_run_id=(
                    ranking_run_id
                ),
                batch_id=batch_id,
            )
        )

        selection = _hydrate_trailer_metadata(
            selection,
            record["source_generation_metadata_json"],
        )

    return GenerationRevisionSelection(
        review_action_id=(
            normalized_review_action_id
        ),
        batch_id=batch_id,
        source_generated_post_id=(
            source_generated_post_id
        ),
        source_version_number=(
            source_version_number
        ),
        target_version_number=(
            source_version_number + 1
        ),
        source_post_text=source_post_text,
        source_text_format=source_text_format,
        editorial_comment=(
            editorial_comment
        ),
        issues=issues,
        selection=selection,
    )