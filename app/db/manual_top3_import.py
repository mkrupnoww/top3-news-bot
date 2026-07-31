from dataclasses import dataclass
from datetime import date
import hashlib
import json
from typing import Any, Mapping

import asyncpg

from app.generation.manual_top3_input import (
    ManualTop3Input,
    ManualTop3NewsItem,
)


@dataclass(frozen=True, slots=True)
class ManualTop3ImportResult:
    """Результат загрузки ручного выпуска TOP-3."""

    batch_id: int
    generated_post_id: int
    publication_date: date
    edition: int
    version_number: int
    batch_status: str
    post_status: str
    news_ids: tuple[int, ...]
    publication_attempt_count: int
    already_imported: bool


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_source_url(
    source_url: str,
) -> str:
    """Нормализует URL перед вычислением хеша."""

    return source_url.strip().rstrip("/")


def _calculate_url_sha256(
    source_url: str,
) -> str:
    """Вычисляет SHA-256 нормализованного URL."""

    normalized_url = _normalize_source_url(
        source_url
    )

    return hashlib.sha256(
        normalized_url.encode("utf-8")
    ).hexdigest()


def _calculate_input_sha256(
    top3_input: ManualTop3Input,
) -> str:
    """Вычисляет стабильный хеш всего входного выпуска."""

    canonical_payload = json.dumps(
        top3_input.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()


async def _find_existing_import(
    connection: asyncpg.Connection,
    *,
    input_sha256: str,
    telegram_chat_id: int,
) -> asyncpg.Record | None:
    """Ищет ранее загруженный идентичный выпуск."""

    return await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.publication_date,
            b.edition,
            b.batch_status,
            p.generated_post_id,
            p.version_number,
            p.post_status,
            ARRAY(
                SELECT bi.news_id
                FROM batch_items AS bi
                WHERE bi.batch_id = b.batch_id
                ORDER BY bi.position
            ) AS news_ids,
            (
                SELECT COUNT(*)::integer
                FROM publication_attempts AS a
                WHERE a.generated_post_id =
                    p.generated_post_id
            ) AS publication_attempt_count
        FROM publication_batches AS b
        JOIN LATERAL (
            SELECT
                gp.generated_post_id,
                gp.version_number,
                gp.post_status
            FROM generated_posts AS gp
            WHERE gp.batch_id = b.batch_id
            ORDER BY gp.version_number DESC
            LIMIT 1
        ) AS p
            ON true
        WHERE b.target_telegram_chat_id = $1
          AND b.metadata->>'manual_input_sha256' = $2
        ORDER BY b.created_at DESC
        LIMIT 1
        """,
        telegram_chat_id,
        input_sha256,
    )


async def _upsert_source(
    connection: asyncpg.Connection,
    *,
    item: ManualTop3NewsItem,
) -> int:
    """Создаёт или обновляет источник новости."""

    source_settings = _encode_json(
        {
            "input_mode": "manual",
            "managed_by": "manual_top3_import",
        }
    )

    source_id = await connection.fetchval(
        """
        INSERT INTO sources (
            source_code,
            source_name,
            source_type,
            base_url,
            default_language,
            settings
        )
        VALUES (
            $1,
            $2,
            'manual',
            $3,
            'ru',
            $4::jsonb
        )
        ON CONFLICT (source_code)
        DO UPDATE SET
            source_name = EXCLUDED.source_name,
            base_url = COALESCE(
                EXCLUDED.base_url,
                sources.base_url
            ),
            settings = (
                sources.settings
                || EXCLUDED.settings
            )
        RETURNING source_id
        """,
        item.source_code,
        item.source_name,
        item.source_base_url,
        source_settings,
    )

    return int(source_id)


async def _upsert_news_item(
    connection: asyncpg.Connection,
    *,
    source_id: int,
    item: ManualTop3NewsItem,
    input_sha256: str,
) -> int:
    """Создаёт либо обновляет новость по хешу URL."""

    normalized_url = _normalize_source_url(
        item.source_url
    )

    url_sha256 = _calculate_url_sha256(
        item.source_url
    )

    raw_payload = _encode_json(
        item.model_dump(mode="json")
    )

    news_metadata = dict(item.metadata)
    news_metadata.update(
        {
            "input_mode": "manual",
            "manual_input_sha256": input_sha256,
            "position": item.position,
        }
    )

    news_id = await connection.fetchval(
        """
        INSERT INTO news_items (
            source_id,
            source_url,
            canonical_url,
            url_sha256,
            raw_title,
            normalized_title,
            raw_summary,
            normalized_summary,
            source_published_at,
            language_code,
            primary_image_url,
            image_credit,
            processing_status,
            raw_payload,
            metadata
        )
        VALUES (
            $1,
            $2,
            $2,
            $3,
            $4,
            $4,
            $5,
            $5,
            $6,
            'ru',
            $7,
            $8,
            'candidate',
            $9::jsonb,
            $10::jsonb
        )
        ON CONFLICT (url_sha256)
            WHERE url_sha256 IS NOT NULL
        DO UPDATE SET
            source_id = EXCLUDED.source_id,
            source_url = EXCLUDED.source_url,
            canonical_url = EXCLUDED.canonical_url,
            raw_title = EXCLUDED.raw_title,
            normalized_title =
                EXCLUDED.normalized_title,
            raw_summary = EXCLUDED.raw_summary,
            normalized_summary =
                EXCLUDED.normalized_summary,
            source_published_at =
                EXCLUDED.source_published_at,
            primary_image_url = COALESCE(
                EXCLUDED.primary_image_url,
                news_items.primary_image_url
            ),
            image_credit = COALESCE(
                EXCLUDED.image_credit,
                news_items.image_credit
            ),
            processing_status = 'candidate',
            raw_payload = (
                news_items.raw_payload
                || EXCLUDED.raw_payload
            ),
            metadata = (
                news_items.metadata
                || EXCLUDED.metadata
            )
        RETURNING news_id
        """,
        source_id,
        normalized_url,
        url_sha256,
        item.title,
        item.summary,
        item.source_published_at,
        item.primary_image_url,
        item.image_credit,
        raw_payload,
        _encode_json(news_metadata),
    )

    return int(news_id)


def _result_from_existing_record(
    record: asyncpg.Record,
) -> ManualTop3ImportResult:
    """Преобразует найденную запись в результат импорта."""

    return ManualTop3ImportResult(
        batch_id=record["batch_id"],
        generated_post_id=(
            record["generated_post_id"]
        ),
        publication_date=record["publication_date"],
        edition=record["edition"],
        version_number=record["version_number"],
        batch_status=record["batch_status"],
        post_status=record["post_status"],
        news_ids=tuple(record["news_ids"]),
        publication_attempt_count=(
            record["publication_attempt_count"]
        ),
        already_imported=True,
    )


async def import_manual_top3(
    pool: asyncpg.Pool,
    *,
    top3_input: ManualTop3Input,
    telegram_chat_id: int,
) -> ManualTop3ImportResult:
    """
    Атомарно загружает ручной выпуск TOP-3.

    Создаются или обновляются:

    - sources;
    - news_items;
    - publication_batches;
    - batch_items;
    - generated_posts.

    Telegram не вызывается, publication_attempts не создаётся.
    """

    input_sha256 = _calculate_input_sha256(
        top3_input
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            # Такой же принцип блокировки используется
            # при создании обычного review draft.
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock($1::bigint)
                """,
                top3_input.publication_date.toordinal(),
            )

            existing_record = (
                await _find_existing_import(
                    connection,
                    input_sha256=input_sha256,
                    telegram_chat_id=telegram_chat_id,
                )
            )

            if existing_record is not None:
                return _result_from_existing_record(
                    existing_record
                )

            edition = await connection.fetchval(
                """
                SELECT
                    COALESCE(MAX(edition), 0)::integer + 1
                FROM publication_batches
                WHERE publication_date = $1
                """,
                top3_input.publication_date,
            )

            batch_metadata = dict(
                top3_input.metadata
            )
            batch_metadata.update(
                {
                    "input_mode": "manual",
                    "manual_input_sha256": (
                        input_sha256
                    ),
                    "news_count": 3,
                    "source_urls": [
                        item.source_url
                        for item in sorted(
                            top3_input.items,
                            key=lambda value: (
                                value.position
                            ),
                        )
                    ],
                }
            )

            batch_id = await connection.fetchval(
                """
                INSERT INTO publication_batches (
                    publication_date,
                    edition,
                    batch_status,
                    target_telegram_chat_id,
                    metadata
                )
                VALUES (
                    $1,
                    $2,
                    'awaiting_review',
                    $3,
                    $4::jsonb
                )
                RETURNING batch_id
                """,
                top3_input.publication_date,
                edition,
                telegram_chat_id,
                _encode_json(batch_metadata),
            )

            news_ids: list[int] = []

            for item in sorted(
                top3_input.items,
                key=lambda value: value.position,
            ):
                source_id = await _upsert_source(
                    connection,
                    item=item,
                )

                news_id = await _upsert_news_item(
                    connection,
                    source_id=source_id,
                    item=item,
                    input_sha256=input_sha256,
                )

                await connection.execute(
                    """
                    INSERT INTO batch_items (
                        batch_id,
                        news_id,
                        position,
                        selection_reason
                    )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4
                    )
                    """,
                    batch_id,
                    news_id,
                    item.position,
                    item.selection_reason,
                )

                news_ids.append(news_id)

            generation_metadata = dict(
                top3_input.metadata
            )
            generation_metadata.update(
                {
                    "input_mode": "manual",
                    "manual_input_sha256": (
                        input_sha256
                    ),
                    "news_ids": news_ids,
                    "news_count": len(news_ids),
                }
            )

            generated_post_id = (
                await connection.fetchval(
                    """
                    INSERT INTO generated_posts (
                        batch_id,
                        version_number,
                        post_status,
                        post_text,
                        text_format,
                        generation_metadata
                    )
                    VALUES (
                        $1,
                        1,
                        'awaiting_review',
                        $2,
                        $3,
                        $4::jsonb
                    )
                    RETURNING generated_post_id
                    """,
                    batch_id,
                    top3_input.post_text,
                    top3_input.text_format,
                    _encode_json(
                        generation_metadata
                    ),
                )
            )

    return ManualTop3ImportResult(
        batch_id=int(batch_id),
        generated_post_id=int(
            generated_post_id
        ),
        publication_date=(
            top3_input.publication_date
        ),
        edition=int(edition),
        version_number=1,
        batch_status="awaiting_review",
        post_status="awaiting_review",
        news_ids=tuple(news_ids),
        publication_attempt_count=0,
        already_imported=False,
    )