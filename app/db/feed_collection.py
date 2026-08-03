from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

import asyncpg

from app.collectors.feed_http import (
    FeedDownloadResult,
)
from app.collectors.feed_parser import (
    FeedParseResult,
    ParsedFeedEntry,
)


@dataclass(frozen=True, slots=True)
class FeedCollectionRun:
    """Запущенный сбор новостей из одного источника."""

    source_id: int
    collection_run_id: int


@dataclass(frozen=True, slots=True)
class FeedPersistenceResult:
    """Итог сохранения разобранной ленты."""

    source_id: int
    collection_run_id: int
    run_status: str
    fetched_count: int
    inserted_count: int
    duplicate_count: int
    rejected_count: int
    news_ids: tuple[int, ...]


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_url(
    value: str,
) -> str:
    """Нормализует URL перед сравнением и хешированием."""

    return value.strip().rstrip("/")


def _calculate_sha256(
    value: str,
) -> str:
    """Возвращает SHA-256 строки."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _calculate_url_sha256(
    source_url: str,
) -> str:
    """Вычисляет хеш нормализованного URL."""

    return _calculate_sha256(
        _normalize_url(source_url)
    )


def _calculate_content_sha256(
    entry: ParsedFeedEntry,
) -> str:
    """Вычисляет хеш основного содержимого новости."""

    canonical_payload = json.dumps(
        {
            "title": entry.title.strip(),
            "summary": (
                entry.summary.strip()
                if entry.summary
                else None
            ),
            "author_name": (
                entry.author_name.strip()
                if entry.author_name
                else None
            ),
            "source_published_at": (
                entry.source_published_at.isoformat()
                if entry.source_published_at
                else None
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return _calculate_sha256(
        canonical_payload
    )


def _normalize_optional_text(
    value: str | None,
) -> str | None:
    """Удаляет пробелы и превращает пустую строку в None."""

    if value is None:
        return None

    normalized_value = value.strip()

    return normalized_value or None


def _build_raw_payload(
    entry: ParsedFeedEntry,
) -> dict[str, Any]:
    """Формирует исходное представление записи ленты."""

    return {
        "external_id": entry.external_id,
        "title": entry.title,
        "source_url": entry.source_url,
        "summary": entry.summary,
        "author_name": entry.author_name,
        "source_published_at": (
            entry.source_published_at.isoformat()
            if entry.source_published_at
            else None
        ),
        "primary_image_url": (
            entry.primary_image_url
        ),
        "categories": list(entry.categories),
    }


async def start_feed_collection_run(
    pool: asyncpg.Pool,
    *,
    source_code: str,
    source_name: str,
    feed_url: str,
    base_url: str | None,
    language_code: str,
    collection_priority: int,
    collector_name: str,
    collector_version: str,
) -> FeedCollectionRun:
    """Создаёт или обновляет источник и начинает журнал запуска."""

    source_settings = _encode_json(
        {
            "collector": "rss_atom_http",
            "managed_by": "feed_collection",
        }
    )

    run_metadata = _encode_json(
        {
            "source_code": source_code,
            "feed_url": feed_url,
            "language_code": language_code,
        }
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            source_id = await connection.fetchval(
                """
                INSERT INTO sources (
                    source_code,
                    source_name,
                    source_type,
                    base_url,
                    feed_url,
                    default_language,
                    is_active,
                    collection_priority,
                    settings
                )
                VALUES (
                    $1,
                    $2,
                    'rss',
                    $3,
                    $4,
                    $5,
                    true,
                    $6,
                    $7::jsonb
                )
                ON CONFLICT (source_code)
                DO UPDATE SET
                    source_name = EXCLUDED.source_name,
                    source_type = 'rss',
                    base_url = COALESCE(
                        EXCLUDED.base_url,
                        sources.base_url
                    ),
                    feed_url = EXCLUDED.feed_url,
                    default_language = (
                        EXCLUDED.default_language
                    ),
                    is_active = true,
                    collection_priority = (
                        EXCLUDED.collection_priority
                    ),
                    settings = (
                        sources.settings
                        || EXCLUDED.settings
                    )
                RETURNING source_id
                """,
                source_code,
                source_name,
                base_url,
                feed_url,
                language_code,
                collection_priority,
                source_settings,
            )

            collection_run_id = await connection.fetchval(
                """
                INSERT INTO collection_runs (
                    source_id,
                    run_status,
                    collector_name,
                    collector_version,
                    run_metadata
                )
                VALUES (
                    $1,
                    'running',
                    $2,
                    $3,
                    $4::jsonb
                )
                RETURNING collection_run_id
                """,
                source_id,
                collector_name,
                collector_version,
                run_metadata,
            )

    return FeedCollectionRun(
        source_id=int(source_id),
        collection_run_id=int(
            collection_run_id
        ),
    )


async def mark_feed_collection_failed(
    pool: asyncpg.Pool,
    *,
    collection_run_id: int,
    error_message: str,
    failed_stage: str,
) -> None:
    """Фиксирует неуспешный запуск сборщика."""

    failure_metadata = _encode_json(
        {
            "failed_stage": failed_stage,
            "error_type": (
                error_message.split(
                    ":",
                    maxsplit=1,
                )[0]
            ),
        }
    )

    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE collection_runs
            SET
                run_status = 'failed',
                finished_at = now(),
                error_message = $2,
                run_metadata = (
                    run_metadata
                    || $3::jsonb
                )
            WHERE collection_run_id = $1
            """,
            collection_run_id,
            error_message[:4000],
            failure_metadata,
        )


async def _find_existing_news_id(
    connection: asyncpg.Connection,
    *,
    source_id: int,
    url_sha256: str,
    external_id: str | None,
) -> int | None:
    """Ищет новость по URL или ID исходной ленты."""

    news_id = await connection.fetchval(
        """
        SELECT news_id
        FROM news_items
        WHERE
            url_sha256 = $1
            OR (
                $3::text IS NOT NULL
                AND source_id = $2
                AND external_id = $3
            )
        ORDER BY
            CASE
                WHEN url_sha256 = $1
                THEN 0
                ELSE 1
            END,
            news_id
        LIMIT 1
        """,
        url_sha256,
        source_id,
        external_id,
    )

    return (
        int(news_id)
        if news_id is not None
        else None
    )


async def _insert_news_item(
    connection: asyncpg.Connection,
    *,
    source_id: int,
    collection_run_id: int,
    language_code: str,
    entry: ParsedFeedEntry,
    feed_type: str,
    feed_title: str | None,
    final_feed_url: str,
) -> int | None:
    """Сохраняет одну новую запись ленты."""

    normalized_url = _normalize_url(
        entry.source_url
    )

    external_id = _normalize_optional_text(
        entry.external_id
    )

    normalized_title = entry.title.strip()

    normalized_summary = _normalize_optional_text(
        entry.summary
    )

    raw_payload = _encode_json(
        _build_raw_payload(entry)
    )

    metadata = _encode_json(
        {
            "collector": "rss_atom_http",
            "feed_type": feed_type,
            "feed_title": feed_title,
            "final_feed_url": final_feed_url,
        }
    )

    news_id = await connection.fetchval(
        """
        INSERT INTO news_items (
            source_id,
            collection_run_id,
            external_id,
            source_url,
            canonical_url,
            url_sha256,
            content_sha256,
            raw_title,
            normalized_title,
            raw_summary,
            normalized_summary,
            author_name,
            source_published_at,
            language_code,
            primary_image_url,
            processing_status,
            raw_payload,
            metadata
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            $4,
            $5,
            $6,
            $7,
            $7,
            $8,
            $8,
            $9,
            $10,
            $11,
            $12,
            'collected',
            $13::jsonb,
            $14::jsonb
        )
        ON CONFLICT DO NOTHING
        RETURNING news_id
        """,
        source_id,
        collection_run_id,
        external_id,
        normalized_url,
        _calculate_url_sha256(
            normalized_url
        ),
        _calculate_content_sha256(
            entry
        ),
        normalized_title,
        normalized_summary,
        _normalize_optional_text(
            entry.author_name
        ),
        entry.source_published_at,
        language_code,
        _normalize_optional_text(
            entry.primary_image_url
        ),
        raw_payload,
        metadata,
    )

    return (
        int(news_id)
        if news_id is not None
        else None
    )


async def persist_feed_collection(
    pool: asyncpg.Pool,
    *,
    run: FeedCollectionRun,
    language_code: str,
    download_result: FeedDownloadResult,
    parse_result: FeedParseResult,
) -> FeedPersistenceResult:
    """Атомарно сохраняет записи и завершает collection_run."""

    inserted_count = 0
    duplicate_count = 0
    rejected_count = (
        parse_result.skipped_count
    )

    news_ids: list[int] = []

    fetched_count = (
        len(parse_result.entries)
        + parse_result.skipped_count
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            # Не допускаем два одновременных запуска
            # одного и того же источника.
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    $1::bigint
                )
                """,
                run.source_id,
            )

            for entry in parse_result.entries:
                external_id = (
                    _normalize_optional_text(
                        entry.external_id
                    )
                )

                url_sha256 = (
                    _calculate_url_sha256(
                        entry.source_url
                    )
                )

                existing_news_id = (
                    await _find_existing_news_id(
                        connection,
                        source_id=run.source_id,
                        url_sha256=url_sha256,
                        external_id=external_id,
                    )
                )

                if existing_news_id is not None:
                    duplicate_count += 1
                    continue

                news_id = await _insert_news_item(
                    connection,
                    source_id=run.source_id,
                    collection_run_id=(
                        run.collection_run_id
                    ),
                    language_code=language_code,
                    entry=entry,
                    feed_type=(
                        parse_result.feed_type
                    ),
                    feed_title=(
                        parse_result.feed_title
                    ),
                    final_feed_url=(
                        download_result.final_url
                    ),
                )

                if news_id is None:
                    # Защита от конкурентного INSERT,
                    # завершившегося между SELECT и INSERT.
                    duplicate_count += 1
                    continue

                inserted_count += 1
                news_ids.append(news_id)

            run_status = (
                "completed_with_errors"
                if rejected_count > 0
                else "completed"
            )

            completion_metadata = _encode_json(
                {
                    "requested_url": (
                        download_result.requested_url
                    ),
                    "final_url": (
                        download_result.final_url
                    ),
                    "status_code": (
                        download_result.status_code
                    ),
                    "content_type": (
                        download_result.content_type
                    ),
                    "bytes_downloaded": (
                        download_result.bytes_downloaded
                    ),
                    "redirect_count": (
                        download_result.redirect_count
                    ),
                    "feed_type": (
                        parse_result.feed_type
                    ),
                    "feed_title": (
                        parse_result.feed_title
                    ),
                    "parsed_entry_count": (
                        len(parse_result.entries)
                    ),
                    "parser_skipped_count": (
                        parse_result.skipped_count
                    ),
                    "inserted_news_ids": news_ids,
                }
            )

            await connection.execute(
                """
                UPDATE collection_runs
                SET
                    run_status = $2,
                    finished_at = now(),
                    fetched_count = $3,
                    inserted_count = $4,
                    duplicate_count = $5,
                    rejected_count = $6,
                    error_message = NULL,
                    run_metadata = (
                        run_metadata
                        || $7::jsonb
                    )
                WHERE collection_run_id = $1
                """,
                run.collection_run_id,
                run_status,
                fetched_count,
                inserted_count,
                duplicate_count,
                rejected_count,
                completion_metadata,
            )

    return FeedPersistenceResult(
        source_id=run.source_id,
        collection_run_id=(
            run.collection_run_id
        ),
        run_status=run_status,
        fetched_count=fetched_count,
        inserted_count=inserted_count,
        duplicate_count=duplicate_count,
        rejected_count=rejected_count,
        news_ids=tuple(news_ids),
    )