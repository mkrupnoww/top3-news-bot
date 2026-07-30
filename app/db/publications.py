from dataclasses import dataclass
from datetime import date
import json
from typing import Any, Mapping

import asyncpg


@dataclass(frozen=True, slots=True)
class PublicationAttempt:
    """Созданная в PostgreSQL попытка публикации."""

    batch_id: int
    generated_post_id: int
    publication_attempt_id: int
    publication_date: date
    edition: int


def _encode_json(payload: Mapping[str, Any]) -> str:
    """Преобразует словарь в JSON для передачи в asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def create_publication_attempt(
    pool: asyncpg.Pool,
    *,
    publication_date: date,
    telegram_chat_id: int,
    post_text: str,
    request_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> PublicationAttempt:
    """Создаёт batch, generated_post и started-попытку публикации."""

    encoded_request = _encode_json(request_payload)
    encoded_metadata = _encode_json(metadata)

    async with pool.acquire() as connection:
        async with connection.transaction():
            # Не допускаем одновременный выбор одинакового edition
            # для одной даты внутри этого сценария.
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock($1::bigint)
                """,
                publication_date.toordinal(),
            )

            edition = await connection.fetchval(
                """
                SELECT COALESCE(MAX(edition), 0)::integer + 1
                FROM publication_batches
                WHERE publication_date = $1
                """,
                publication_date,
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
                    'publishing',
                    $3,
                    $4::jsonb
                )
                RETURNING batch_id
                """,
                publication_date,
                edition,
                telegram_chat_id,
                encoded_metadata,
            )

            generated_post_id = await connection.fetchval(
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
                    'approved',
                    $2,
                    'plain_text',
                    $3::jsonb
                )
                RETURNING generated_post_id
                """,
                batch_id,
                post_text,
                encoded_metadata,
            )

            publication_attempt_id = await connection.fetchval(
                """
                INSERT INTO publication_attempts (
                    generated_post_id,
                    attempt_number,
                    attempt_status,
                    telegram_chat_id,
                    request_payload
                )
                VALUES (
                    $1,
                    1,
                    'started',
                    $2,
                    $3::jsonb
                )
                RETURNING publication_attempt_id
                """,
                generated_post_id,
                telegram_chat_id,
                encoded_request,
            )

    return PublicationAttempt(
        batch_id=batch_id,
        generated_post_id=generated_post_id,
        publication_attempt_id=publication_attempt_id,
        publication_date=publication_date,
        edition=edition,
    )


async def mark_publication_published(
    pool: asyncpg.Pool,
    publication: PublicationAttempt,
    *,
    telegram_message_id: int,
    response_payload: Mapping[str, Any],
) -> None:
    """Фиксирует успешную публикацию во всех связанных таблицах."""

    encoded_response = _encode_json(response_payload)

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                UPDATE publication_attempts
                SET
                    attempt_status = 'published',
                    telegram_message_id = $2,
                    response_payload = $3::jsonb,
                    finished_at = now(),
                    telegram_error_code = NULL,
                    error_message = NULL
                WHERE publication_attempt_id = $1
                """,
                publication.publication_attempt_id,
                telegram_message_id,
                encoded_response,
            )

            await connection.execute(
                """
                UPDATE generated_posts
                SET post_status = 'published'
                WHERE generated_post_id = $1
                """,
                publication.generated_post_id,
            )

            await connection.execute(
                """
                UPDATE publication_batches
                SET
                    batch_status = 'published',
                    published_at = now(),
                    error_message = NULL
                WHERE batch_id = $1
                """,
                publication.batch_id,
            )


async def mark_publication_failed(
    pool: asyncpg.Pool,
    publication: PublicationAttempt,
    *,
    error_message: str,
    telegram_error_code: int | None = None,
) -> None:
    """Фиксирует неуспешную попытку публикации."""

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                UPDATE publication_attempts
                SET
                    attempt_status = 'failed',
                    telegram_error_code = $2,
                    error_message = $3,
                    finished_at = now()
                WHERE publication_attempt_id = $1
                """,
                publication.publication_attempt_id,
                telegram_error_code,
                error_message,
            )

            await connection.execute(
                """
                UPDATE generated_posts
                SET post_status = 'failed'
                WHERE generated_post_id = $1
                """,
                publication.generated_post_id,
            )

            await connection.execute(
                """
                UPDATE publication_batches
                SET
                    batch_status = 'failed',
                    error_message = $2
                WHERE batch_id = $1
                """,
                publication.batch_id,
                error_message,
            )