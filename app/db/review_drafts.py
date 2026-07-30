from dataclasses import dataclass
from datetime import date
import json
from typing import Any, Literal, Mapping

import asyncpg


TextFormat = Literal[
    "markdown",
    "markdown_v2",
    "html",
    "plain_text",
]


@dataclass(frozen=True, slots=True)
class ReviewDraft:
    """Пост, подготовленный для ручной проверки."""

    batch_id: int
    generated_post_id: int
    publication_date: date
    edition: int
    version_number: int
    batch_status: str
    post_status: str


def _encode_json(payload: Mapping[str, Any]) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def create_review_draft(
    pool: asyncpg.Pool,
    *,
    publication_date: date,
    telegram_chat_id: int,
    post_text: str,
    text_format: TextFormat,
    metadata: Mapping[str, Any],
) -> ReviewDraft:
    """
    Создаёт batch и generated_post для ручной проверки.

    Сообщение в Telegram не отправляется, а запись
    publication_attempts не создаётся.
    """

    normalized_post_text = post_text.strip()

    if not normalized_post_text:
        raise ValueError("Текст публикации не может быть пустым.")

    encoded_metadata = _encode_json(metadata)

    async with pool.acquire() as connection:
        async with connection.transaction():
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
                    'awaiting_review',
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
                    'awaiting_review',
                    $2,
                    $3,
                    $4::jsonb
                )
                RETURNING generated_post_id
                """,
                batch_id,
                normalized_post_text,
                text_format,
                encoded_metadata,
            )

    return ReviewDraft(
        batch_id=batch_id,
        generated_post_id=generated_post_id,
        publication_date=publication_date,
        edition=edition,
        version_number=1,
        batch_status="awaiting_review",
        post_status="awaiting_review",
    )