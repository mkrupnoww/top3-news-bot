from dataclasses import dataclass
from datetime import date
import json
from typing import Literal

import asyncpg


ReviewDecision = Literal["approve", "reject"]


@dataclass(frozen=True, slots=True)
class ReviewDraftPreview:
    """Черновик, ожидающий ручной проверки."""

    batch_id: int
    generated_post_id: int
    publication_date: date
    edition: int
    version_number: int
    post_text: str
    text_format: str


@dataclass(frozen=True, slots=True)
class ReviewDecisionResult:
    """Результат ручного решения по черновику."""

    batch_id: int
    generated_post_id: int
    review_action_id: int | None
    decision: ReviewDecision
    post_status: str
    batch_status: str
    already_processed: bool


async def get_latest_review_draft(
    pool: asyncpg.Pool,
) -> ReviewDraftPreview | None:
    """Возвращает последний черновик со статусом awaiting_review."""

    query = """
        SELECT
            b.batch_id,
            b.publication_date,
            b.edition,
            p.generated_post_id,
            p.version_number,
            p.post_text,
            p.text_format
        FROM publication_batches AS b
        JOIN generated_posts AS p
            ON p.batch_id = b.batch_id
        WHERE b.batch_status = 'awaiting_review'
          AND p.post_status = 'awaiting_review'
        ORDER BY
            b.publication_date DESC,
            b.edition DESC,
            p.version_number DESC
        LIMIT 1
    """

    async with pool.acquire() as connection:
        record = await connection.fetchrow(query)

    if record is None:
        return None

    return ReviewDraftPreview(
        batch_id=record["batch_id"],
        generated_post_id=record["generated_post_id"],
        publication_date=record["publication_date"],
        edition=record["edition"],
        version_number=record["version_number"],
        post_text=record["post_text"],
        text_format=record["text_format"],
    )


async def record_human_review_decision(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
    reviewer_telegram_user_id: int,
    decision: ReviewDecision,
) -> ReviewDecisionResult:
    """
    Фиксирует решение человека по черновику.

    Разрешены только переходы:

    awaiting_review -> approved
    awaiting_review -> rejected
    """

    if decision not in {"approve", "reject"}:
        raise ValueError(
            f"Неподдерживаемое решение: {decision}"
        )

    target_status = (
        "approved"
        if decision == "approve"
        else "rejected"
    )

    review_details = json.dumps(
        {
            "source": "telegram_inline_keyboard",
            "generated_post_id": generated_post_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            reviewer = await connection.fetchrow(
                """
                SELECT
                    telegram_user_id,
                    user_role
                FROM bot_users
                WHERE telegram_user_id = $1
                  AND is_active = true
                  AND user_role IN ('admin', 'reviewer')
                """,
                reviewer_telegram_user_id,
            )

            if reviewer is None:
                raise PermissionError(
                    "Пользователь не имеет права "
                    "проверять публикации."
                )

            record = await connection.fetchrow(
                """
                SELECT
                    p.generated_post_id,
                    p.batch_id,
                    p.post_status,
                    b.batch_status
                FROM generated_posts AS p
                JOIN publication_batches AS b
                    ON b.batch_id = p.batch_id
                WHERE p.generated_post_id = $1
                FOR UPDATE OF p, b
                """,
                generated_post_id,
            )

            if record is None:
                raise LookupError(
                    "Черновик не найден: "
                    f"generated_post_id={generated_post_id}"
                )

            if (
                record["post_status"] == target_status
                and record["batch_status"] == target_status
            ):
                return ReviewDecisionResult(
                    batch_id=record["batch_id"],
                    generated_post_id=generated_post_id,
                    review_action_id=None,
                    decision=decision,
                    post_status=target_status,
                    batch_status=target_status,
                    already_processed=True,
                )

            if record["post_status"] != "awaiting_review":
                raise ValueError(
                    "Пост уже не ожидает проверки: "
                    f"post_status={record['post_status']}"
                )

            if record["batch_status"] != "awaiting_review":
                raise ValueError(
                    "Подборка уже не ожидает проверки: "
                    f"batch_status={record['batch_status']}"
                )

            review_action_id = await connection.fetchval(
                """
                INSERT INTO review_actions (
                    generated_post_id,
                    reviewer_type,
                    reviewer_telegram_user_id,
                    decision,
                    requires_human_review,
                    comment_text,
                    review_details
                )
                VALUES (
                    $1,
                    'human',
                    $2,
                    $3,
                    false,
                    $4,
                    $5::jsonb
                )
                RETURNING review_action_id
                """,
                generated_post_id,
                reviewer_telegram_user_id,
                decision,
                (
                    "Решение принято через "
                    "inline-кнопку Telegram-бота."
                ),
                review_details,
            )

            await connection.execute(
                """
                UPDATE generated_posts
                SET post_status = $2
                WHERE generated_post_id = $1
                """,
                generated_post_id,
                target_status,
            )

            if decision == "approve":
                await connection.execute(
                    """
                    UPDATE publication_batches
                    SET
                        batch_status = 'approved',
                        approved_at = now(),
                        approved_by_telegram_user_id = $2,
                        error_message = NULL
                    WHERE batch_id = $1
                    """,
                    record["batch_id"],
                    reviewer_telegram_user_id,
                )
            else:
                await connection.execute(
                    """
                    UPDATE publication_batches
                    SET
                        batch_status = 'rejected',
                        approved_at = NULL,
                        approved_by_telegram_user_id = NULL,
                        error_message = NULL
                    WHERE batch_id = $1
                    """,
                    record["batch_id"],
                )

    return ReviewDecisionResult(
        batch_id=record["batch_id"],
        generated_post_id=generated_post_id,
        review_action_id=review_action_id,
        decision=decision,
        post_status=target_status,
        batch_status=target_status,
        already_processed=False,
    )