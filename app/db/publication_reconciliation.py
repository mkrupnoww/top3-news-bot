from dataclasses import dataclass
from datetime import UTC, datetime
import json

import asyncpg


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Результат ручной сверки публикации."""

    publication_attempt_id: int
    generated_post_id: int
    batch_id: int
    telegram_message_id: int
    previous_attempt_status: str
    current_attempt_status: str
    already_reconciled: bool


async def reconcile_unknown_publication(
    pool: asyncpg.Pool,
    *,
    publication_attempt_id: int,
    expected_telegram_message_id: int,
    confirmed_by_telegram_user_id: int,
    confirmation_note: str,
) -> ReconciliationResult:
    """
    Подтверждает существование уже отправленного Telegram-сообщения.

    Повторная отправка не выполняется. Разрешён только безопасный переход
    publication_attempts.attempt_status из unknown в published.
    """

    reconciliation_time = datetime.now(UTC)

    reconciliation_payload = json.dumps(
        {
            "reconciliation": {
                "method": "manual_channel_confirmation",
                "confirmed_by_telegram_user_id": (
                    confirmed_by_telegram_user_id
                ),
                "confirmed_at": reconciliation_time.isoformat(),
                "confirmation_note": confirmation_note,
                "expected_telegram_message_id": (
                    expected_telegram_message_id
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            administrator_exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM bot_users
                    WHERE telegram_user_id = $1
                      AND user_role = 'admin'
                      AND is_active = true
                )
                """,
                confirmed_by_telegram_user_id,
            )

            if not administrator_exists:
                raise PermissionError(
                    "Пользователь Telegram не является "
                    "активным администратором."
                )

            record = await connection.fetchrow(
                """
                SELECT
                    a.publication_attempt_id,
                    a.generated_post_id,
                    a.attempt_status,
                    a.telegram_message_id,
                    p.batch_id,
                    p.post_status,
                    b.batch_status
                FROM publication_attempts AS a
                JOIN generated_posts AS p
                    ON p.generated_post_id = a.generated_post_id
                JOIN publication_batches AS b
                    ON b.batch_id = p.batch_id
                WHERE a.publication_attempt_id = $1
                FOR UPDATE OF a, p, b
                """,
                publication_attempt_id,
            )

            if record is None:
                raise LookupError(
                    "Попытка публикации не найдена: "
                    f"publication_attempt_id={publication_attempt_id}"
                )

            stored_message_id = record["telegram_message_id"]

            if stored_message_id is None:
                raise ValueError(
                    "У попытки публикации отсутствует "
                    "telegram_message_id."
                )

            if stored_message_id != expected_telegram_message_id:
                raise ValueError(
                    "telegram_message_id не совпадает: "
                    f"stored={stored_message_id}, "
                    f"expected={expected_telegram_message_id}"
                )

            previous_attempt_status = record["attempt_status"]

            if previous_attempt_status == "published":
                return ReconciliationResult(
                    publication_attempt_id=record[
                        "publication_attempt_id"
                    ],
                    generated_post_id=record["generated_post_id"],
                    batch_id=record["batch_id"],
                    telegram_message_id=stored_message_id,
                    previous_attempt_status=previous_attempt_status,
                    current_attempt_status="published",
                    already_reconciled=True,
                )

            if previous_attempt_status != "unknown":
                raise ValueError(
                    "Ручная сверка разрешена только для статуса "
                    f"unknown: current_status={previous_attempt_status}"
                )

            if record["post_status"] != "published":
                raise ValueError(
                    "Связанный generated_post должен иметь статус "
                    f"published: current_status={record['post_status']}"
                )

            if record["batch_status"] != "published":
                raise ValueError(
                    "Связанный publication_batch должен иметь статус "
                    f"published: current_status={record['batch_status']}"
                )

            await connection.execute(
                """
                UPDATE publication_attempts
                SET
                    attempt_status = 'published',
                    response_payload = (
                        response_payload || $2::jsonb
                    ),
                    telegram_error_code = NULL,
                    error_message = NULL,
                    finished_at = COALESCE(finished_at, now())
                WHERE publication_attempt_id = $1
                """,
                publication_attempt_id,
                reconciliation_payload,
            )

            await connection.execute(
                """
                UPDATE generated_posts
                SET post_status = 'published'
                WHERE generated_post_id = $1
                """,
                record["generated_post_id"],
            )

            await connection.execute(
                """
                UPDATE publication_batches
                SET
                    batch_status = 'published',
                    published_at = COALESCE(published_at, now()),
                    error_message = NULL
                WHERE batch_id = $1
                """,
                record["batch_id"],
            )

    return ReconciliationResult(
        publication_attempt_id=record["publication_attempt_id"],
        generated_post_id=record["generated_post_id"],
        batch_id=record["batch_id"],
        telegram_message_id=stored_message_id,
        previous_attempt_status=previous_attempt_status,
        current_attempt_status="published",
        already_reconciled=False,
    )