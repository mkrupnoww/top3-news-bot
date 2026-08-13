from dataclasses import dataclass
import json
from typing import Any, Mapping

import asyncpg

from app.db.review_queue import (
    ReviewDraftPreview,
)
from app.db.users import BotUser


@dataclass(frozen=True, slots=True)
class ReviewDeliveryReservation:
    """Зарезервированная доставка review в Telegram."""

    review_delivery_attempt_id: int
    generated_post_id: int
    telegram_user_id: int
    telegram_chat_id: int
    attempt_number: int
    delivery_status: str
    created_new: bool

    @property
    def should_send(self) -> bool:
        """Показывает, нужно ли вызывать Telegram."""

        return (
            self.created_new
            and self.delivery_status == "reserved"
        )


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует Mapping в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def list_active_reviewers(
    pool: asyncpg.Pool,
) -> tuple[BotUser, ...]:
    """
    Возвращает активных пользователей,
    имеющих право review.
    """

    query = """
        SELECT
            telegram_user_id,
            telegram_username,
            display_name,
            user_role
        FROM bot_users
        WHERE is_active = true
          AND user_role IN (
              'admin',
              'reviewer'
          )
        ORDER BY
            CASE
                WHEN user_role = 'admin'
                THEN 0
                ELSE 1
            END,
            telegram_user_id
    """

    async with pool.acquire() as connection:
        records = await connection.fetch(query)

    return tuple(
        BotUser(
            telegram_user_id=(
                record["telegram_user_id"]
            ),
            telegram_username=(
                record["telegram_username"]
            ),
            display_name=(
                record["display_name"]
            ),
            user_role=record["user_role"],
        )
        for record in records
    )


async def get_review_draft_by_id(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
) -> ReviewDraftPreview | None:
    """
    Возвращает конкретный generated_post,
    если он действительно ожидает review.

    Черновик с уже зафиксированным
    changes_required не возвращается.
    """

    if generated_post_id <= 0:
        raise ValueError(
            "generated_post_id должен "
            "быть больше нуля."
        )

    query = """
        SELECT
            b.batch_id,
            b.publication_date,
            b.edition,
            p.generated_post_id,
            p.version_number,
            p.post_text,
            p.text_format,
            p.image_path,
            p.image_sha256
        FROM publication_batches AS b
        JOIN generated_posts AS p
          ON p.batch_id = b.batch_id
        WHERE p.generated_post_id = $1
          AND b.batch_status = 'awaiting_review'
          AND p.post_status = 'awaiting_review'
          AND NOT EXISTS (
              SELECT 1
              FROM review_actions AS ra
              WHERE ra.generated_post_id =
                    p.generated_post_id
                AND ra.reviewer_type = 'human'
                AND ra.decision =
                    'changes_required'
                AND ra.requested_action =
                    'regenerate_text'
          )
    """

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            query,
            generated_post_id,
        )

    if record is None:
        return None

    return ReviewDraftPreview(
        batch_id=record["batch_id"],
        generated_post_id=(
            record["generated_post_id"]
        ),
        publication_date=(
            record["publication_date"]
        ),
        edition=record["edition"],
        version_number=(
            record["version_number"]
        ),
        post_text=record["post_text"],
        text_format=record["text_format"],
        image_path=record["image_path"],
        image_sha256=record["image_sha256"],
    )


async def reserve_review_delivery(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
    telegram_user_id: int,
    request_payload: Mapping[str, Any],
) -> ReviewDeliveryReservation:
    """
    Резервирует автоматическую доставку review.

    Для личного сообщения Telegram chat_id
    совпадает с telegram_user_id.

    Если для пары post/user уже существует
    reserved/sent/unknown, новый Telegram-call
    блокируется.
    """

    if generated_post_id <= 0:
        raise ValueError(
            "generated_post_id должен "
            "быть больше нуля."
        )

    if telegram_user_id <= 0:
        raise ValueError(
            "telegram_user_id должен "
            "быть больше нуля."
        )

    encoded_request = _encode_json(
        request_payload
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            post_record = await connection.fetchrow(
                """
                SELECT
                    p.generated_post_id,
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

            if post_record is None:
                raise LookupError(
                    "generated_post не найден: "
                    f"generated_post_id="
                    f"{generated_post_id}"
                )

            if (
                post_record["post_status"]
                != "awaiting_review"
                or post_record["batch_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Review delivery разрешён "
                    "только для awaiting_review: "
                    "post_status="
                    f"{post_record['post_status']}, "
                    "batch_status="
                    f"{post_record['batch_status']}"
                )

            reviewer_record = (
                await connection.fetchrow(
                    """
                    SELECT
                        telegram_user_id,
                        user_role,
                        is_active
                    FROM bot_users
                    WHERE telegram_user_id = $1
                    FOR SHARE
                    """,
                    telegram_user_id,
                )
            )

            if reviewer_record is None:
                raise LookupError(
                    "bot_user не найден: "
                    "telegram_user_id="
                    f"{telegram_user_id}"
                )

            if (
                reviewer_record["is_active"]
                is not True
            ):
                raise ValueError(
                    "Review delivery запрещён "
                    "неактивному bot_user."
                )

            if reviewer_record["user_role"] not in {
                "admin",
                "reviewer",
            }:
                raise ValueError(
                    "Review delivery разрешён "
                    "только admin/reviewer: "
                    f"user_role="
                    f"{reviewer_record['user_role']}"
                )

            existing = await connection.fetchrow(
                """
                SELECT
                    review_delivery_attempt_id,
                    generated_post_id,
                    telegram_user_id,
                    telegram_chat_id,
                    attempt_number,
                    delivery_status
                FROM review_delivery_attempts
                WHERE generated_post_id = $1
                  AND telegram_user_id = $2
                  AND delivery_status IN (
                      'reserved',
                      'sent',
                      'unknown'
                  )
                ORDER BY
                    attempt_number DESC
                LIMIT 1
                """,
                generated_post_id,
                telegram_user_id,
            )

            if existing is not None:
                return ReviewDeliveryReservation(
                    review_delivery_attempt_id=(
                        existing[
                            "review_delivery_attempt_id"
                        ]
                    ),
                    generated_post_id=(
                        existing[
                            "generated_post_id"
                        ]
                    ),
                    telegram_user_id=(
                        existing[
                            "telegram_user_id"
                        ]
                    ),
                    telegram_chat_id=(
                        existing[
                            "telegram_chat_id"
                        ]
                    ),
                    attempt_number=(
                        existing["attempt_number"]
                    ),
                    delivery_status=(
                        existing["delivery_status"]
                    ),
                    created_new=False,
                )

            attempt_number = (
                await connection.fetchval(
                    """
                    SELECT
                        COALESCE(
                            MAX(attempt_number),
                            0
                        )::integer + 1
                    FROM review_delivery_attempts
                    WHERE generated_post_id = $1
                      AND telegram_user_id = $2
                    """,
                    generated_post_id,
                    telegram_user_id,
                )
            )

            record = await connection.fetchrow(
                """
                INSERT INTO review_delivery_attempts (
                    generated_post_id,
                    telegram_user_id,
                    attempt_number,
                    delivery_status,
                    telegram_chat_id,
                    request_payload
                )
                VALUES (
                    $1,
                    $2,
                    $3,
                    'reserved',
                    $2,
                    $4::jsonb
                )
                RETURNING
                    review_delivery_attempt_id,
                    generated_post_id,
                    telegram_user_id,
                    telegram_chat_id,
                    attempt_number,
                    delivery_status
                """,
                generated_post_id,
                telegram_user_id,
                attempt_number,
                encoded_request,
            )

    if record is None:
        raise RuntimeError(
            "Reservation review delivery "
            "не была создана."
        )

    return ReviewDeliveryReservation(
        review_delivery_attempt_id=(
            record[
                "review_delivery_attempt_id"
            ]
        ),
        generated_post_id=(
            record["generated_post_id"]
        ),
        telegram_user_id=(
            record["telegram_user_id"]
        ),
        telegram_chat_id=(
            record["telegram_chat_id"]
        ),
        attempt_number=(
            record["attempt_number"]
        ),
        delivery_status=(
            record["delivery_status"]
        ),
        created_new=True,
    )


async def mark_review_delivery_sent(
    pool: asyncpg.Pool,
    reservation: ReviewDeliveryReservation,
    *,
    telegram_message_id: int,
    response_payload: Mapping[str, Any],
) -> None:
    """Фиксирует подтверждённую Telegram-доставку."""

    if telegram_message_id <= 0:
        raise ValueError(
            "telegram_message_id должен "
            "быть больше нуля."
        )

    encoded_response = _encode_json(
        response_payload
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            UPDATE review_delivery_attempts
            SET
                delivery_status = 'sent',
                telegram_message_id = $2,
                response_payload = $3::jsonb,
                telegram_error_code = NULL,
                error_type = NULL,
                error_message = NULL,
                sent_at = now(),
                failed_at = NULL
            WHERE review_delivery_attempt_id = $1
              AND delivery_status = 'reserved'
            RETURNING review_delivery_attempt_id
            """,
            reservation.review_delivery_attempt_id,
            telegram_message_id,
            encoded_response,
        )

    if record is None:
        raise RuntimeError(
            "Review delivery нельзя перевести "
            "из reserved в sent: "
            "review_delivery_attempt_id="
            f"{reservation.review_delivery_attempt_id}"
        )


async def mark_review_delivery_failed(
    pool: asyncpg.Pool,
    reservation: ReviewDeliveryReservation,
    *,
    error_type: str,
    error_message: str,
    telegram_error_code: int | None = None,
) -> None:
    """
    Фиксирует ошибку без подтверждённого message_id.

    После failed новая попытка разрешена.
    """

    normalized_error_type = error_type.strip()
    normalized_error_message = (
        error_message.strip()
    )

    if not normalized_error_type:
        raise ValueError(
            "error_type не может быть пустым."
        )

    if not normalized_error_message:
        raise ValueError(
            "error_message не может быть пустым."
        )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            UPDATE review_delivery_attempts
            SET
                delivery_status = 'failed',
                telegram_message_id = NULL,
                response_payload = '{}'::jsonb,
                telegram_error_code = $2,
                error_type = $3,
                error_message = $4,
                sent_at = NULL,
                failed_at = now()
            WHERE review_delivery_attempt_id = $1
              AND delivery_status = 'reserved'
            RETURNING review_delivery_attempt_id
            """,
            reservation.review_delivery_attempt_id,
            telegram_error_code,
            normalized_error_type,
            normalized_error_message,
        )

    if record is None:
        raise RuntimeError(
            "Review delivery нельзя перевести "
            "из reserved в failed: "
            "review_delivery_attempt_id="
            f"{reservation.review_delivery_attempt_id}"
        )


async def mark_review_delivery_unknown(
    pool: asyncpg.Pool,
    reservation: ReviewDeliveryReservation,
    *,
    telegram_message_id: int | None,
    response_payload: Mapping[str, Any],
    error_type: str,
    error_message: str,
) -> None:
    """
    Фиксирует Telegram-доставку с неопределённым исходом.

    telegram_message_id может отсутствовать, если
    сетевой сбой произошёл после отправки запроса,
    но до получения подтверждённого ответа Telegram.

    После unknown автоматический retry запрещён.
    """

    if (
        telegram_message_id is not None
        and telegram_message_id <= 0
    ):
        raise ValueError(
            "telegram_message_id должен "
            "быть больше нуля или None."
        )

    normalized_error_type = error_type.strip()
    normalized_error_message = (
        error_message.strip()
    )

    if not normalized_error_type:
        raise ValueError(
            "error_type не может быть пустым."
        )

    if not normalized_error_message:
        raise ValueError(
            "error_message не может быть пустым."
        )

    encoded_response = _encode_json(
        response_payload
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            UPDATE review_delivery_attempts
            SET
                delivery_status = 'unknown',
                telegram_message_id = $2,
                response_payload = $3::jsonb,
                telegram_error_code = NULL,
                error_type = $4,
                error_message = $5,
                sent_at = CASE
                    WHEN $2::bigint IS NOT NULL
                    THEN now()
                    ELSE NULL
                END,
                failed_at = NULL
            WHERE review_delivery_attempt_id = $1
              AND delivery_status = 'reserved'
            RETURNING review_delivery_attempt_id
            """,
            reservation.review_delivery_attempt_id,
            telegram_message_id,
            encoded_response,
            normalized_error_type,
            normalized_error_message,
        )

    if record is None:
        raise RuntimeError(
            "Review delivery нельзя перевести "
            "из reserved в unknown: "
            "review_delivery_attempt_id="
            f"{reservation.review_delivery_attempt_id}"
        )
