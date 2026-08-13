import asyncio

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.review_delivery import (
    get_review_draft_by_id,
    list_active_reviewers,
    mark_review_delivery_failed,
    mark_review_delivery_sent,
    mark_review_delivery_unknown,
    reserve_review_delivery,
)


class _SingleConnectionAcquire:
    """Context manager одной asyncpg connection."""

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    async def __aenter__(
        self,
    ) -> asyncpg.Connection:
        return self._connection

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None


class _SingleConnectionPool:
    """Pool-like wrapper для тестовой транзакции."""

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    def acquire(
        self,
    ) -> _SingleConnectionAcquire:
        return _SingleConnectionAcquire(
            self._connection
        )


async def test_lifecycle(
    connection: asyncpg.Connection,
) -> None:
    """Проверяет reservation/idempotency/retry."""

    pool = _SingleConnectionPool(
        connection
    )

    generated_post_id = await connection.fetchval(
        """
        SELECT p.generated_post_id
        FROM generated_posts AS p
        JOIN publication_batches AS b
          ON b.batch_id = p.batch_id
        WHERE p.post_status = 'awaiting_review'
          AND b.batch_status = 'awaiting_review'
        ORDER BY
            b.publication_date DESC,
            b.edition DESC,
            p.version_number DESC
        LIMIT 1
        """
    )

    if generated_post_id is None:
        raise AssertionError(
            "Нет generated_post со статусом "
            "awaiting_review для теста."
        )

    draft = await get_review_draft_by_id(
        pool,
        generated_post_id=(
            int(generated_post_id)
        ),
    )

    if draft is None:
        raise AssertionError(
            "Не удалось загрузить точный "
            "review draft."
        )

    reviewers = await list_active_reviewers(
        pool
    )

    if not reviewers:
        raise AssertionError(
            "Нет активных admin/reviewer "
            "для теста."
        )

    reviewer = reviewers[0]

    request_payload = {
        "transport": "native_photo",
        "generated_post_id": (
            draft.generated_post_id
        ),
        "test": True,
    }

    first = await reserve_review_delivery(
        pool,
        generated_post_id=(
            draft.generated_post_id
        ),
        telegram_user_id=(
            reviewer.telegram_user_id
        ),
        request_payload=request_payload,
    )

    assert first.created_new is True
    assert first.should_send is True
    assert first.delivery_status == "reserved"
    assert first.attempt_number == 1

    duplicate_reserved = (
        await reserve_review_delivery(
            pool,
            generated_post_id=(
                draft.generated_post_id
            ),
            telegram_user_id=(
                reviewer.telegram_user_id
            ),
            request_payload=request_payload,
        )
    )

    assert (
        duplicate_reserved.created_new
        is False
    )

    assert (
        duplicate_reserved.should_send
        is False
    )

    assert (
        duplicate_reserved
        .review_delivery_attempt_id
        == first.review_delivery_attempt_id
    )

    print("Reserved duplicate blocking: OK")

    await mark_review_delivery_failed(
        pool,
        first,
        error_type="SyntheticNetworkError",
        error_message=(
            "Synthetic Telegram failure."
        ),
        telegram_error_code=None,
    )

    failed_status = await connection.fetchval(
        """
        SELECT delivery_status
        FROM review_delivery_attempts
        WHERE review_delivery_attempt_id = $1
        """,
        first.review_delivery_attempt_id,
    )

    assert failed_status == "failed"

    print("Failed without error code: OK")

    retry = await reserve_review_delivery(
        pool,
        generated_post_id=(
            draft.generated_post_id
        ),
        telegram_user_id=(
            reviewer.telegram_user_id
        ),
        request_payload=request_payload,
    )

    assert retry.created_new is True
    assert retry.should_send is True
    assert retry.attempt_number == 2

    print("Retry after failed: OK")

    await mark_review_delivery_sent(
        pool,
        retry,
        telegram_message_id=123456789,
        response_payload={
            "message_id": 123456789,
            "transport": "native_photo",
            "test": True,
        },
    )

    sent_status = await connection.fetchval(
        """
        SELECT delivery_status
        FROM review_delivery_attempts
        WHERE review_delivery_attempt_id = $1
        """,
        retry.review_delivery_attempt_id,
    )

    assert sent_status == "sent"

    duplicate_sent = (
        await reserve_review_delivery(
            pool,
            generated_post_id=(
                draft.generated_post_id
            ),
            telegram_user_id=(
                reviewer.telegram_user_id
            ),
            request_payload=request_payload,
        )
    )

    assert duplicate_sent.created_new is False
    assert duplicate_sent.should_send is False
    assert duplicate_sent.delivery_status == "sent"

    print("Sent duplicate blocking: OK")

    synthetic_user_id = (
        900_000_000_000_000_001
    )

    await connection.execute(
        """
        INSERT INTO bot_users (
            telegram_user_id,
            display_name,
            user_role,
            is_active
        )
        VALUES (
            $1,
            'Synthetic Review Delivery User',
            'reviewer',
            true
        )
        """,
        synthetic_user_id,
    )

    unknown_reservation = (
        await reserve_review_delivery(
            pool,
            generated_post_id=(
                draft.generated_post_id
            ),
            telegram_user_id=(
                synthetic_user_id
            ),
            request_payload={
                "transport": "native_photo",
                "generated_post_id": (
                    draft.generated_post_id
                ),
                "test": True,
                "scenario": (
                    "unknown_without_message_id"
                ),
            },
        )
    )

    assert unknown_reservation.should_send is True

    await mark_review_delivery_unknown(
        pool,
        unknown_reservation,
        telegram_message_id=None,
        response_payload={},
        error_type="SyntheticNetworkTimeout",
        error_message=(
            "Telegram response was not received."
        ),
    )

    unknown_record = await connection.fetchrow(
        """
        SELECT
            delivery_status,
            telegram_message_id,
            sent_at,
            failed_at
        FROM review_delivery_attempts
        WHERE review_delivery_attempt_id = $1
        """,
        (
            unknown_reservation
            .review_delivery_attempt_id
        ),
    )

    assert unknown_record is not None
    assert (
        unknown_record["delivery_status"]
        == "unknown"
    )
    assert (
        unknown_record["telegram_message_id"]
        is None
    )
    assert unknown_record["sent_at"] is None
    assert unknown_record["failed_at"] is None

    duplicate_unknown = (
        await reserve_review_delivery(
            pool,
            generated_post_id=(
                draft.generated_post_id
            ),
            telegram_user_id=(
                synthetic_user_id
            ),
            request_payload={
                "transport": "native_photo",
                "generated_post_id": (
                    draft.generated_post_id
                ),
                "test": True,
            },
        )
    )

    assert (
        duplicate_unknown.created_new
        is False
    )
    assert (
        duplicate_unknown.should_send
        is False
    )
    assert (
        duplicate_unknown.delivery_status
        == "unknown"
    )

    print(
        "Unknown without message ID "
        "duplicate blocking: OK"
    )

    print()
    print(
        "generated_post_id="
        f"{draft.generated_post_id}"
    )
    print(
        "telegram_user_id="
        f"{reviewer.telegram_user_id}"
    )


async def main() -> int:
    """Запускает DB lifecycle с полным rollback."""

    print("Review delivery lifecycle test")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print()

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        async with (
            database_pool.acquire()
            as connection
        ):
            transaction = (
                connection.transaction()
            )

            await transaction.start()

            try:
                await test_lifecycle(
                    connection
                )
            finally:
                await transaction.rollback()
    finally:
        await close_database_pool(
            database_pool
        )

    print()
    print("Database changes=rolled_back")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print(
        "Review delivery lifecycle test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
