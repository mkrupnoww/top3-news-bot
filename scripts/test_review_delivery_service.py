import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import tempfile

import asyncpg
from aiogram.types import (
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from app.bot.review_delivery_service import (
    deliver_generated_post_to_reviewers,
)
from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


SUCCESS_USER_ID = (
    900_000_000_000_000_101
)

TIMEOUT_USER_ID = (
    900_000_000_000_000_102
)


class _SingleConnectionAcquire:
    """Context manager одной connection."""

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
    """Pool-like wrapper для rollback test."""

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


class FakeSuccessBot:
    """Fake Telegram с успешным send_photo."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[
            dict[str, Any]
        ] = []

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: FSInputFile,
        caption: str,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup,
    ) -> object:
        self.call_count += 1

        self.calls.append(
            {
                "chat_id": chat_id,
                "photo": photo,
                "caption": caption,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
            }
        )

        return SimpleNamespace(
            message_id=700_000
            + self.call_count
        )


class FakeTimeoutBot:
    """Fake Telegram с неопределённым timeout."""

    def __init__(self) -> None:
        self.call_count = 0

    async def send_photo(
        self,
        *,
        chat_id: int,
        photo: FSInputFile,
        caption: str,
        parse_mode: str | None,
        reply_markup: InlineKeyboardMarkup,
    ) -> object:
        self.call_count += 1

        raise TimeoutError(
            "Synthetic Telegram timeout."
        )


def build_keyboard(
    generated_post_id: int,
) -> InlineKeyboardMarkup:
    """Минимальная review keyboard."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Одобрить",
                    callback_data=(
                        "review:approve:"
                        f"{generated_post_id}"
                    ),
                )
            ]
        ]
    )


async def prepare_test_post(
    connection: asyncpg.Connection,
    *,
    image_path: Path,
    image_sha256: str,
) -> int:
    """Подготавливает реальный awaiting_review post."""

    generated_post_id = await connection.fetchval(
        """
        SELECT
            p.generated_post_id
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
            "Нет awaiting_review generated_post."
        )

    await connection.execute(
        """
        UPDATE generated_posts
        SET
            image_path = $2,
            image_sha256 = $3
        WHERE generated_post_id = $1
        """,
        generated_post_id,
        str(image_path),
        image_sha256,
    )

    return int(
        generated_post_id
    )


async def disable_real_reviewers(
    connection: asyncpg.Connection,
) -> None:
    """Временно отключает реальные review recipients."""

    await connection.execute(
        """
        UPDATE bot_users
        SET is_active = false
        WHERE is_active = true
          AND user_role IN (
              'admin',
              'reviewer'
          )
        """
    )


async def insert_reviewer(
    connection: asyncpg.Connection,
    *,
    telegram_user_id: int,
    display_name: str,
) -> None:
    """Создаёт synthetic reviewer."""

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
            $2,
            'reviewer',
            true
        )
        """,
        telegram_user_id,
        display_name,
    )


async def set_only_active_reviewer(
    connection: asyncpg.Connection,
    *,
    telegram_user_id: int,
) -> None:
    """Оставляет активным одного synthetic reviewer."""

    await connection.execute(
        """
        UPDATE bot_users
        SET is_active = (
            telegram_user_id = $1
        )
        WHERE user_role IN (
            'admin',
            'reviewer'
        )
        """,
        telegram_user_id,
    )


async def run_test(
    connection: asyncpg.Connection,
    *,
    image_path: Path,
    image_sha256: str,
) -> None:
    """Проверяет production delivery service."""

    pool = _SingleConnectionPool(
        connection
    )

    generated_post_id = (
        await prepare_test_post(
            connection,
            image_path=image_path,
            image_sha256=image_sha256,
        )
    )

    keyboard = build_keyboard(
        generated_post_id
    )

    await disable_real_reviewers(
        connection
    )

    await insert_reviewer(
        connection,
        telegram_user_id=(
            SUCCESS_USER_ID
        ),
        display_name=(
            "Synthetic Success Reviewer"
        ),
    )

    success_bot = FakeSuccessBot()

    result = (
        await deliver_generated_post_to_reviewers(
            pool,
            bot=success_bot,
            generated_post_id=(
                generated_post_id
            ),
            reply_markup=keyboard,
        )
    )

    assert result.reviewer_count == 1
    assert result.sent_count == 1
    assert result.failed_count == 0
    assert result.unknown_count == 0
    assert success_bot.call_count == 1

    sent_row = await connection.fetchrow(
        """
        SELECT
            delivery_status,
            telegram_message_id
        FROM review_delivery_attempts
        WHERE generated_post_id = $1
          AND telegram_user_id = $2
        ORDER BY attempt_number DESC
        LIMIT 1
        """,
        generated_post_id,
        SUCCESS_USER_ID,
    )

    assert sent_row is not None
    assert (
        sent_row["delivery_status"]
        == "sent"
    )
    assert (
        sent_row["telegram_message_id"]
        == 700001
    )

    print("Successful delivery lifecycle: OK")

    duplicate_result = (
        await deliver_generated_post_to_reviewers(
            pool,
            bot=success_bot,
            generated_post_id=(
                generated_post_id
            ),
            reply_markup=keyboard,
        )
    )

    assert duplicate_result.reviewer_count == 1
    assert duplicate_result.skipped_count == 1
    assert success_bot.call_count == 1

    print("Successful duplicate blocking: OK")

    await insert_reviewer(
        connection,
        telegram_user_id=(
            TIMEOUT_USER_ID
        ),
        display_name=(
            "Synthetic Timeout Reviewer"
        ),
    )

    await set_only_active_reviewer(
        connection,
        telegram_user_id=(
            TIMEOUT_USER_ID
        ),
    )

    timeout_bot = FakeTimeoutBot()

    timeout_result = (
        await deliver_generated_post_to_reviewers(
            pool,
            bot=timeout_bot,
            generated_post_id=(
                generated_post_id
            ),
            reply_markup=keyboard,
        )
    )

    assert timeout_result.reviewer_count == 1
    assert timeout_result.sent_count == 0
    assert timeout_result.failed_count == 0
    assert timeout_result.unknown_count == 1
    assert timeout_bot.call_count == 1

    unknown_row = await connection.fetchrow(
        """
        SELECT
            delivery_status,
            telegram_message_id,
            sent_at,
            failed_at
        FROM review_delivery_attempts
        WHERE generated_post_id = $1
          AND telegram_user_id = $2
        ORDER BY attempt_number DESC
        LIMIT 1
        """,
        generated_post_id,
        TIMEOUT_USER_ID,
    )

    assert unknown_row is not None
    assert (
        unknown_row["delivery_status"]
        == "unknown"
    )
    assert (
        unknown_row["telegram_message_id"]
        is None
    )
    assert unknown_row["sent_at"] is None
    assert unknown_row["failed_at"] is None

    print("Timeout becomes unknown: OK")

    duplicate_timeout_result = (
        await deliver_generated_post_to_reviewers(
            pool,
            bot=timeout_bot,
            generated_post_id=(
                generated_post_id
            ),
            reply_markup=keyboard,
        )
    )

    assert (
        duplicate_timeout_result
        .skipped_count
        == 1
    )

    assert timeout_bot.call_count == 1

    print("Unknown duplicate blocking: OK")

    print()
    print(
        "generated_post_id="
        f"{generated_post_id}"
    )


async def main() -> int:
    """Запускает integration test с rollback."""

    print("Review delivery service test")
    print("Telegram requests=not_performed")
    print("OpenAI requests=not_performed")
    print()

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = (
                Path(temp_dir)
                / "review_delivery_test.png"
            )

            image_bytes = (
                b"\x89PNG\r\n\x1a\n"
                b"synthetic-review-delivery-data"
            )

            image_path.write_bytes(
                image_bytes
            )

            image_sha256 = hashlib.sha256(
                image_bytes
            ).hexdigest()

            async with (
                database_pool.acquire()
                as connection
            ):
                transaction = (
                    connection.transaction()
                )

                await transaction.start()

                try:
                    await run_test(
                        connection,
                        image_path=image_path,
                        image_sha256=(
                            image_sha256
                        ),
                    )
                finally:
                    await transaction.rollback()

    finally:
        await close_database_pool(
            database_pool
        )

    print()
    print("Database changes=rolled_back")
    print("Telegram requests=not_performed")
    print("OpenAI requests=not_performed")
    print(
        "Review delivery service test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )
