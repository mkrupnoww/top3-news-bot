import asyncio
from datetime import UTC, datetime
import hashlib

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.review_drafts import (
    create_review_draft,
)
from app.db.review_queue import (
    get_latest_review_draft,
)


TEST_IMAGE_PATH = (
    "/tmp/top3-news-review-image-test.png"
)

TEST_IMAGE_SHA256 = hashlib.sha256(
    b"top3-news-review-image-test"
).hexdigest()


async def cleanup_test_draft(
    database_pool,
    *,
    batch_id: int,
    generated_post_id: int,
) -> None:
    """Удаляет только записи этого теста."""

    async with database_pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                DELETE FROM review_actions
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM publication_attempts
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM batch_items
                WHERE batch_id = $1
                """,
                batch_id,
            )

            await connection.execute(
                """
                DELETE FROM generated_posts
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM publication_batches
                WHERE batch_id = $1
                """,
                batch_id,
            )


async def main() -> None:
    """
    Проверяет передачу image_path/image_sha256
    из generated_posts в ReviewDraftPreview.

    OpenAI и Telegram не вызываются.
    """

    settings = get_settings()
    now = datetime.now(UTC)

    database_pool = await create_database_pool(
        settings
    )

    draft = None
    cleanup_completed = False

    try:
        draft = await create_review_draft(
            database_pool,
            publication_date=now.date(),
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            post_text=(
                "Тест image fields "
                "для ReviewDraftPreview."
            ),
            text_format="plain_text",
            metadata={
                "technical_test": True,
                "scenario": (
                    "review_draft_image_fields_test"
                ),
                "created_at": now.isoformat(),
            },
        )

        async with database_pool.acquire() as connection:
            await connection.execute(
                """
                UPDATE generated_posts
                SET
                    image_path = $2,
                    image_sha256 = $3
                WHERE generated_post_id = $1
                """,
                draft.generated_post_id,
                TEST_IMAGE_PATH,
                TEST_IMAGE_SHA256,
            )

        review_draft = await get_latest_review_draft(
            database_pool
        )

        if review_draft is None:
            raise RuntimeError(
                "get_latest_review_draft() "
                "не вернул тестовый черновик."
            )

        if (
            review_draft.generated_post_id
            != draft.generated_post_id
        ):
            raise RuntimeError(
                "get_latest_review_draft() "
                "вернул другой generated_post: "
                f"expected={draft.generated_post_id}, "
                "actual="
                f"{review_draft.generated_post_id}"
            )

        if (
            review_draft.image_path
            != TEST_IMAGE_PATH
        ):
            raise RuntimeError(
                "image_path не передан "
                "в ReviewDraftPreview: "
                f"expected={TEST_IMAGE_PATH!r}, "
                "actual="
                f"{review_draft.image_path!r}"
            )

        if (
            review_draft.image_sha256
            != TEST_IMAGE_SHA256
        ):
            raise RuntimeError(
                "image_sha256 не передан "
                "в ReviewDraftPreview: "
                f"expected={TEST_IMAGE_SHA256!r}, "
                "actual="
                f"{review_draft.image_sha256!r}"
            )

        print(
            "Review draft image fields test: OK"
        )
        print(
            f"batch_id={draft.batch_id}"
        )
        print(
            "generated_post_id="
            f"{draft.generated_post_id}"
        )
        print(
            "image_path="
            f"{review_draft.image_path}"
        )
        print(
            "image_sha256="
            f"{review_draft.image_sha256}"
        )
        print(
            "OpenAI requests: not performed"
        )
        print(
            "Telegram requests: not performed"
        )

    finally:
        if draft is not None:
            await cleanup_test_draft(
                database_pool,
                batch_id=draft.batch_id,
                generated_post_id=(
                    draft.generated_post_id
                ),
            )

            cleanup_completed = True

        await close_database_pool(
            database_pool
        )

        print(
            "cleanup_completed="
            f"{str(cleanup_completed).lower()}"
        )


if __name__ == "__main__":
    asyncio.run(main())