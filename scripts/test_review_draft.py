import asyncio
from datetime import UTC, datetime

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.review_drafts import create_review_draft


async def main() -> None:
    """Создаёт тестовый пост для ручной проверки без публикации."""

    settings = get_settings()
    now = datetime.now(UTC)

    post_text = (
        "🎬 Предпросмотр TOP 3 Movie News\n\n"
        "Это тестовый черновик будущей публикации.\n\n"
        "Он сохранён в PostgreSQL со статусом "
        "awaiting_review, но ещё не отправлен "
        "в Telegram-канал.\n\n"
        f"Время создания: {now:%Y-%m-%d %H:%M:%S} UTC"
    )

    metadata = {
        "technical_test": True,
        "scenario": "review_draft_test",
        "script": "scripts.test_review_draft",
        "created_at": now.isoformat(),
        "batch_items_expected": False,
    }

    database_pool = await create_database_pool(settings)

    try:
        draft = await create_review_draft(
            database_pool,
            publication_date=now.date(),
            telegram_chat_id=settings.telegram_channel_id,
            post_text=post_text,
            text_format="plain_text",
            metadata=metadata,
        )

        attempt_count = await database_pool.fetchval(
            """
            SELECT COUNT(*)::integer
            FROM publication_attempts
            WHERE generated_post_id = $1
            """,
            draft.generated_post_id,
        )

        if attempt_count != 0:
            raise RuntimeError(
                "Для черновика неожиданно создана "
                "попытка публикации."
            )

    finally:
        await close_database_pool(database_pool)

    print("Review draft created")
    print(f"batch_id={draft.batch_id}")
    print(f"generated_post_id={draft.generated_post_id}")
    print(f"publication_date={draft.publication_date}")
    print(f"edition={draft.edition}")
    print(f"version_number={draft.version_number}")
    print(f"batch_status={draft.batch_status}")
    print(f"post_status={draft.post_status}")
    print(f"publication_attempt_count={attempt_count}")
    print("Telegram publication: not performed")
    print("Review draft test: OK")


if __name__ == "__main__":
    asyncio.run(main())