import asyncio
from datetime import UTC, datetime

from aiogram import Bot
from aiogram.enums import ChatType

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.publications import (
    create_publication_attempt,
    mark_publication_failed,
    mark_publication_published,
)


async def main() -> None:
    """Публикует тестовое сообщение с полным аудитом в PostgreSQL."""

    settings = get_settings()
    now = datetime.now(UTC)

    post_text = (
        "✅ DB-тест публикации TOP 3 Movie News\n\n"
        "Сообщение отправлено через Telegram Bot API, "
        "а весь жизненный цикл публикации сохранён "
        "в PostgreSQL.\n\n"
        "Это техническое тестовое сообщение. Его можно удалить.\n\n"
        f"Время проверки: {now:%Y-%m-%d %H:%M:%S} UTC"
    )

    request_payload = {
        "chat_id": settings.telegram_channel_id,
        "text": post_text,
        "text_format": "plain_text",
        "disable_notification": True,
    }

    metadata = {
        "technical_test": True,
        "scenario": "database_publication_test",
        "script": "scripts.test_database_publication",
        "created_at": now.isoformat(),
        "batch_items_expected": False,
    }

    database_pool = await create_database_pool(settings)

    try:
        publication = await create_publication_attempt(
            database_pool,
            publication_date=now.date(),
            telegram_chat_id=settings.telegram_channel_id,
            post_text=post_text,
            request_payload=request_payload,
            metadata=metadata,
        )

        print("Database publication records created")
        print(f"batch_id={publication.batch_id}")
        print(f"generated_post_id={publication.generated_post_id}")
        print(
            "publication_attempt_id="
            f"{publication.publication_attempt_id}"
        )
        print(f"publication_date={publication.publication_date}")
        print(f"edition={publication.edition}")

        try:
            async with Bot(
                token=settings.telegram_bot_token.get_secret_value()
            ).context() as bot:
                chat = await bot.get_chat(
                    settings.telegram_channel_id
                )

                if chat.type != ChatType.CHANNEL:
                    raise RuntimeError(
                        "TELEGRAM_CHANNEL_ID указывает не на канал: "
                        f"chat_type={chat.type}"
                    )

                message = await bot.send_message(
                    chat_id=settings.telegram_channel_id,
                    text=post_text,
                    disable_notification=True,
                )

            response_payload = {
                "message_id": message.message_id,
                "chat_id": message.chat.id,
                "chat_type": message.chat.type.value,
                "chat_title": message.chat.title,
                "message_date": message.date.isoformat(),
            }

            await mark_publication_published(
                database_pool,
                publication,
                telegram_message_id=message.message_id,
                response_payload=response_payload,
            )

        except Exception as error:
            await mark_publication_failed(
                database_pool,
                publication,
                error_message=(
                    f"{type(error).__name__}: {error}"
                ),
            )
            raise

        print(f"telegram_message_id={message.message_id}")
        print("Database publication status: published")
        print("Test publication: OK")

    finally:
        await close_database_pool(database_pool)


if __name__ == "__main__":
    asyncio.run(main())