import argparse
import asyncio
from datetime import UTC, datetime
from typing import Any

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
    mark_publication_unknown,
)


def enum_value(value: Any) -> str:
    """Возвращает строковое значение Enum или обычной строки."""

    return str(getattr(value, "value", value))


def parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы тестового сценария."""

    parser = argparse.ArgumentParser(
        description=(
            "Публикует техническое сообщение и сохраняет "
            "жизненный цикл публикации в PostgreSQL."
        ),
    )

    parser.add_argument(
        "--simulate-finalization-failure",
        action="store_true",
        help=(
            "После успешной отправки в Telegram имитирует "
            "сбой финальной записи published."
        ),
    )

    return parser.parse_args()


async def main(
    *,
    simulate_finalization_failure: bool,
) -> int:
    """Публикует тестовое сообщение с полным аудитом в PostgreSQL."""

    settings = get_settings()
    now = datetime.now(UTC)

    post_text = (
        "✅ DB-тест публикации TOP 3 Movie News\n\n"
        "Сообщение отправлено через Telegram Bot API, "
        "а жизненный цикл публикации сохраняется "
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
        "simulate_finalization_failure": (
            simulate_finalization_failure
        ),
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

        # До получения message_id любая ошибка означает,
        # что подтверждённой публикации в Telegram нет.
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

        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"

            await mark_publication_failed(
                database_pool,
                publication,
                error_message=error_text,
            )

            raise

        response_payload = {
            "message_id": message.message_id,
            "chat_id": message.chat.id,
            "chat_type": enum_value(message.chat.type),
            "chat_title": message.chat.title,
            "message_date": message.date.isoformat(),
        }

        # После получения message_id повторная отправка опасна:
        # Telegram уже подтвердил создание сообщения.
        try:
            if simulate_finalization_failure:
                raise RuntimeError(
                    "Simulated database finalization failure"
                )

            await mark_publication_published(
                database_pool,
                publication,
                telegram_message_id=message.message_id,
                response_payload=response_payload,
            )

        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"

            await mark_publication_unknown(
                database_pool,
                publication,
                telegram_message_id=message.message_id,
                response_payload=response_payload,
                error_message=error_text,
            )

            print(f"telegram_message_id={message.message_id}")
            print("Database publication status: unknown")
            print(
                "Telegram confirmed the publication, "
                "but database finalization requires review."
            )

            return 2

        print(f"telegram_message_id={message.message_id}")
        print("Database publication status: published")
        print("Test publication: OK")

        return 0

    finally:
        await close_database_pool(database_pool)


if __name__ == "__main__":
    arguments = parse_arguments()

    raise SystemExit(
        asyncio.run(
            main(
                simulate_finalization_failure=(
                    arguments.simulate_finalization_failure
                ),
            )
        )
    )