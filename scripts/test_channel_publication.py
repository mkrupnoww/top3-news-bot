import asyncio
from datetime import UTC, datetime

from aiogram import Bot
from aiogram.enums import ChatType

from app.config import get_settings


async def main() -> None:
    """Проверяет доступ к каналу и отправляет тестовое сообщение."""

    settings = get_settings()

    async with Bot(
        token=settings.telegram_bot_token.get_secret_value()
    ).context() as bot:
        chat = await bot.get_chat(settings.telegram_channel_id)

        if chat.type != ChatType.CHANNEL:
            raise RuntimeError(
                "TELEGRAM_CHANNEL_ID указывает не на Telegram-канал: "
                f"chat_type={chat.type}"
            )

        message = await bot.send_message(
            chat_id=settings.telegram_channel_id,
            text=(
                "✅ Тестовая публикация TOP 3 Movie News\n\n"
                "Бот успешно подключён к тестовому каналу "
                "и может публиковать сообщения.\n\n"
                "Это техническое тестовое сообщение. Его можно удалить.\n\n"
                f"Время проверки: "
                f"{datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC"
            ),
            disable_notification=True,
        )

    print("Telegram channel access: OK")
    print(f"channel_id={chat.id}")
    print(f"channel_title={chat.title}")
    print(f"message_id={message.message_id}")
    print("Test publication: OK")


if __name__ == "__main__":
    asyncio.run(main())