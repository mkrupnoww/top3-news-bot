import asyncio

from aiogram import Bot

from app.config import get_settings


async def main() -> None:
    settings = get_settings()
    expected_username = settings.telegram_bot_username.lstrip("@").lower()

    async with Bot(
        token=settings.telegram_bot_token.get_secret_value()
    ).context() as bot:
        bot_info = await bot.get_me()

    actual_username = (bot_info.username or "").lower()

    print("Telegram Bot API connection: OK")
    print(f"bot_id={bot_info.id}")
    print(f"bot_username={bot_info.username}")
    print(f"bot_name={bot_info.first_name}")

    if actual_username != expected_username:
        raise RuntimeError(
            "Telegram bot username mismatch: "
            f"expected @{expected_username}, "
            f"received @{actual_username}"
        )

    print("Bot username verification: OK")


if __name__ == "__main__":
    asyncio.run(main())