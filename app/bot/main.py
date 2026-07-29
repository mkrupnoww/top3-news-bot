import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher

from app.bot.handlers import router
from app.config import get_settings


async def main() -> None:
    """Запускает Telegram-бота в режиме long polling."""

    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        stream=sys.stdout,
    )

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    async with Bot(
        token=settings.telegram_bot_token.get_secret_value()
    ).context() as bot:
        bot_info = await bot.get_me()

        logging.info(
            "Starting Telegram bot @%s, id=%s",
            bot_info.username,
            bot_info.id,
        )

        # На этапе разработки удаляем возможный старый webhook
        # и сбрасываем накопившиеся тестовые обновления.
        await bot.delete_webhook(drop_pending_updates=True)

        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
            close_bot_session=False,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Telegram bot stopped by user")