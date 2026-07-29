import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher

from app.bot.handlers import router
from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


async def main() -> None:
    """Запускает Telegram-бота и подключение к PostgreSQL."""

    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        stream=sys.stdout,
    )

    logger = logging.getLogger(__name__)

    database_pool = await create_database_pool(settings)

    try:
        logger.info(
            "PostgreSQL connection pool created: "
            "database=%s, user=%s, schema=%s",
            settings.db_name,
            settings.db_user,
            settings.db_schema,
        )

        dispatcher = Dispatcher()
        dispatcher.include_router(router)

        async with Bot(
            token=settings.telegram_bot_token.get_secret_value()
        ).context() as bot:
            bot_info = await bot.get_me()

            logger.info(
                "Starting Telegram bot @%s, id=%s",
                bot_info.username,
                bot_info.id,
            )

            await bot.delete_webhook(
                drop_pending_updates=True,
            )

            await dispatcher.start_polling(
                bot,
                allowed_updates=(
                    dispatcher.resolve_used_update_types()
                ),
                close_bot_session=False,
                db_pool=database_pool,
            )

    finally:
        await close_database_pool(database_pool)
        logger.info("PostgreSQL connection pool closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info(
            "Telegram bot stopped by user"
        )