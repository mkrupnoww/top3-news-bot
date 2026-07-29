import logging

import asyncpg
from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from app.db.users import get_active_bot_user


logger = logging.getLogger(__name__)

router = Router(name=__name__)


@router.message(CommandStart())
async def handle_start(
    message: Message,
    db_pool: asyncpg.Pool,
) -> None:
    """Проверяет пользователя и обрабатывает команду /start."""

    telegram_user = message.from_user

    if telegram_user is None:
        await message.answer(
            "Не удалось определить пользователя Telegram."
        )
        return

    bot_user = await get_active_bot_user(
        pool=db_pool,
        telegram_user_id=telegram_user.id,
    )

    if bot_user is None:
        logger.warning(
            "Access denied for Telegram user id=%s, username=%s",
            telegram_user.id,
            telegram_user.username,
        )

        await message.answer(
            "Доступ к этому боту не предоставлен."
        )
        return

    logger.info(
        "Authorized Telegram user id=%s, role=%s",
        bot_user.telegram_user_id,
        bot_user.user_role,
    )

    username = (
        f"@{telegram_user.username}"
        if telegram_user.username
        else "не указан"
    )

    await message.answer(
        "TOP 3 Movie News запущен.\n\n"
        f"Пользователь: {bot_user.display_name}\n"
        f"Роль: {bot_user.user_role}\n"
        f"Username: {username}\n"
        f"Telegram user ID: {bot_user.telegram_user_id}\n\n"
        "Доступ подтверждён через PostgreSQL.\n"
        "Сейчас бот работает в тестовом режиме."
    )