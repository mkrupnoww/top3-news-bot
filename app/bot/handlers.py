from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message


router = Router(name=__name__)


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Обрабатывает команду /start и показывает данные пользователя."""

    if message.from_user is None:
        await message.answer(
            "Не удалось определить пользователя Telegram."
        )
        return

    username = (
        f"@{message.from_user.username}"
        if message.from_user.username
        else "не указан"
    )

    await message.answer(
        "TOP 3 Movie News запущен.\n\n"
        f"Имя: {message.from_user.full_name}\n"
        f"Username: {username}\n"
        f"Telegram user ID: {message.from_user.id}\n\n"
        "Сейчас бот работает в тестовом режиме."
    )