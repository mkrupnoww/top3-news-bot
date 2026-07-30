import logging

import asyncpg
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.db.review_queue import (
    get_latest_review_draft,
    record_human_review_decision,
)
from app.db.users import BotUser, get_active_bot_user


logger = logging.getLogger(__name__)

router = Router(name=__name__)


def build_review_keyboard(
    generated_post_id: int,
) -> InlineKeyboardMarkup:
    """Создаёт кнопки ручного решения по черновику."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Одобрить",
                    callback_data=(
                        f"review:approve:{generated_post_id}"
                    ),
                ),
                InlineKeyboardButton(
                    text="❌ Отклонить",
                    callback_data=(
                        f"review:reject:{generated_post_id}"
                    ),
                ),
            ]
        ]
    )


async def get_authorized_user(
    pool: asyncpg.Pool,
    telegram_user_id: int,
) -> BotUser | None:
    """Возвращает активного пользователя бота."""

    return await get_active_bot_user(
        pool=pool,
        telegram_user_id=telegram_user_id,
    )


def can_review(bot_user: BotUser) -> bool:
    """Проверяет право пользователя принимать решение."""

    return bot_user.user_role in {"admin", "reviewer"}


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

    bot_user = await get_authorized_user(
        db_pool,
        telegram_user.id,
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
        "Команда проверки черновика: /review"
    )


@router.message(Command("review"))
async def handle_review(
    message: Message,
    db_pool: asyncpg.Pool,
) -> None:
    """Показывает последний черновик на ручную проверку."""

    telegram_user = message.from_user
    if telegram_user is None:
        await message.answer(
            "Не удалось определить пользователя Telegram."
        )
        return

    bot_user = await get_authorized_user(
        db_pool,
        telegram_user.id,
    )

    if bot_user is None:
        await message.answer(
            "Доступ к этому боту не предоставлен."
        )
        return

    if not can_review(bot_user):
        await message.answer(
            "У вашей роли нет права проверять публикации."
        )
        return

    draft = await get_latest_review_draft(db_pool)

    if draft is None:
        await message.answer(
            "Сейчас нет черновиков со статусом awaiting_review."
        )
        return

    if len(draft.post_text) > 4096:
        await message.answer(
            "Черновик превышает лимит Telegram в 4096 символов. "
            "Его нельзя одобрить до сокращения."
        )
        return

    await message.answer(
        "Черновик на проверку\n\n"
        f"Дата публикации: {draft.publication_date}\n"
        f"Выпуск: {draft.edition}\n"
        f"Batch ID: {draft.batch_id}\n"
        f"Generated post ID: {draft.generated_post_id}\n"
        f"Версия: {draft.version_number}\n"
        f"Формат: {draft.text_format}"
    )

    await message.answer(
        draft.post_text,
        reply_markup=build_review_keyboard(
            draft.generated_post_id
        ),
    )


@router.callback_query(F.data.startswith("review:"))
async def handle_review_callback(
    callback: CallbackQuery,
    db_pool: asyncpg.Pool,
) -> None:
    """Обрабатывает одобрение или отклонение черновика."""

    callback_data = callback.data
    if callback_data is None:
        await callback.answer(
            "Некорректные данные кнопки.",
            show_alert=True,
        )
        return

    parts = callback_data.split(":")
    if len(parts) != 3:
        await callback.answer(
            "Некорректный формат команды.",
            show_alert=True,
        )
        return

    _, action, generated_post_id_text = parts

    if action not in {"approve", "reject"}:
        await callback.answer(
            "Неизвестное действие.",
            show_alert=True,
        )
        return

    try:
        generated_post_id = int(generated_post_id_text)
    except ValueError:
        await callback.answer(
            "Некорректный ID черновика.",
            show_alert=True,
        )
        return

    bot_user = await get_authorized_user(
        db_pool,
        callback.from_user.id,
    )

    if bot_user is None:
        await callback.answer(
            "Доступ не предоставлен.",
            show_alert=True,
        )
        return

    if not can_review(bot_user):
        await callback.answer(
            "Недостаточно прав.",
            show_alert=True,
        )
        return

    try:
        result = await record_human_review_decision(
            db_pool,
            generated_post_id=generated_post_id,
            reviewer_telegram_user_id=(
                bot_user.telegram_user_id
            ),
            decision=action,
        )
    except (
        LookupError,
        PermissionError,
        ValueError,
    ) as error:
        logger.warning(
            "Review decision rejected: user_id=%s, "
            "generated_post_id=%s, error=%s",
            bot_user.telegram_user_id,
            generated_post_id,
            error,
        )
        await callback.answer(
            str(error),
            show_alert=True,
        )
        return
    except Exception:
        logger.exception(
            "Review decision failed: user_id=%s, "
            "generated_post_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
        )
        await callback.answer(
            "Не удалось сохранить решение.",
            show_alert=True,
        )
        return

    if callback.message is not None:
        await callback.message.edit_reply_markup(
            reply_markup=None
        )

    if result.already_processed:
        await callback.answer(
            "Это решение уже было сохранено.",
            show_alert=True,
        )
        return

    if result.decision == "approve":
        result_text = (
            "✅ Черновик одобрен.\n\n"
            f"Generated post ID: {result.generated_post_id}\n"
            f"Статус поста: {result.post_status}\n"
            f"Статус подборки: {result.batch_status}\n\n"
            "Публикация в канал пока не выполнялась."
        )
    else:
        result_text = (
            "❌ Черновик отклонён.\n\n"
            f"Generated post ID: {result.generated_post_id}\n"
            f"Статус поста: {result.post_status}\n"
            f"Статус подборки: {result.batch_status}"
        )

    await callback.answer("Решение сохранено.")

    if callback.message is not None:
        await callback.message.answer(result_text)

    logger.info(
        "Review decision saved: user_id=%s, "
        "generated_post_id=%s, decision=%s, "
        "review_action_id=%s",
        bot_user.telegram_user_id,
        result.generated_post_id,
        result.decision,
        result.review_action_id,
    )