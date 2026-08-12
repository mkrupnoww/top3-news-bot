import inspect
import logging

import asyncpg
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from app.bot.review_source_messages import (
    build_review_source_messages,
)
from app.bot.review_preview import (
    ReviewImageUnavailableError,
    send_review_draft_preview,
)
from app.config import get_settings
from app.db.review_queue import (
    get_latest_review_draft,
    record_human_review_decision,
)
from app.db.review_sources import (
    get_review_sources,
)
from app.db.users import BotUser, get_active_bot_user
from app.generation.openai_factory import (
    create_openai_generation_runtime,
)
from app.generation.openai_revision_pipeline import (
    run_reserved_openai_generation_revision,
)
from app.publication import (
    PublicationStateUncertainError,
    publish_approved_post,
)


logger = logging.getLogger(__name__)

router = Router(name=__name__)


class ReviewRevisionStates(StatesGroup):
    """Состояния диалога редакционной доработки."""

    waiting_for_comment = State()


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
                    text="✏️ Изменить",
                    callback_data=(
                        f"review:change:{generated_post_id}"
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


def build_publish_keyboard(
    generated_post_id: int,
) -> InlineKeyboardMarkup:
    """Создаёт кнопку публикации одобренного поста."""

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📢 Опубликовать в канал",
                    callback_data=(
                        f"publish:approved:{generated_post_id}"
                    ),
                )
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


def can_publish(bot_user: BotUser) -> bool:
    """Проверяет право пользователя публиковать в канал."""

    return bot_user.user_role == "admin"


async def remove_callback_keyboard(
    callback: CallbackQuery,
) -> None:
    """Удаляет inline-кнопки, не меняя результат операции."""

    if callback.message is None:
        return

    try:
        await callback.message.edit_reply_markup(
            reply_markup=None
        )
    except Exception:
        logger.exception(
            "Не удалось удалить inline-клавиатуру: "
            "callback_data=%s",
            callback.data,
        )


async def send_callback_message(
    callback: CallbackQuery,
    text: str,
) -> None:
    """Отправляет результат операции пользователю."""

    if callback.message is not None:
        await callback.message.answer(text)
        return

    await callback.bot.send_message(
        chat_id=callback.from_user.id,
        text=text,
    )


async def close_sdk_client(
    sdk_client: object,
) -> None:
    """Закрывает совместимый OpenAI SDK-клиент."""

    close_method = getattr(
        sdk_client,
        "close",
        None,
    )

    if close_method is None:
        return

    try:
        close_result = close_method()

        if inspect.isawaitable(close_result):
            await close_result
    except Exception:
        logger.exception(
            "Не удалось закрыть OpenAI SDK-клиент."
        )


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
    state: FSMContext,
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

    await state.clear()

    draft = await get_latest_review_draft(
        db_pool
    )

    if draft is None:
        await message.answer(
            "Сейчас нет черновиков со статусом "
            "awaiting_review."
        )
        return

    if len(draft.post_text) > 4096:
        await message.answer(
            "Черновик превышает лимит Telegram "
            "в 4096 символов. Его нельзя одобрить "
            "до сокращения."
        )
        return

    try:
        source_items = await get_review_sources(
            db_pool,
            generated_post_id=(
                draft.generated_post_id
            ),
        )

        source_messages = (
            build_review_source_messages(
                source_items
            )
        )

    except (LookupError, ValueError) as error:
        logger.warning(
            "Review source validation failed: "
            "generated_post_id=%s, error=%s",
            draft.generated_post_id,
            error,
        )

        await message.answer(
            "Не удалось подготовить досье "
            "источников выпуска.\n\n"
            f"Generated post ID: "
            f"{draft.generated_post_id}\n"
            f"Ошибка: {error}\n\n"
            "Одобрение и публикация заблокированы."
        )
        return

    except Exception:
        logger.exception(
            "Review source loading failed: "
            "generated_post_id=%s",
            draft.generated_post_id,
        )

        await message.answer(
            "Не удалось загрузить источники выпуска.\n\n"
            "Одобрение и публикация заблокированы. "
            "Подробности сохранены в журнале."
        )
        return

    await message.answer(
        "Черновик на проверку\n\n"
        f"Дата публикации: "
        f"{draft.publication_date}\n"
        f"Выпуск: {draft.edition}\n"
        f"Batch ID: {draft.batch_id}\n"
        f"Generated post ID: "
        f"{draft.generated_post_id}\n"
        f"Версия: {draft.version_number}\n"
        f"Формат: {draft.text_format}\n"
        f"Источников: {len(source_items)}"
    )

    await message.answer(
        "Сначала проверьте источники "
        "трёх выбранных новостей."
    )

    for source_message in source_messages:
        await message.answer(
            source_message
        )

    await message.answer(
        "Итоговый текст публикации:"
    )

    try:
        await send_review_draft_preview(
            message,
            draft=draft,
            reply_markup=build_review_keyboard(
                draft.generated_post_id
            ),
        )

    except ReviewImageUnavailableError as error:
        logger.warning(
            "Review image unavailable: "
            "generated_post_id=%s, error=%s",
            draft.generated_post_id,
            error,
        )

        await message.answer(
            "Изображение выпуска ещё не готово.\n\n"
            f"Generated post ID: "
            f"{draft.generated_post_id}\n"
            f"Причина: {error}\n\n"
            "Одобрение заблокировано до "
            "успешной генерации PNG."
        )
        return

    except Exception:
        logger.exception(
            "Review Rich Message failed: "
            "generated_post_id=%s",
            draft.generated_post_id,
        )

        await message.answer(
            "Не удалось сформировать preview "
            "публикации с изображением.\n\n"
            "Одобрение заблокировано. "
            "Подробности сохранены в журнале."
        )
        return


@router.callback_query(F.data.startswith("review:"))
async def handle_review_callback(
    callback: CallbackQuery,
    db_pool: asyncpg.Pool,
    state: FSMContext,
) -> None:
    """Обрабатывает решение человека по черновику."""

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

    if action not in {"approve", "change", "reject"}:
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

    if action == "change":
        await state.clear()
        await state.set_state(
            ReviewRevisionStates.waiting_for_comment
        )
        await state.update_data(
            generated_post_id=generated_post_id
        )

        await remove_callback_keyboard(callback)

        await callback.answer(
            "Введите замечания."
        )

        await send_callback_message(
            callback,
            (
                "✏️ Напишите замечания к черновику "
                "одним сообщением.\n\n"
                "Можно перечислить несколько замечаний "
                "с новой строки. После отправки бот "
                "сохранит changes_required и создаст "
                "новую версию текста через OpenAI.\n\n"
                "Чтобы отменить ввод без изменений, "
                "отправьте /cancel."
            ),
        )

        logger.info(
            "Review revision input requested: "
            "user_id=%s, generated_post_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
        )
        return

    await state.clear()

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

    await remove_callback_keyboard(callback)

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

        publish_keyboard = (
            build_publish_keyboard(
                result.generated_post_id
            )
            if can_publish(bot_user)
            else None
        )
    else:
        result_text = (
            "❌ Черновик отклонён.\n\n"
            f"Generated post ID: {result.generated_post_id}\n"
            f"Статус поста: {result.post_status}\n"
            f"Статус подборки: {result.batch_status}"
        )
        publish_keyboard = None

    await callback.answer("Решение сохранено.")

    if callback.message is not None:
        await callback.message.answer(
            result_text,
            reply_markup=publish_keyboard,
        )

    logger.info(
        "Review decision saved: user_id=%s, "
        "generated_post_id=%s, decision=%s, "
        "review_action_id=%s",
        bot_user.telegram_user_id,
        result.generated_post_id,
        result.decision,
        result.review_action_id,
    )


@router.message(Command("cancel"))
async def handle_cancel(
    message: Message,
    state: FSMContext,
) -> None:
    """Отменяет незавершённый ввод замечаний."""

    current_state = await state.get_state()

    if current_state is None:
        await message.answer(
            "Сейчас нет активного ввода замечаний."
        )
        return

    await state.clear()

    await message.answer(
        "Ввод замечаний отменён. "
        "Черновик не изменён.\n\n"
        "Для повторной проверки используйте /review."
    )


@router.message(
    ReviewRevisionStates.waiting_for_comment,
    F.text,
)
async def handle_revision_comment(
    message: Message,
    db_pool: asyncpg.Pool,
    state: FSMContext,
) -> None:
    """Сохраняет замечания и создаёт новую версию."""

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
        await state.clear()
        return

    comment_text = (
        message.text.strip()
        if message.text is not None
        else ""
    )

    if not comment_text:
        await message.answer(
            "Замечания не могут быть пустыми. "
            "Напишите текст или отправьте /cancel."
        )
        return

    state_data = await state.get_data()

    generated_post_id = state_data.get(
        "generated_post_id"
    )

    if not isinstance(generated_post_id, int):
        await state.clear()
        await message.answer(
            "Не удалось определить черновик "
            "для доработки. Выполните /review заново."
        )
        return

    issues = tuple(
        line.strip()
        for line in comment_text.splitlines()
        if line.strip()
    )

    if not issues:
        await message.answer(
            "Замечания не могут быть пустыми. "
            "Напишите текст или отправьте /cancel."
        )
        return

    try:
        review_result = (
            await record_human_review_decision(
                db_pool,
                generated_post_id=generated_post_id,
                reviewer_telegram_user_id=(
                    bot_user.telegram_user_id
                ),
                decision="changes_required",
                requested_action="regenerate_text",
                comment_text=comment_text,
                issues=issues,
            )
        )
    except (
        LookupError,
        PermissionError,
        ValueError,
    ) as error:
        logger.warning(
            "Changes required rejected: user_id=%s, "
            "generated_post_id=%s, error=%s",
            bot_user.telegram_user_id,
            generated_post_id,
            error,
        )

        await message.answer(
            "Не удалось сохранить замечания:\n"
            f"{error}\n\n"
            "Исправьте сообщение или отправьте /cancel."
        )
        return
    except Exception:
        logger.exception(
            "Changes required failed: user_id=%s, "
            "generated_post_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
        )

        await message.answer(
            "Не удалось сохранить замечания. "
            "Подробности сохранены в журнале."
        )
        return

    review_action_id = (
        review_result.review_action_id
    )

    if not isinstance(review_action_id, int):
        logger.error(
            "Changes required has no review_action_id: "
            "user_id=%s, generated_post_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
        )

        await message.answer(
            "Замечания сохранены, но не удалось "
            "определить Review action ID. "
            "Автоматическая доработка не запущена."
        )
        return

    await message.answer(
        "Замечания сохранены. "
        "Создаю новую версию текста…"
    )

    runtime = None

    try:
        settings = get_settings()

        runtime = create_openai_generation_runtime(
            settings
        )

        revision_result = (
            await run_reserved_openai_generation_revision(
                db_pool,
                generator=runtime.generator,
                review_action_id=review_action_id,
            )
        )
    except Exception:
        logger.exception(
            "OpenAI review revision failed: "
            "user_id=%s, generated_post_id=%s, "
            "review_action_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
            review_action_id,
        )

        await message.answer(
            "⚠️ Замечания сохранены, но новую версию "
            "создать не удалось.\n\n"
            "Повторно отправьте точно такое же сообщение, "
            "чтобы безопасно повторить revision, "
            "либо отправьте /cancel."
        )
        return
    finally:
        if runtime is not None:
            await close_sdk_client(
                runtime.sdk_client
            )

    await state.clear()

    if not revision_result.completed:
        logger.error(
            "Revision pipeline returned non-completed "
            "result: user_id=%s, review_action_id=%s, "
            "revision_status=%s",
            bot_user.telegram_user_id,
            review_action_id,
            revision_result.revision_status,
        )

        await message.answer(
            "Замечания сохранены, но новая версия "
            "не перешла в completed. "
            "Проверьте журнал сервиса."
        )
        return

    await message.answer(
        "✅ Новая версия текста создана.\n\n"
        f"Исходный Generated post ID: "
        f"{generated_post_id}\n"
        f"Новый Generated post ID: "
        f"{revision_result.generated_post_id}\n"
        f"Версия: "
        f"{revision_result.target_version_number}\n"
        "Статус: awaiting_review\n\n"
        "Отправьте /review, чтобы проверить "
        "новую версию."
    )

    logger.info(
        "Review revision completed: "
        "user_id=%s, source_generated_post_id=%s, "
        "generated_post_id=%s, review_action_id=%s, "
        "generation_revision_id=%s, model_called=%s",
        bot_user.telegram_user_id,
        generated_post_id,
        revision_result.generated_post_id,
        review_action_id,
        revision_result.generation_revision_id,
        revision_result.model_called,
    )


@router.callback_query(F.data.startswith("publish:"))
async def handle_publish_callback(
    callback: CallbackQuery,
    db_pool: asyncpg.Pool,
) -> None:
    """Публикует одобренный пост в Telegram-канал."""

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

    if action != "approved":
        await callback.answer(
            "Неизвестное действие публикации.",
            show_alert=True,
        )
        return

    try:
        generated_post_id = int(generated_post_id_text)
    except ValueError:
        await callback.answer(
            "Некорректный ID поста.",
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

    if not can_publish(bot_user):
        await callback.answer(
            "Публикация доступна только администратору.",
            show_alert=True,
        )
        return

    await callback.answer("Публикация запущена.")

    # Удаляем кнопку сразу. Параллельный повтор дополнительно
    # блокируется транзакцией и статусами в PostgreSQL.
    await remove_callback_keyboard(callback)

    settings = get_settings()

    try:
        result = await publish_approved_post(
            db_pool,
            bot_token=(
                settings.telegram_bot_token.get_secret_value()
            ),
            generated_post_id=generated_post_id,
            disable_notification=True,
        )
    except PublicationStateUncertainError as error:
        logger.exception(
            "Publication state is uncertain: "
            "user_id=%s, generated_post_id=%s, "
            "publication_attempt_id=%s, "
            "telegram_message_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
            error.publication_attempt_id,
            error.telegram_message_id,
        )

        await send_callback_message(
            callback,
            (
                "⚠️ Telegram подтвердил отправку, но состояние "
                "не удалось надёжно сохранить в PostgreSQL.\n\n"
                f"Generated post ID: {generated_post_id}\n"
                "Publication attempt ID: "
                f"{error.publication_attempt_id}\n"
                "Telegram message ID: "
                f"{error.telegram_message_id}\n\n"
                "Повторная публикация запрещена до ручной сверки."
            ),
        )
        return
    except (LookupError, ValueError) as error:
        logger.warning(
            "Approved publication rejected: user_id=%s, "
            "generated_post_id=%s, error=%s",
            bot_user.telegram_user_id,
            generated_post_id,
            error,
        )

        await send_callback_message(
            callback,
            f"Публикация не выполнена:\n{error}",
        )
        return
    except Exception:
        logger.exception(
            "Approved publication failed: user_id=%s, "
            "generated_post_id=%s",
            bot_user.telegram_user_id,
            generated_post_id,
        )

        await send_callback_message(
            callback,
            (
                "Не удалось опубликовать пост. "
                "Подробности сохранены в журнале и PostgreSQL."
            ),
        )
        return

    if result.requires_review:
        result_text = (
            "⚠️ Сообщение отправлено в канал, но требуется "
            "ручная сверка состояния.\n\n"
            f"Generated post ID: {result.generated_post_id}\n"
            "Publication attempt ID: "
            f"{result.publication_attempt_id}\n"
            "Telegram message ID: "
            f"{result.telegram_message_id}\n"
            f"Статус БД: {result.database_status}\n\n"
            "Повторная отправка запрещена."
        )
    else:
        result_text = (
            "✅ Пост опубликован в канале.\n\n"
            f"Generated post ID: {result.generated_post_id}\n"
            "Publication attempt ID: "
            f"{result.publication_attempt_id}\n"
            "Telegram message ID: "
            f"{result.telegram_message_id}\n"
            f"Статус БД: {result.database_status}"
        )

    await send_callback_message(
        callback,
        result_text,
    )

    logger.info(
        "Approved post publication completed: "
        "user_id=%s, generated_post_id=%s, "
        "publication_attempt_id=%s, "
        "telegram_message_id=%s, database_status=%s",
        bot_user.telegram_user_id,
        result.generated_post_id,
        result.publication_attempt_id,
        result.telegram_message_id,
        result.database_status,
    )