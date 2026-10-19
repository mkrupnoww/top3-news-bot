"""Автопубликация готового выпуска без новых генераций или ручного approve."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging

import asyncpg
from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter, TelegramUnauthorizedError
from aiogram.types import FSInputFile

from app.config import Settings
from app.db.auto_publications import (
    auto_delivery_statuses, claim_auto_post, finalize_auto_publication,
    find_due_auto_post, finish_auto_delivery, get_auto_plan, reserve_auto_delivery,
)
from app.generation.post_contract import MAXIMUM_POST_LENGTH
from app.publication.approved_service import _resolve_image_file
from app.publication.telegram_text import prepare_telegram_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoPublicationResult:
    generated_post_id: int | None = None
    published: bool = False
    channel_statuses: dict[int, str] = field(default_factory=dict)
    skipped_reason: str | None = None


async def run_auto_publication(
    pool: asyncpg.Pool, *, settings: Settings, now: datetime | None = None,
    dry_run: bool = False,
) -> AutoPublicationResult:
    if not settings.auto_publication_enabled:
        return AutoPublicationResult(skipped_reason='disabled')
    now = now or datetime.now(UTC)
    plan = await get_auto_plan(pool, now=now)
    if plan is None:
        source = await find_due_auto_post(pool, now=now)
    else:
        async with pool.acquire() as connection:
            source = await connection.fetchrow(
                'SELECT * FROM generated_posts WHERE generated_post_id=$1', plan['generated_post_id'],
            )
    if source is None:
        return AutoPublicationResult(skipped_reason='no_unanswered_daily_post_due')
    post_id = source['generated_post_id']
    if len(source['post_text']) > MAXIMUM_POST_LENGTH:
        raise ValueError('Слишком длинный текст для автопубликации')
    image = _resolve_image_file(image_path=source['image_path'], expected_sha256=source['image_sha256'])
    caption = prepare_telegram_text(source['post_text'], text_format=source['text_format'])
    if dry_run:
        return AutoPublicationResult(generated_post_id=post_id, skipped_reason='dry_run')

    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    try:
        if plan is None:
            plan = await claim_auto_post(
                pool, now=now, source_post=source,
                target_chat_ids=(settings.channel_prod_id, settings.channel_prod_vip_id),
            )
            if plan is None:
                return AutoPublicationResult(skipped_reason='review_or_publication_won_race')

        # Старые кнопки больше не должны предлагать ручное решение.
        async with pool.acquire() as connection:
            previews = await connection.fetch(
                """SELECT telegram_chat_id, telegram_message_id FROM review_delivery_attempts
                   WHERE generated_post_id=$1 AND delivery_status='sent'""", post_id,
            )
        for preview in previews:
            try:
                await bot.edit_message_reply_markup(
                    chat_id=preview['telegram_chat_id'], message_id=preview['telegram_message_id'], reply_markup=None,
                )
            except Exception:
                logger.warning('Не удалось убрать review-кнопки поста %s', post_id, exc_info=True)

        for chat_id in plan['target_chat_ids']:
            request = {
                'publication_mode': 'daily_no_response', 'chat_id': chat_id,
                'source_post_text': source['post_text'], 'source_text_format': source['text_format'],
                'photo': {'path': source['image_path'], 'sha256': source['image_sha256']},
                'caption': {'text': caption.text, 'text_format': caption.text_format, 'parse_mode': caption.parse_mode},
                'disable_notification': True,
            }
            attempt_id = await reserve_auto_delivery(
                pool, generated_post_id=post_id, chat_id=chat_id, request_payload=request,
            )
            if attempt_id is None:
                continue
            sending = False
            try:
                chat = await bot.get_chat(chat_id)
                if chat.type != ChatType.CHANNEL:
                    raise ValueError('Получатель автопубликации не является каналом')
                sending = True
                message = await bot.send_photo(
                    chat_id=chat_id, photo=FSInputFile(image), caption=caption.text,
                    parse_mode=caption.parse_mode, disable_notification=True,
                )
            except Exception as error:
                # Потеря ответа sendPhoto не доказывает отсутствие сообщения.
                rejected = isinstance(error, (TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter, TelegramUnauthorizedError))
                status = 'failed' if not sending or rejected else 'unknown'
                logger.exception('Auto publication post=%s channel=%s status=%s', post_id, chat_id, status)
                try:
                    await finish_auto_delivery(pool, attempt_id=attempt_id, status=status, error=f'{type(error).__name__}: {error}')
                except Exception:
                    logger.exception('Не удалось завершить попытку %s; started блокирует дубликат', attempt_id)
                continue

            response = {'message_id': message.message_id, 'chat_id': message.chat.id, 'transport': 'native_photo'}
            try:
                await finish_auto_delivery(pool, attempt_id=attempt_id, status='published', message_id=message.message_id, response=response)
            except Exception as error:
                logger.exception('Сбой фиксации отправленного сообщения: attempt=%s', attempt_id)
                try:
                    await finish_auto_delivery(
                        pool, attempt_id=attempt_id, status='unknown', message_id=message.message_id,
                        response=response, error=f'{type(error).__name__}: {error}',
                    )
                except Exception:
                    logger.exception('Попытка %s требует ручной сверки; автоматический повтор запрещён', attempt_id)

        complete = await finalize_auto_publication(pool, generated_post_id=post_id)
        statuses = await auto_delivery_statuses(pool, generated_post_id=post_id)
        return AutoPublicationResult(generated_post_id=post_id, published=complete, channel_statuses=statuses)
    finally:
        await bot.session.close()
