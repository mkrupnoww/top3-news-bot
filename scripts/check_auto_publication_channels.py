"""Проверка двух каналов только через getChat/getChatMember, без публикаций."""
import asyncio
from aiogram import Bot
from aiogram.enums import ChatType
from app.config import get_settings


async def main():
    settings = get_settings()
    if not settings.auto_publication_enabled:
        raise ValueError('AUTO_PUBLICATION_ENABLED выключен')
    async with Bot(token=settings.telegram_bot_token.get_secret_value()).context() as bot:
        me = await bot.get_me()
        for label, chat_id in (('CHANNEL_PROD', settings.channel_prod_id), ('CHANNEL_PROD_VIP', settings.channel_prod_vip_id)):
            chat = await bot.get_chat(chat_id)
            member = await bot.get_chat_member(chat_id, me.id)
            if chat.type != ChatType.CHANNEL or member.status not in {'administrator', 'creator'}:
                raise ValueError(f'{label}: бот должен быть администратором канала')
            if member.status != 'creator' and not getattr(member, 'can_post_messages', False):
                raise ValueError(f'{label}: нет can_post_messages')
            print(f'{label}: доступ и право публикации подтверждены; сообщений не отправлено')


if __name__ == '__main__':
    asyncio.run(main())
