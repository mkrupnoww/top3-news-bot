from dataclasses import dataclass

import asyncpg


@dataclass(frozen=True, slots=True)
class BotUser:
    """Активный пользователь, которому разрешена работа с ботом."""

    telegram_user_id: int
    telegram_username: str | None
    display_name: str
    user_role: str


async def get_active_bot_user(
    pool: asyncpg.Pool,
    telegram_user_id: int,
) -> BotUser | None:
    """Возвращает активного пользователя Telegram или None."""

    query = """
        SELECT
            telegram_user_id,
            telegram_username,
            display_name,
            user_role
        FROM bot_users
        WHERE telegram_user_id = $1
          AND is_active = true
    """

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            query,
            telegram_user_id,
        )

    if record is None:
        return None

    return BotUser(
        telegram_user_id=record["telegram_user_id"],
        telegram_username=record["telegram_username"],
        display_name=record["display_name"],
        user_role=record["user_role"],
    )