import asyncpg

from app.config import Settings


async def create_database_pool(
    settings: Settings,
) -> asyncpg.Pool:
    """Создаёт пул подключений приложения к PostgreSQL."""

    return await asyncpg.create_pool(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password.get_secret_value(),
        min_size=1,
        max_size=5,
        command_timeout=10,
        server_settings={
            "application_name": "top3-news-bot",
            "search_path": f"{settings.db_schema},public",
        },
    )


async def close_database_pool(
    pool: asyncpg.Pool | None,
) -> None:
    """Корректно закрывает пул подключений."""

    if pool is not None:
        await pool.close()