import asyncio

import asyncpg

from app.config import get_settings


async def main() -> None:
    settings = get_settings()

    connection = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password.get_secret_value(),
        server_settings={
            "application_name": "top3-news-bot-db-check",
            "search_path": f"{settings.db_schema},public",
        },
        command_timeout=10,
    )

    try:
        connection_info = await connection.fetchrow(
            """
            SELECT
                current_user AS db_user,
                current_database() AS db_name,
                current_schema() AS db_schema,
                current_setting('search_path') AS search_path
            """
        )

        latest_migration = await connection.fetchrow(
            """
            SELECT
                version,
                description,
                applied_at,
                applied_by
            FROM schema_migrations
            ORDER BY applied_at DESC
            LIMIT 1
            """
        )

        table_count = await connection.fetchval(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = $1
              AND table_type = 'BASE TABLE'
            """,
            settings.db_schema,
        )

        print("PostgreSQL connection: OK")
        print(f"db_user={connection_info['db_user']}")
        print(f"db_name={connection_info['db_name']}")
        print(f"db_schema={connection_info['db_schema']}")
        print(f"search_path={connection_info['search_path']}")
        print(f"table_count={table_count}")

        if latest_migration is None:
            raise RuntimeError("No database migrations found")

        print(f"latest_migration={latest_migration['version']}")
        print(f"migration_applied_by={latest_migration['applied_by']}")
        print("PostgreSQL schema verification: OK")

    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())