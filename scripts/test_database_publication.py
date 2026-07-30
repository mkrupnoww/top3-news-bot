import argparse
import asyncio
from datetime import UTC, datetime

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.publication import publish_text_to_channel


def parse_arguments() -> argparse.Namespace:
    """Разбирает аргументы тестового сценария."""

    parser = argparse.ArgumentParser(
        description=(
            "Публикует техническое сообщение и сохраняет "
            "жизненный цикл публикации в PostgreSQL."
        ),
    )

    parser.add_argument(
        "--simulate-finalization-failure",
        action="store_true",
        help=(
            "После успешной отправки в Telegram имитирует "
            "сбой финальной записи published."
        ),
    )

    return parser.parse_args()


async def main(
    *,
    simulate_finalization_failure: bool,
) -> int:
    """Запускает технический тест рабочего сервиса публикации."""

    settings = get_settings()
    now = datetime.now(UTC)

    post_text = (
        "✅ DB-тест публикации TOP 3 Movie News\n\n"
        "Сообщение отправлено через рабочий модуль публикации, "
        "а жизненный цикл сохранён в PostgreSQL.\n\n"
        "Это техническое тестовое сообщение. Его можно удалить.\n\n"
        f"Время проверки: {now:%Y-%m-%d %H:%M:%S} UTC"
    )

    metadata = {
        "technical_test": True,
        "scenario": "database_publication_test",
        "script": "scripts.test_database_publication",
        "service": "app.publication.service",
        "created_at": now.isoformat(),
        "batch_items_expected": False,
        "simulate_finalization_failure": (
            simulate_finalization_failure
        ),
    }

    database_pool = await create_database_pool(settings)

    try:
        result = await publish_text_to_channel(
            database_pool,
            bot_token=(
                settings.telegram_bot_token.get_secret_value()
            ),
            telegram_chat_id=settings.telegram_channel_id,
            publication_date=now.date(),
            post_text=post_text,
            metadata=metadata,
            disable_notification=True,
            simulate_finalization_failure=(
                simulate_finalization_failure
            ),
        )
    finally:
        await close_database_pool(database_pool)

    print("Publication workflow completed")
    print(f"batch_id={result.batch_id}")
    print(f"generated_post_id={result.generated_post_id}")
    print(
        "publication_attempt_id="
        f"{result.publication_attempt_id}"
    )
    print(f"publication_date={result.publication_date}")
    print(f"edition={result.edition}")
    print(
        "telegram_message_id="
        f"{result.telegram_message_id}"
    )
    print(
        "database_status="
        f"{result.database_status}"
    )
    print(
        "requires_review="
        f"{str(result.requires_review).lower()}"
    )

    if result.requires_review:
        print(
            "Telegram подтвердил публикацию, "
            "но требуется ручная сверка."
        )
        return 2

    print("Test publication: OK")
    return 0


if __name__ == "__main__":
    arguments = parse_arguments()

    raise SystemExit(
        asyncio.run(
            main(
                simulate_finalization_failure=(
                    arguments.simulate_finalization_failure
                ),
            )
        )
    )