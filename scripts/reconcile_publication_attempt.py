import argparse
import asyncio

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.publication_reconciliation import (
    reconcile_unknown_publication,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры ручной сверки публикации."""

    parser = argparse.ArgumentParser(
        description=(
            "Подтверждает существование уже опубликованного "
            "Telegram-сообщения без его повторной отправки."
        ),
    )

    parser.add_argument(
        "--attempt-id",
        type=int,
        required=True,
        help="publication_attempt_id из PostgreSQL.",
    )

    parser.add_argument(
        "--message-id",
        type=int,
        required=True,
        help="Подтверждённый telegram_message_id.",
    )

    parser.add_argument(
        "--confirmed-by",
        type=int,
        required=True,
        help=(
            "Telegram ID активного администратора, "
            "подтвердившего сообщение."
        ),
    )

    parser.add_argument(
        "--note",
        required=True,
        help="Краткое пояснение ручной проверки.",
    )

    return parser.parse_args()


async def main(arguments: argparse.Namespace) -> int:
    """Выполняет ручную сверку попытки публикации."""

    settings = get_settings()
    database_pool = await create_database_pool(settings)

    try:
        result = await reconcile_unknown_publication(
            database_pool,
            publication_attempt_id=arguments.attempt_id,
            expected_telegram_message_id=arguments.message_id,
            confirmed_by_telegram_user_id=arguments.confirmed_by,
            confirmation_note=arguments.note,
        )
    finally:
        await close_database_pool(database_pool)

    print("Publication reconciliation completed")
    print(
        "publication_attempt_id="
        f"{result.publication_attempt_id}"
    )
    print(f"generated_post_id={result.generated_post_id}")
    print(f"batch_id={result.batch_id}")
    print(
        "telegram_message_id="
        f"{result.telegram_message_id}"
    )
    print(
        "previous_attempt_status="
        f"{result.previous_attempt_status}"
    )
    print(
        "current_attempt_status="
        f"{result.current_attempt_status}"
    )
    print(
        "already_reconciled="
        f"{str(result.already_reconciled).lower()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )