import argparse
import asyncio

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.publication import publish_approved_post


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры публикации одобренного поста."""

    parser = argparse.ArgumentParser(
        description=(
            "Публикует существующий generated_post "
            "со статусом approved."
        ),
    )

    parser.add_argument(
        "--generated-post-id",
        type=int,
        required=True,
        help="ID одобренного generated_posts.",
    )

    return parser.parse_args()


async def main(arguments: argparse.Namespace) -> int:
    """Публикует одобренный пост через рабочий сервис."""

    settings = get_settings()
    database_pool = await create_database_pool(settings)

    try:
        result = await publish_approved_post(
            database_pool,
            bot_token=(
                settings.telegram_bot_token.get_secret_value()
            ),
            generated_post_id=(
                arguments.generated_post_id
            ),
            disable_notification=True,
        )
    finally:
        await close_database_pool(database_pool)

    print("Approved publication completed")
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

    return 2 if result.requires_review else 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )