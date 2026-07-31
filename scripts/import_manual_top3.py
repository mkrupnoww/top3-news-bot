import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.manual_top3_import import (
    import_manual_top3,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.manual_top3_input import (
    load_manual_top3_input,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры загрузки выпуска."""

    parser = argparse.ArgumentParser(
        description=(
            "Загружает ручной выпуск TOP-3 "
            "в PostgreSQL со статусом awaiting_review. "
            "Публикация в Telegram не выполняется."
        ),
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Путь к JSON-файлу выпуска TOP-3.",
    )

    parser.add_argument(
        "--allow-example",
        action="store_true",
        help=(
            "Разрешить загрузку файла, у которого "
            "metadata.example_only=true."
        ),
    )

    return parser.parse_args()


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Валидирует и загружает выпуск."""

    top3_input = load_manual_top3_input(
        arguments.input_file
    )

    is_example = (
        top3_input.metadata.get("example_only")
        is True
    )

    if is_example and not arguments.allow_example:
        print("Manual TOP-3 import refused")
        print(
            "Файл помечен metadata.example_only=true."
        )
        print(
            "Для контролируемого теста добавьте "
            "--allow-example."
        )
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        result = await import_manual_top3(
            database_pool,
            top3_input=top3_input,
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
        )
    finally:
        await close_database_pool(
            database_pool
        )

    if result.already_imported:
        print(
            "Manual TOP-3 was already imported"
        )
    else:
        print(
            "Manual TOP-3 imported successfully"
        )

    print(
        f"already_imported="
        f"{str(result.already_imported).lower()}"
    )
    print(f"batch_id={result.batch_id}")
    print(
        f"generated_post_id="
        f"{result.generated_post_id}"
    )
    print(
        f"publication_date="
        f"{result.publication_date}"
    )
    print(f"edition={result.edition}")
    print(
        f"version_number="
        f"{result.version_number}"
    )
    print(
        f"batch_status="
        f"{result.batch_status}"
    )
    print(
        f"post_status="
        f"{result.post_status}"
    )
    print(
        "news_ids="
        + ",".join(
            str(news_id)
            for news_id in result.news_ids
        )
    )
    print(
        "publication_attempt_count="
        f"{result.publication_attempt_count}"
    )
    print("Review command: /review")
    print("Telegram publication: not performed")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )