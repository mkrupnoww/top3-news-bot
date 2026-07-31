import argparse
from pathlib import Path

from pydantic import ValidationError

from app.generation.manual_top3_input import (
    load_manual_top3_input,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает путь к JSON-файлу выпуска."""

    parser = argparse.ArgumentParser(
        description=(
            "Проверяет JSON-файл ручного выпуска TOP-3. "
            "PostgreSQL и Telegram не изменяются."
        ),
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Путь к JSON-файлу выпуска TOP-3.",
    )

    return parser.parse_args()


def main(arguments: argparse.Namespace) -> int:
    """Проверяет структуру и выводит краткое резюме."""

    try:
        top3_input = load_manual_top3_input(
            arguments.input_file
        )
    except (
        FileNotFoundError,
        ValueError,
        ValidationError,
    ) as error:
        print("Manual TOP-3 input is invalid")
        print(error)
        return 1

    print("Manual TOP-3 input is valid")
    print(
        f"publication_date="
        f"{top3_input.publication_date}"
    )
    print(
        f"text_format="
        f"{top3_input.text_format}"
    )
    print(
        f"post_length="
        f"{len(top3_input.post_text)}"
    )
    print(
        f"item_count="
        f"{len(top3_input.items)}"
    )

    for item in sorted(
        top3_input.items,
        key=lambda news_item: news_item.position,
    ):
        print(
            f"{item.position}. "
            f"{item.title} "
            f"[{item.source_name}]"
        )
        print(
            f"   published_at="
            f"{item.source_published_at.isoformat()}"
        )
        print(
            f"   source_url="
            f"{item.source_url}"
        )

    print("Database changes: not performed")
    print("Telegram publication: not performed")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(parse_arguments())
    )