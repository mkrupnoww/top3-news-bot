import argparse
import asyncio

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.review_sources import (
    get_review_sources,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает ID сформированного поста."""

    parser = argparse.ArgumentParser(
        description=(
            "Показывает источники трёх новостей, "
            "связанных с generated_post."
        ),
    )

    parser.add_argument(
        "--generated-post-id",
        type=int,
        required=True,
        help="ID записи generated_posts.",
    )

    return parser.parse_args()


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Читает и выводит досье источников."""

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        items = await get_review_sources(
            database_pool,
            generated_post_id=(
                arguments.generated_post_id
            ),
        )
    finally:
        await close_database_pool(
            database_pool
        )

    print("Review sources loaded")
    print(
        "generated_post_id="
        f"{arguments.generated_post_id}"
    )
    print(f"news_count={len(items)}")

    for item in items:
        print()
        print(
            f"{item.position}. {item.title}"
        )
        print(f"   news_id={item.news_id}")
        print(
            f"   source={item.source_name}"
        )
        print(
            "   published_at="
            + (
                item.source_published_at.isoformat()
                if item.source_published_at
                else "unknown"
            )
        )
        print(
            f"   source_url={item.source_url}"
        )
        print(
            "   selection_reason="
            + (
                item.selection_reason
                or "not specified"
            )
        )
        print(
            "   primary_image_url="
            + (
                item.primary_image_url
                or "not specified"
            )
        )
        print(
            "   image_credit="
            + (
                item.image_credit
                or "not specified"
            )
        )

    print()
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Review sources test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )