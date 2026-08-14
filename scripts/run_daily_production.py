import argparse
import asyncio
from datetime import date, datetime, timezone

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.workflows.daily_production import (
    run_daily_production_workflow,
)


def _parse_date(
    value: str,
) -> date:
    """Парсит YYYY-MM-DD."""

    try:
        return date.fromisoformat(
            value
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "publication-date должен иметь "
            "формат YYYY-MM-DD."
        ) from error


def _parse_as_of(
    value: str,
) -> datetime:
    """Парсит timezone-aware ISO datetime и приводит к UTC."""

    normalized = value.strip()

    if normalized.endswith("Z"):
        normalized = (
            normalized[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "as-of должен быть ISO datetime "
            "с часовым поясом."
        ) from error

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise argparse.ArgumentTypeError(
            "as-of обязан содержать "
            "часовой пояс."
        )

    return parsed.astimezone(
        timezone.utc
    )


def build_parser() -> argparse.ArgumentParser:
    """Создаёт CLI parser production workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Run TOP-3 daily production workflow "
            "through Telegram human review."
        )
    )

    parser.add_argument(
        "--publication-date",
        required=True,
        type=_parse_date,
        help="Logical publication date YYYY-MM-DD.",
    )

    parser.add_argument(
        "--as-of",
        required=True,
        type=_parse_as_of,
        help=(
            "Fixed UTC-aware ranking cutoff, "
            "for example 2026-08-14T10:30:00+00:00."
        ),
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=500,
        help="Maximum ranking candidates. Default: 500.",
    )

    return parser


async def async_main() -> int:
    """Запускает production daily workflow."""

    args = build_parser().parse_args()

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    try:
        result = (
            await run_daily_production_workflow(
                pool,
                settings=settings,
                publication_date=(
                    args.publication_date
                ),
                as_of=args.as_of,
                candidate_limit=(
                    args.candidate_limit
                ),
            )
        )
    finally:
        await close_database_pool(
            pool
        )

    print("Daily production workflow: OK")
    print(
        "daily_workflow_run_id="
        f"{result.daily_workflow_run_id}"
    )
    print(
        "publication_date="
        f"{result.publication_date}"
    )
    print(
        "as_of="
        f"{result.as_of.isoformat()}"
    )
    print(
        "workflow_status="
        f"{result.workflow_status}"
    )
    print(
        "ranking_run_id="
        f"{result.ranking_run_id}"
    )
    print(
        "batch_id="
        f"{result.batch_id}"
    )
    print(
        "generated_post_id="
        f"{result.generated_post_id}"
    )
    print(
        "image_generation_id="
        f"{result.image_generation_id}"
    )
    print(
        "ranking_model_called="
        f"{str(result.ranking_model_called).lower()}"
    )
    print(
        "generation_model_called="
        f"{str(result.generation_model_called).lower()}"
    )
    print(
        "image_model_called="
        f"{str(result.image_model_called).lower()}"
    )
    print(
        "reviewer_count="
        f"{result.reviewer_count}"
    )
    print(
        "review_sent_count="
        f"{result.review_sent_count}"
    )
    print(
        "review_failed_count="
        f"{result.review_failed_count}"
    )
    print(
        "review_unknown_count="
        f"{result.review_unknown_count}"
    )
    print(
        "review_skipped_count="
        f"{result.review_skipped_count}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            async_main()
        )
    )