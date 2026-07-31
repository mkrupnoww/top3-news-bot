import argparse
import asyncio
from datetime import datetime, timezone
import re

from app.config import get_settings
from app.db.news_candidates import (
    select_news_candidates,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


_SOURCE_CODE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_-]{1,63}$"
)


def parse_as_of(
    value: str,
) -> datetime:
    """Разбирает ISO 8601 дату с часовым поясом."""

    normalized_value = value.strip()

    if normalized_value.endswith("Z"):
        normalized_value = (
            normalized_value[:-1]
            + "+00:00"
        )

    try:
        parsed_value = datetime.fromisoformat(
            normalized_value
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Некорректная дата. Используйте ISO 8601, "
            "например 2026-07-31T11:21:00Z."
        ) from error

    if (
        parsed_value.tzinfo is None
        or parsed_value.utcoffset() is None
    ):
        raise argparse.ArgumentTypeError(
            "Дата должна содержать часовой пояс."
        )

    return parsed_value.astimezone(
        timezone.utc
    )


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры временного окна."""

    parser = argparse.ArgumentParser(
        description=(
            "Показывает новости-кандидаты из PostgreSQL "
            "за заданное временное окно. "
            "Данные не изменяются."
        ),
    )

    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        default=None,
        help=(
            "Конец окна в ISO 8601. "
            "По умолчанию используется текущее время UTC."
        ),
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help=(
            "Размер окна в часах. "
            "По умолчанию: 24."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help=(
            "Максимум кандидатов. "
            "По умолчанию: 500."
        ),
    )

    parser.add_argument(
        "--source-code",
        action="append",
        default=None,
        help=(
            "Фильтр по source_code. "
            "Параметр можно повторить."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет аргументы до подключения к БД."""

    if arguments.window_hours <= 0:
        raise ValueError(
            "--window-hours должен быть больше нуля."
        )

    if arguments.limit <= 0:
        raise ValueError(
            "--limit должен быть больше нуля."
        )

    if arguments.limit > 5000:
        raise ValueError(
            "--limit не может превышать 5000."
        )

    if arguments.source_code:
        normalized_codes: list[str] = []

        for source_code in arguments.source_code:
            normalized_code = (
                source_code.strip().lower()
            )

            if not _SOURCE_CODE_PATTERN.fullmatch(
                normalized_code
            ):
                raise ValueError(
                    "Некорректный --source-code: "
                    f"{source_code}"
                )

            normalized_codes.append(
                normalized_code
            )

        arguments.source_code = normalized_codes


def _truncate(
    value: str | None,
    *,
    max_length: int,
) -> str:
    """Сокращает длинный текст для терминала."""

    if value is None:
        return "not specified"

    normalized_value = value.strip()

    if not normalized_value:
        return "not specified"

    if len(normalized_value) <= max_length:
        return normalized_value

    return (
        normalized_value[: max_length - 1].rstrip()
        + "…"
    )


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Читает и выводит кандидатов."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print("Candidate selection refused")
        print(error)
        print("Database changes: not performed")
        return 2

    as_of = (
        arguments.as_of
        or datetime.now(timezone.utc)
    )

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        result = await select_news_candidates(
            database_pool,
            as_of=as_of,
            window_hours=arguments.window_hours,
            limit=arguments.limit,
            source_codes=(
                tuple(arguments.source_code)
                if arguments.source_code
                else None
            ),
        )
    finally:
        await close_database_pool(
            database_pool
        )

    print("Candidate selection completed")
    print(
        f"window_start="
        f"{result.window_start.isoformat()}"
    )
    print(
        f"window_end="
        f"{result.window_end.isoformat()}"
    )
    print(
        f"window_hours="
        f"{result.window_hours}"
    )
    print(
        f"candidate_count="
        f"{len(result.candidates)}"
    )

    for position, candidate in enumerate(
        result.candidates,
        start=1,
    ):
        print()
        print(
            f"{position}. "
            f"{_truncate(candidate.title, max_length=400)}"
        )
        print(
            f"   news_id={candidate.news_id}"
        )
        print(
            f"   source="
            f"{candidate.source_name} "
            f"[{candidate.source_code}]"
        )
        print(
            f"   collection_priority="
            f"{candidate.collection_priority}"
        )
        print(
            f"   processing_status="
            f"{candidate.processing_status}"
        )
        print(
            f"   published_at="
            f"{candidate.source_published_at.isoformat()}"
        )
        print(
            f"   age_hours="
            f"{candidate.age_hours:.2f}"
        )
        print(
            f"   author="
            f"{_truncate(candidate.author_name, max_length=200)}"
        )
        print(
            f"   source_url="
            f"{candidate.source_url}"
        )
        print(
            "   primary_image_url="
            f"{_truncate(candidate.primary_image_url, max_length=500)}"
        )
        print(
            f"   summary="
            f"{_truncate(candidate.summary, max_length=500)}"
        )

    print()
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Candidate selection test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )