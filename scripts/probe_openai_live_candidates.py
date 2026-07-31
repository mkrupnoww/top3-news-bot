import argparse
import asyncio
from datetime import datetime, timezone
import inspect

from app.config import get_settings
from app.db.news_candidates import (
    select_news_candidates,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.openai_factory import (
    create_openai_ranking_runtime,
)
from app.ranking.score_formula import (
    calculate_individual_score,
    create_score_components,
)


def parse_as_of(
    value: str,
) -> datetime:
    """Разбирает дату ISO 8601 с часовым поясом."""

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
            "Некорректная дата --as-of."
        ) from error

    if (
        parsed_value.tzinfo is None
        or parsed_value.utcoffset() is None
    ):
        raise argparse.ArgumentTypeError(
            "--as-of должен содержать часовой пояс."
        )

    return parsed_value.astimezone(
        timezone.utc
    )


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры живого теста."""

    parser = argparse.ArgumentParser(
        description=(
            "Оценивает реальные кандидаты из "
            "PostgreSQL одним запросом OpenAI. "
            "Данные не изменяются."
        ),
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Подтверждает выполнение одного "
            "реального платного API-запроса."
        ),
    )

    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        default=None,
        help=(
            "Конец временного окна в ISO 8601. "
            "По умолчанию используется текущее UTC."
        ),
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Размер временного окна.",
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

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Максимальное число кандидатов.",
    )

    parser.add_argument(
        "--top-size",
        type=int,
        default=3,
        help="Размер итогового TOP.",
    )

    return parser.parse_args()


async def close_sdk_client(
    sdk_client: object,
) -> None:
    """Закрывает AsyncOpenAI-клиент."""

    close_method = getattr(
        sdk_client,
        "close",
        None,
    )

    if close_method is None:
        return

    close_result = close_method()

    if inspect.isawaitable(close_result):
        await close_result


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет числовые параметры."""

    if arguments.window_hours <= 0:
        raise ValueError(
            "--window-hours должен быть больше нуля."
        )

    if arguments.limit <= 0:
        raise ValueError(
            "--limit должен быть больше нуля."
        )

    if arguments.limit > 100:
        raise ValueError(
            "--limit не может превышать 100."
        )

    if arguments.top_size <= 0:
        raise ValueError(
            "--top-size должен быть больше нуля."
        )


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет одну пакетную оценку."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print("OpenAI candidate probe refused")
        print(f"error={error}")
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    if not arguments.confirm_live_request:
        print("OpenAI candidate probe refused")
        print(
            "Use --confirm-live-request "
            "to perform one paid API request."
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
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
        candidate_result = (
            await select_news_candidates(
                database_pool,
                as_of=as_of,
                window_hours=(
                    arguments.window_hours
                ),
                limit=arguments.limit,
                source_codes=(
                    tuple(arguments.source_code)
                    if arguments.source_code
                    else None
                ),
            )
        )
    finally:
        await close_database_pool(
            database_pool
        )

    candidates = candidate_result.candidates

    if not candidates:
        print("OpenAI candidate probe refused")
        print(
            "В заданном временном окне "
            "нет кандидатов."
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    if arguments.top_size > len(candidates):
        print("OpenAI candidate probe refused")
        print(
            "--top-size превышает количество "
            f"кандидатов: {len(candidates)}"
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    runtime = create_openai_ranking_runtime(
        settings
    )

    print("OpenAI candidate probe started")
    print(
        f"model="
        f"{runtime.evaluator.metadata.model_name}"
    )
    print(
        "window_started_at="
        f"{candidate_result.window_start.isoformat()}"
    )
    print(
        "window_finished_at="
        f"{candidate_result.window_end.isoformat()}"
    )
    print(
        f"candidate_count={len(candidates)}"
    )

    try:
        assessments = (
            await runtime.evaluator.evaluate(
                candidates
            )
        )
    except Exception as error:
        print()
        print("OpenAI candidate probe: FAILED")
        print(
            f"error_type="
            f"{type(error).__name__}"
        )
        print(f"error={error}")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 1
    finally:
        await close_sdk_client(
            runtime.sdk_client
        )

    candidate_by_news_id = {
        candidate.news_id: candidate
        for candidate in candidates
    }

    ranked_items = []

    for assessment in assessments:
        candidate = candidate_by_news_id[
            assessment.news_id
        ]

        components = create_score_components(
            f_score=assessment.f_score,
            m_score=assessment.m_score,
            r_score=assessment.r_score,
            h_score=assessment.h_score,
            q_score=assessment.q_score,
        )

        calculated = calculate_individual_score(
            components
        )

        ranked_items.append(
            (
                calculated.individual_score,
                candidate,
                components,
                assessment.explanation,
            )
        )

    ranked_items.sort(
        key=lambda item: (
            -item[0],
            item[1].news_id,
        )
    )

    print()
    print("Full ranking:")

    for rank_position, item in enumerate(
        ranked_items,
        start=1,
    ):
        (
            individual_score,
            candidate,
            components,
            explanation,
        ) = item

        print()
        print(
            f"{rank_position}. "
            f"news_id={candidate.news_id} "
            f"score={individual_score}"
        )
        print(f"   title={candidate.title}")
        print(
            f"   source="
            f"{candidate.source_name} "
            f"[{candidate.source_code}]"
        )
        print(
            "   scores="
            f"F:{components.f_score} "
            f"M:{components.m_score} "
            f"R:{components.r_score} "
            f"H:{components.h_score} "
            f"Q:{components.q_score}"
        )
        print(
            f"   explanation={explanation}"
        )
        print(
            f"   source_url="
            f"{candidate.source_url}"
        )

    print()
    print(
        f"TOP-{arguments.top_size}:"
    )

    for rank_position, item in enumerate(
        ranked_items[
            :arguments.top_size
        ],
        start=1,
    ):
        individual_score, candidate, _, _ = item

        print(
            f"{rank_position}. "
            f"news_id={candidate.news_id} "
            f"score={individual_score}"
        )
        print(
            f"   title={candidate.title}"
        )

    print()
    print("OpenAI requests: performed=1")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("OpenAI candidate probe: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )