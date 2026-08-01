import argparse
import asyncio
from datetime import datetime, timezone
import inspect
from typing import Any

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.openai_factory import (
    create_openai_ranking_runtime,
)
from app.ranking.openai_pipeline import (
    ReservedOpenAIRankingResult,
    run_reserved_openai_ranking,
)


def parse_as_of(
    value: str,
) -> datetime:
    """Разбирает ISO 8601 с обязательным часовым поясом."""

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
            "--as-of должен содержать "
            "часовой пояс."
        )

    return parsed_value.astimezone(
        timezone.utc
    )


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры защищённого запуска."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет защищённое ранжирование "
            "реальных новостей через OpenAI, "
            "сохраняет результат в PostgreSQL "
            "и блокирует повторный платный запрос."
        ),
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Разрешает реальный платный запрос "
            "OpenAI, если request_key ещё "
            "не был зарезервирован."
        ),
    )

    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        required=True,
        help=(
            "Конец временного окна "
            "в ISO 8601 с часовым поясом."
        ),
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help=(
            "Размер временного окна "
            "в часах. По умолчанию 24."
        ),
    )

    parser.add_argument(
        "--source-code",
        action="append",
        default=None,
        help=(
            "Фильтр по source_code. "
            "Параметр можно повторять."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help=(
            "Максимальное число кандидатов. "
            "По умолчанию 20."
        ),
    )

    parser.add_argument(
        "--top-size",
        type=int,
        default=3,
        help=(
            "Количество первых новостей "
            "для отдельного блока TOP."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет числовые параметры."""

    if arguments.window_hours <= 0:
        raise ValueError(
            "--window-hours должен быть "
            "больше нуля."
        )

    if arguments.window_hours > 168:
        raise ValueError(
            "--window-hours не может "
            "превышать 168."
        )

    if arguments.limit <= 0:
        raise ValueError(
            "--limit должен быть "
            "больше нуля."
        )

    if arguments.limit > 100:
        raise ValueError(
            "--limit не может "
            "превышать 100."
        )

    if arguments.top_size <= 0:
        raise ValueError(
            "--top-size должен быть "
            "больше нуля."
        )

    if arguments.top_size > 100:
        raise ValueError(
            "--top-size не может "
            "превышать 100."
        )


async def close_sdk_client(
    sdk_client: object,
) -> None:
    """Закрывает совместимый OpenAI SDK-клиент."""

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


async def load_persisted_ranking(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> tuple[asyncpg.Record, ...]:
    """Читает сохранённый рейтинг из PostgreSQL."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                ns.news_id,
                ns.rank_position,
                ns.individual_score,
                ns.f_score,
                ns.m_score,
                ns.r_score,
                ns.h_score,
                ns.q_score,
                ns.score_explanation,
                COALESCE(
                    NULLIF(n.normalized_title, ''),
                    NULLIF(n.raw_title, ''),
                    'Без заголовка'
                ) AS title,
                n.source_url,
                s.source_code,
                s.source_name
            FROM top3_news.news_scores AS ns
            JOIN top3_news.news_items AS n
              ON n.news_id = ns.news_id
            JOIN top3_news.sources AS s
              ON s.source_id = n.source_id
            WHERE ns.ranking_run_id = $1
            ORDER BY
                ns.rank_position,
                ns.news_id
            """,
            ranking_run_id,
        )

    return tuple(records)


def print_run_summary(
    result: ReservedOpenAIRankingResult,
) -> None:
    """Печатает состояние защищённого запуска."""

    selection = result.candidate_selection

    print()
    print("Protected OpenAI ranking result:")
    print(
        "ranking_run_id="
        f"{result.ranking_run_id}"
    )
    print(
        "request_key="
        f"{result.request_key.value}"
    )
    print(
        "request_key_version="
        f"{result.request_key.version}"
    )
    print(
        "run_status="
        f"{result.run_status}"
    )
    print(
        "created_new="
        f"{str(result.reservation.created_new).lower()}"
    )
    print(
        "model_called="
        f"{str(result.model_called).lower()}"
    )
    print(
        "duplicate_request_blocked="
        f"{str(result.duplicate_request_blocked).lower()}"
    )
    print(
        "candidate_count="
        f"{len(selection.candidates)}"
    )
    print(
        "window_started_at="
        f"{selection.window_start.isoformat()}"
    )
    print(
        "window_finished_at="
        f"{selection.window_end.isoformat()}"
    )


def print_persisted_ranking(
    records: tuple[asyncpg.Record, ...],
    *,
    top_size: int,
) -> None:
    """Печатает полный сохранённый рейтинг."""

    print()
    print("Persisted full ranking:")

    for record in records:
        print()
        print(
            f"{record['rank_position']}. "
            f"news_id={record['news_id']} "
            f"score={record['individual_score']}"
        )
        print(
            f"   title={record['title']}"
        )
        print(
            "   source="
            f"{record['source_name']} "
            f"[{record['source_code']}]"
        )
        print(
            "   scores="
            f"F:{record['f_score']} "
            f"M:{record['m_score']} "
            f"R:{record['r_score']} "
            f"H:{record['h_score']} "
            f"Q:{record['q_score']}"
        )
        print(
            "   explanation="
            f"{record['score_explanation']}"
        )
        print(
            "   source_url="
            f"{record['source_url']}"
        )

    actual_top_size = min(
        top_size,
        len(records),
    )

    print()
    print(f"TOP-{actual_top_size}:")

    for record in records[:actual_top_size]:
        print(
            f"{record['rank_position']}. "
            f"news_id={record['news_id']} "
            f"score={record['individual_score']}"
        )
        print(
            f"   title={record['title']}"
        )


def print_openai_telemetry(
    result: ReservedOpenAIRankingResult,
) -> None:
    """Печатает usage и расчёт стоимости нового вызова."""

    evaluation = result.evaluation

    if evaluation is None:
        print()
        print(
            "OpenAI telemetry: "
            "existing reservation reused"
        )
        return

    usage = evaluation.model_response.usage

    cost_estimate = (
        evaluation
        .model_response
        .cost_estimate
    )

    if usage is None:
        raise RuntimeError(
            "OpenAI usage отсутствует "
            "в завершённом результате."
        )

    if cost_estimate is None:
        raise RuntimeError(
            "Расчёт стоимости отсутствует "
            "в завершённом результате."
        )

    print()
    print("OpenAI token usage:")
    print(
        "input_tokens="
        f"{usage.input_tokens}"
    )
    print(
        "regular_input_tokens="
        f"{usage.regular_input_tokens}"
    )
    print(
        "cached_input_tokens="
        f"{usage.cached_input_tokens}"
    )
    print(
        "cache_write_tokens="
        f"{usage.cache_write_tokens}"
    )
    print(
        "output_tokens="
        f"{usage.output_tokens}"
    )
    print(
        "reasoning_tokens="
        f"{usage.reasoning_tokens}"
    )
    print(
        "total_tokens="
        f"{usage.total_tokens}"
    )

    print()
    print("OpenAI estimated cost:")
    print(
        "model="
        f"{cost_estimate.model_name}"
    )
    print(
        "pricing_version="
        f"{cost_estimate.pricing_version}"
    )
    print(
        "regular_input_cost_usd="
        f"{cost_estimate.regular_input_cost_usd}"
    )
    print(
        "cached_input_cost_usd="
        f"{cost_estimate.cached_input_cost_usd}"
    )
    print(
        "cache_write_cost_usd="
        f"{cost_estimate.cache_write_cost_usd}"
    )
    print(
        "output_cost_usd="
        f"{cost_estimate.output_cost_usd}"
    )
    print(
        "total_cost_usd="
        f"{cost_estimate.total_cost_usd}"
    )


def normalize_source_codes(
    value: Any,
) -> tuple[str, ...] | None:
    """Преобразует параметры source-code в tuple."""

    if value is None:
        return None

    source_codes = tuple(
        str(source_code).strip()
        for source_code in value
        if str(source_code).strip()
    )

    return source_codes or None


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет управляемый защищённый запуск."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(
            "Protected OpenAI ranking refused"
        )
        print(f"error={error}")
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print(
            "Telegram publication: "
            "not performed"
        )
        return 2

    if not arguments.confirm_live_request:
        print(
            "Protected OpenAI ranking refused"
        )
        print(
            "Use --confirm-live-request "
            "to permit a paid API request."
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print(
            "Telegram publication: "
            "not performed"
        )
        return 2

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    runtime = None
    result = None

    try:
        runtime = (
            create_openai_ranking_runtime(
                settings
            )
        )

        print(
            "Protected OpenAI ranking started"
        )
        print(
            "model="
            f"{runtime.evaluator.metadata.model_name}"
        )
        print(
            "as_of="
            f"{arguments.as_of.isoformat()}"
        )
        print(
            "window_hours="
            f"{arguments.window_hours}"
        )
        print(
            "limit="
            f"{arguments.limit}"
        )

        result = (
            await run_reserved_openai_ranking(
                database_pool,
                evaluator=runtime.evaluator,
                as_of=arguments.as_of,
                window_hours=(
                    arguments.window_hours
                ),
                limit=arguments.limit,
                source_codes=(
                    normalize_source_codes(
                        arguments.source_code
                    )
                ),
            )
        )

        persisted_records = (
            await load_persisted_ranking(
                database_pool,
                ranking_run_id=(
                    result.ranking_run_id
                ),
            )
        )

        if not persisted_records:
            raise RuntimeError(
                "Завершённый ranking_run "
                "не содержит news_scores."
            )

        print_run_summary(result)

        print_persisted_ranking(
            persisted_records,
            top_size=arguments.top_size,
        )

        print_openai_telemetry(result)

    except Exception as error:
        print()
        print(
            "Protected OpenAI ranking: FAILED"
        )
        print(
            "error_type="
            f"{type(error).__name__}"
        )
        print(f"error={error}")
        print(
            "OpenAI requests: "
            "possibly attempted"
        )
        print(
            "Database changes: reservation "
            "or failure record may be stored"
        )
        print(
            "Telegram publication: "
            "not performed"
        )

        notes = getattr(
            error,
            "__notes__",
            None,
        )

        if notes:
            for note in notes:
                print(f"error_note={note}")

        return 1

    finally:
        if runtime is not None:
            await close_sdk_client(
                runtime.sdk_client
            )

        await close_database_pool(
            database_pool
        )

    if result is None:
        raise RuntimeError(
            "Защищённый запуск "
            "не вернул результат."
        )

    print()
    print(
        "OpenAI requests: performed=1"
        if result.model_called
        else (
            "OpenAI requests: performed=0 "
            "(duplicate blocked)"
        )
    )

    print(
        "Database changes: "
        "ranking saved"
        if result.reservation.created_new
        else (
            "Database changes: "
            "existing ranking reused"
        )
    )

    print(
        "Telegram publication: "
        "not performed"
    )

    print(
        "Protected OpenAI ranking: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )