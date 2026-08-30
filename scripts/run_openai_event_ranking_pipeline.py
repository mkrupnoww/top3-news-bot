import argparse
import asyncio
from datetime import datetime, timezone
import inspect
import json
from typing import Any

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.openai_event_factory import (
    create_openai_event_ranking_runtime,
)
from app.ranking.openai_event_pipeline import (
    ReservedOpenAIEventRankingResult,
    run_reserved_openai_event_ranking,
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
    """Разбирает параметры защищённого запуска v4."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет защищённое event-level "
            "ранжирование киноновостей через OpenAI, "
            "рассчитывает полную формулу "
            "top3_cinema_v5 и сохраняет результат "
            "в PostgreSQL."
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
            "Конец строгого 24-часового окна "
            "в ISO 8601 с часовым поясом."
        ),
    )

    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help=(
            "Размер окна. Для полной формулы v4 "
            "допускается только 24."
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
            "Максимальное число публикаций-кандидатов. "
            "По умолчанию 20."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет параметры до обращения к API и БД."""

    if arguments.window_hours != 24.0:
        raise ValueError(
            "--window-hours для top3_cinema_v5 "
            "должен быть равен 24."
        )

    if arguments.limit < 3:
        raise ValueError(
            "--limit должен быть не меньше 3."
        )

    if arguments.limit > 100:
        raise ValueError(
            "--limit не может превышать 100."
        )


def normalize_source_codes(
    value: Any,
) -> tuple[str, ...] | None:
    """Преобразует повторяемый source-code в tuple."""

    if value is None:
        return None

    source_codes = tuple(
        str(source_code).strip()
        for source_code in value
        if str(source_code).strip()
    )

    return source_codes or None


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


def decode_jsonb(
    value: Any,
) -> Any:
    """Декодирует jsonb, возвращённый asyncpg."""

    if isinstance(value, str):
        return json.loads(value)

    return value


async def load_persisted_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> asyncpg.Record:
    """Загружает итоговые параметры ranking_run."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                ranking_run_id,
                request_key,
                run_status,
                formula_version,
                model_name,
                prompt_version,
                window_started_at,
                window_finished_at,
                candidate_count,
                scored_count,
                eligible_count,
                error_message,
                started_at,
                finished_at,
                parameters->'openai_usage'
                    AS openai_usage,
                parameters->'openai_cost'
                    AS openai_cost,
                parameters->'winner_news_ids'
                    AS winner_news_ids,
                parameters->>'event_count'
                    AS event_count,
                parameters->>'combination_count'
                    AS combination_count,
                parameters->'audience_maxima'
                    AS audience_maxima,
                parameters->'story_cluster_verification'
                    AS story_cluster_verification
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            ranking_run_id,
        )

    if record is None:
        raise LookupError(
            "ranking_run не найден после запуска: "
            f"ranking_run_id={ranking_run_id}"
        )

    return record


async def load_persisted_events(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> tuple[asyncpg.Record, ...]:
    """Читает сохранённые инфоповоды и оценки."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                ns.news_id,
                ns.rank_position,
                ns.individual_score,
                ns.is_eligible,
                ns.exclusion_reason,
                ns.selected_for_top3,
                ns.top3_position,
                ns.age_hours,
                ns.f_score,
                ns.u_score,
                ns.i_score,
                ns.m_score,
                ns.v_score,
                ns.c_score,
                ns.s_score,
                ns.r_score,
                ns.k_score,
                ns.n_score,
                ns.e_score,
                ns.x_score,
                ns.h_score,
                ns.q_score,
                ns.resonance_confidence,
                ns.score_explanation,
                re.event_title,
                re.event_time_utc,
                re.macro_topic,
                re.event_details->>'story_cluster_key'
                    AS story_cluster_key,
                re.source_weight_sum,
                COALESCE(
                    NULLIF(n.normalized_title, ''),
                    NULLIF(n.raw_title, ''),
                    'Без заголовка'
                ) AS representative_title,
                n.source_url,
                s.source_code,
                s.source_name,
                array_agg(
                    rem.news_id
                    ORDER BY rem.news_id
                ) AS member_news_ids
            FROM top3_news.news_scores AS ns
            JOIN top3_news.ranking_events AS re
              ON re.ranking_event_id
                 = ns.ranking_event_id
             AND re.ranking_run_id
                 = ns.ranking_run_id
            JOIN top3_news.ranking_event_members AS rem
              ON rem.ranking_event_id
                 = re.ranking_event_id
             AND rem.ranking_run_id
                 = re.ranking_run_id
            JOIN top3_news.news_items AS n
              ON n.news_id = ns.news_id
            JOIN top3_news.sources AS s
              ON s.source_id = n.source_id
            WHERE ns.ranking_run_id = $1
            GROUP BY
                ns.score_id,
                re.ranking_event_id,
                n.news_id,
                s.source_id
            ORDER BY
                ns.rank_position,
                ns.news_id
            """,
            ranking_run_id,
        )

    return tuple(records)


async def load_winner_combination(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> asyncpg.Record:
    """Читает победившую комбинацию TOP-3."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                rc.combination_id,
                rc.combination_rank,
                rc.mean_individual_score,
                rc.diversity_score,
                rc.final_top_score,
                rc.mean_m_score,
                rc.mean_q_score,
                rc.mean_f_score,
                rc.distinct_macro_topic_count,
                rc.selection_reason,
                array_agg(
                    ns.news_id
                    ORDER BY ci.position
                ) AS news_ids
            FROM top3_news.ranking_combinations AS rc
            JOIN top3_news.ranking_combination_items AS ci
              ON ci.combination_id
                 = rc.combination_id
             AND ci.ranking_run_id
                 = rc.ranking_run_id
            JOIN top3_news.news_scores AS ns
              ON ns.score_id = ci.score_id
             AND ns.ranking_run_id
                 = ci.ranking_run_id
            WHERE rc.ranking_run_id = $1
              AND rc.is_winner = true
            GROUP BY rc.combination_id
            """,
            ranking_run_id,
        )

    if record is None:
        raise RuntimeError(
            "Победившая комбинация TOP-3 "
            "не найдена."
        )

    return record


def print_run_summary(
    result: ReservedOpenAIEventRankingResult,
    run_record: asyncpg.Record,
) -> None:
    """Печатает состояние защищённого запуска."""

    selection = result.candidate_selection

    print()
    print("Protected OpenAI event ranking result:")
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
        f"{run_record['run_status']}"
    )
    print(
        "formula_version="
        f"{run_record['formula_version']}"
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
        f"{run_record['candidate_count']}"
    )
    print(
        "scored_event_count="
        f"{run_record['scored_count']}"
    )
    print(
        "eligible_count="
        f"{run_record['eligible_count']}"
    )
    print(
        "combination_count="
        f"{run_record['combination_count']}"
    )
    print(
        "window_started_at="
        f"{selection.window_start.isoformat()}"
    )
    print(
        "window_finished_at="
        f"{selection.window_end.isoformat()}"
    )
    print("audience_metrics_supplied=0")
    print(
        "missing_audience_metrics_policy="
        "R=0, confidence=unavailable"
    )

    verification = decode_jsonb(
        run_record["story_cluster_verification"]
    )

    if isinstance(verification, dict):
        print(
            "story_cluster_verification_attempted="
            f"{str(verification.get('attempted')).lower()}"
        )
        print(
            "story_cluster_verification_succeeded="
            f"{str(verification.get('succeeded')).lower()}"
        )
        print(
            "story_cluster_verification_degraded="
            f"{str(verification.get('degraded')).lower()}"
        )
        print(
            "story_cluster_verification_skipped_reason="
            f"{verification.get('skipped_reason')}"
        )
        print(
            "story_cluster_counts="
            f"{verification.get('cluster_count_before')}->"
            f"{verification.get('cluster_count_after')}"
        )
        print(
            "story_cluster_multi_event_counts="
            f"{verification.get('multi_event_cluster_count_before')}->"
            f"{verification.get('multi_event_cluster_count_after')}"
        )
        print(
            "story_cluster_verifier_event_count="
            f"{verification.get('verifier_event_count')}"
        )


def print_persisted_events(
    records: tuple[asyncpg.Record, ...],
) -> None:
    """Печатает полный event-level рейтинг."""

    print()
    print("Persisted event ranking:")

    for record in records:
        selected_marker = (
            f"TOP-{record['top3_position']}"
            if record["selected_for_top3"]
            else "not_selected"
        )

        member_news_ids = ",".join(
            str(news_id)
            for news_id
            in record["member_news_ids"]
        )

        print()
        print(
            f"{record['rank_position']}. "
            f"news_id={record['news_id']} "
            f"B={record['individual_score']} "
            f"{selected_marker}"
        )
        print(
            "   event_title="
            f"{record['event_title']}"
        )
        print(
            "   macro_topic="
            f"{record['macro_topic']}"
        )
        print(
            "   story_cluster_key="
            f"{record['story_cluster_key']}"
        )
        print(
            "   event_time_utc="
            f"{record['event_time_utc'].isoformat()}"
        )
        print(
            "   member_news_ids="
            f"{member_news_ids}"
        )
        print(
            "   representative_title="
            f"{record['representative_title']}"
        )
        print(
            "   source="
            f"{record['source_name']} "
            f"[{record['source_code']}]"
        )
        print(
            "   base_scores="
            f"F:{record['f_score']} "
            f"U:{record['u_score']} "
            f"I:{record['i_score']} "
            f"M:{record['m_score']} "
            f"R:{record['r_score']} "
            f"H:{record['h_score']} "
            f"Q:{record['q_score']}"
        )
        print(
            "   resonance_scores="
            f"V:{record['v_score']} "
            f"C:{record['c_score']} "
            f"S:{record['s_score']} "
            f"confidence:"
            f"{record['resonance_confidence']}"
        )
        print(
            "   hook_scores="
            f"K:{record['k_score']} "
            f"N:{record['n_score']} "
            f"E:{record['e_score']} "
            f"X:{record['x_score']}"
        )
        print(
            "   eligible="
            f"{str(record['is_eligible']).lower()} "
            f"exclusion_reason="
            f"{record['exclusion_reason']}"
        )
        print(
            "   explanation="
            f"{record['score_explanation']}"
        )
        print(
            "   source_url="
            f"{record['source_url']}"
        )


def print_winner(
    winner: asyncpg.Record,
) -> None:
    """Печатает победивший TOP-3."""

    news_ids = ",".join(
        str(news_id)
        for news_id in winner["news_ids"]
    )

    print()
    print("Winning TOP-3 combination:")
    print(
        "combination_id="
        f"{winner['combination_id']}"
    )
    print(
        "combination_rank="
        f"{winner['combination_rank']}"
    )
    print(f"news_ids={news_ids}")
    print(
        "mean_individual_score="
        f"{winner['mean_individual_score']}"
    )
    print(
        "diversity_score="
        f"{winner['diversity_score']}"
    )
    print(
        "final_top_score="
        f"{winner['final_top_score']}"
    )
    print(
        "distinct_macro_topic_count="
        f"{winner['distinct_macro_topic_count']}"
    )
    print(
        "tie_break_means="
        f"M:{winner['mean_m_score']} "
        f"Q:{winner['mean_q_score']} "
        f"F:{winner['mean_f_score']}"
    )
    print(
        "selection_reason="
        f"{winner['selection_reason']}"
    )


def print_openai_telemetry(
    run_record: asyncpg.Record,
) -> None:
    """Печатает сохранённые usage и стоимость."""

    usage = decode_jsonb(
        run_record["openai_usage"]
    )

    cost = decode_jsonb(
        run_record["openai_cost"]
    )

    if not isinstance(usage, dict):
        print()
        print("OpenAI token usage: unavailable")
        return

    if not isinstance(cost, dict):
        print()
        print("OpenAI estimated cost: unavailable")
        return

    print()
    print("OpenAI token usage:")
    print(
        "input_tokens="
        f"{usage['input_tokens']}"
    )
    print(
        "regular_input_tokens="
        f"{usage['regular_input_tokens']}"
    )
    print(
        "cached_input_tokens="
        f"{usage['cached_input_tokens']}"
    )
    print(
        "cache_write_tokens="
        f"{usage['cache_write_tokens']}"
    )
    print(
        "output_tokens="
        f"{usage['output_tokens']}"
    )
    print(
        "reasoning_tokens="
        f"{usage['reasoning_tokens']}"
    )
    print(
        "total_tokens="
        f"{usage['total_tokens']}"
    )

    print()
    print("OpenAI estimated cost:")
    print(
        "model="
        f"{cost['model_name']}"
    )
    print(
        "pricing_version="
        f"{cost['pricing_version']}"
    )
    print(
        "regular_input_cost_usd="
        f"{cost['regular_input_cost_usd']}"
    )
    print(
        "cached_input_cost_usd="
        f"{cost['cached_input_cost_usd']}"
    )
    print(
        "cache_write_cost_usd="
        f"{cost['cache_write_cost_usd']}"
    )
    print(
        "output_cost_usd="
        f"{cost['output_cost_usd']}"
    )
    print(
        "total_cost_usd="
        f"{cost['total_cost_usd']}"
    )


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет управляемый защищённый запуск v4."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(
            "Protected OpenAI event ranking refused"
        )
        print(f"error={error}")
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    if not arguments.confirm_live_request:
        print(
            "Protected OpenAI event ranking refused"
        )
        print(
            "Use --confirm-live-request "
            "to permit a paid API request."
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    runtime = None
    result = None

    try:
        runtime = (
            create_openai_event_ranking_runtime(
                settings
            )
        )

        print(
            "Protected OpenAI event ranking started"
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
        print("audience_metrics_supplied=0")

        result = (
            await run_reserved_openai_event_ranking(
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
                audience_metrics=(),
            )
        )

        run_record = await load_persisted_run(
            database_pool,
            ranking_run_id=(
                result.ranking_run_id
            ),
        )

        if run_record["run_status"] != "completed":
            raise RuntimeError(
                "ranking_run не завершён: "
                f"status={run_record['run_status']}, "
                f"error={run_record['error_message']}"
            )

        event_records = (
            await load_persisted_events(
                database_pool,
                ranking_run_id=(
                    result.ranking_run_id
                ),
            )
        )

        if not event_records:
            raise RuntimeError(
                "Завершённый ranking_run "
                "не содержит event scores."
            )

        winner = await load_winner_combination(
            database_pool,
            ranking_run_id=(
                result.ranking_run_id
            ),
        )

        print_run_summary(
            result,
            run_record,
        )

        print_persisted_events(
            event_records
        )

        print_winner(winner)

        print_openai_telemetry(
            run_record
        )

    except Exception as error:
        print()
        print(
            "Protected OpenAI event ranking: FAILED"
        )
        print(
            "error_type="
            f"{type(error).__name__}"
        )
        print(f"error={error}")
        print(
            "OpenAI requests: possibly attempted"
        )
        print(
            "Database changes: reservation, "
            "result, or failure may be stored"
        )
        print("Telegram publication: not performed")

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
            "Защищённый event-запуск "
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
        "Database changes: event ranking saved"
        if result.reservation.created_new
        else (
            "Database changes: existing "
            "event ranking reused"
        )
    )

    print("Telegram publication: not performed")
    print("Protected OpenAI event ranking: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )