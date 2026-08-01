import argparse
import asyncio
from datetime import date
import inspect
import json
from typing import Any

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.openai_factory import (
    create_openai_generation_runtime,
)
from app.generation.openai_pipeline import (
    ReservedOpenAIGenerationResult,
    run_reserved_openai_generation,
)


def parse_publication_date(
    value: str,
) -> date:
    """Разбирает дату публикации YYYY-MM-DD."""

    normalized_value = value.strip()

    try:
        return date.fromisoformat(
            normalized_value
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--publication-date должен быть "
            "датой в формате YYYY-MM-DD."
        ) from error


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры защищённого запуска."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет защищённую генерацию "
            "Telegram-поста через OpenAI, "
            "сохраняет черновик в PostgreSQL "
            "и блокирует повторный платный запрос. "
            "Публикация в Telegram не выполняется."
        ),
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Разрешает реальный платный запрос "
            "OpenAI, если generation_request_key "
            "ещё не был зарезервирован."
        ),
    )

    parser.add_argument(
        "--ranking-run-id",
        type=int,
        required=True,
        help=(
            "Идентификатор завершённого "
            "ranking_run с сохранённым TOP-3."
        ),
    )

    parser.add_argument(
        "--publication-date",
        type=parse_publication_date,
        required=True,
        help=(
            "Дата выпуска в формате YYYY-MM-DD."
        ),
    )

    parser.add_argument(
        "--telegram-chat-id",
        type=int,
        default=None,
        help=(
            "Целевой Telegram chat_id. "
            "Если не указан, используется "
            "TELEGRAM_CHANNEL_ID из .env. "
            "Сообщение в Telegram не отправляется."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет параметры запуска."""

    if arguments.ranking_run_id <= 0:
        raise ValueError(
            "--ranking-run-id должен быть "
            "больше нуля."
        )

    if (
        arguments.telegram_chat_id
        is not None
        and arguments.telegram_chat_id == 0
    ):
        raise ValueError(
            "--telegram-chat-id не может "
            "быть равен нулю."
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


def decode_jsonb(
    value: Any,
) -> Any:
    """Декодирует jsonb, полученный от asyncpg."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "PostgreSQL вернул некорректный "
                "JSON в поле jsonb."
            ) from error

    return value


async def load_persisted_generation(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> asyncpg.Record:
    """Читает выпуск и generated_post из PostgreSQL."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_id,
                b.publication_date,
                b.edition,
                b.batch_status,
                b.ranking_run_id,
                b.target_telegram_chat_id,
                b.generation_request_key,
                b.error_message,

                b.metadata->>'generator_name'
                    AS generator_name,

                b.metadata->>'generator_version'
                    AS generator_version,

                b.metadata->>'prompt_version'
                    AS prompt_version,

                b.metadata->>'model_name'
                    AS model_name,

                b.metadata->>'text_format'
                    AS batch_text_format,

                b.metadata->'openai_usage'
                    AS openai_usage,

                b.metadata->'openai_cost'
                    AS openai_cost,

                b.metadata->'failure'
                    AS failure,

                gp.generated_post_id,
                gp.version_number,
                gp.post_status,
                gp.post_text,
                gp.text_format,
                gp.text_model_name,
                gp.text_prompt_version,

                gp.generation_metadata
                    ->'generated_items'
                    AS generated_items,

                gp.generation_metadata
                    ->>'generation_request_key'
                    AS post_request_key,

                gp.generation_metadata
                    ->>'completion_version'
                    AS completion_version,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.batch_items AS bi
                    WHERE bi.batch_id = b.batch_id
                ) AS batch_item_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS p
                    WHERE p.batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .publication_attempts AS pa
                    JOIN
                        top3_news
                        .generated_posts AS p
                      ON p.generated_post_id =
                         pa.generated_post_id
                    WHERE p.batch_id = b.batch_id
                ) AS publication_attempt_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news.review_actions AS ra
                    JOIN
                        top3_news
                        .generated_posts AS p
                      ON p.generated_post_id =
                         ra.generated_post_id
                    WHERE p.batch_id = b.batch_id
                ) AS review_action_count

            FROM
                top3_news
                .publication_batches AS b
            LEFT JOIN
                top3_news.generated_posts AS gp
              ON gp.batch_id = b.batch_id
             AND gp.version_number = 1
            WHERE b.batch_id = $1
            """,
            batch_id,
        )

    if record is None:
        raise RuntimeError(
            "publication_batch не найден "
            f"после запуска: batch_id={batch_id}"
        )

    return record


def print_run_summary(
    result: ReservedOpenAIGenerationResult,
) -> None:
    """Печатает состояние защищённого запуска."""

    print()
    print(
        "Protected OpenAI generation result:"
    )
    print(
        f"batch_id={result.batch_id}"
    )
    print(
        "generation_request_key="
        f"{result.request_key.value}"
    )
    print(
        "request_key_version="
        f"{result.request_key.version}"
    )
    print(
        "batch_status="
        f"{result.batch_status}"
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
        "ranking_run_id="
        f"{result.selection.ranking_run_id}"
    )
    print(
        "news_ids="
        + ",".join(
            str(news_id)
            for news_id
            in result.selection.news_ids
        )
    )


def print_persisted_generation(
    record: asyncpg.Record,
) -> None:
    """Печатает сохранённый выпуск и пост."""

    generated_items = decode_jsonb(
        record["generated_items"]
    )

    print()
    print("Persisted generation:")
    print(
        f"batch_id={record['batch_id']}"
    )
    print(
        "publication_date="
        f"{record['publication_date']}"
    )
    print(
        f"edition={record['edition']}"
    )
    print(
        "batch_status="
        f"{record['batch_status']}"
    )
    print(
        "ranking_run_id="
        f"{record['ranking_run_id']}"
    )
    print(
        "target_telegram_chat_id="
        f"{record['target_telegram_chat_id']}"
    )
    print(
        "generator_name="
        f"{record['generator_name']}"
    )
    print(
        "generator_version="
        f"{record['generator_version']}"
    )
    print(
        "model_name="
        f"{record['model_name']}"
    )
    print(
        "prompt_version="
        f"{record['prompt_version']}"
    )
    print(
        "batch_item_count="
        f"{record['batch_item_count']}"
    )
    print(
        "generated_post_count="
        f"{record['generated_post_count']}"
    )
    print(
        "publication_attempt_count="
        f"{record['publication_attempt_count']}"
    )
    print(
        "review_action_count="
        f"{record['review_action_count']}"
    )

    if record["generated_post_id"] is None:
        print("generated_post_id=none")

        failure = decode_jsonb(
            record["failure"]
        )

        if isinstance(failure, dict):
            print(
                "failure_error_type="
                f"{failure.get('error_type')}"
            )
            print(
                "failure_error_message="
                f"{failure.get('error_message')}"
            )

        if record["error_message"]:
            print(
                "error_message="
                f"{record['error_message']}"
            )

        return

    print(
        "generated_post_id="
        f"{record['generated_post_id']}"
    )
    print(
        "version_number="
        f"{record['version_number']}"
    )
    print(
        "post_status="
        f"{record['post_status']}"
    )
    print(
        "text_format="
        f"{record['text_format']}"
    )
    print(
        "text_model_name="
        f"{record['text_model_name']}"
    )
    print(
        "text_prompt_version="
        f"{record['text_prompt_version']}"
    )
    print(
        "completion_version="
        f"{record['completion_version']}"
    )

    if isinstance(generated_items, list):
        print(
            "generated_item_count="
            f"{len(generated_items)}"
        )

        for item in generated_items:
            if not isinstance(item, dict):
                continue

            print()
            print(
                "Generated item "
                f"{item.get('position')}:"
            )
            print(
                "news_id="
                f"{item.get('news_id')}"
            )
            print(
                "headline="
                f"{item.get('headline')}"
            )

    print()
    print("Generated Telegram post:")
    print()
    print(record["post_text"])


def print_openai_telemetry(
    record: asyncpg.Record,
    *,
    model_called: bool,
) -> None:
    """Печатает сохранённые usage и стоимость."""

    usage = decode_jsonb(
        record["openai_usage"]
    )

    cost = decode_jsonb(
        record["openai_cost"]
    )

    if not isinstance(usage, dict):
        print()
        print(
            "OpenAI telemetry: "
            "not stored"
        )
        return

    if not isinstance(cost, dict):
        raise RuntimeError(
            "OpenAI cost отсутствует "
            "при сохранённом token usage."
        )

    print()
    print(
        "OpenAI telemetry source="
        + (
            "new live request"
            if model_called
            else "existing reservation"
        )
    )

    print()
    print("OpenAI token usage:")
    print(
        "input_tokens="
        f"{usage.get('input_tokens')}"
    )
    print(
        "regular_input_tokens="
        f"{usage.get('regular_input_tokens')}"
    )
    print(
        "cached_input_tokens="
        f"{usage.get('cached_input_tokens')}"
    )
    print(
        "cache_write_tokens="
        f"{usage.get('cache_write_tokens')}"
    )
    print(
        "output_tokens="
        f"{usage.get('output_tokens')}"
    )
    print(
        "reasoning_tokens="
        f"{usage.get('reasoning_tokens')}"
    )
    print(
        "total_tokens="
        f"{usage.get('total_tokens')}"
    )

    print()
    print("OpenAI estimated cost:")
    print(
        "model="
        f"{cost.get('model_name')}"
    )
    print(
        "pricing_version="
        f"{cost.get('pricing_version')}"
    )
    print(
        "regular_input_cost_usd="
        f"{cost.get('regular_input_cost_usd')}"
    )
    print(
        "cached_input_cost_usd="
        f"{cost.get('cached_input_cost_usd')}"
    )
    print(
        "cache_write_cost_usd="
        f"{cost.get('cache_write_cost_usd')}"
    )
    print(
        "output_cost_usd="
        f"{cost.get('output_cost_usd')}"
    )
    print(
        "total_cost_usd="
        f"{cost.get('total_cost_usd')}"
    )


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет управляемый защищённый запуск."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print(
            "Protected OpenAI generation refused"
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
            "Protected OpenAI generation refused"
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

    telegram_chat_id = (
        arguments.telegram_chat_id
        if arguments.telegram_chat_id
        is not None
        else settings.telegram_channel_id
    )

    if telegram_chat_id == 0:
        print(
            "Protected OpenAI generation refused"
        )
        print(
            "error=Telegram chat_id "
            "не может быть равен нулю."
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print(
            "Telegram publication: "
            "not performed"
        )
        return 2

    database_pool = await create_database_pool(
        settings
    )

    runtime = None
    result = None

    try:
        runtime = (
            create_openai_generation_runtime(
                settings
            )
        )

        print(
            "Protected OpenAI generation started"
        )
        print(
            "model="
            f"{runtime.generator.metadata.model_name}"
        )
        print(
            "ranking_run_id="
            f"{arguments.ranking_run_id}"
        )
        print(
            "publication_date="
            f"{arguments.publication_date}"
        )
        print(
            "telegram_chat_id="
            f"{telegram_chat_id}"
        )
        print(
            "Telegram publication: "
            "disabled"
        )

        result = (
            await run_reserved_openai_generation(
                database_pool,
                generator=runtime.generator,
                ranking_run_id=(
                    arguments.ranking_run_id
                ),
                publication_date=(
                    arguments.publication_date
                ),
                telegram_chat_id=(
                    telegram_chat_id
                ),
            )
        )

        persisted_record = (
            await load_persisted_generation(
                database_pool,
                batch_id=result.batch_id,
            )
        )

        print_run_summary(result)

        print_persisted_generation(
            persisted_record
        )

        print_openai_telemetry(
            persisted_record,
            model_called=result.model_called,
        )

        if result.batch_status != "awaiting_review":
            raise RuntimeError(
                "Выпуск не перешёл "
                "в awaiting_review: "
                f"batch_status="
                f"{result.batch_status}"
            )

        if (
            persisted_record[
                "generated_post_id"
            ]
            is None
        ):
            raise RuntimeError(
                "Выпуск awaiting_review "
                "не содержит generated_post."
            )

        if (
            persisted_record[
                "generated_post_count"
            ]
            != 1
        ):
            raise RuntimeError(
                "Ожидался ровно один "
                "generated_post."
            )

        if (
            persisted_record[
                "publication_attempt_count"
            ]
            != 0
        ):
            raise RuntimeError(
                "Во время генерации появилась "
                "попытка публикации."
            )

    except Exception as error:
        print()
        print(
            "Protected OpenAI generation: FAILED"
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
            "Database changes: reservation, "
            "generated post or failure record "
            "may be stored"
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
            "Защищённый запуск генерации "
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
        "generation saved"
        if result.reservation.created_new
        else (
            "Database changes: "
            "existing generation reused"
        )
    )

    print(
        "Post status: awaiting_review"
    )
    print(
        "Telegram publication: "
        "not performed"
    )
    print(
        "Protected OpenAI generation: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )