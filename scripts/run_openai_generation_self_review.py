import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import difflib
import inspect
import json
from typing import Any

import asyncpg

from app.config import get_settings
from app.db.generation_selection import (
    load_generation_top3,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.openai_factory import (
    create_openai_generation_runtime,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры диагностического запуска."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет один диагностический self-review "
            "существующего Telegram-поста через OpenAI. "
            "Модель может самостоятельно использовать "
            "web_search. База данных и Telegram не "
            "изменяются."
        ),
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Разрешает один реальный платный запрос "
            "OpenAI."
        ),
    )

    parser.add_argument(
        "--generated-post-id",
        type=int,
        required=True,
        help=(
            "Идентификатор существующего generated_post, "
            "который нужно проверить вторым проходом."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет параметры запуска."""

    if arguments.generated_post_id <= 0:
        raise ValueError(
            "--generated-post-id должен быть больше нуля."
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


async def load_source_post(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
) -> asyncpg.Record:
    """Читает пост и ranking_run без изменений БД."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                gp.generated_post_id,
                gp.batch_id,
                gp.version_number,
                gp.post_status,
                gp.post_text,
                gp.text_format,
                b.batch_status,
                b.ranking_run_id
            FROM top3_news.generated_posts AS gp
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = gp.batch_id
            WHERE gp.generated_post_id = $1
            """,
            generated_post_id,
        )

    if record is None:
        raise LookupError(
            "Не найден generated_post: "
            f"{generated_post_id}"
        )

    post_text = record["post_text"]

    if not isinstance(post_text, str):
        raise ValueError(
            "generated_post.post_text отсутствует."
        )

    if not post_text.strip():
        raise ValueError(
            "generated_post.post_text пустой."
        )

    text_format = record["text_format"]

    if text_format != "markdown":
        raise ValueError(
            "Self-review сейчас поддерживает только "
            "text_format=markdown: "
            f"actual={text_format!r}"
        )

    return record


def serialize_telemetry(value: Any) -> str:
    """Печатает dataclass/dict телеметрии безопасно."""

    if value is None:
        return "null"

    if is_dataclass(value):
        value = asdict(value)

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
            sort_keys=True,
        )

    return str(value)


def print_diff(
    source_post_text: str,
    reviewed_post_text: str,
) -> None:
    """Показывает текстовый diff двух версий."""

    diff = difflib.unified_diff(
        source_post_text.splitlines(),
        reviewed_post_text.splitlines(),
        fromfile="original",
        tofile="self_review",
        lineterm="",
    )

    print("----- DIFF -----")

    diff_lines = list(diff)

    if not diff_lines:
        print("No textual changes")
        return

    for line in diff_lines:
        print(line)


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет один read-only self-review поста."""

    try:
        validate_arguments(arguments)
    except ValueError as error:
        print("Protected OpenAI self-review refused")
        print(f"error={error}")
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    if not arguments.confirm_live_request:
        print("Protected OpenAI self-review refused")
        print(
            "Use --confirm-live-request to permit "
            "one paid API request."
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

    try:
        source_record = await load_source_post(
            database_pool,
            generated_post_id=(
                arguments.generated_post_id
            ),
        )

        selection = await load_generation_top3(
            database_pool,
            ranking_run_id=int(
                source_record["ranking_run_id"]
            ),
        )

        runtime = create_openai_generation_runtime(
            settings
        )

        source_post_text = str(
            source_record["post_text"]
        ).strip()

        print("Protected OpenAI self-review started")
        print(
            "model="
            f"{runtime.generator.metadata.model_name}"
        )
        print(
            "generated_post_id="
            f"{source_record['generated_post_id']}"
        )
        print(
            "version_number="
            f"{source_record['version_number']}"
        )
        print(
            "post_status="
            f"{source_record['post_status']}"
        )
        print(
            "batch_id="
            f"{source_record['batch_id']}"
        )
        print(
            "batch_status="
            f"{source_record['batch_status']}"
        )
        print(
            "ranking_run_id="
            f"{source_record['ranking_run_id']}"
        )
        print(
            "news_ids="
            f"{selection.news_ids}"
        )
        print("Database changes: disabled")
        print("Telegram publication: disabled")

        result = (
            await runtime.generator
            .generate_self_review_detailed(
                selection.items,
                source_post_text=source_post_text,
            )
        )

        reviewed_post_text = (
            result.payload.post_text.strip()
        )
        model_response = result.model_response

        print("Self-review completed")
        print(
            "changed="
            f"{reviewed_post_text != source_post_text}"
        )
        print(
            "web_search_used="
            f"{model_response.web_search_used}"
        )
        print(
            "web_search_call_count="
            f"{model_response.web_search_call_count}"
        )
        print(
            "web_search_tool_cost_usd="
            f"{model_response.web_search_call_count * 0.01:.4f}"
        )
        print(
            "web_source_count="
            f"{len(model_response.web_source_urls)}"
        )

        for index, url in enumerate(
            model_response.web_source_urls,
            start=1,
        ):
            print(f"web_source_{index}={url}")

        print(
            "usage="
            f"{serialize_telemetry(model_response.usage)}"
        )
        print(
            "cost_estimate="
            f"{serialize_telemetry(model_response.cost_estimate)}"
        )

        print("----- ORIGINAL -----")
        print(source_post_text)
        print("----- SELF REVIEW -----")
        print(reviewed_post_text)
        print_diff(
            source_post_text,
            reviewed_post_text,
        )

        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 0

    finally:
        if runtime is not None:
            await close_sdk_client(
                runtime.sdk_client
            )

        await close_database_pool(
            database_pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )