import argparse
import asyncio
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
from app.generation.openai_revision_pipeline import (
    ReservedOpenAIGenerationRevisionResult,
    run_reserved_openai_generation_revision,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры защищённого запуска."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет защищённую редакционную "
            "ревизию Telegram-поста через OpenAI "
            "по сохранённому human review_action, "
            "создаёт новую версию generated_post "
            "и блокирует повторный платный запрос. "
            "Публикация в Telegram не выполняется."
        ),
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Разрешает реальный платный запрос "
            "OpenAI, если для review_action "
            "ещё нет активной или завершённой "
            "revision reservation."
        ),
    )

    parser.add_argument(
        "--review-action-id",
        type=int,
        required=True,
        help=(
            "Идентификатор human review_action "
            "с decision=changes_required и "
            "requested_action=regenerate_text."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Проверяет параметры запуска."""

    if arguments.review_action_id <= 0:
        raise ValueError(
            "--review-action-id должен быть "
            "больше нуля."
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


async def load_persisted_revision(
    pool: asyncpg.Pool,
    *,
    generation_revision_id: int,
) -> asyncpg.Record:
    """Читает сохранённую revision и версии поста."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                grr.generation_revision_id,
                grr.batch_id,
                grr.source_generated_post_id,
                grr.review_action_id,
                grr.target_version_number,
                grr.revision_status,
                grr.revision_request_key,
                grr.request_key_version,
                grr.requested_action,
                grr.editorial_comment,
                grr.issues,
                grr.model_name,
                grr.generator_version,
                grr.prompt_version,
                grr.text_format,
                grr.openai_usage,
                grr.openai_cost,
                grr.error_type,
                grr.error_message,
                grr.completed_at,
                grr.failed_at,

                b.publication_date,
                b.edition,
                b.batch_status,
                b.ranking_run_id,
                b.target_telegram_chat_id,

                source_gp.version_number
                    AS source_version_number,
                source_gp.post_status
                    AS source_post_status,
                source_gp.post_text
                    AS source_post_text,

                target_gp.generated_post_id
                    AS generated_post_id,
                target_gp.version_number,
                target_gp.post_status,
                target_gp.post_text,
                target_gp.text_format
                    AS target_text_format,
                target_gp.text_model_name,
                target_gp.text_prompt_version,

                target_gp.generation_metadata
                    ->>'generation_mode'
                    AS generation_mode,

                target_gp.generation_metadata
                    ->>'completion_version'
                    AS completion_version,

                target_gp.generation_metadata
                    ->'generated_items'
                    AS generated_items,

                ra.reviewer_type,
                ra.reviewer_telegram_user_id,
                ra.decision,
                ra.requested_action
                    AS review_requested_action,
                ra.comment_text,
                ra.issues
                    AS review_issues,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp
                    WHERE gp.batch_id = grr.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .generation_revision_requests AS r
                    WHERE r.batch_id = grr.batch_id
                      AND r.review_action_id =
                          grr.review_action_id
                ) AS revision_request_count,

                (
                    SELECT COUNT(*)::integer
                    FROM
                        top3_news
                        .publication_attempts AS pa
                    JOIN top3_news.generated_posts AS gp
                      ON gp.generated_post_id =
                         pa.generated_post_id
                    WHERE gp.batch_id = grr.batch_id
                ) AS publication_attempt_count

            FROM
                top3_news
                .generation_revision_requests AS grr
            JOIN top3_news.publication_batches AS b
              ON b.batch_id = grr.batch_id
            JOIN top3_news.generated_posts AS source_gp
              ON source_gp.generated_post_id =
                 grr.source_generated_post_id
            JOIN top3_news.review_actions AS ra
              ON ra.review_action_id =
                 grr.review_action_id
            LEFT JOIN top3_news.generated_posts AS target_gp
              ON target_gp.generated_post_id =
                 grr.generated_post_id
            WHERE grr.generation_revision_id = $1
            """,
            generation_revision_id,
        )

    if record is None:
        raise RuntimeError(
            "generation_revision_request "
            "не найден после запуска: "
            "generation_revision_id="
            f"{generation_revision_id}"
        )

    return record


def print_run_summary(
    result: ReservedOpenAIGenerationRevisionResult,
) -> None:
    """Печатает состояние защищённого запуска."""

    print()
    print(
        "Protected OpenAI generation "
        "revision result:"
    )
    print(
        "generation_revision_id="
        f"{result.generation_revision_id}"
    )
    print(
        f"batch_id={result.batch_id}"
    )
    print(
        "source_generated_post_id="
        f"{result.source_generated_post_id}"
    )
    print(
        "review_action_id="
        f"{result.review_action_id}"
    )
    print(
        "target_version_number="
        f"{result.target_version_number}"
    )
    print(
        "revision_request_key="
        f"{result.request_key.value}"
    )
    print(
        "request_key_version="
        f"{result.request_key.version}"
    )
    print(
        "revision_status="
        f"{result.revision_status}"
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
        f"{result.revision_selection.ranking_run_id}"
    )
    print(
        "news_ids="
        + ",".join(
            str(news_id)
            for news_id
            in result.revision_selection.news_ids
        )
    )


def print_persisted_revision(
    record: asyncpg.Record,
) -> None:
    """Печатает сохранённую revision и пост."""

    generated_items = decode_jsonb(
        record["generated_items"]
    )

    review_issues = decode_jsonb(
        record["review_issues"]
    )

    print()
    print("Persisted generation revision:")
    print(
        "generation_revision_id="
        f"{record['generation_revision_id']}"
    )
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
        "source_generated_post_id="
        f"{record['source_generated_post_id']}"
    )
    print(
        "source_version_number="
        f"{record['source_version_number']}"
    )
    print(
        "source_post_status="
        f"{record['source_post_status']}"
    )
    print(
        "review_action_id="
        f"{record['review_action_id']}"
    )
    print(
        "reviewer_type="
        f"{record['reviewer_type']}"
    )
    print(
        "reviewer_telegram_user_id="
        f"{record['reviewer_telegram_user_id']}"
    )
    print(
        "decision="
        f"{record['decision']}"
    )
    print(
        "requested_action="
        f"{record['review_requested_action']}"
    )
    print(
        "editorial_comment="
        f"{record['comment_text']}"
    )

    if isinstance(review_issues, list):
        print(
            "issue_count="
            f"{len(review_issues)}"
        )

        for index, issue in enumerate(
            review_issues,
            start=1,
        ):
            print(
                f"issue_{index}={issue}"
            )

    print(
        "revision_status="
        f"{record['revision_status']}"
    )
    print(
        "revision_request_key="
        f"{record['revision_request_key']}"
    )
    print(
        "request_key_version="
        f"{record['request_key_version']}"
    )
    print(
        "model_name="
        f"{record['model_name']}"
    )
    print(
        "generator_version="
        f"{record['generator_version']}"
    )
    print(
        "revision_prompt_version="
        f"{record['prompt_version']}"
    )
    print(
        "text_format="
        f"{record['text_format']}"
    )
    print(
        "generated_post_count="
        f"{record['generated_post_count']}"
    )
    print(
        "revision_request_count="
        f"{record['revision_request_count']}"
    )
    print(
        "publication_attempt_count="
        f"{record['publication_attempt_count']}"
    )

    if record["revision_status"] == "failed":
        print(
            "failure_error_type="
            f"{record['error_type']}"
        )
        print(
            "failure_error_message="
            f"{record['error_message']}"
        )

        return

    if record["generated_post_id"] is None:
        print("generated_post_id=none")
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
        "target_text_format="
        f"{record['target_text_format']}"
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
        "generation_mode="
        f"{record['generation_mode']}"
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
    print("Revised Telegram post:")
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
            "new live revision request"
            if model_called
            else "existing revision"
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
            "Protected OpenAI generation "
            "revision refused"
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
            "Protected OpenAI generation "
            "revision refused"
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
            create_openai_generation_runtime(
                settings
            )
        )

        print(
            "Protected OpenAI generation "
            "revision started"
        )
        print(
            "model="
            f"{runtime.generator.metadata.model_name}"
        )
        print(
            "review_action_id="
            f"{arguments.review_action_id}"
        )
        print(
            "Telegram publication: "
            "disabled"
        )

        result = (
            await run_reserved_openai_generation_revision(
                database_pool,
                generator=runtime.generator,
                review_action_id=(
                    arguments.review_action_id
                ),
            )
        )

        persisted_record = (
            await load_persisted_revision(
                database_pool,
                generation_revision_id=(
                    result.generation_revision_id
                ),
            )
        )

        print_run_summary(result)

        print_persisted_revision(
            persisted_record
        )

        print_openai_telemetry(
            persisted_record,
            model_called=result.model_called,
        )

        if (
            result.revision_status
            != "completed"
        ):
            raise RuntimeError(
                "Revision не перешла "
                "в completed: "
                f"revision_status="
                f"{result.revision_status}"
            )

        if (
            persisted_record[
                "batch_status"
            ]
            != "awaiting_review"
        ):
            raise RuntimeError(
                "Batch после revision должен "
                "оставаться awaiting_review."
            )

        if (
            persisted_record[
                "source_post_status"
            ]
            != "superseded"
        ):
            raise RuntimeError(
                "Исходная версия после revision "
                "не перешла в superseded."
            )

        if (
            persisted_record[
                "generated_post_id"
            ]
            is None
        ):
            raise RuntimeError(
                "Completed revision не содержит "
                "новую generated_post."
            )

        if (
            persisted_record[
                "post_status"
            ]
            != "awaiting_review"
        ):
            raise RuntimeError(
                "Новая версия после revision "
                "не находится в awaiting_review."
            )

        if (
            persisted_record[
                "version_number"
            ]
            != (
                persisted_record[
                    "source_version_number"
                ]
                + 1
            )
        ):
            raise RuntimeError(
                "Номер новой версии "
                "не равен source version + 1."
            )

        if (
            persisted_record[
                "publication_attempt_count"
            ]
            != 0
        ):
            raise RuntimeError(
                "Во время revision появилась "
                "попытка публикации."
            )

    except Exception as error:
        print()
        print(
            "Protected OpenAI generation "
            "revision: FAILED"
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
            "Database changes: revision "
            "reservation, generated post "
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
            "Защищённый revision-запуск "
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
        "revision saved"
        if result.reservation.created_new
        else (
            "Database changes: "
            "existing revision reused"
        )
    )

    print(
        "Revision status: completed"
    )
    print(
        "Post status: awaiting_review"
    )
    print(
        "Telegram publication: "
        "not performed"
    )
    print(
        "Protected OpenAI generation "
        "revision: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )