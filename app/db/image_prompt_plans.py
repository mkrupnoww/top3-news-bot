"""Reservation и сохранение ImagePromptPlan до Image API."""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import asyncpg

from app.db.image_generation_reservation import (
    _load_initial_context,
    _load_regenerate_context,
    _validate_initial_context,
    _validate_regenerate_context,
    _validate_current_top3,
    _validate_batch_items,
)
from app.generation.image_generator import ImageGenerationNewsItem
from app.generation.image_prompt_agent import (
    IMAGE_PROMPT_AGENT_PROMPT_VERSION,
    IMAGE_PROMPT_AGENT_VERSION,
    IMAGE_PROMPT_NORMAL_VERSION,
    IMAGE_PROMPT_RECOVERY_VERSION,
    ImagePromptAgent,
    ImagePromptPlan,
)
from app.generation.moderation_diagnostics import is_moderation_failure


def encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


@dataclass(frozen=True, slots=True)
class SavedImagePrompt:
    image_prompt_id: int
    image_prompt_version: str
    plan: ImagePromptPlan


def _saved_prompt(row: asyncpg.Record) -> SavedImagePrompt:
    if row["prompt_status"] != "completed":
        raise RuntimeError(
            "Image Prompt Agent reservation ещё не завершена; повторный платный "
            f"вызов запрещён: image_prompt_id={row['image_prompt_id']}"
        )
    plan = ImagePromptPlan.model_validate(decode_json(row["structured_plan"]))
    if plan.final_image_prompt != row["final_image_prompt"]:
        raise ValueError("Сохранённый exact prompt расходится со structured plan.")
    return SavedImagePrompt(row["image_prompt_id"], row["image_prompt_version"], plan)


async def prepare_saved_image_prompt(
    pool: asyncpg.Pool,
    *,
    agent: ImagePromptAgent,
    batch_id: int,
    generated_post_id: int,
    ranking_run_id: int,
    request_kind: str,
    review_action_id: int | None,
    editorial_comment: str | None,
    issues: tuple[str, ...],
    items: tuple[ImageGenerationNewsItem, ...],
) -> SavedImagePrompt:
    """Резервирует агент, вызывает один Responses request и фиксирует exact prompt.

    Lock совпадает с existing image reservation. Между транзакциями сохраняется
    reserved запись: конкурентный или orphan запуск не повторяет платный вызов.
    Завершённый план переживает failure Image API и повторно используется.
    """

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock($1::bigint)", generated_post_id)
            post = await connection.fetchrow(
                """SELECT gp.post_text, gp.generation_metadata, gp.batch_id,
                          b.ranking_run_id
                   FROM top3_news.generated_posts gp
                   JOIN top3_news.publication_batches b USING (batch_id)
                   WHERE gp.generated_post_id=$1 FOR UPDATE OF gp, b""",
                generated_post_id,
            )
            if post is None or post["batch_id"] != batch_id or post["ranking_run_id"] != ranking_run_id:
                raise ValueError("Image Prompt Agent получил чужой batch/post/ranking context.")
            latest_image = await connection.fetchrow(
                """SELECT * FROM top3_news.image_generation_requests
                   WHERE generated_post_id=$1 AND request_kind=$2
                     AND review_action_id IS NOT DISTINCT FROM $3::bigint
                   ORDER BY image_generation_id DESC LIMIT 1""",
                generated_post_id, request_kind, review_action_id,
            )
            if latest_image and not is_moderation_failure(dict(latest_image)):
                if latest_image["image_prompt_id"] is not None:
                    row = await connection.fetchrow(
                        "SELECT * FROM top3_news.image_prompt_plans WHERE image_prompt_id=$1",
                        latest_image["image_prompt_id"],
                    )
                    # Текст после reservation не может незаметно изменить visual input.
                    if decode_json(row["input_payload"])["approved_post_text"] != post["post_text"]:
                        raise ValueError("Post text изменился после image prompt reservation.")
                    return _saved_prompt(row)
                if latest_image["image_status"] in {"reserved", "completed"}:
                    raise RuntimeError("У post уже есть исторический active image request без Prompt Agent.")

            # Проверяем существующие контракты до нового платного text request.
            if request_kind == "initial":
                context = await _load_initial_context(
                    connection, batch_id=batch_id, generated_post_id=generated_post_id
                )
                _validate_initial_context(
                    context, batch_id=batch_id, generated_post_id=generated_post_id,
                    ranking_run_id=ranking_run_id,
                )
            else:
                context = await _load_regenerate_context(
                    connection, batch_id=batch_id, generated_post_id=generated_post_id,
                    review_action_id=review_action_id,
                )
                _validate_regenerate_context(
                    context, batch_id=batch_id, generated_post_id=generated_post_id,
                    ranking_run_id=ranking_run_id, review_action_id=review_action_id,
                    editorial_comment=editorial_comment, issues=issues,
                )
            news_ids = await _validate_current_top3(
                connection, ranking_run_id=ranking_run_id, batch_id=batch_id, items=items
            )
            await _validate_batch_items(connection, batch_id=batch_id, expected_news_ids=news_ids)

            previous: dict[str, Any] | None = None
            source_prompt_id = None
            source_image_id = None
            mode = "normal_creative"
            attempt_kind = "initial" if request_kind == "initial" else "editorial_revision"
            if latest_image and is_moderation_failure(dict(latest_image)):
                mode = "moderation_recovery"
                attempt_kind = "moderation_recovery"
                source_image_id = latest_image["image_generation_id"]
                source_prompt_id = latest_image["image_prompt_id"]
                source_plan = None
                if source_prompt_id is not None:
                    source_row = await connection.fetchrow(
                        "SELECT structured_plan FROM top3_news.image_prompt_plans WHERE image_prompt_id=$1",
                        source_prompt_id,
                    )
                    source_plan = decode_json(source_row["structured_plan"])
                image_payload = decode_json(latest_image["request_payload"])
                previous = {
                    "image_generation_id": source_image_id,
                    "image_prompt_id": source_prompt_id,
                    "final_image_prompt": image_payload["model_request"]["prompt"],
                    "structured_plan": source_plan,
                    "moderation_diagnostics": (decode_json(latest_image["response_metadata"]) or {}).get(
                        "openai_error", {"error_code": "moderation_blocked", "request_id": None,
                                         "moderation_stage": None, "moderation_categories": None}
                    ),
                }

            metadata = decode_json(post["generation_metadata"])
            generated_items = metadata.get("generated_items", [])
            if not isinstance(generated_items, list):
                raise ValueError("Некорректный generated_items финального поста.")
            attempt_number = await connection.fetchval(
                """SELECT COALESCE(MAX(attempt_number), 0)+1 FROM top3_news.image_prompt_plans
                   WHERE generated_post_id=$1 AND request_kind=$2
                     AND review_action_id IS NOT DISTINCT FROM $3::bigint""",
                generated_post_id, request_kind, review_action_id,
            )
            instructions, payload = agent.build_input(
                mode=mode, items=items, post_text=post["post_text"],
                generated_items=generated_items, attempt_number=attempt_number,
                previous=previous, editorial_comment=editorial_comment, issues=issues,
            )
            original_rows = await connection.fetch(
                """SELECT news_id, raw_title, raw_summary, article_text
                   FROM top3_news.news_items WHERE news_id=ANY($1::bigint[])""", list(news_ids),
            )
            by_id = {row["news_id"]: dict(row) for row in original_rows}
            payload["original_news_texts"] = [by_id[news_id] for news_id in news_ids]
            # Номер reservation не участвует в ключе: повтор после crash использует
            # ту же completed/reserved запись, а новый moderation result — новый key.
            key_payload = {k: v for k, v in payload.items() if k != "attempt_number"}
            key_payload.update({
                "batch_id": batch_id, "generated_post_id": generated_post_id,
                "ranking_run_id": ranking_run_id, "request_kind": request_kind,
                "review_action_id": review_action_id, "agent_model": agent.model_name,
                "agent_version": IMAGE_PROMPT_AGENT_VERSION,
                "agent_prompt_version": IMAGE_PROMPT_AGENT_PROMPT_VERSION,
                "instructions": instructions,
            })
            prompt_key = sha256(encode_json(key_payload).encode("utf-8")).hexdigest()
            existing = await connection.fetchrow(
                """SELECT * FROM top3_news.image_prompt_plans WHERE prompt_request_key=$1
                   AND prompt_status IN ('reserved', 'completed')""", prompt_key,
            )
            if existing:
                return _saved_prompt(existing)
            reserved = await connection.fetchval(
                """SELECT image_prompt_id FROM top3_news.image_prompt_plans
                   WHERE generated_post_id=$1 AND prompt_status='reserved'""", generated_post_id,
            )
            if reserved is not None:
                raise RuntimeError(f"Незавершённая prompt reservation: image_prompt_id={reserved}")
            daily_id = await connection.fetchval(
                """SELECT daily_workflow_run_id FROM top3_news.daily_workflow_runs
                   WHERE batch_id=$1 ORDER BY daily_workflow_run_id DESC LIMIT 1""", batch_id,
            )
            image_prompt_version = (
                IMAGE_PROMPT_RECOVERY_VERSION if mode == "moderation_recovery" else IMAGE_PROMPT_NORMAL_VERSION
            )
            if request_kind == "regenerate" and source_prompt_id is None:
                source_prompt_id = await connection.fetchval(
                    """SELECT image_prompt_id FROM top3_news.image_generation_requests
                       WHERE batch_id=$1 AND image_status='completed' AND image_prompt_id IS NOT NULL
                       ORDER BY image_generation_id DESC LIMIT 1""", batch_id,
                )
            image_prompt_id = await connection.fetchval(
                """INSERT INTO top3_news.image_prompt_plans (
                       batch_id, generated_post_id, daily_workflow_run_id, review_action_id,
                       source_prompt_id, source_image_generation_id, prompt_request_key,
                       request_kind, attempt_kind, attempt_number, prompt_mode, agent_model,
                       agent_version, agent_prompt_version, image_prompt_version,
                       agent_instructions, input_payload)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17::jsonb)
                   RETURNING image_prompt_id""",
                batch_id, generated_post_id, daily_id, review_action_id, source_prompt_id,
                source_image_id, prompt_key, request_kind, attempt_kind, attempt_number,
                mode, agent.model_name, IMAGE_PROMPT_AGENT_VERSION,
                IMAGE_PROMPT_AGENT_PROMPT_VERSION, image_prompt_version, instructions, encode_json(payload),
            )

    try:
        result = await agent.generate(instructions=instructions, payload=payload)
        async with pool.acquire() as connection:
            updated = await connection.execute(
                """UPDATE top3_news.image_prompt_plans SET prompt_status='completed',
                       final_image_prompt=$2, structured_plan=$3::jsonb, response_metadata=$4::jsonb,
                       openai_usage=$5::jsonb, openai_cost=$6::jsonb, completed_at=now()
                   WHERE image_prompt_id=$1 AND prompt_status='reserved'""",
                image_prompt_id, result.plan.final_image_prompt, encode_json(result.plan.model_dump()),
                encode_json(result.response_metadata), encode_json(result.usage), encode_json(result.cost),
            )
            if updated != "UPDATE 1":
                raise RuntimeError("Image prompt reservation не была completed атомарно.")
    except Exception as error:
        telemetry = getattr(error, "image_prompt_telemetry", {})
        try:
            async with pool.acquire() as connection:
                await connection.execute(
                    """UPDATE top3_news.image_prompt_plans SET prompt_status='failed',
                           error_type=$2, error_message=$3, failed_at=now(),
                           response_metadata=$4::jsonb, openai_usage=$5::jsonb, openai_cost=$6::jsonb
                       WHERE image_prompt_id=$1 AND prompt_status='reserved'""",
                    image_prompt_id, type(error).__name__, str(error) or repr(error),
                    encode_json(telemetry["response_metadata"]) if telemetry else None,
                    encode_json(telemetry["usage"]) if telemetry else None,
                    encode_json(telemetry["cost"]) if telemetry else None,
                )
        except Exception as persistence_error:
            error.add_note(f"Не удалось сохранить failure prompt reservation: {persistence_error}")
        raise
    return SavedImagePrompt(image_prompt_id, image_prompt_version, result.plan)
