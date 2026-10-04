"""Один контролируемый платный smoke: agent + image, без Telegram и повторов.

Запускается на cloud-001 в изолированной тестовой БД с копией production TOP-3.
Ключ читается только на сервере из production .env и никогда не печатается.
"""

import asyncio
from decimal import Decimal
import json
from pathlib import Path

from app.config import Settings, get_settings
from app.db.generation_selection import load_generation_batch_selection
from app.db.pool import create_database_pool
from app.generation.openai_image_factory import create_openai_image_generation_runtime
from app.generation.openai_image_pipeline import run_reserved_openai_image_generation
from scripts import test_image_generation_reservation as fixture


async def main():
    test_settings = get_settings()
    if test_settings.app_env != "testing" or test_settings.db_port != 55432:
        raise RuntimeError("Smoke требует изолированную server test DB порта 55432.")
    production = Settings(_env_file="/opt/top3-news-bot/.env")
    settings = test_settings.model_copy(update={
        "openai_api_key": production.openai_api_key,
        "openai_image_prompt_agent_model": "gpt-6.1-sol",
        "openai_image_model": "gpt-image-2.5-flare",
        "openai_image_size": "1024x1024", "openai_image_quality": "medium",
        "openai_timeout_seconds": 300.0, "openai_max_retries": 0,
    })
    pool = await create_database_pool(settings)
    batches = set()
    runtime = None
    batch = post = ranking_id = None
    directory = Path.cwd().parent / "live-smoke"
    directory.mkdir(mode=0o700, exist_ok=True)
    try:
        async with pool.acquire() as connection:
            source = await connection.fetchrow(
                """SELECT b.batch_id, b.ranking_run_id, gp.post_text, gp.generation_metadata
                   FROM top3_news.publication_batches b
                   JOIN top3_news.generated_posts gp USING (batch_id)
                   JOIN top3_news.ranking_runs rr USING (ranking_run_id)
                   WHERE b.publication_date <= CURRENT_DATE AND rr.run_status='completed'
                   ORDER BY b.publication_date DESC, b.edition DESC, gp.version_number DESC LIMIT 1"""
            )
            if source is None:
                raise RuntimeError("Нет production TOP-3 в тестовой копии.")
        ranking_id = source["ranking_run_id"]
        selection = await load_generation_batch_selection(
            pool, ranking_run_id=ranking_id, batch_id=source["batch_id"]
        )
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT raw_title, raw_summary, article_text FROM top3_news.news_items WHERE news_id=ANY($1::bigint[])",
                list(selection.news_ids),
            )
        # Токенов не больше UTF-8 bytes. 100k input bytes + output cap 5k
        # дают верхнюю text-estimate $0.25, с большим запасом до лимита $1.
        byte_bound = len(json.dumps([dict(row) for row in rows], ensure_ascii=False).encode())
        byte_bound += len(source["post_text"].encode()) + 15000
        text_upper = Decimal(byte_bound) * Decimal("0.0000025") + Decimal("0.05")
        if text_upper > Decimal("0.25"):
            raise RuntimeError("Исходный payload слишком большой для бюджета smoke; API не вызывается.")
        print(f"Smoke source_batch={source['batch_id']} news_ids={selection.news_ids}", flush=True)
        print(f"Agent cost upper estimate={text_upper:.4f}; image reserve=0.30; budget=1.00; max_retries=0", flush=True)
        batch, post = await fixture.create_test_batch_and_post(
            pool, selection=selection, telegram_chat_id=settings.telegram_channel_id,
            existing_image=False, test_name="paid_image_prompt_agent_smoke", created_batch_ids=batches,
        )
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE top3_news.generated_posts SET post_text=$2, generation_metadata=$3::jsonb WHERE generated_post_id=$1",
                post, source["post_text"], source["generation_metadata"],
            )
        runtime = create_openai_image_generation_runtime(settings)
        try:
            result = await run_reserved_openai_image_generation(
                pool, generator=runtime.generator, selection=selection, batch_id=batch,
                generated_post_id=post, output_dir=directory / "images",
            )
            print(f"Live image status={result.image_status}; PNG={result.artifact.file_path}", flush=True)
        finally:
            async with pool.acquire() as connection:
                plans = await connection.fetch("SELECT * FROM top3_news.image_prompt_plans WHERE batch_id=$1", batch)
                images = await connection.fetch("SELECT * FROM top3_news.image_generation_requests WHERE batch_id=$1", batch)
            trace = {"source_batch_id": source["batch_id"], "test_batch_id": batch,
                     "news_ids": list(selection.news_ids), "prompts": [dict(row) for row in plans],
                     "images": [dict(row) for row in images]}
            (directory / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str))
            total = Decimal("0")
            for row in [*plans, *images]:
                if row["openai_cost"]:
                    total += Decimal(json.loads(row["openai_cost"])["total_cost_usd"])
            print(f"Reported model cost USD={total:.8f}; calls: agent={len(plans)}, image={len(images)}", flush=True)
            print("Trace exported; Telegram calls=0; retries=0", flush=True)
            if total > Decimal("1.00"):
                raise RuntimeError("Фактическая стоимость smoke превышает $1; дальнейшие вызовы запрещены.")
    finally:
        if runtime is not None:
            await runtime.sdk_client.close()
        try:
            if batches:
                await fixture.cleanup_test_batches(pool, created_batch_ids=batches, ranking_run_id=ranking_id)
        finally:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
