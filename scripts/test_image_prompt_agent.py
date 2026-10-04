"""Бесплатные unit и PostgreSQL integration проверки нового image pipeline."""

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import httpx
from openai import BadRequestError

from app.config import get_settings
from app.db.image_prompt_plans import prepare_saved_image_prompt
from app.db.image_generation_reservation import reserve_image_generation
from app.db.pool import create_database_pool
from app.generation.image_generator import (
    ImageGenerationNewsItem, OpenAIMovieNewsImageGenerator,
)
from app.generation.image_prompt_agent import ImagePromptAgent, ImagePromptPlan
from app.generation.moderation_diagnostics import extract_image_error_diagnostics
from app.generation.openai_image_pipeline import run_reserved_openai_image_generation
from app.generation.image_request_key import create_image_request_key
from app.generation.editorial_image_fallback import LocalEditorialImageGenerator
from scripts import test_event_ranking_run_completion as ranking_fixture
from scripts import test_image_generation_reservation as fixture
from scripts.test_openai_image_pipeline import SyntheticImageClient, build_png_bytes


class FakeResponses:
    def __init__(self):
        self.calls = []
        self.pause = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.invalid = False

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if self.pause:
            await self.release.wait()
        payload = json.loads(kwargs["input"])
        strategies = ["editorial_scene", "public_event", "film_set"]
        if payload["mode"] != "normal_creative":
            strategies = [f"title_card_{payload['attempt_number']}", "press_conference", "professional_portrait"]
        plan = {
            "mode": payload["mode"],
            "blocks": [{"position": news["position"], "news_id": news["news_id"],
                        "strategy": strategies[i], "visual_concept": f"Сюжет блока {i+1}",
                        "avoid": ["официальный логотип"]} for i, news in enumerate(payload["news"])],
            "overall_style": "Яркая насыщенная редакционная иллюстрация.",
            "final_image_prompt": f"Квадратная иллюстрация 1024×1024. Три блока: 1, 2, 3. Концепция {strategies[0]}.",
        }
        if self.invalid:
            plan["blocks"][0]["news_id"] = 999999999
        return SimpleNamespace(
            id="resp_synthetic", _request_id="req_agent_synthetic", status="completed",
            output_text=json.dumps(plan, ensure_ascii=False),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500, total_tokens=1500,
                                  input_tokens_details=SimpleNamespace(cached_tokens=100, cache_write_tokens=0),
                                  output_tokens_details=SimpleNamespace(reasoning_tokens=100)),
        )


def moderation_error(*, details=True):
    body = {"code": "moderation_blocked"}
    if details:
        body["moderation_details"] = {"moderation_stage": "output", "categories": ["synthetic_category"]}
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"),
                              headers={"x-request-id": "req_image_synthetic"})
    return BadRequestError("moderation_blocked", response=response, body=body)


class CheckingImageClient(SyntheticImageClient):
    def __init__(self, pool, post_id, *, block=False):
        super().__init__(image_bytes=build_png_bytes(width=64, height=64))
        self.pool, self.post_id, self.block = pool, post_id, block

    async def create_image(self, request):
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT p.prompt_status, p.final_image_prompt, p.completed_at,
                          r.image_status, r.reserved_at, r.request_payload
                   FROM top3_news.image_generation_requests r
                   JOIN top3_news.image_prompt_plans p USING (image_prompt_id)
                   WHERE r.generated_post_id=$1 AND r.image_status='reserved'""", self.post_id,
            )
        assert row is not None and row["prompt_status"] == "completed"
        assert row["completed_at"] <= row["reserved_at"]
        assert row["final_image_prompt"] == request.prompt
        assert json.loads(row["request_payload"])["model_request"]["prompt"] == request.prompt
        assert "recovery_examples" not in request.prompt and "approved_post_text" not in request.prompt
        assert request.model == "gpt-image-2.5-flare" and request.quality == "medium"
        if self.block:
            self.requests.append(request)
            raise moderation_error()
        return await super().create_image(request)


def unit_tests():
    items = tuple(ImageGenerationNewsItem(i, i, f"Новость {i}", "Подтверждённые факты.") for i in (1, 2, 3))
    agent = ImagePromptAgent(client=SimpleNamespace(responses=FakeResponses()))
    common = dict(items=items, post_text="Финальный TOP-3.", generated_items=[],
                  attempt_number=1, editorial_comment=None, issues=())
    normal, payload = agent.build_input(mode="normal_creative", previous=None, **common)
    assert "recovery_examples" not in payload and "previous_attempt" not in payload
    assert "FAILED_STRATEGY" not in normal and "PARTIALLY_BLOCKED" not in normal
    previous = {"final_image_prompt": "Отклонённая концепция", "moderation_diagnostics": {"moderation_stage": None}}
    recovery, recovery_payload = agent.build_input(mode="moderation_recovery", previous=previous, **common)
    assert recovery_payload["previous_attempt"] == previous
    assert "PARTIALLY_BLOCKED" in recovery_payload["recovery_examples"]
    assert "strategy" in recovery and "ДРУГОЙ" in recovery
    safe, _ = agent.build_input(mode="safe_editorial_fallback", previous=previous, **common)
    assert "SAFE_EDITORIAL_FALLBACK" in safe
    schema = ImagePromptPlan.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    block = schema["$defs"]["ImageBlockPlan"]
    assert block["additionalProperties"] is False
    assert set(block["required"]) == set(block["properties"])
    diagnostics = extract_image_error_diagnostics(moderation_error())
    assert diagnostics == {"error_code": "moderation_blocked", "request_id": "req_image_synthetic",
                           "moderation_stage": "output", "moderation_categories": ["synthetic_category"]}
    missing = extract_image_error_diagnostics(moderation_error(details=False))
    assert missing["moderation_stage"] is None and missing["moderation_categories"] is None
    assert extract_image_error_diagnostics(RuntimeError())["error_code"] is None
    card = LocalEditorialImageGenerator(post_text=fixture.SOURCE_POST_TEXT)
    request = card.build_request(items=items)
    assert "Тестовый заголовок первой новости" in request.prompt
    assert len(card.headlines) == 3
    print("Unit: prompt isolation, strict schema, optional moderation diagnostics, SAFE preparation: OK")


async def main():
    unit_tests()
    settings = get_settings()
    if settings.app_env != "testing":
        raise RuntimeError("Этот тест запускается только с APP_ENV=testing и изолированной БД.")
    pool = await create_database_pool(settings)
    runs, batches, news = set(), set(), ()
    ranking_id = None
    try:
        news = await ranking_fixture.create_test_news_items(pool)
        ranking_fixture.configure_test_news_ids(news)
        selection = await fixture.create_test_ranking_selection(pool, created_run_ids=runs)
        ranking_id = selection.ranking_run_id
        items = tuple(ImageGenerationNewsItem(i.position, i.news_id, i.title, i.summary) for i in selection.items)

        async def new_post(label):
            return await fixture.create_test_batch_and_post(
                pool, selection=selection, telegram_chat_id=settings.telegram_channel_id,
                existing_image=False, test_name=label, created_batch_ids=batches,
            )

        def generator_for(responses, image_client):
            return OpenAIMovieNewsImageGenerator(
                client=image_client, model_name="gpt-image-2.5-flare", size="64x64",
                prompt_agent=ImagePromptAgent(client=SimpleNamespace(responses=responses)),
            )

        with tempfile.TemporaryDirectory(prefix="top3-prompt-tests-") as directory:
            batch, post = await new_post("image_prompt_recovery")
            responses = FakeResponses()
            client = CheckingImageClient(pool, post, block=True)
            generator = generator_for(responses, client)
            kwargs = dict(generator=generator, selection=selection, batch_id=batch,
                          generated_post_id=post, output_dir=Path(directory))
            try:
                await run_reserved_openai_image_generation(pool, **kwargs)
                raise AssertionError("Первый synthetic moderation отказ не произошёл.")
            except BadRequestError:
                pass
            assert len(responses.calls) == 1
            first_input = json.loads(responses.calls[0]["input"])
            assert first_input["mode"] == "normal_creative" and "recovery_examples" not in first_input
            assert first_input["approved_post_text"] == fixture.SOURCE_POST_TEXT
            assert len(first_input["original_news_texts"]) == 3
            assert responses.calls[0]["store"] is False
            assert responses.calls[0]["reasoning"] == {"effort": "low"}
            assert responses.calls[0]["text"]["format"]["strict"] is True
            client.block = False
            result = await run_reserved_openai_image_generation(pool, **kwargs)
            assert result.completed
            assert len(responses.calls) == 2 and len(client.requests) == 2
            recovery_input = json.loads(responses.calls[1]["input"])
            assert recovery_input["mode"] == "moderation_recovery"
            assert recovery_input["previous_attempt"]["final_image_prompt"] == client.requests[0].prompt
            assert recovery_input["previous_attempt"]["moderation_diagnostics"]["moderation_stage"] == "output"
            async with pool.acquire() as connection:
                rows = await connection.fetch("SELECT * FROM top3_news.image_prompt_plans WHERE generated_post_id=$1 ORDER BY image_prompt_id", post)
                requests = await connection.fetch("SELECT * FROM top3_news.image_generation_requests WHERE generated_post_id=$1 ORDER BY image_generation_id", post)
            assert len(rows) == 2 and all(row["prompt_status"] == "completed" for row in rows)
            assert rows[1]["source_prompt_id"] == rows[0]["image_prompt_id"]
            assert rows[1]["source_image_generation_id"] == requests[0]["image_generation_id"]
            assert [row["attempt_number"] for row in rows] == [1, 2]
            assert rows[1]["attempt_kind"] == "moderation_recovery"
            assert json.loads(rows[0]["openai_usage"])["cached_input_tokens"] == 100
            assert json.loads(rows[0]["openai_cost"])["total_cost_usd"] == "0.00681000"
            repeated = await run_reserved_openai_image_generation(pool, **kwargs)
            assert repeated.duplicate_request_blocked and not repeated.model_called
            assert len(responses.calls) == 2 and len(client.requests) == 2
            print("Integration: save before Image API, recovery parent/history, telemetry, completed idempotency: OK")

            # Реальный конкурентный запуск блокируется persisted reserved строкой.
            batch, post = await new_post("image_prompt_concurrency")
            paused = FakeResponses()
            paused.pause = True
            agent = ImagePromptAgent(client=SimpleNamespace(responses=paused))
            prompt_kwargs = dict(agent=agent, batch_id=batch, generated_post_id=post,
                                 ranking_run_id=ranking_id, request_kind="initial", review_action_id=None,
                                 editorial_comment=None, issues=(), items=items)
            task = asyncio.create_task(prepare_saved_image_prompt(pool, **prompt_kwargs))
            await paused.started.wait()
            try:
                await prepare_saved_image_prompt(pool, **prompt_kwargs)
                raise AssertionError("Конкурентный prompt agent не был заблокирован.")
            except RuntimeError as error:
                assert "повторный платный" in str(error)
            paused.release.set()
            first = await task
            again = await prepare_saved_image_prompt(pool, **prompt_kwargs)
            assert first.image_prompt_id == again.image_prompt_id and len(paused.calls) == 1
            print("Integration: concurrent reservation and reuse before image reservation: OK")

            guard_generator = generator_for(paused, CheckingImageClient(pool, post))
            wrong_request = guard_generator.build_request(items=items, final_image_prompt="Подменённый prompt")
            key = create_image_request_key(
                batch_id=batch, ranking_run_id=ranking_id, request_kind="initial", review_action_id=None,
                metadata=guard_generator.metadata, model_request=wrong_request, items=items,
            )
            try:
                await reserve_image_generation(
                    pool, request_key=key, batch_id=batch, generated_post_id=post, ranking_run_id=ranking_id,
                    request_kind="initial", review_action_id=None, editorial_comment=None, issues=(),
                    metadata=guard_generator.metadata, model_request=wrong_request, items=items,
                    image_prompt_id=first.image_prompt_id,
                )
                raise AssertionError("Image reservation приняла подменённый exact prompt.")
            except ValueError:
                pass
            async with pool.acquire() as connection:
                assert await connection.fetchval("SELECT count(*) FROM top3_news.image_generation_requests WHERE generated_post_id=$1", post) == 0
                assert await connection.fetchval("SELECT prompt_status FROM top3_news.image_prompt_plans WHERE image_prompt_id=$1", first.image_prompt_id) == "completed"
            print("Integration: reservation rollback preserves prompt, mismatched exact prompt blocked: OK")

            # Invalid output не вызывает image; фактическая agent usage сохраняется.
            batch, post = await new_post("image_prompt_invalid")
            invalid = FakeResponses()
            invalid.invalid = True
            client = CheckingImageClient(pool, post)
            try:
                await run_reserved_openai_image_generation(
                    pool, generator=generator_for(invalid, client), selection=selection,
                    batch_id=batch, generated_post_id=post, output_dir=Path(directory),
                )
                raise AssertionError("Некорректный news_id был принят.")
            except ValueError:
                pass
            assert not client.requests
            async with pool.acquire() as connection:
                row = await connection.fetchrow("SELECT * FROM top3_news.image_prompt_plans WHERE generated_post_id=$1", post)
                assert row["prompt_status"] == "failed" and row["openai_cost"] is not None
                assert await connection.fetchval("SELECT count(*) FROM top3_news.image_generation_requests WHERE generated_post_id=$1", post) == 0
            print("Integration: invalid agent output, persisted failure/cost, no Image API: OK")

            # Editorial regenerate создаёт новый план с отдельным kind и parent.
            batch, post = await new_post("image_prompt_editorial")
            responses = FakeResponses()
            client = CheckingImageClient(pool, post)
            generator = generator_for(responses, client)
            kwargs = dict(generator=generator, selection=selection, batch_id=batch,
                          generated_post_id=post, output_dir=Path(directory))
            await run_reserved_openai_image_generation(pool, **kwargs)
            reviewer = await fixture.load_test_reviewer(pool)
            review = await fixture.create_regenerate_review_action(pool, generated_post_id=post,
                                                                    reviewer_telegram_user_id=reviewer,
                                                                    test_name="prompt_editorial")
            await run_reserved_openai_image_generation(
                pool, **kwargs, request_kind="regenerate", review_action_id=review,
                editorial_comment=fixture.EDITORIAL_COMMENT, issues=fixture.IMAGE_ISSUES,
            )
            async with pool.acquire() as connection:
                rows = await connection.fetch("SELECT * FROM top3_news.image_prompt_plans WHERE generated_post_id=$1 ORDER BY image_prompt_id", post)
            assert rows[1]["attempt_kind"] == "editorial_revision"
            assert rows[1]["source_prompt_id"] == rows[0]["image_prompt_id"]
            print("Integration: editorial revision distinct from moderation recovery: OK")
    finally:
        try:
            if batches:
                await fixture.cleanup_test_batches(pool, created_batch_ids=batches, ranking_run_id=ranking_id)
            await ranking_fixture.cleanup_test_runs(pool, created_run_ids=runs)
            await ranking_fixture.cleanup_test_news_items(pool, news_ids=news)
        finally:
            await pool.close()
    print("Image Prompt Agent tests: OK; real OpenAI requests: 0; Telegram calls: 0")


if __name__ == "__main__":
    asyncio.run(main())
