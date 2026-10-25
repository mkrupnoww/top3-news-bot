"""Интеграция на read-only snapshot production в отдельной БД cloud-001.
Ни Telegram, ни OpenAI не вызываются; настоящие text/image reservations сохраняются.
"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from app.config import get_settings
from app.db.pool import create_database_pool, close_database_pool
from app.db.published_news_history import load_published_news_history
from app.db.review_queue import record_human_review_decision
from app.db.review_delivery import get_review_draft_by_id
from app.generation.openai_generator import OpenAITelegramPostGenerator, GenerationModelResponse, OpenAIGeneratedNewsPayload, build_top3_post_text
from app.generation.openai_pipeline import run_reserved_openai_generation
from app.generation.openai_image_pipeline import run_reserved_openai_image_generation
import app.generation.editorial_replacement as replacement
from scripts.test_openai_generation_pipeline import build_telemetry, FakeOfficialTrailerEnricher
from scripts.test_openai_image_pipeline import SyntheticImageClient, build_generator, build_png_bytes, EXPECTED_COST_PAYLOAD


class FakeTextClient:
    def __init__(self): self.calls = 0
    async def create_response(self, request):
        self.calls += 1
        await asyncio.sleep(0)
        news = json.loads(request.input_text)['news']
        items = []
        for n in news:
            lead = 'Новость о кино ' + str(n['news_id'])
            items.append(OpenAIGeneratedNewsPayload(position=n['position'],news_id=n['news_id'],headline=lead,
                body=lead+': редакция опубликовала подтвержденные сведения о событии. Материал содержит подробности проекта и описывает его значение для индустрии.'))
        telemetry = build_telemetry(model_name=request.model)
        return GenerationModelResponse(output_text=json.dumps(dict(post_text=build_top3_post_text(items),items=[i.model_dump() for i in items]),ensure_ascii=False),usage=telemetry.usage,cost_estimate=telemetry.cost_estimate)


class FakeSDK:
    async def close(self): pass


async def main():
    settings = get_settings()
    if settings.app_env!='testing' or settings.db_port!=55432 or settings.db_name!='top3_history_test':
        raise RuntimeError('Только изолированная top3_history_test на cloud-001:55432')
    pool = await create_database_pool(settings)
    reviewer = await pool.fetchval("SELECT telegram_user_id FROM bot_users WHERE user_role='admin' AND is_active ORDER BY telegram_user_id LIMIT 1")
    assert reviewer
    with TemporaryDirectory() as tmp:
        async def reset():
            await pool.execute('TRUNCATE editorial_replacement_requests')
            await pool.execute("DELETE FROM publication_batches WHERE metadata ? 'source_review_action_id'")
            await pool.execute('DELETE FROM review_actions WHERE generated_post_id IN (168,169)')
            await pool.execute('DELETE FROM publication_attempts WHERE generated_post_id IN (168,169)')
            await pool.execute("UPDATE generated_posts SET post_status=CASE WHEN generated_post_id=168 THEN 'awaiting_review' ELSE 'superseded' END WHERE generated_post_id IN (168,169)")
            await pool.execute("UPDATE publication_batches SET batch_status='awaiting_review',published_at=NULL,approved_at=NULL,approved_by_telegram_user_id=NULL,error_message=NULL WHERE batch_id=165")
            action = await record_human_review_decision(pool,generated_post_id=168,reviewer_telegram_user_id=reviewer,decision='changes_required',requested_action='replace_news',comment_text='Замени третью новость. Эта новость уже была вчера.',issues=('Замени третью новость.',))
            return action.review_action_id

        async def text_runner(*args, **kwargs):
            return await run_reserved_openai_generation(*args, **kwargs,trailer_enricher=FakeOfficialTrailerEnricher())
        image_client = SyntheticImageClient(image_bytes=build_png_bytes())
        def runtime_factory(settings):
            return SimpleNamespace(generator=build_generator(image_client),sdk_client=FakeSDK())
        async def image_runner(*args, **kwargs):
            assert await get_review_draft_by_id(pool,generated_post_id=kwargs['generated_post_id']) is None
            try:
                await record_human_review_decision(pool,generated_post_id=kwargs['generated_post_id'],reviewer_telegram_user_id=reviewer,decision='approve')
            except ValueError:
                pass
            else:
                raise AssertionError('Недоделанную замену можно одобрить')
            return await run_reserved_openai_image_generation(*args, **kwargs,output_dir=tmp,cost_estimator=lambda **kwargs:EXPECTED_COST_PAYLOAD)
        client = FakeTextClient()
        generator = OpenAITelegramPostGenerator(client=client,model_name='test-editorial-no-network')
        async def run(action, **kwargs):
            return await replacement.run_editorial_replacement(pool,generator=generator,settings=settings,review_action_id=action,positions=(3,),text_runner=text_runner,image_runtime_factory=runtime_factory,image_runner=kwargs.pop('image_runner',image_runner),**kwargs)

        history = await load_published_news_history(pool,as_of=__import__('datetime').datetime(2026,10,25,7,30,tzinfo=__import__('datetime').UTC))
        yesterday = next(i for i in history if i['batch_id']==164)
        assert {x['news_id'] for x in yesterday['items']}=={8236,8257,8201}
        assert 'Тарантино' in yesterday['published_text']
        assert all(i['publication_date']<'2026-10-25' for i in history)
        print('Published history reads actual delivered issue: OK')

        action = await reset()
        before = client.calls
        r = await run(action)
        assert {8269,8285}.issubset(r.news_ids) and 8262 not in r.news_ids
        assert await pool.fetchval('SELECT batch_status FROM publication_batches WHERE batch_id=165')=='superseded'
        actual_ids = await pool.fetch('SELECT news_id FROM batch_items WHERE batch_id=$1 ORDER BY position',r.batch_id)
        assert tuple(x['news_id'] for x in actual_ids)==r.news_ids
        assert await get_review_draft_by_id(pool,generated_post_id=168) is None
        new = await get_review_draft_by_id(pool,generated_post_id=r.generated_post_id)
        assert new and new.image_path and Path(new.image_path).exists()
        assert sha256(Path(new.image_path).read_bytes()).hexdigest()==new.image_sha256
        assert await pool.fetchval("SELECT count(*) FROM review_actions WHERE generated_post_id=$1 AND reviewer_type='human'",r.generated_post_id)==1
        text_count, image_count = client.calls, len(image_client.requests)
        assert text_count>before and image_count==1
        repeated = await run(action)
        assert repeated==r and client.calls==text_count and len(image_client.requests)==image_count
        print('Actual batch IDs, new image, manual review, inherited human action, repeated request: OK')

        action = await reset()
        async def failing_image(*args, **kwargs):
            raise RuntimeError('Synthetic image failure before request')
        try:
            await run(action,image_runner=failing_image)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Image failure not propagated')
        plan = await pool.fetchrow('SELECT * FROM editorial_replacement_requests WHERE review_action_id=$1',action)
        assert plan['replacement_status']=='text_ready'
        assert await get_review_draft_by_id(pool,generated_post_id=plan['generated_post_id']) is None
        assert await pool.fetchval('SELECT post_status FROM generated_posts WHERE generated_post_id=168')=='awaiting_review'
        count = client.calls
        r = await run(action)
        assert r.completed and client.calls==count
        print('Image failure keeps old content intact and new draft hidden; resume does not regenerate text: OK')

        action = await reset()
        original = replacement._activate
        async def broken_activate(*args, **kwargs):
            raise RuntimeError('Synthetic process interruption after image completion')
        with patch.object(replacement,'_activate',broken_activate):
            try: await run(action)
            except RuntimeError: pass
            else: raise AssertionError('Interruption not simulated')
        counts = client.calls,len(image_client.requests)
        r = await run(action)
        assert r.completed and counts==(client.calls,len(image_client.requests))
        print('Resume after image completion: no duplicate text/image calls: OK')

        action = await reset()
        results = await asyncio.gather(run(action),run(action),return_exceptions=True)
        assert any(isinstance(x,replacement.EditorialReplacementResult) for x in results)
        assert any(isinstance(x,ValueError) for x in results)
        print('Concurrent replacement is serialized: OK')

        # Чисто текстовое замечание по-прежнему использует старый revision pipeline.
        await reset()
        await pool.execute('DELETE FROM review_actions WHERE generated_post_id=168')
        old_action = await record_human_review_decision(pool,generated_post_id=168,reviewer_telegram_user_id=reviewer,decision='changes_required',requested_action='regenerate_text',comment_text='Исправь дату',issues=('Исправь дату',))
        from app.db.generation_revision_selection import load_generation_revision_selection
        old = await load_generation_revision_selection(pool,review_action_id=old_action.review_action_id)
        assert old.news_ids==(8269,8285,8262)
        print('Ordinary text revisions preserve original selection: OK')

        # Handler направляет конкретное замечание в replacement, не в перефразирование.
        await reset()
        await pool.execute('DELETE FROM review_actions WHERE generated_post_id=168')
        import app.bot.handlers as handlers
        class State:
            async def get_data(self): return {'generated_post_id':168}
            async def clear(self): pass
        class Message:
            from_user=SimpleNamespace(id=reviewer)
            text='Замени третью новость. Эта новость уже была вчера.'
            async def answer(self,*args,**kwargs): pass
        called=[]
        async def route(*args,**kwargs):
            ra=await pool.fetchrow('SELECT * FROM review_actions WHERE review_action_id=$1',kwargs['review_action_id'])
            assert ra['requested_action']=='replace_news' and kwargs['positions']==(3,)
            called.append(True)
            return SimpleNamespace(batch_id=1,generated_post_id=2,news_ids=(1,2,3))
        with patch.object(handlers,'create_openai_generation_runtime',lambda s:SimpleNamespace(generator=generator,sdk_client=FakeSDK())),patch.object(handlers,'run_editorial_replacement',route):
            await handlers.handle_revision_comment(Message(),pool,State())
        assert called==[True]
        print('Real Telegram handler routes replacement intent correctly: OK')
    await close_database_pool(pool)
    print('PASS: editorial replacement integration; real Telegram/OpenAI calls=0')


if __name__=='__main__':
    asyncio.run(main())
