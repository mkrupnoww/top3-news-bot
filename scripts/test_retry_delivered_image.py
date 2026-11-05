"""Image-only retry: реальные reservations/PNG и fake Image API/Telegram, только cloud-001 clone."""
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import httpx
from openai import BadRequestError

from app.config import get_settings
from app.db.pool import create_database_pool,close_database_pool
from app.generation.image_generator import OpenAIMovieNewsImageGenerator
from app.generation.image_openai_usage import build_openai_image_cost_payload
from app.publication.telegram_text import prepare_telegram_text
from scripts.retry_delivered_image import run
from scripts.test_openai_image_pipeline import SyntheticImageClient,build_png_bytes


class FakeSDK:
    async def close(self): pass


class FakeBot:
    calls=[]
    def __init__(self,**kwargs): self.session=FakeSDK()
    async def edit_message_media(self,**kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message_id=kwargs['message_id'])


class FakeClient:
    def __init__(self,budget,*,reserve=Decimal('0.49')):
        self.budget=budget;self.reserve=reserve;self.calls=0
        self.delegate=SyntheticImageClient(image_bytes=build_png_bytes(width=64,height=64))
    async def create_image(self,request):
        assert 'retry_after_output_moderation_blocked' in request.prompt
        call=self.budget.reserve('images',self.reserve)
        self.calls+=1
        if self.calls==1:
            call.update(status='uncertain');self.budget.save()
            raise BadRequestError('Synthetic moderation_blocked',response=httpx.Response(400,request=httpx.Request('POST','https://example.invalid/images')),body={'code':'moderation_blocked'})
        response=await self.delegate.create_image(request)
        cost=build_openai_image_cost_payload(request.model,response.usage)
        self.budget.complete(call,Decimal(cost['total_cost_usd']),SimpleNamespace(),asdict(response.usage))
        return response


async def main():
    settings=get_settings()
    if settings.app_env!='testing' or settings.db_port!=55432 or settings.db_name!='top3_image_retry_test':
        raise RuntimeError('Только отдельная БД cloud-001:55432/top3_image_retry_test')
    pool=await create_database_pool(settings)
    old=dict(await pool.fetchrow('SELECT * FROM generated_posts WHERE generated_post_id=180'))
    async def reset():
        await pool.execute('UPDATE daily_workflow_runs SET image_generation_id=182 WHERE daily_workflow_run_id=159')
        await pool.execute("DELETE FROM image_generation_requests WHERE generated_post_id=180 AND request_kind='regenerate'")
        await pool.execute("DELETE FROM review_actions WHERE generated_post_id=180 AND requested_action='regenerate_image'")
        await pool.execute("UPDATE generated_posts SET post_status='awaiting_review',image_path=$1,image_sha256=$2,image_model_name=$3,image_prompt_version=$4 WHERE generated_post_id=180",old['image_path'],old['image_sha256'],old['image_model_name'],old['image_prompt_version'])
        await pool.execute("UPDATE publication_batches SET batch_status='awaiting_review',published_at=NULL,metadata=metadata-'editorial_replacement_pending' WHERE batch_id=176")
        FakeBot.calls=[]
    with TemporaryDirectory() as tmp:
        clients=[]
        def factory(s,budget):
            client=FakeClient(budget);clients.append(client)
            return SimpleNamespace(generator=OpenAIMovieNewsImageGenerator(client=client,model_name='gpt-image-2.5-flare',size='64x64'),sdk_client=FakeSDK())
        from app.generation.openai_image_pipeline import run_reserved_openai_image_generation
        async def runner(*args,**kwargs):
            return await run_reserved_openai_image_generation(*args,**kwargs,output_dir=Path(tmp)/'images')
        result=await run(pool,settings=settings,workflow_id=159,post_id=180,directory=Path(tmp)/'success',maximum=Decimal('1'),runtime_factory=factory,bot_factory=FakeBot,image_runner=runner)
        assert result['preview_updated'] and clients[0].calls==2 and len(FakeBot.calls)==1
        assert FakeBot.calls[0]['message_id']==544
        assert FakeBot.calls[0]['media'].caption==prepare_telegram_text(old['post_text'],text_format=old['text_format']).text
        assert [b.text for b in FakeBot.calls[0]['reply_markup'].inline_keyboard[0]]==['✅ В тест','✏️ Изменить','❌ Отклонить']
        fresh=await pool.fetchrow('SELECT * FROM generated_posts WHERE generated_post_id=180')
        assert fresh['post_text']==old['post_text'] and fresh['image_sha256']!=old['image_sha256']
        assert await pool.fetchval('SELECT image_generation_id FROM daily_workflow_runs WHERE daily_workflow_run_id=159')==result['image_generation_id']
        assert Decimal(result['accounted_usd'])<=1
        print('Real safe-prompt retry, new PNG, same text/message/buttons, workflow image pointer: OK')
        await reset()
        stopped=[]
        def expensive_factory(s,budget):
            client=FakeClient(budget,reserve=Decimal('0.60'));stopped.append(client)
            return SimpleNamespace(generator=OpenAIMovieNewsImageGenerator(client=client,model_name='gpt-image-2.5-flare',size='64x64'),sdk_client=FakeSDK())
        try:
            await run(pool,settings=settings,workflow_id=159,post_id=180,directory=Path(tmp)/'budget-stop',maximum=Decimal('1'),runtime_factory=expensive_factory,bot_factory=FakeBot,image_runner=runner)
        except RuntimeError as e:
            assert 'Retry budget would be exceeded' in str(e)
        else: raise AssertionError('Budget exceeded')
        assert stopped[0].calls==1 and not FakeBot.calls
        assert await pool.fetchval('SELECT image_sha256 FROM generated_posts WHERE generated_post_id=180')==old['image_sha256']
        assert await pool.fetchval("SELECT metadata->>'editorial_replacement_pending' FROM publication_batches WHERE batch_id=176")=='false'
        print('Budget stops before next paid call; old image remains reviewable: OK')
        await reset()
        await pool.execute("UPDATE generated_posts SET post_status='published' WHERE generated_post_id=180")
        try:
            await run(pool,settings=settings,workflow_id=159,post_id=180,directory=Path(tmp)/'published',maximum=Decimal('1'),runtime_factory=factory,bot_factory=FakeBot)
        except ValueError: pass
        else: raise AssertionError('Published post accepted')
        assert not (Path(tmp)/'published').exists()
        print('Already published post rejected before budget/API: OK')
    await close_database_pool(pool)
    print('PASS: delivered-image retry; real OpenAI/Telegram calls=0')


if __name__=='__main__': asyncio.run(main())
