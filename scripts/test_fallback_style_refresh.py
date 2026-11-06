"""Обновление старого fallback без изменения текста/отбора: реальные reservations и fake API."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from decimal import Decimal

from app.config import get_settings
from app.db.pool import create_database_pool,close_database_pool
from app.generation.image_generator import OpenAIMovieNewsImageGenerator,OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
from app.generation.openai_image_pipeline import run_reserved_openai_image_generation
from scripts.retry_delivered_image import run
from scripts.test_retry_delivered_image import FakeClient,FakeSDK,FakeBot


async def main():
    settings=get_settings()
    if settings.app_env!='testing' or settings.db_port!=55432 or settings.db_name!='top3_style_test':
        raise RuntimeError('Только изолированная cloud-001:55432/top3_style_test')
    pool=await create_database_pool(settings)
    before=await pool.fetchrow('SELECT * FROM generated_posts WHERE generated_post_id=181')
    assert before['image_prompt_version']=='movie_news_image_moderation_fallback_v9'
    ids=await pool.fetch('SELECT news_id FROM batch_items WHERE batch_id=177 ORDER BY position')
    with TemporaryDirectory() as tmp:
        async def runner(*args,**kwargs):
            return await run_reserved_openai_image_generation(*args,**kwargs,output_dir=Path(tmp)/'images')
        clients=[]
        def factory(settings,budget):
            client=FakeClient(budget);client.calls=1  # Для style smoke достаточно одного успешного fake image response.
            clients.append(client)
            return SimpleNamespace(generator=OpenAIMovieNewsImageGenerator(client=client,model_name=settings.openai_image_model,size='64x64'),sdk_client=FakeSDK())
        FakeBot.calls=[]
        result=await run(pool,settings=settings,workflow_id=162,post_id=181,directory=Path(tmp)/'refresh',maximum=Decimal('1'),refresh_fallback_style=True,runtime_factory=factory,bot_factory=FakeBot,image_runner=runner)
        after=await pool.fetchrow('SELECT * FROM generated_posts WHERE generated_post_id=181')
        assert after['post_text']==before['post_text']
        assert after['image_prompt_version']==OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
        assert after['image_sha256']!=before['image_sha256']
        assert ids==await pool.fetch('SELECT news_id FROM batch_items WHERE batch_id=177 ORDER BY position')
        assert len(FakeBot.calls)==1 and FakeBot.calls[0]['message_id']==545
        assert await pool.fetchval("SELECT review_details->>'operator_image_update_complete' FROM review_actions WHERE review_action_id=$1",result['review_action_id'])=='true'
        try:
            await run(pool,settings=settings,workflow_id=162,post_id=181,directory=Path(tmp)/'same-version',maximum=Decimal('1'),refresh_fallback_style=True,runtime_factory=factory,bot_factory=FakeBot,image_runner=runner)
        except ValueError: pass
        else:raise AssertionError('Текущая версия ошибочно признана устаревшей')
        assert not (Path(tmp)/'same-version').exists()
    await close_database_pool(pool)
    print('PASS: old fallback refresh, preserved text/selection/preview, completed operator marker, same-version no-op guard; API calls=0')


if __name__=='__main__':asyncio.run(main())
