"""Изолированная БД: failed ranking -> тот же ID и frozen request -> completion."""
import asyncio
from dataclasses import replace
import json
from uuid import uuid4

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from app.db.daily_background_ranking_recovery import reopen_background_ranking_workflow
from app.ranking.background_response import BackgroundRankingError
from app.ranking.openai_event_evaluator import OpenAIEventRankingEvaluator
import app.ranking.openai_event_pipeline as pipeline
from scripts.test_openai_event_pipeline import FakeStructuredEventRankingClient


class CachedFixture(FakeStructuredEventRankingClient):
    def __init__(self):
        super().__init__()
        self.first_request = None
        self.resume_only = False

    def resume_saved_response_only(self):
        self.resume_only = True

    async def create_response(self,request):
        if self.first_request is None:
            self.first_request = request
            raise BackgroundRankingError('Synthetic interrupted GET; response ID saved')
        assert self.resume_only
        if not self.requests:
            assert request == self.first_request
        return await super().create_response(request)


async def main():
    settings = Settings()
    if settings.app_env != 'testing' or settings.db_port != 55432:
        raise RuntimeError('Requires isolated testing PostgreSQL on 55432')
    pool = await create_database_pool(settings)
    original_select = pipeline.select_news_candidates
    try:
        async with pool.acquire() as c:
            w = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=89')
        original_selection = await original_select(pool,as_of=w['as_of'],limit=500)
        selection = replace(original_selection,candidates=original_selection.candidates[:3])
        assert len(selection.candidates)==3
        async def fixed_selection(*args,**kwargs): return selection
        pipeline.select_news_candidates = fixed_selection
        client = CachedFixture()
        evaluator = OpenAIEventRankingEvaluator(client=client,model_name='test_background_resume_'+uuid4().hex)
        reserved_id = None
        async def observer(reservation):
            nonlocal reserved_id
            reserved_id = reservation.ranking_run_id
        try:
            await pipeline.run_reserved_openai_event_ranking(pool,evaluator=evaluator,as_of=w['as_of'],reservation_observer=observer)
        except BackgroundRankingError:
            pass
        else:
            raise AssertionError('Expected first read failure')
        async with pool.acquire() as c:
            row = await c.fetchrow('SELECT * FROM top3_news.ranking_runs WHERE ranking_run_id=$1',reserved_id)
            assert row['run_status']=='failed'
            saved_key = row['request_key']
            assert json.loads(row['parameters'])['background_snapshot']['request']['input_text']==client.first_request.input_text
            await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='failed',current_stage='failed',
                ranking_run_id=$1,batch_id=NULL,generated_post_id=NULL,image_generation_id=NULL,
                error_type='BackgroundRankingError',error_message='Synthetic GET',finished_at=now()
                WHERE daily_workflow_run_id=89""",reserved_id)
        assert await reopen_background_ranking_workflow(pool,daily_workflow_run_id=89)==reserved_id
        async def forbidden_selection(*args,**kwargs):
            raise AssertionError('Recovery reselected candidates instead of frozen snapshot')
        pipeline.select_news_candidates = forbidden_selection
        result = await pipeline.run_reserved_openai_event_ranking(pool,evaluator=evaluator,
            as_of=w['as_of'],resume_ranking_run_id=reserved_id)
        assert result.ranking_run_id==reserved_id and result.request_key.value==saved_key
        assert result.completion.run_status=='completed'
        async with pool.acquire() as c:
            params = json.loads(await c.fetchval('SELECT parameters FROM top3_news.ranking_runs WHERE ranking_run_id=$1',reserved_id))
            assert len(params['background_recoveries'])==1
            assert params['background_recoveries'][0]['failure']['error_type']=='BackgroundRankingError'
        assert await reopen_background_ranking_workflow(pool,daily_workflow_run_id=89) is None
        print('PASS: frozen candidates/request, same ranking ID/key, recovery history, completed ranking not replayed; paid calls=0')
    finally:
        pipeline.select_news_candidates = original_select
        await close_database_pool(pool)


if __name__ == '__main__':
    asyncio.run(main())
