"""Reopen только незавершённого background ranking; вызывается под daily lock."""
import json

from app.ranking.event_evaluator import EVENT_EVALUATOR_VERSION


async def reopen_background_ranking_workflow(pool, *, daily_workflow_run_id):
    async with pool.acquire() as c, c.transaction():
        w = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1 FOR UPDATE',daily_workflow_run_id)
        if (w['workflow_status'] not in ('failed','running') or w['ranking_run_id'] is None
                or any(w[k] is not None for k in ('batch_id','generated_post_id','image_generation_id'))):
            return None
        if w['workflow_status']=='failed' and w['error_type'] not in ('BackgroundRankingError','APIConnectionError','APITimeoutError'):
            return None
        r = await c.fetchrow('SELECT * FROM top3_news.ranking_runs WHERE ranking_run_id=$1 FOR UPDATE',w['ranking_run_id'])
        params = json.loads(r['parameters'])
        if (r['run_status'] not in ('failed','running') or not params.get('background_snapshot')
                or params.get('evaluator_version') != EVENT_EVALUATOR_VERSION):
            return None
        await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='running',
            current_stage='ranking',error_type=NULL,error_message=NULL,finished_at=NULL
            WHERE daily_workflow_run_id=$1""",daily_workflow_run_id)
        return int(r['ranking_run_id'])
