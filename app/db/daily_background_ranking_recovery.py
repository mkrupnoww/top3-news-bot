"""Reopen только незавершённого background ranking; вызывается под daily lock."""
import json

from app.ranking.event_evaluator import EVENT_EVALUATOR_VERSION
from app.ranking.request_identity import PREPARED_REQUEST_MISMATCH, same_event_ranking_request


def is_jsonb_order_failure(workflow, snapshot) -> bool:
    """Разрешает только доказанный старый сбой сравнения строк после JSONB round trip."""
    if workflow['error_type'] != 'ValueError' or workflow['error_message'] != PREPARED_REQUEST_MISMATCH:
        return False
    from app.ranking.background_snapshot import decode_snapshot
    from app.ranking.openai_event_evaluator import OpenAIEventRankingEvaluator

    class NoNetworkClient:
        async def create_response(self, request):
            raise RuntimeError('Проверка snapshot не должна вызывать API')
    try:
        selection, saved, _ = decode_snapshot(snapshot)
        rebuilt = OpenAIEventRankingEvaluator(client=NoNetworkClient(), model_name=saved.model).build_request(selection)
        return saved != rebuilt and same_event_ranking_request(saved, rebuilt)
    except (KeyError, TypeError, ValueError):
        return False


async def reopen_background_ranking_workflow(pool, *, daily_workflow_run_id):
    async with pool.acquire() as c, c.transaction():
        w = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1 FOR UPDATE',daily_workflow_run_id)
        if (w['workflow_status'] not in ('failed','running') or w['ranking_run_id'] is None
                or any(w[k] is not None for k in ('batch_id','generated_post_id','image_generation_id'))):
            return None
        possible_jsonb_failure = w['error_type']=='ValueError' and w['error_message']==PREPARED_REQUEST_MISMATCH
        if w['workflow_status']=='failed' and w['error_type'] not in ('BackgroundRankingError','APIConnectionError','APITimeoutError') and not possible_jsonb_failure:
            return None
        r = await c.fetchrow('SELECT * FROM top3_news.ranking_runs WHERE ranking_run_id=$1 FOR UPDATE',w['ranking_run_id'])
        params = json.loads(r['parameters'])
        if (r['run_status'] not in ('failed','running') or not params.get('background_snapshot')
                or params.get('evaluator_version') != EVENT_EVALUATOR_VERSION):
            return None
        if possible_jsonb_failure and not is_jsonb_order_failure(w, params['background_snapshot']):
            return None
        await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='running',
            current_stage='ranking',error_type=NULL,error_message=NULL,finished_at=NULL
            WHERE daily_workflow_run_id=$1""",daily_workflow_run_id)
        return int(r['ranking_run_id'])
