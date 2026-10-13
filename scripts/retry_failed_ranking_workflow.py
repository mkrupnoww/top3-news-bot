"""Разрешённое восстановление network-failed ranking с лимитом до $1."""
import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from app.ranking.event_evaluator import EVENT_EVALUATOR_VERSION
from scripts.retry_failed_text_workflow import RetryBudget, BudgetClient
import app.workflows.daily_production as daily


async def reopen(pool, workflow_id, directory, *, check_only=False):
    workflow = await daily.load_daily_workflow(pool, daily_workflow_run_id=workflow_id)
    async with daily._daily_workflow_lock(pool, publication_date=workflow.publication_date):
        async with pool.acquire() as c, c.transaction():
            row = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1 FOR UPDATE', workflow_id)
            if not (row['workflow_status'] == 'failed' and row['error_type'] == 'APIConnectionError'
                    and row['ranking_run_id'] is not None and row['batch_id'] is None
                    and row['generated_post_id'] is None and row['image_generation_id'] is None):
                raise ValueError('Expected ranking connection failure without text/image required')
            ranking = await c.fetchrow('SELECT * FROM top3_news.ranking_runs WHERE ranking_run_id=$1 FOR UPDATE', row['ranking_run_id'])
            params = json.loads(ranking['parameters'])
            if not (ranking['run_status'] == 'failed' and params['evaluator_version'] != EVENT_EVALUATOR_VERSION
                    and params.get('failure', {}).get('error_type') == 'APIConnectionError'):
                raise ValueError('Expected old failed ranking; never silently repeat an uncertain new request')
            if check_only:
                print('Preflight passed: network-failed ranking, no downstream stages, new request identity')
                return row
            (directory/'state-before.json').write_text(json.dumps({'workflow':dict(row),'ranking':dict(ranking)},default=str,indent=2))
            await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='running',
                current_stage='ranking',ranking_run_id=NULL,error_type=NULL,error_message=NULL,
                finished_at=NULL,updated_at=now() WHERE daily_workflow_run_id=$1""", workflow_id)
            return row


async def run(workflow_id, directory, maximum, *, check_only=False):
    settings = Settings()
    pool = await create_database_pool(settings)
    try:
        if check_only:
            await reopen(pool, workflow_id, directory, check_only=True)
            return
        budget = RetryBudget(directory, maximum)
        row = await reopen(pool, workflow_id, directory)
        original_generation = daily.create_openai_generation_runtime
        original_image = daily.create_openai_image_generation_runtime
        original_ranking = daily.create_openai_event_ranking_runtime
        daily.create_openai_generation_runtime = lambda s: original_generation(s, client_factory=lambda **kw: BudgetClient(budget, **kw))
        daily.create_openai_image_generation_runtime = lambda s: original_image(s, client_factory=lambda **kw: BudgetClient(budget, **kw))
        daily.create_openai_event_ranking_runtime = lambda s: original_ranking(s, client_factory=lambda **kw: BudgetClient(budget, ranking=True, **kw))
        result = await daily.run_daily_production_workflow(pool, settings=settings,
            publication_date=row['publication_date'], as_of=row['as_of'],
            progress=lambda message: print(message, flush=True))
        (directory/'result.json').write_text(json.dumps(asdict(result),default=str,indent=2))
        print(json.dumps(asdict(result),default=str),flush=True)
        print('Retry estimated/accounted USD:', budget.total, flush=True)
    finally:
        await close_database_pool(pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow-id', type=int, required=True)
    parser.add_argument('--ledger-directory', type=Path, required=True)
    parser.add_argument('--max-usd', type=Decimal, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.workflow_id,args.ledger_directory,args.max_usd,check_only=args.check_only))
