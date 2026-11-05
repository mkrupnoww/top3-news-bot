"""Продолжить существующий ranking response и выпуск; дополнительные вызовы ограничены $1."""
import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from app.ranking.background_snapshot import decode_snapshot
from app.ranking.request_identity import same_event_ranking_request
from scripts.retry_failed_text_workflow import RetryBudget, BudgetClient
import app.workflows.daily_production as daily


async def run(workflow_id: int, directory: Path, maximum: Decimal):
    settings=Settings()
    budget=RetryBudget(directory,maximum)
    pool=await create_database_pool(settings)
    originals=(daily.create_openai_event_ranking_runtime,daily.create_openai_generation_runtime,daily.create_openai_image_generation_runtime)
    try:
        row=await pool.fetchrow('SELECT * FROM daily_workflow_runs WHERE daily_workflow_run_id=$1',workflow_id)
        if row is None or row['batch_id'] is not None or row['generated_post_id'] is not None or row['ranking_run_id'] is None:
            raise ValueError('Ожидается существующий незавершённый ranking без поста')
        rank=await pool.fetchrow('SELECT * FROM ranking_runs WHERE ranking_run_id=$1',row['ranking_run_id'])
        params=json.loads(rank['parameters'])
        selection,saved,_=decode_snapshot(params['background_snapshot'])
        (directory/'state-before.json').write_text(json.dumps({'workflow':dict(row),'ranking':dict(rank)},default=str,ensure_ascii=False,indent=2))
        def ranking_factory(s):
            runtime=originals[0](s,client_factory=lambda **kw:BudgetClient(budget,ranking=True,**kw))
            if not same_event_ranking_request(saved,runtime.evaluator.build_request(selection)):
                raise ValueError('Сохранённые факты/инструкции действительно изменились')
            # Дополнительный независимый запрет нового primary POST.
            runtime.evaluator.resume_saved_response_only()
            return runtime
        daily.create_openai_event_ranking_runtime=ranking_factory
        daily.create_openai_generation_runtime=lambda s:originals[1](s,client_factory=lambda **kw:BudgetClient(budget,**kw))
        daily.create_openai_image_generation_runtime=lambda s:originals[2](s,client_factory=lambda **kw:BudgetClient(budget,**kw))
        result=await daily.run_daily_production_workflow(pool,settings=settings,publication_date=row['publication_date'],as_of=row['as_of'],progress=lambda message:print(message,flush=True))
        (directory/'result.json').write_text(json.dumps(asdict(result),default=str,ensure_ascii=False,indent=2))
        print(json.dumps(asdict(result),default=str,ensure_ascii=False),flush=True)
        print('Additional recovery cost/reservations USD:',budget.total,flush=True)
    finally:
        daily.create_openai_event_ranking_runtime,daily.create_openai_generation_runtime,daily.create_openai_image_generation_runtime=originals
        await close_database_pool(pool)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workflow-id',type=int,required=True)
    parser.add_argument('--ledger-directory',type=Path,required=True)
    parser.add_argument('--max-usd',type=Decimal,required=True)
    args=parser.parse_args()
    asyncio.run(run(args.workflow_id,args.ledger_directory,args.max_usd))
