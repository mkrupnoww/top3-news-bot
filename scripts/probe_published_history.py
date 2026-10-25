"""Один платный диагностический запрос; только чтение production, без публикаций."""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, UTC, timedelta
from decimal import Decimal
import json
from pathlib import Path

from app.config import get_settings
from app.db.news_candidates import NewsCandidate, CandidateSelectionResult
from app.db.pool import create_database_pool, close_database_pool
from app.db.published_news_history import load_published_news_history
from app.ranking.openai_event_client import OpenAIResponsesEventRankingClient
from app.ranking.openai_event_evaluator import OpenAIEventRankingEvaluator, OpenAIEventRankingPayload
from scripts.retry_failed_text_workflow import RetryBudget, BudgetClient


async def main(output_dir: Path):
    settings=get_settings()
    budget=RetryBudget(output_dir,Decimal('1'))
    pool=await create_database_pool(settings)
    as_of=datetime(2026,10,25,7,30,tzinfo=UTC)
    try:
        async with pool.acquire() as connection, connection.transaction(readonly=True):
            rows=await connection.fetch("""SELECT n.*,s.source_code,s.source_name,s.collection_priority,s.settings
               FROM news_items n JOIN sources s USING(source_id) WHERE n.news_id=ANY($1::bigint[]) ORDER BY n.news_id""",[8262,8269,8285])
        history=await load_published_news_history(pool,as_of=as_of)
    finally:
        await close_database_pool(pool)
    candidates=[]
    for row in rows:
        config=json.loads(row['settings']) if isinstance(row['settings'],str) else row['settings']
        candidates.append(NewsCandidate(news_id=row['news_id'],source_id=row['source_id'],source_code=row['source_code'],source_name=row['source_name'],collection_priority=row['collection_priority'],processing_status='candidate',title=row['normalized_title'] or row['raw_title'],summary=row['normalized_summary'] or row['raw_summary'],author_name=row['author_name'],source_published_at=row['source_published_at'],age_hours=(as_of-row['source_published_at']).total_seconds()/3600,source_url=row['source_url'],primary_image_url=row['primary_image_url'],source_weight=config['ranking']['source_weight']))
    selection=CandidateSelectionResult(as_of-timedelta(hours=24),as_of,24,tuple(candidates),history)
    sdk=BudgetClient(budget,api_key=settings.openai_api_key.get_secret_value(),timeout=120)
    client=OpenAIResponsesEventRankingClient(client=sdk)
    evaluator=OpenAIEventRankingEvaluator(client=client,model_name=settings.openai_ranking_model)
    request=evaluator.build_request(selection)
    (output_dir/'request.json').write_text(json.dumps(dict(instructions=request.instructions,input=json.loads(request.input_text)),ensure_ascii=False,indent=2))
    try:
        response=await client.create_response(request)  # Ровно один POST; retry/repair здесь не выполняются.
    finally:
        await sdk.close()
    payload=OpenAIEventRankingPayload.model_validate_json(response.output_text)
    outcomes={m.news_id:e.duplicate_of_published_news_id for e in payload.events for m in e.members}
    assert outcomes.keys()=={8262,8269,8285},outcomes
    assert outcomes[8262]==8257,outcomes
    assert outcomes[8269] is None and outcomes[8285] is None,outcomes
    print(json.dumps(dict(passed=True,classification=outcomes,accounted_usd=str(budget.total),paid_posts=len(budget.calls),evidence=str(output_dir)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--paid',action='store_true',required=True,help='Один платный запрос в пределах согласованного $1')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(main(args.output_dir))
