"""Recovery identity/history checks on isolated PostgreSQL only, never production."""
import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from scripts.retry_failed_ranking_workflow import reopen


async def main(fixture_file):
    settings = Settings()
    if settings.app_env != 'testing' or settings.db_port != 55432:
        raise RuntimeError('Requires isolated testing PostgreSQL on 55432')
    fixture = json.loads(fixture_file.read_text())
    old = fixture['workflow']
    workflow_id, ranking_id = old['daily_workflow_run_id'], old['ranking_run_id']
    pool = await create_database_pool(settings)
    try:
        async with pool.acquire() as c:
            await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='failed',
                current_stage='failed',ranking_run_id=$2,batch_id=NULL,generated_post_id=NULL,
                image_generation_id=NULL,error_type='APIConnectionError',error_message='Connection error.'
                WHERE daily_workflow_run_id=$1""",workflow_id,ranking_id)
            before = dict(await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1',workflow_id))
            ranking_before = dict(await c.fetchrow('SELECT * FROM top3_news.ranking_runs WHERE ranking_run_id=$1',ranking_id))
        with TemporaryDirectory() as directory:
            path = Path(directory)
            await reopen(pool,workflow_id,path,check_only=True)
            async with pool.acquire() as c:
                assert dict(await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1',workflow_id)) == before
            await reopen(pool,workflow_id,path)
            async with pool.acquire() as c:
                current = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1',workflow_id)
                assert current['workflow_status']=='running' and current['current_stage']=='ranking'
                assert current['ranking_run_id'] is None and current['batch_id'] is None
                assert current['publication_date']==before['publication_date'] and current['as_of']==before['as_of']
                assert dict(await c.fetchrow('SELECT * FROM top3_news.ranking_runs WHERE ranking_run_id=$1',ranking_id)) == ranking_before
            assert json.loads((path/'state-before.json').read_text())['workflow']['ranking_run_id']==ranking_id
            try:
                await reopen(pool,workflow_id,path)
            except ValueError:
                pass
            else:
                raise AssertionError('Duplicate retry accepted')
        print('PASS: read-only preflight, preserved cutoff/date/failed ranking, audited reopening, duplicate rejection; API/Telegram=0')
    finally:
        await close_database_pool(pool)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixture-file',type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(main(args.fixture_file))
