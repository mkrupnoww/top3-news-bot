"""Проверки manual retry: бюджет до сети и сохранение истории в тестовой БД."""
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import tempfile

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from scripts import retry_failed_text_workflow as retry


@dataclass
class StubResult:
    workflow_status: str = 'test_only'


async def main():
    s = Settings()
    assert s.app_env == 'testing' and s.db_port == 55432
    with tempfile.TemporaryDirectory() as parent:
        budget = retry.RetryBudget(Path(parent)/'budget', Decimal('1'))
        call = budget.reserve('synthetic', Decimal('.7'))
        try:
            budget.reserve('synthetic', Decimal('.4'))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Budget overrun accepted')
        budget.complete(call, Decimal('.1'), object(), {})
        assert budget.total == Decimal('.1')
        assert len(budget.calls) == 1
        try:
            retry.RetryBudget(Path(parent)/'budget', Decimal('1'))
        except FileExistsError:
            pass
        else:
            raise AssertionError('Duplicate ledger accepted')

        pool = await create_database_pool(s)
        try:
            async with pool.acquire() as c:
                before = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=82')
                assert before['workflow_status'] == 'failed'
            async def no_paid_workflow(pool, **kwargs):
                assert kwargs['as_of'] == before['as_of']
                assert kwargs['publication_date'] == before['publication_date']
                async with pool.acquire() as c:
                    current = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=82')
                    assert current['ranking_run_id'] == before['ranking_run_id']
                    assert current['workflow_status'] == 'running' and current['current_stage'] == 'generation'
                    assert current['batch_id'] is None
                    assert await c.fetchval('SELECT batch_status FROM top3_news.publication_batches WHERE batch_id=$1', before['batch_id']) == 'failed'
                return StubResult()
            retry.daily.run_daily_production_workflow = no_paid_workflow
            directory = Path(parent)/'manual-retry'
            await retry.run(82, directory, Decimal('1'))
            assert json.loads((directory/'cost-ledger.json').read_text())['calls'] == []
            assert json.loads((directory/'state-before.json').read_text())['workflow']['batch_id'] == before['batch_id']
            print('PASS: budget, duplicate blocking, failed history, saved ranking/date/as_of; paid/Telegram calls=0')
        finally:
            await close_database_pool(pool)


if __name__ == '__main__':
    asyncio.run(main())
