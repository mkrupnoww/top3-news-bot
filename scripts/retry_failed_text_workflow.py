"""Явный однократный retry text failure: saved ranking, отдельный cost ledger."""
import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from app.generation.openai_image_client import _extract_usage
from app.generation.image_openai_usage import build_openai_image_cost_payload
from app.generation.openai_generator import OPENAI_POST_GENERATOR_VERSION
from app.ranking.openai_usage import extract_response_usage, calculate_openai_cost, get_model_pricing
import app.workflows.daily_production as daily


class RetryBudget:
    """Резервирует консервативный allowance до каждого запроса, retries=0."""
    def __init__(self, directory: Path, maximum: Decimal):
        if not maximum.is_finite() or not Decimal('0') < maximum <= Decimal('1'):
            raise ValueError('Authorized retry budget must be in (0, 1] USD')
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.directory, self.maximum = directory, maximum
        self.calls = []
        self.save()

    def save(self):
        payload = {'maximum_usd': str(self.maximum), 'accounted_usd': str(self.total), 'calls': self.calls}
        (self.directory/'cost-ledger.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    @property
    def total(self):
        return sum((Decimal(c['accounted_usd']) for c in self.calls), Decimal('0'))

    def reserve(self, kind: str, amount: Decimal):
        if self.total + amount > self.maximum:
            raise RuntimeError(f'Retry budget would be exceeded: spent/reserved={self.total}, next={amount}')
        call = {'kind': kind, 'status': 'reserved', 'accounted_usd': str(amount)}
        self.calls.append(call)
        self.save()
        return call

    def complete(self, call, cost, response, usage):
        call.update(status='completed', accounted_usd=str(cost), usage=usage,
                    request_id=getattr(response, '_request_id', None))
        self.save()
        if self.total > self.maximum:
            raise RuntimeError('Actual usage exceeded conservative retry allowance; no further calls')


class BudgetClient:
    def __init__(self, budget, *, deny_ranking=False, **kwargs):
        self.budget, self.deny_ranking = budget, deny_ranking
        kwargs['max_retries'] = 0
        self.raw = AsyncOpenAI(**kwargs)
        self.responses = SimpleNamespace(create=self.create_response)
        self.images = SimpleNamespace(generate=self.create_image)

    async def close(self):
        await self.raw.close()

    async def create_response(self, **kwargs):
        if self.deny_ranking:
            raise RuntimeError('Paid ranking replay is forbidden; reuse saved ranking')
        if kwargs['model'] != 'gpt-6.1-sol':
            raise ValueError('Unexpected retry text model')
        kwargs.update(max_output_tokens=5000, reasoning={'effort': 'low'}, max_tool_calls=1)
        size = len(json.dumps(kwargs, ensure_ascii=False).encode('utf-8')) + 1024
        # Worst input/cache-write rate; bounded output includes reasoning.
        allowance = Decimal(size) * Decimal('0.0000025') + Decimal('0.05')
        if kwargs.get('tools'):
            # One hosted search, conservative 128K search context + tool fee.
            allowance += Decimal('0.34')
        call = self.budget.reserve('responses', allowance)
        try:
            response = await self.raw.responses.create(**kwargs)
            usage = extract_response_usage(response)
            cost = calculate_openai_cost(usage, get_model_pricing(kwargs['model'])).total_cost_usd
            searches = sum(getattr(item, 'type', None) == 'web_search_call' for item in response.output)
            cost += Decimal(searches) * Decimal('0.01')
            self.budget.complete(call, cost, response, asdict(usage))
            (self.budget.directory/f"response-{len(self.budget.calls)}.json").write_text(response.output_text)
            return response
        except Exception as error:
            if call['status'] == 'reserved':
                call.update(status='uncertain', error_type=type(error).__name__)
                self.budget.save()
            raise

    async def create_image(self, **kwargs):
        if (kwargs['model'], kwargs['size'], kwargs['quality'], kwargs['n']) != (
            'gpt-image-2.5-flare', '1024x1024', 'medium', 1
        ):
            raise ValueError('Unexpected retry image settings')
        allowance = Decimal('0.45') + Decimal(len(kwargs['prompt'].encode('utf-8')) + 1024)*Decimal('0.000005')
        call = self.budget.reserve('images', allowance)
        try:
            response = await self.raw.images.generate(**kwargs)
            usage = _extract_usage(response)
            if usage is None:
                raise RuntimeError('Image usage missing; retain reserved allowance')
            cost = build_openai_image_cost_payload(kwargs['model'], usage)
            self.budget.complete(call, Decimal(cost['total_cost_usd']), response, asdict(usage))
            return response
        except Exception as error:
            if call['status'] == 'reserved':
                call.update(status='uncertain', error_type=type(error).__name__)
                self.budget.save()
            raise


async def run(workflow_id: int, directory: Path, maximum: Decimal):
    settings = Settings()
    budget = RetryBudget(directory, maximum)
    pool = await create_database_pool(settings)
    try:
        async with pool.acquire() as c:
            row = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1', workflow_id)
        if row is None:
            raise ValueError('Workflow does not exist')
        async with daily._daily_workflow_lock(pool, publication_date=row['publication_date']):
            async with pool.acquire() as c, c.transaction():
                row = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1 FOR UPDATE', workflow_id)
                assert row['workflow_status'] == 'failed'
                assert row['error_message'] == 'Ответ модели не соответствует схеме Telegram-поста.'
                assert row['generated_post_id'] is None and row['image_generation_id'] is None
                assert row['ranking_run_id'] is not None and row['batch_id'] is not None
                batch = await c.fetchrow('SELECT * FROM top3_news.publication_batches WHERE batch_id=$1 FOR UPDATE', row['batch_id'])
                assert batch['batch_status'] == 'failed'
                assert json.loads(batch['metadata'])['generator_version'] != OPENAI_POST_GENERATOR_VERSION
                assert not await c.fetchval('SELECT EXISTS (SELECT 1 FROM top3_news.generated_posts WHERE batch_id=$1)', row['batch_id'])
                assert await c.fetchval("SELECT run_status='completed' FROM top3_news.ranking_runs WHERE ranking_run_id=$1", row['ranking_run_id'])
                (directory/'state-before.json').write_text(json.dumps({'workflow':dict(row),'batch':dict(batch)}, ensure_ascii=False, default=str, indent=2))
                await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='running',
                    current_stage='generation',batch_id=NULL,error_type=NULL,error_message=NULL,
                    finished_at=NULL,updated_at=now() WHERE daily_workflow_run_id=$1""", workflow_id)
        # Original failed batch is preserved. Generator v9 makes a new request key.
        original_generation = daily.create_openai_generation_runtime
        original_image = daily.create_openai_image_generation_runtime
        original_ranking = daily.create_openai_event_ranking_runtime
        daily.create_openai_generation_runtime = lambda s: original_generation(s, client_factory=lambda **kw: BudgetClient(budget, **kw))
        daily.create_openai_image_generation_runtime = lambda s: original_image(s, client_factory=lambda **kw: BudgetClient(budget, **kw))
        daily.create_openai_event_ranking_runtime = lambda s: original_ranking(s, client_factory=lambda **kw: BudgetClient(budget, deny_ranking=True, **kw))
        result = await daily.run_daily_production_workflow(pool, settings=settings,
            publication_date=row['publication_date'], as_of=row['as_of'],
            progress=lambda message: print(message, flush=True))
        (directory/'result.json').write_text(json.dumps(asdict(result), default=str, indent=2))
        print(json.dumps(asdict(result), default=str), flush=True)
        print('Retry estimated/accounted USD:', budget.total, flush=True)
    finally:
        await close_database_pool(pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow-id', type=int, required=True)
    parser.add_argument('--ledger-directory', type=Path, required=True)
    parser.add_argument('--max-usd', type=Decimal, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.workflow_id, args.ledger_directory, args.max_usd))
