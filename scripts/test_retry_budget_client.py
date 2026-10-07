"""SDK adapter budget checks without network or DB."""
import asyncio
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts import retry_failed_text_workflow as retry
from scripts.test_openai_image_factory import FakeImagesResource


async def main():
    clients = []
    class FakeSDK:
        def __init__(self, **kwargs):
            assert kwargs['max_retries'] == 0
            self.calls = []
            self.responses = SimpleNamespace(create=self.create)
            self.images = FakeImagesResource()
            clients.append(self)
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs['max_output_tokens'] == 5000
            assert kwargs['max_tool_calls'] == 1 and kwargs['reasoning']['effort'] == 'low'
            return SimpleNamespace(output=[],output_text='{}',usage=SimpleNamespace(
                input_tokens=1000,output_tokens=500,total_tokens=1500,
                input_tokens_details=SimpleNamespace(cached_tokens=100,cache_write_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=100)))
        async def close(self):
            pass
    retry.AsyncOpenAI = FakeSDK
    with TemporaryDirectory() as directory:
        budget = retry.RetryBudget(Path(directory)/'ledger', Decimal('1'))
        client = retry.BudgetClient(budget, api_key='fake',max_retries=2)
        await client.responses.create(model='gpt-6.1-sol',input='Fake data')
        await client.images.generate(model='gpt-image-2.5-flare',size='1024x1024',quality='medium',n=1,prompt='Fake image')
        assert budget.total == Decimal('.03441000')
        assert len(clients[0].calls) == len(clients[0].images.calls) == 1
        denied = retry.BudgetClient(budget, deny_ranking=True, api_key='fake')
        try:
            await denied.responses.create(model='gpt-6.1-sol')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Ranking replay accepted')
        assert not clients[1].calls
        budget.maximum = budget.total
        try:
            await client.responses.create(model='gpt-6.1-sol',input='Stop before SDK')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Over-budget SDK call dispatched')
        assert len(clients[0].calls) == 1
        print('PASS: retry-free SDK, bounded tokens/tools, usage cost, saved-ranking guard and budget stop before dispatch')


if __name__ == '__main__':
    asyncio.run(main())
