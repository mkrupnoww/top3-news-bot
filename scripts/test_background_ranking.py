"""Обрыв GET, restart/checkpoint, неизвестный POST и terminal failure без сети."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import httpx
from openai import APIConnectionError

from app.ranking.background_response import BackgroundRankingResponses, BackgroundRankingError


class Resource:
    def __init__(self, *, terminal='completed', lost_post=False):
        self.posts = self.reads = 0
        self.terminal, self.lost_post = terminal, lost_post

    async def create(self, **kwargs):
        self.posts += 1
        assert kwargs['background'] is True and kwargs['store'] is True
        if self.lost_post:
            raise APIConnectionError(request=httpx.Request('POST','https://example.invalid'))
        return SimpleNamespace(id='resp_saved',status='queued')

    async def retrieve(self, response_id, **kwargs):
        assert response_id == 'resp_saved'
        self.reads += 1
        if self.reads == 1:
            raise APIConnectionError(request=httpx.Request('GET','https://example.invalid'))
        return SimpleNamespace(id=response_id,status=self.terminal,output_text='{"events":[]}',
            usage=SimpleNamespace(input_tokens=10,output_tokens=20,total_tokens=30), error=None)


async def main():
    with TemporaryDirectory() as root:
        directory = Path(root)/'successful'
        resource = Resource()
        runner = BackgroundRankingResponses(resource,directory,poll_seconds=0)
        response = await runner.create(model='fake',input='same')
        assert response.status == 'completed' and resource.posts == 1 and resource.reads == 2
        response = await BackgroundRankingResponses(resource,directory,poll_seconds=0).create(model='fake',input='same')
        assert response.output_text == '{"events":[]}' and response.usage.total_tokens == 30
        assert resource.posts == 1 and resource.reads == 2

        interrupted_dir = Path(root)/'interrupted'
        resource = Resource()
        try:
            await BackgroundRankingResponses(resource,interrupted_dir,poll_seconds=0,deadline_seconds=0).create(model='fake')
        except BackgroundRankingError:
            pass
        else:
            raise AssertionError('Deadline ignored')
        assert resource.posts == 1
        await BackgroundRankingResponses(resource,interrupted_dir,poll_seconds=0).create(model='fake')
        assert resource.posts == 1  # После restart читается тот же response ID.

        resource = Resource()
        runner = BackgroundRankingResponses(resource,Path(root)/'resume_missing',poll_seconds=0)
        runner.require_existing_first = True
        try:
            await runner.create(model='fake')
        except BackgroundRankingError:
            pass
        else:
            raise AssertionError('Recovery dispatched fresh POST')
        assert resource.posts == resource.reads == 0

        resource = Resource(lost_post=True)
        runner = BackgroundRankingResponses(resource,Path(root)/'unknown',poll_seconds=0)
        try:
            await runner.create(model='fake')
        except APIConnectionError:
            pass
        else:
            raise AssertionError('Expected unknown POST outcome')
        try:
            await runner.create(model='fake')
        except BackgroundRankingError:
            pass
        else:
            raise AssertionError('Unknown POST replayed')
        assert resource.posts == 1

        for terminal in ('failed','incomplete','cancelled'):
            resource = Resource(terminal=terminal)
            try:
                await BackgroundRankingResponses(resource,Path(root)/terminal,poll_seconds=0).create(model='fake')
            except BackgroundRankingError:
                pass
            else:
                raise AssertionError('Terminal failure accepted as ranking')
            assert resource.posts == 1
        print('PASS: transient GET, cached result, resume same response, deadline, unknown POST, terminal failures; API calls=0')


if __name__ == '__main__':
    asyncio.run(main())
