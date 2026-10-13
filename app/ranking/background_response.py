"""Долгое ранжирование: один POST, восстановимые GET и локальный checkpoint."""
import asyncio
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace

from openai import APIConnectionError, APIStatusError


class BackgroundRankingError(RuntimeError):
    """Результат фонового запроса не завершён; повторный POST запрещён."""


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


class BackgroundRankingResponses:
    def __init__(self, resource, directory: Path, *, poll_seconds=10, deadline_seconds=1800):
        self.resource, self.directory = resource, directory
        self.poll_seconds, self.deadline_seconds = poll_seconds, deadline_seconds

    async def create(self, **kwargs):
        key = sha256(json.dumps(kwargs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self.directory / f'{key}.json'

        def save(state):
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, ensure_ascii=False, default=lambda v:
                v.model_dump() if hasattr(v, 'model_dump') else vars(v)))
            temporary.chmod(0o600)
            temporary.replace(path)

        with (self.directory / f'{key}.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BackgroundRankingError('Background ranking already has an active reader') from error
            state = json.loads(path.read_text()) if path.exists() else None
            if state and state.get('completed'):
                return _namespace(state['completed'])
            if state and not state.get('response_id'):
                raise BackgroundRankingError(f'Unknown POST outcome; inspect checkpoint {path}')
            if state is None:
                state = {'status': 'creating', 'created_at': time.time()}
                save(state)  # Сохраняется ДО POST; неизвестный исход не переигрывается.
                kwargs.update(background=True, store=True, timeout=60)
                response = await self.resource.create(**kwargs)
                state.update(response_id=response.id, status=response.status)
                save(state)
                print(f'[ranking] background response saved: {response.id}', flush=True)
            else:
                response = None
                print(f"[ranking] resume background response: {state['response_id']}", flush=True)
            deadline = time.monotonic() + self.deadline_seconds
            while True:
                if response is not None:
                    status = response.status
                    if status == 'completed':
                        state.update(status=status, completed={
                            'output_text': response.output_text, 'usage': response.usage,
                            '_request_id': getattr(response, '_request_id', None)})
                        save(state)
                        return response
                    if status not in ('queued', 'in_progress'):
                        state.update(status=status, error=str(getattr(response, 'error', None)),
                                     incomplete_details=str(getattr(response, 'incomplete_details', None)))
                        save(state)
                        raise BackgroundRankingError(f"Background response {state['response_id']} ended: {status}; {state.get('error')}; {state.get('incomplete_details')}")
                if time.monotonic() >= deadline:
                    raise BackgroundRankingError(f"Polling deadline; resume response {state['response_id']} from {path}")
                await asyncio.sleep(self.poll_seconds)
                try:
                    response = await self.resource.retrieve(state['response_id'], timeout=30)
                except (APIConnectionError, APIStatusError) as error:
                    if isinstance(error, APIStatusError) and error.status_code not in (408, 429) and error.status_code < 500:
                        raise
                    state.update(last_read_error=type(error).__name__)
                    save(state)
                    response = None
                    print(f"[ranking] transient GET error; keep response {state['response_id']}: {type(error).__name__}", flush=True)
