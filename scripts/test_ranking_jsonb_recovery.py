"""Реальный PostgreSQL JSONB round trip и защищённый reopen на отдельной cloud-001 БД."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json

from app.config import get_settings
from app.db.pool import create_database_pool, close_database_pool
from app.db.daily_background_ranking_recovery import reopen_background_ranking_workflow, is_jsonb_order_failure
from app.ranking.background_snapshot import encode_snapshot, decode_snapshot
from app.ranking.request_identity import same_event_ranking_request, PREPARED_REQUEST_MISMATCH
from scripts.test_openai_event_ranking_evaluator import build_selection, build_valid_payload, build_response, build_evaluator


class TransactionPool:
    def __init__(self, connection): self.connection = connection
    @asynccontextmanager
    async def acquire(self): yield self.connection


async def main():
    settings=get_settings()
    if settings.app_env!='testing' or settings.db_port!=55432 or settings.db_name!='top3_recovery_test':
        raise RuntimeError('Только изолированная top3_recovery_test на cloud-001:55432')
    pool=await create_database_pool(settings)
    async with pool.acquire() as connection:
        selection=replace(build_selection(),published_history=({'batch_id':7,'publication_date':'2026-08-01','published_text':'Сохранённый выпуск','items':[{'news_id':999,'position':1,'title':'История','summary':'Факты','source_url':'https://example.invalid/999'}]},))
        payload=build_valid_payload()
        for e in payload['events']: e['duplicate_of_published_news_id']=None
        evaluator,client=build_evaluator(build_response(payload))
        original=evaluator.build_request(selection)
        snapshot=json.loads(await connection.fetchval('SELECT ($1::jsonb)::text',encode_snapshot(selection,original,())))
        restored,saved,_=decode_snapshot(snapshot)
        rebuilt=evaluator.build_request(restored)
        assert original==saved and saved!=rebuilt
        assert same_event_ranking_request(saved,rebuilt)
        await evaluator.evaluate_prepared_request(restored,saved)
        assert client.requests[0].input_text==original.input_text  # Исходный checkpoint ключ не меняется.
        print('Real JSONB key reordering accepted; original outbound input preserved: OK')

        for field in ('model','instructions'):
            assert not same_event_ranking_request(saved,replace(rebuilt,**{field:getattr(rebuilt,field)+' changed'}))
        for mutate in (
            lambda p:p['expected_news_ids'].reverse(),
            lambda p:p['published_history'][0].update(published_text='Другие факты'),
            lambda p:p.update(expected_news_count=True),
            lambda p:p.update(expected_news_count=float(p['expected_news_count'])),
        ):
            changed=json.loads(rebuilt.input_text);mutate(changed)
            assert not same_event_ranking_request(saved,replace(rebuilt,input_text=json.dumps(changed)))
        assert not same_event_ranking_request(saved,replace(rebuilt,input_text='{invalid'))
        assert not same_event_ranking_request(saved,replace(rebuilt,input_text='{"x":1,"x":2}'))
        print('Changed model, prompt, history, list order, numeric/bool types, invalid JSON rejected: OK')
        error={'error_type':'ValueError','error_message':PREPARED_REQUEST_MISMATCH}
        assert is_jsonb_order_failure(error,snapshot)
        corrupt=json.loads(json.dumps(snapshot));corrupt['selection']['published_history'][0]['published_text']='changed'
        assert not is_jsonb_order_failure(error,corrupt)
        assert not is_jsonb_order_failure(dict(error_type='ValueError',error_message='other error'),snapshot)

        workflow_id=159
        # Реальные incident-строки скопированы в эту изолированную БД; каждая ветка откатывается.
        for alteration,expected in (
            (None,265),
            ("UPDATE daily_workflow_runs SET error_message='unrelated ValueError' WHERE daily_workflow_run_id=159",None),
            ("UPDATE ranking_runs SET parameters=jsonb_set(parameters,'{evaluator_version}','\"other_version\"'::jsonb) WHERE ranking_run_id=265",None),
            ("UPDATE ranking_runs SET parameters=parameters - 'background_snapshot' WHERE ranking_run_id=265",None),
            ("UPDATE ranking_runs SET parameters=jsonb_set(parameters,'{background_snapshot,selection,published_history,0,published_text}','\"changed facts\"'::jsonb) WHERE ranking_run_id=265",None),
        ):
            tx=connection.transaction();await tx.start()
            try:
                if alteration: await connection.execute(alteration)
                got=await reopen_background_ranking_workflow(TransactionPool(connection),daily_workflow_run_id=workflow_id)
                assert got==expected,(alteration,got)
                state=await connection.fetchval('SELECT workflow_status FROM daily_workflow_runs WHERE daily_workflow_run_id=$1',workflow_id)
                assert state==('running' if expected else 'failed')
            finally: await tx.rollback()
        print('Incident workflow resumes only for proved order-only mismatch; unrelated/mutated snapshots stay blocked: OK')
    await close_database_pool(pool)
    print('PASS: JSONB recovery regression; API calls=0; incident state rolled back')


if __name__=='__main__': asyncio.run(main())
