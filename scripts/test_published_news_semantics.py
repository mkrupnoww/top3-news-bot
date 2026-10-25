"""Контракт истории, программное исключение и различение editorial intent без сети."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json

from app.generation.editorial_intent import replacement_positions, ReplacementPositionRequired
from app.ranking.background_snapshot import encode_snapshot, decode_snapshot
from app.ranking.event_formula_pipeline import calculate_event_scores
from app.ranking.openai_event_client import OPENAI_EVENT_RANKING_RESPONSE_SCHEMA
from app.ranking.openai_event_evaluator import OpenAIEventRankingPayload, _build_events
from scripts.test_openai_event_ranking_evaluator import build_selection, build_valid_payload, build_evaluator, build_response


async def main():
    for comment, expected in (
        ('Замени третью новость. Эта новость уже была вчера.', (3,)),
        ('Замени первую новость', (1,)), ('Убери вторую новость', (2,)),
        ('Замени новость №3', (3,)), ('Замени 3-ю новость', (3,)),
        ('Замени все три новости', (1,2,3)),
        ('Замени формулировку третьей новости', ()), ('Не заменяй третью новость', ()),
        ('Убери повтор слова в третьей новости', ()), ('Исправь дату в третьей новости', ()),
    ):
        assert replacement_positions(comment) == expected, comment
    for comment in ('Замени эту новость', 'Это дубль вчерашней новости'):
        try:
            replacement_positions(comment)
        except ReplacementPositionRequired:
            pass
        else:
            raise AssertionError('Неоднозначная замена не запросила номер')

    history = ({'batch_id':7,'publication_date':'2026-08-01','published_text':'Вчера показали первые 17 минут фильма.',
                'items':[{'news_id':999,'title':'Первые 17 минут','summary':'Показ на фестивале.'}]},)
    selection = replace(build_selection(), published_history=history)
    payload = build_valid_payload()
    for event in payload['events']:
        event['duplicate_of_published_news_id'] = None
    repeated = payload['events'][0]
    repeated['duplicate_of_published_news_id'] = 999
    repeated['q_score'] = 1  # Даже при ошибочном Q модели код обязан исключить дубль.
    repeated['q_reason'] = 'Другой источник описал тот же показ без новых фактов.'
    evaluator, client = build_evaluator(build_response(payload))
    request = evaluator.build_request(selection)
    assert json.loads(request.input_text)['published_history'] == list(history)
    snapshot = json.loads(encode_snapshot(selection, request, ()))
    restored, restored_request, _ = decode_snapshot(snapshot)
    assert restored.published_history == history and restored_request == request
    result = await evaluator.evaluate_prepared_request(selection, request)
    event = next(e for e in result.events if e.representative_news_id == repeated['representative_news_id'])
    assert event.q_score == 0 and 'news_id=999' in event.q_reason
    scores = calculate_event_scores(selection=selection, events=result.events).scores
    assert not next(s for s in scores if s.news_id==event.representative_news_id).is_eligible
    assert all(e.q_score > 0 for e in result.events if e.representative_news_id != event.representative_news_id)
    assert set(m.news_id for e in result.events for m in e.members)=={101,102,103}

    # Нельзя сослаться на придуманную опубликованную новость.
    bad = deepcopy(payload)
    bad['events'][0]['duplicate_of_published_news_id'] = 123456
    parsed = OpenAIEventRankingPayload.model_validate(bad)
    try:
        _build_events(payload=parsed, selection=selection, story_cluster_keys_by_representative={e.representative_news_id:'test' for e in parsed.events})
    except ValueError:
        pass
    else:
        raise AssertionError('Неизвестная ссылка на историю принята')
    schema = OPENAI_EVENT_RANKING_RESPONSE_SCHEMA
    def check_schema(node):
        if isinstance(node, dict):
            if node.get('type')=='object':
                assert set(node['properties']) == set(node['required'])
                assert node['additionalProperties'] is False
            for value in node.values(): check_schema(value)
        elif isinstance(node,list):
            for value in node: check_schema(value)
    check_schema(schema)
    print('PASS: editorial intent, frozen history/request, hard duplicate veto, coverage, invalid history ID, strict schema; API calls=0')


if __name__=='__main__':
    asyncio.run(main())
