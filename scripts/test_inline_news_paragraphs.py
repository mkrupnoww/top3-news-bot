"""Цельный абзац с жирным началом; legacy items, trailer и строгий общий лимит."""
import asyncio
import json
from pydantic import ValidationError

from app.generation.openai_generator import (
    OpenAIGeneratedNewsPayload, OpenAIGeneratedPostPayload,
    OpenAITelegramPostGenerator, build_top3_post_text,
)
from app.generation.openai_client import OPENAI_GENERATION_RESPONSE_SCHEMA
from app.generation.post_contract import MAXIMUM_POST_LENGTH, TARGET_POST_LENGTH_MIN, TARGET_POST_LENGTH_MAX
from app.generation.post_integrity import build_deterministic_integrity_fallback, validate_generated_post_integrity
from app.publication.telegram_text import prepare_telegram_text
from scripts.test_openai_post_generator import FakeStructuredGenerationClient, build_news_items


def paragraphs():
    # Вымышленные факты только для offline fixture, не для публикации.
    return [
        ('Студия перенесла премьеру',
         'Студия перенесла премьеру фильма «Маяк» с ноября на февраль, чтобы завершить съёмки и монтаж после смены режиссёра. Прокатчик подтвердил новую дату, а уже начатая рекламная кампания продолжится с обновлёнными материалами. Изменение касается кинотеатрального релиза.'),
        ('Фестиваль расширил конкурс',
         'Фестиваль расширил конкурс дебютных фильмов: в программу добавили отдельную секцию для режиссёров, которые сняли первую полнометражную картину. Победитель получит средства на следующий проект. Отбор проведёт прежняя комиссия; заявки принимаются до конца месяца.'),
        ('Компания открыла павильон',
         'Компания открыла павильон для съёмок сериалов на территории действующей киностудии. Площадка оборудована системой виртуальных декораций, позволяющей менять фон без выезда съёмочной группы. Первый проект начнёт работу осенью; студия также будет сдавать павильон другим продюсерам.'),
    ]


async def main():
    items = [OpenAIGeneratedNewsPayload(position=i,news_id=200+i,headline=lead,body=body)
             for i,(lead,body) in enumerate(paragraphs(),1)]
    post = build_top3_post_text(items)
    assert TARGET_POST_LENGTH_MIN==920 and TARGET_POST_LENGTH_MAX==960 and MAXIMUM_POST_LENGTH==1000
    assert TARGET_POST_LENGTH_MIN <= len(post) <= TARGET_POST_LENGTH_MAX, len(post)
    for item in items:
        assert item.body.startswith(item.headline)
        section = f'**{item.headline}**{item.body[len(item.headline):]}'
        assert section in post and post.count(item.headline)==1
        assert f'**{item.headline}**\n' not in post
    payload = OpenAIGeneratedPostPayload(post_text=post,items=items)
    assert not validate_generated_post_integrity(payload)
    prepared = prepare_telegram_text(post,text_format='markdown')
    assert prepared.parse_mode=='HTML'
    assert f'<b>{items[0].headline}</b>{items[0].body[len(items[0].headline):]}' in prepared.text
    client = FakeStructuredGenerationClient(json.dumps(payload.model_dump(),ensure_ascii=False))
    generator = OpenAITelegramPostGenerator(client=client,model_name='gpt-6.1-sol')
    result = await generator.generate_detailed(build_news_items())
    assert result.payload.post_text==post
    reviewed = await generator.generate_self_review_detailed(build_news_items(),source_post_text=post)
    assert reviewed.payload.post_text==post
    revision = generator.build_revision_request(build_news_items(),source_post_text=post,
        editorial_comment='Сохрани новый формат.',issues=('Не возвращай отдельные заголовки.',))
    revised = await generator.generate_prepared_revision_request(build_news_items(),revision,source_post_text=post,
        editorial_comment='Сохрани новый формат.',issues=('Не возвращай отдельные заголовки.',))
    assert revised.payload.post_text==post
    for request in client.requests:
        normalized=' '.join(request.instructions.split())
        assert '920–960' in normalized and '300 символов' in normalized
    trailer_items = [item.model_copy(update={'official_trailer_url':'https://www.youtube.com/watch?v=testvideo01',
                     'official_trailer_channel_name':'Test Studio'}) if item.position==1 else item for item in items]
    compacted = build_deterministic_integrity_fallback(payload.model_copy(update={'items':trailer_items}))
    assert len(compacted.post_text)<=1000
    assert 'https://www.youtube.com/watch?v=testvideo01' in compacted.post_text
    assert [v.news_id for v in compacted.items]==[201,202,203]
    assert not validate_generated_post_integrity(compacted)
    legacy = items[0].model_copy(update={'headline':'Ранее сохранённый заголовок','body':'Ранее сохранённый текст.'})
    assert '**Ранее сохранённый заголовок** Ранее сохранённый текст.' in build_top3_post_text([legacy,*items[1:]])
    schema=OPENAI_GENERATION_RESPONSE_SCHEMA['properties']['items']['items']['properties']
    assert schema['body']['maxLength']==300 and 'headline' in schema['body']['description']
    try:
        OpenAIGeneratedPostPayload(post_text='Т'*1001,items=items)
    except ValidationError:
        pass
    else:
        raise AssertionError('Final 1000-character limit was weakened')
    print(f'PASS: inline paragraphs ({len(post)} chars), no duplicated lead, generation/review/revision, trailer compaction, legacy data and final limit; API/Telegram=0')


if __name__=='__main__':
    asyncio.run(main())
