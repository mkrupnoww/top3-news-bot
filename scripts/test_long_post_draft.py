"""Регрессия: длинный model draft не обходит лимит финального Telegram-поста."""
import asyncio
import json

from pydantic import ValidationError

from app.generation.openai_generator import (
    OpenAIGeneratedPostPayload, OpenAITelegramPostGenerator, _parse_response,
)
from app.generation.post_contract import MAXIMUM_POST_LENGTH
from app.generation.post_integrity import validate_generated_post_integrity
from scripts.test_openai_post_generator import (
    FakeStructuredGenerationClient, build_news_items, build_valid_response,
    build_expected_post_text,
)
from scripts.test_text_integrity_repair import _overflow_payload


async def main():
    data = json.loads(build_valid_response())
    data['post_text'] = 'Длинный черновик. ' * 100
    client = FakeStructuredGenerationClient(json.dumps(data, ensure_ascii=False))
    generator = OpenAITelegramPostGenerator(client=client, model_name='gpt-6.1-sol')
    result = await generator.generate_self_review_detailed(
        build_news_items(), source_post_text=build_expected_post_text()
    )
    assert result.payload.post_text == build_expected_post_text()
    assert len(client.requests) == 1
    assert not validate_generated_post_integrity(result.payload)

    overflow = _overflow_payload()
    data = overflow.model_dump()
    data['post_text'] = 'Длинный черновик. ' * 100
    compacted = _parse_response(json.dumps(data, ensure_ascii=False))
    assert len(compacted.post_text) <= MAXIMUM_POST_LENGTH
    assert [x.news_id for x in compacted.items] == [x.news_id for x in overflow.items]
    assert [x.official_trailer_url for x in compacted.items] == [x.official_trailer_url for x in overflow.items]
    assert not validate_generated_post_integrity(compacted)
    OpenAIGeneratedPostPayload.model_validate(compacted.model_dump())

    for change in ('extra', 'body', 'duplicate', 'missing', 'empty'):
        invalid = json.loads(build_valid_response())
        invalid['post_text'] = 'Т' * (MAXIMUM_POST_LENGTH + 1)
        if change == 'extra':
            invalid['unknown'] = True
        elif change == 'body':
            invalid['items'][0]['body'] = 'Т' * 1000
        elif change == 'duplicate':
            invalid['items'][1]['news_id'] = invalid['items'][0]['news_id']
        elif change == 'missing':
            del invalid['items']
        else:
            invalid['post_text'] = ''
        try:
            _parse_response(json.dumps(invalid))
        except (ValueError, ValidationError):
            pass
        else:
            raise AssertionError(f'Invalid payload accepted: {change}')
    try:
        OpenAIGeneratedPostPayload(post_text='Т' * 1001, items=compacted.items)
    except ValidationError:
        pass
    else:
        raise AssertionError('Final payload length constraint was weakened')
    print('PASS: long self-review draft, canonical overflow compaction, trailer/IDs, strict final limit and invalid structure')


if __name__ == '__main__':
    asyncio.run(main())
