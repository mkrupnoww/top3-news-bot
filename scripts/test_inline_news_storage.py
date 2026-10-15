"""Existing completion integration checks with real 942-character inline payload."""
import asyncio
from dataclasses import replace

from app.config import Settings
from app.generation.openai_generator import OpenAIGeneratedNewsPayload, OpenAIGeneratedPostPayload, build_top3_post_text
from scripts import test_generation_completion as completion
from scripts.test_inline_news_paragraphs import paragraphs


async def main():
    settings=Settings()
    if settings.app_env!='testing' or settings.db_port!=55432:
        raise RuntimeError('Requires isolated testing PostgreSQL on 55432')
    original=completion.build_generation_result
    def inline_result(selection,*,model_name,post_suffix=''):
        result=original(selection,model_name=model_name,post_suffix=post_suffix)
        items=[OpenAIGeneratedNewsPayload(position=i,news_id=selection.items[i-1].news_id,
                    headline=lead,body=body) for i,(lead,body) in enumerate(paragraphs(),1)]
        assert all(len(item.body)>210 for item in items)
        payload=OpenAIGeneratedPostPayload(post_text=build_top3_post_text(items)+post_suffix,items=items)
        return replace(result,payload=payload,
            model_response=replace(result.model_response,output_text=payload.model_dump_json()))
    completion.build_generation_result=inline_result
    try:
        await completion.main()
        print('PASS: inline paragraphs longer than old 210 limit persisted through production completion SQL; API/Telegram=0')
    finally:
        completion.build_generation_result=original


if __name__=='__main__':
    asyncio.run(main())
