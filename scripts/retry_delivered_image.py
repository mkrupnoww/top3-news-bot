"""Повтор безопасного image stage доставленного поста с отдельным согласованным бюджетом."""
import argparse
import asyncio
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from aiogram import Bot
from aiogram.types import FSInputFile, InputMediaPhoto

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
from app.db.generation_selection import load_generation_batch_selection
from app.generation.openai_image_factory import create_openai_image_generation_runtime
from app.generation.openai_image_pipeline import run_reserved_openai_image_generation
from app.generation.image_generator import OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
from app.publication.telegram_text import prepare_telegram_text
from scripts.retry_failed_text_workflow import RetryBudget, BudgetClient
import app.workflows.daily_production as daily

COMMENT='Нужна нормальная картинка вместо универсальной.'
ISSUES=(COMMENT,)


def image_runtime(settings, budget):
    return create_openai_image_generation_runtime(settings,client_factory=lambda **kw:BudgetClient(budget,**kw))


async def run(pool, *, settings, workflow_id, post_id, directory, maximum,
              check_only=False, refresh_fallback_style=False, runtime_factory=image_runtime, bot_factory=Bot,
              image_runner=run_reserved_openai_image_generation):
    comment='Вернуть насыщенные цвета, контраст и фотографическую реалистичность.' if refresh_fallback_style else COMMENT
    issues=(comment,)
    row=await pool.fetchrow('SELECT * FROM daily_workflow_runs WHERE daily_workflow_run_id=$1',workflow_id)
    if row is None or row['generated_post_id']!=post_id:
        raise ValueError('Неверная связь workflow/post')
    async with daily._daily_workflow_lock(pool,publication_date=row['publication_date']):
        async with pool.acquire() as c,c.transaction():
            post=await c.fetchrow('SELECT p.*,b.batch_status FROM generated_posts p JOIN publication_batches b USING(batch_id) WHERE p.generated_post_id=$1 FOR UPDATE OF p,b',post_id)
            if post is None or post['post_status']!='awaiting_review' or post['batch_status']!='awaiting_review':
                raise ValueError('Повтор разрешён только до одобрения/отклонения/публикации')
            if await c.fetchval('SELECT EXISTS(SELECT 1 FROM publication_attempts WHERE generated_post_id=$1)',post_id):
                raise ValueError('У поста уже есть попытка публикации; требуется отдельная сверка')
            deliveries=await c.fetch("""SELECT r.* FROM review_delivery_attempts r JOIN bot_users u ON u.telegram_user_id=r.telegram_user_id
                 WHERE generated_post_id=$1 AND delivery_status='sent' AND u.is_active AND u.user_role='admin'""",post_id)
            if len(deliveries)!=1:
                raise ValueError('Нужна одна подтверждённая доставка активному владельцу')
            if await c.fetchval("SELECT EXISTS(SELECT 1 FROM image_generation_requests WHERE generated_post_id=$1 AND image_status='reserved')",post_id):
                raise ValueError('Image request уже выполняется; повтор запрещён')
            blocked=await c.fetchval("SELECT EXISTS(SELECT 1 FROM image_generation_requests WHERE generated_post_id=$1 AND image_status='failed' AND error_message LIKE '%moderation_blocked%')",post_id)
            old_fallback = (refresh_fallback_style and post['image_model_name']==settings.openai_image_model
                and (post['image_prompt_version'] or '').startswith('movie_news_image_moderation_fallback_')
                and post['image_prompt_version']!=OPENAI_IMAGE_FALLBACK_PROMPT_VERSION)
            if not blocked or not (post['image_model_name']=='local_static_png_asset' or old_fallback):
                raise ValueError('Ожидалась универсальная картинка или явно выбранный старый fallback после moderation block')
            selection=await load_generation_batch_selection(pool,ranking_run_id=row['ranking_run_id'],batch_id=post['batch_id'])
            if check_only:
                print('Preflight OK: unpublished delivered post, source text/selection preserved, moderation-safe regeneration supported')
                return
            budget=RetryBudget(directory,maximum)
            (directory/'state-before.json').write_text(json.dumps({'workflow':dict(row),'post':dict(post),'deliveries':[dict(x) for x in deliveries]},default=str,ensure_ascii=False,indent=2))
            review_id=await c.fetchval("""INSERT INTO review_actions(generated_post_id,reviewer_type,reviewer_telegram_user_id,
                 decision,requested_action,requires_human_review,comment_text,issues,review_details)
                 VALUES($1,'human',$2,'changes_required','regenerate_image',true,$3,$4::jsonb,$5::jsonb)
                 RETURNING review_action_id""",post_id,deliveries[0]['telegram_user_id'],comment,json.dumps(issues,ensure_ascii=False),
                 json.dumps({'source':'explicit_codex_chat_request','additional_budget_usd':str(maximum),'resume_after':'first_moderation_block','refresh_fallback_style':refresh_fallback_style,'audit_directory':str(directory)},ensure_ascii=False))
            await c.execute("UPDATE publication_batches SET metadata=jsonb_set(metadata,'{editorial_replacement_pending}','true'::jsonb) WHERE batch_id=$1",post['batch_id'])
        runtime=None;bot=None;result=None
        try:
            runtime=runtime_factory(settings,budget)
            runtime.generator.set_moderation_safe_editorial_fallback(True)
            for attempt in range(1,3):
                print(f'Moderation-safe Image API attempt {attempt}/2; additional budget <= ${maximum}',flush=True)
                try:
                    result=await image_runner(pool,generator=runtime.generator,selection=selection,batch_id=post['batch_id'],generated_post_id=post_id,
                         request_kind='regenerate',review_action_id=review_id,editorial_comment=comment,issues=issues)
                    if not result.completed:
                        raise RuntimeError('Image generation не завершена')
                    break
                except Exception:
                    failed=await pool.fetchrow("SELECT image_status,error_type,error_message FROM image_generation_requests WHERE review_action_id=$1 ORDER BY image_generation_id DESC LIMIT 1",review_id)
                    if attempt==1 and failed and failed['image_status']=='failed' and failed['error_type']=='BadRequestError' and 'moderation_blocked' in failed['error_message']:
                        continue
                    raise
            fresh=await pool.fetchrow('SELECT * FROM generated_posts WHERE generated_post_id=$1',post_id)
            if fresh['post_text']!=post['post_text'] or fresh['text_format']!=post['text_format']:
                raise RuntimeError('Текст изменился во время генерации; preview не обновлён')
            path=Path(fresh['image_path'])
            if sha256(path.read_bytes()).hexdigest()!=fresh['image_sha256']:
                raise RuntimeError('SHA-256 нового изображения не совпадает')
            (directory/'image-ready.json').write_text(json.dumps({'image_generation_id':result.image_generation_id,'image_path':str(path),'image_sha256':fresh['image_sha256'],'review_action_id':review_id},ensure_ascii=False,indent=2))
            prepared=prepare_telegram_text(fresh['post_text'],text_format=fresh['text_format'])
            bot=bot_factory(token=settings.telegram_bot_token.get_secret_value())
            delivery=deliveries[0]
            await bot.edit_message_media(chat_id=delivery['telegram_chat_id'],message_id=delivery['telegram_message_id'],
                media=InputMediaPhoto(media=FSInputFile(path),caption=prepared.text,parse_mode=prepared.parse_mode),
                reply_markup=daily.build_review_keyboard(post_id))
            async with pool.acquire() as c,c.transaction():
                await c.execute('UPDATE daily_workflow_runs SET image_generation_id=$2 WHERE daily_workflow_run_id=$1 AND generated_post_id=$3',workflow_id,result.image_generation_id,post_id)
                await c.execute("UPDATE review_actions SET review_details=review_details || jsonb_build_object('operator_image_update_complete',true,'completed_image_generation_id',$2::bigint) WHERE review_action_id=$1",review_id,result.image_generation_id)
                await c.execute("UPDATE publication_batches SET metadata=jsonb_set(metadata,'{editorial_replacement_pending}','false'::jsonb) WHERE batch_id=$1",post['batch_id'])
                await c.execute("UPDATE review_delivery_attempts SET response_payload=response_payload || $2::jsonb WHERE review_delivery_attempt_id=$1",delivery['review_delivery_attempt_id'],json.dumps({'operator_image_update':{'image_generation_id':result.image_generation_id,'review_action_id':review_id}}))
            payload={'post_id':post_id,'image_generation_id':result.image_generation_id,'image_path':str(path),'review_message_id':delivery['telegram_message_id'],'review_action_id':review_id,'new_api_calls':len(budget.calls),'accounted_usd':str(budget.total),'text_unchanged':True,'preview_updated':True,'requires_human_review':True}
            (directory/'result.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
            print(json.dumps(payload,ensure_ascii=False),flush=True)
            return payload
        except Exception:
            # При неудачной генерации прежняя картинка остаётся доступной для review.
            current=await pool.fetchrow('SELECT image_sha256 FROM generated_posts WHERE generated_post_id=$1',post_id)
            if current['image_sha256']==post['image_sha256']:
                await pool.execute("UPDATE publication_batches SET metadata=jsonb_set(metadata,'{editorial_replacement_pending}','false'::jsonb) WHERE batch_id=$1",post['batch_id'])
            raise
        finally:
            if runtime is not None: await runtime.sdk_client.close()
            if bot is not None: await bot.session.close()


async def main(args):
    settings=Settings();pool=await create_database_pool(settings)
    try:
        await run(pool,settings=settings,workflow_id=args.workflow_id,post_id=args.post_id,directory=args.audit_directory,maximum=args.max_usd,check_only=args.check_only,refresh_fallback_style=args.refresh_fallback_style)
    finally: await close_database_pool(pool)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workflow-id',type=int,required=True)
    parser.add_argument('--post-id',type=int,required=True)
    parser.add_argument('--audit-directory',type=Path,required=True)
    parser.add_argument('--max-usd',type=Decimal,required=True)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--refresh-fallback-style',action='store_true',help='Обновить явно устаревший сгенерированный fallback новым промптом')
    asyncio.run(main(parser.parse_args()))
