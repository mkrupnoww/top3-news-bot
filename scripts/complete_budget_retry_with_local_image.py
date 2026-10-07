"""Завершает budget-stopped retry локальной PNG и обычным Telegram review."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from app.config import Settings
from app.db.pool import create_database_pool, close_database_pool
import app.workflows.daily_production as daily


async def run(workflow_id, batch_id, post_id, directory, *, check_only=False):
    settings = Settings()
    pool = await create_database_pool(settings)
    bot = None
    reopened = False
    try:
        workflow = await daily.load_daily_workflow(pool, daily_workflow_run_id=workflow_id)
        async with daily._daily_workflow_lock(pool, publication_date=workflow.publication_date):
            async with pool.acquire() as c, c.transaction():
                row = await c.fetchrow('SELECT * FROM top3_news.daily_workflow_runs WHERE daily_workflow_run_id=$1 FOR UPDATE', workflow_id)
                if not (row['workflow_status'] == 'failed' and row['error_type'] == 'RuntimeError'
                        and row['error_message'].startswith('Retry budget would be exceeded:')
                        and row['batch_id'] == batch_id and row['generated_post_id'] == post_id):
                    raise ValueError('Expected budget-stopped workflow identity/state required')
                post = await c.fetchrow('SELECT * FROM top3_news.generated_posts WHERE generated_post_id=$1 FOR UPDATE', post_id)
                if not (post['batch_id'] == batch_id and post['post_status'] == 'awaiting_review'
                        and 0 < len(post['post_text']) <= 1000):
                    raise ValueError('Expected completed text awaiting review required')
                if await c.fetchval('SELECT EXISTS (SELECT 1 FROM top3_news.review_delivery_attempts WHERE generated_post_id=$1)', post_id):
                    raise ValueError('Review delivery already attempted; inspect existing delivery first')
                failed = await daily._load_image_request_state(pool,
                    image_generation_id=row['image_generation_id'],
                    expected_batch_id=batch_id, expected_generated_post_id=post_id)
                if failed.image_status != 'failed':
                    raise ValueError('Expected failed image required')
                if not daily.VERSATILE_OPTION_IMAGE_PATH.is_file():
                    raise FileNotFoundError(daily.VERSATILE_OPTION_IMAGE_PATH)
                active = await daily.load_active_daily_workflow_selection(pool, daily_workflow_run_id=workflow_id)
                combination = await daily.load_generation_combination(pool,
                    ranking_run_id=row['ranking_run_id'], combination_id=active.combination_id)
                if check_only:
                    print('Preflight passed: saved text, failed budget retry, no delivery, local PNG available')
                    return
                directory.mkdir(mode=0o700, parents=True, exist_ok=False)
                (directory/'state-before.json').write_text(json.dumps(dict(row), default=str, indent=2))
                await c.execute("""UPDATE top3_news.daily_workflow_runs SET workflow_status='running',
                    current_stage='image', error_type=NULL, error_message=NULL, finished_at=NULL,
                    updated_at=now() WHERE daily_workflow_run_id=$1""", workflow_id)
            reopened = True

            async def observer(reservation):
                await daily.checkpoint_image_reservation(pool, daily_workflow_run_id=workflow_id,
                    image_generation_id=reservation.image_generation_id)

            # Только чтение прежней PNG. OpenAI runtime/client здесь не создаётся.
            result = await daily.run_reserved_openai_image_generation(pool,
                generator=daily._create_versatile_option_generator(), selection=combination.selection,
                batch_id=batch_id, generated_post_id=post_id, request_kind='initial',
                cost_estimator=None, reservation_observer=observer)
            state = await daily.load_generation_workflow_state(pool, batch_id=batch_id)
            if state.image_state_inconsistent or not state.ready_for_review_delivery:
                raise RuntimeError('Local image completion did not produce review-ready state')
            await daily.mark_daily_workflow_stage(pool, daily_workflow_run_id=workflow_id, stage='review_delivery')
            bot = daily.Bot(token=settings.telegram_bot_token.get_secret_value())
            delivery = await daily.deliver_generated_post_to_reviewers(pool, bot=bot,
                generated_post_id=post_id, reply_markup=daily.build_review_keyboard(post_id))
            daily._validate_review_delivery(delivery)
            completed = await daily.complete_daily_workflow(pool, daily_workflow_run_id=workflow_id)
            payload = {'workflow': asdict(completed), 'image_generation_id': result.image_generation_id,
                       'delivery': asdict(delivery), 'additional_paid_calls': 0}
            (directory/'result.json').write_text(json.dumps(payload, default=str, indent=2))
            print(json.dumps(payload, default=str), flush=True)
    except Exception as error:
        if reopened:
            await daily._fail_workflow_after_error(pool, daily_workflow_run_id=workflow_id, error=error)
        raise
    finally:
        if bot is not None:
            await bot.session.close()
        await close_database_pool(pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow-id', type=int, required=True)
    parser.add_argument('--batch-id', type=int, required=True)
    parser.add_argument('--post-id', type=int, required=True)
    parser.add_argument('--audit-directory', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.workflow_id, args.batch_id, args.post_id, args.audit_directory,
                    check_only=args.check_only))
