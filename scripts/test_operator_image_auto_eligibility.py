"""Операторская image-правка сохраняет auto только после готового PNG/preview; кнопка отменяет."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime,UTC

from app.config import get_settings
from app.db.pool import create_database_pool,close_database_pool
from app.db.auto_publications import find_due_auto_post,claim_auto_post
from app.db.review_interactions import record_review_button_interaction


class TxPool:
    def __init__(self,c):self.c=c
    @asynccontextmanager
    async def acquire(self):yield self.c


async def main():
    settings=get_settings()
    if settings.app_env!='testing' or settings.db_port!=55432 or settings.db_name!='top3_image_retry_test':
        raise RuntimeError('Только isolated cloud-001:55432/top3_image_retry_test')
    pool=await create_database_pool(settings)
    async with pool.acquire() as c:
        for case,expected in [('complete',True),('pending',False),('in_progress',False),('wrong_hash',False),('button',False)]:
            tx=c.transaction();await tx.start()
            try:
                await c.execute("UPDATE review_actions SET review_details=review_details || '{\"operator_image_update_complete\":true}'::jsonb WHERE review_action_id=127")
                if case=='pending':await c.execute("UPDATE review_actions SET review_details=review_details-'operator_image_update_complete' WHERE review_action_id=127")
                if case=='in_progress':await c.execute("UPDATE publication_batches SET metadata=jsonb_set(metadata,'{editorial_replacement_pending}','true'::jsonb) WHERE batch_id=176")
                if case=='wrong_hash':await c.execute("UPDATE generated_posts SET image_sha256=$1 WHERE generated_post_id=180",'0'*64)
                if case=='button':
                    reviewer=await c.fetchval('SELECT reviewer_telegram_user_id FROM review_actions WHERE review_action_id=127')
                    await record_review_button_interaction(TxPool(c),generated_post_id=180,telegram_user_id=reviewer,action='change')
                now=datetime(2026,11,5,12,tzinfo=UTC)
                source=await find_due_auto_post(TxPool(c),now=now)
                assert (source is not None)==expected,case
                if source is not None:
                    assert source['generated_post_id']==180
                    assert await claim_auto_post(TxPool(c),now=now,source_post=source,target_chat_ids=(-1009000000001,-1009000000002)) is not None
                print(case,'OK')
            finally:await tx.rollback()
    await close_database_pool(pool)
    print('PASS: complete operator image allows auto; pending/mismatched/button states block it; API calls=0')


if __name__=='__main__':asyncio.run(main())
