"""Реальный PostgreSQL + fake Telegram. Запуск ТОЛЬКО в изолированной БД cloud-001."""
import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from aiogram.enums import ChatType
from aiogram.exceptions import TelegramForbiddenError
from aiogram.methods import SendPhoto
from PIL import Image
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.db.auto_publications import claim_auto_post, find_due_auto_post, get_auto_plan
from app.db.pool import create_database_pool, close_database_pool
from app.db.review_interactions import record_review_button_interaction
from app.db.review_queue import record_human_review_decision
import app.publication.auto_service as service

NOW = datetime(2090, 3, 12, 12, tzinfo=UTC)
EARLY = NOW - timedelta(hours=4)
CHANNELS = (-1009000000001, -1009000000002)
REVIEWER = 990000001


class FakeSession:
    async def close(self):
        pass


class FakeBot:
    sends = []
    outcomes = {}

    def __init__(self, *, token):
        self.session = FakeSession()

    async def get_chat(self, chat_id):
        return SimpleNamespace(type=ChatType.CHANNEL)

    async def edit_message_reply_markup(self, **kwargs):
        pass

    async def send_photo(self, **kwargs):
        type(self).sends.append(kwargs)
        outcome = type(self).outcomes.pop(kwargs['chat_id'], None)
        if outcome is not None:
            raise outcome
        await asyncio.sleep(0)  # Даёт конкурирующему runner возможность работать.
        return SimpleNamespace(message_id=len(type(self).sends), chat=SimpleNamespace(id=kwargs['chat_id']))


async def main():
    logging.basicConfig(level=logging.CRITICAL)  # Ожидаемые synthetic errors проверяем ниже.
    base = get_settings()
    if base.app_env != 'testing' or base.db_port != 55432 or base.db_name != 'top3_auto_test':
        raise RuntimeError('Тест разрешён только APP_ENV=testing, DB_PORT=55432, DB_NAME=top3_auto_test')
    settings = base.model_copy(update=dict(auto_publication_enabled=True, channel_prod_id=CHANNELS[0], channel_prod_vip_id=CHANNELS[1]))
    pool = await create_database_pool(base)
    checks = 0
    with TemporaryDirectory() as tmp:
        image = Path(tmp) / 'test.png'
        Image.new('RGB', (32, 32), 'white').save(image)
        digest = sha256(image.read_bytes()).hexdigest()

        async def fixture(*, delivery='sent', sent_at=EARLY, date=NOW.date(), status='awaiting_review', workflow=True):
            FakeBot.sends = []
            FakeBot.outcomes = {}
            await pool.execute('TRUNCATE daily_workflow_runs, publication_batches CASCADE')
            await pool.execute("INSERT INTO bot_users(telegram_user_id, display_name, user_role) VALUES($1,'Тест','admin') ON CONFLICT DO NOTHING", REVIEWER)
            batch = await pool.fetchval("INSERT INTO publication_batches(publication_date, edition, batch_status, target_telegram_chat_id) VALUES($1,1,$2,$3) RETURNING batch_id", date, status, -1009000000099)
            post = await pool.fetchval("""INSERT INTO generated_posts(batch_id,version_number,post_status,post_text,text_format,image_path,image_sha256,created_at)
                VALUES($1,1,$2,'**Содержательный тестовый выпуск**: три новости.','markdown',$3,$4,$5) RETURNING generated_post_id""", batch, status, str(image), digest, EARLY)
            if workflow:
                await pool.execute("""INSERT INTO daily_workflow_runs(publication_date,workflow_version,workflow_status,current_stage,as_of,target_telegram_chat_id,batch_id,generated_post_id,started_at,finished_at)
                    VALUES($1,'auto-test','awaiting_review','awaiting_review',$2,$3,$4,$5,$2,$2)""", date, EARLY, -1009000000099, batch, post)
            if delivery is not None:
                # sent с настоящим state constraint; другие состояния строим через штатную заготовку.
                await pool.execute("""INSERT INTO review_delivery_attempts(generated_post_id,telegram_user_id,attempt_number,delivery_status,telegram_chat_id,telegram_message_id,sent_at)
                    VALUES($1,$2,1,'sent',$2,100,$3)""", post, REVIEWER, sent_at)
                if delivery == 'failed':
                    await pool.execute("""UPDATE review_delivery_attempts SET delivery_status='failed',telegram_message_id=NULL,sent_at=NULL,failed_at=now(),error_type='Synthetic',error_message='Synthetic' WHERE generated_post_id=$1""", post)
                elif delivery == 'unknown':
                    await pool.execute("UPDATE review_delivery_attempts SET delivery_status='unknown',error_type='Synthetic',error_message='Synthetic' WHERE generated_post_id=$1", post)
            return post, batch

        async def run(**kwargs):
            return await service.run_auto_publication(pool, settings=settings, now=NOW, **kwargs)

        with patch.object(service, 'Bot', FakeBot):
            post, batch = await fixture()
            r = await service.run_auto_publication(pool, settings=settings, now=NOW-timedelta(microseconds=1))
            assert r.generated_post_id is None and not FakeBot.sends
            r = await service.run_auto_publication(pool, settings=settings.model_copy(update={'auto_publication_enabled':False}), now=NOW)
            assert r.skipped_reason == 'disabled'
            r = await run(dry_run=True)
            assert r.skipped_reason == 'dry_run' and not FakeBot.sends
            assert await get_auto_plan(pool, now=NOW) is None
            checks += 1

            r = await run()
            assert r.published and set(r.channel_statuses) == set(CHANNELS)
            assert [x['chat_id'] for x in FakeBot.sends] == list(CHANNELS)
            assert FakeBot.sends[0]['caption'] == FakeBot.sends[1]['caption']
            assert FakeBot.sends[0]['photo'].path == FakeBot.sends[1]['photo'].path
            assert await pool.fetchval('SELECT batch_status FROM publication_batches WHERE batch_id=$1', batch) == 'published'
            assert not await pool.fetchval("SELECT count(*) FROM review_actions WHERE generated_post_id=$1 AND reviewer_type='human'", post)
            assert (await run()).published and len(FakeBot.sends) == 2
            checks += 1

            for action in ('approve','change','reject'):
                post, batch = await fixture()
                await record_review_button_interaction(pool, generated_post_id=post, telegram_user_id=REVIEWER, action=action)
                assert (await run()).generated_post_id is None and not FakeBot.sends
                # Сохраняется обычное поведение решения: approve не отправляет в каналы.
                if action != 'change':
                    result = await record_human_review_decision(pool, generated_post_id=post, reviewer_telegram_user_id=REVIEWER, decision=action)
                    assert result.post_status == ('approved' if action=='approve' else 'rejected')
                checks += 1

            for args in ({'delivery':None}, {'delivery':'failed'}, {'delivery':'unknown'}, {'sent_at':NOW}, {'date':NOW.date()-timedelta(days=1)}, {'status':'approved'}, {'workflow':False}):
                await fixture(**args)
                assert (await run()).generated_post_id is None and not FakeBot.sends, args
                checks += 1

            # Другая версия того же batch уже получила человеческое решение.
            post, batch = await fixture()
            old = await pool.fetchval("INSERT INTO generated_posts(batch_id, version_number, post_status, post_text) VALUES($1,2,'superseded','old') RETURNING generated_post_id", batch)
            await pool.execute("INSERT INTO review_actions(generated_post_id,reviewer_type,reviewer_telegram_user_id,decision) VALUES($1,'human',$2,'comment')", old, REVIEWER)
            assert (await run()).generated_post_id is None
            checks += 1

            post, batch = await fixture()
            FakeBot.outcomes[CHANNELS[1]] = TelegramForbiddenError(method=SendPhoto(chat_id=CHANNELS[1], photo='fake'), message='Synthetic forbidden')
            r = await run()
            assert not r.published and r.channel_statuses == {CHANNELS[0]:'published', CHANNELS[1]:'failed'}
            assert await pool.fetchval('SELECT batch_status FROM publication_batches WHERE batch_id=$1', batch) == 'publishing'
            assert (await run()).published
            assert [x['chat_id'] for x in FakeBot.sends] == [*CHANNELS, CHANNELS[1]]
            checks += 1

            post, batch = await fixture()
            FakeBot.outcomes[CHANNELS[0]] = TimeoutError('Synthetic lost response')
            r = await run()
            assert not r.published and r.channel_statuses == {CHANNELS[0]:'unknown', CHANNELS[1]:'published'}
            await run()
            assert len(FakeBot.sends) == 2  # Даже unknown без message_id не отправляется ещё раз.
            checks += 1

            post, batch = await fixture()
            original_finish = service.finish_auto_delivery
            async def fail_first_finalization(pool, **kwargs):
                if kwargs['status'] == 'published' and len(FakeBot.sends) == 1:
                    raise RuntimeError('Synthetic database failure after Telegram success')
                await original_finish(pool, **kwargs)
            with patch.object(service, 'finish_auto_delivery', fail_first_finalization):
                r = await run()
            assert r.channel_statuses == {CHANNELS[0]:'unknown', CHANNELS[1]:'published'}
            assert await pool.fetchval("SELECT telegram_message_id FROM publication_attempts WHERE generated_post_id=$1 AND telegram_chat_id=$2", post, CHANNELS[0]) == 1
            await run()
            assert len(FakeBot.sends) == 2
            checks += 1

            post, batch = await fixture()
            results = await asyncio.gather(run(), run())
            assert any(r.published for r in results) and len(FakeBot.sends) == 2
            checks += 1

            # Click уже начался: claim ждёт p/b и обязан увидеть новый comment после ожидания.
            post, batch = await fixture()
            source = await find_due_auto_post(pool, now=NOW)
            async with pool.acquire() as connection:
                async with connection.transaction():
                    await connection.fetchrow('SELECT * FROM generated_posts WHERE generated_post_id=$1 FOR UPDATE', post)
                    claiming = asyncio.create_task(claim_auto_post(pool, now=NOW, source_post=source, target_chat_ids=CHANNELS))
                    await asyncio.sleep(0.1)
                    await connection.execute("INSERT INTO review_actions(generated_post_id,reviewer_type,reviewer_telegram_user_id,decision) VALUES($1,'human',$2,'comment')", post, REVIEWER)
                assert await claiming is None
            assert await get_auto_plan(pool, now=NOW) is None
            checks += 1

            # Обратная гонка: после claim кнопка уже не может поменять решение.
            post, batch = await fixture()
            source = await find_due_auto_post(pool, now=NOW)
            plan = await claim_auto_post(pool, now=NOW, source_post=source, target_chat_ids=CHANNELS)
            assert plan is not None
            try:
                await record_review_button_interaction(pool, generated_post_id=post, telegram_user_id=REVIEWER, action='change')
            except ValueError:
                pass
            else:
                raise AssertionError('Click changed claimed post')
            # Имитируем остановку процесса после сохранения started перед sendPhoto.
            await pool.execute("INSERT INTO publication_attempts(generated_post_id,attempt_number,attempt_status,telegram_chat_id) VALUES($1,1,'started',$2)", post, CHANNELS[0])
            r = await run()
            assert not r.published and [x['chat_id'] for x in FakeBot.sends] == [CHANNELS[1]]
            checks += 1

            # Проверка файлов до claim: испорченное изображение не блокирует ручной review.
            post, batch = await fixture()
            await pool.execute("UPDATE generated_posts SET image_sha256=$2 WHERE generated_post_id=$1", post, '0'*64)
            try:
                await run()
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid image accepted')
            assert await get_auto_plan(pool, now=NOW) is None and not FakeBot.sends
            checks += 1

        # Настоящий handler: change записывается до ожидания комментария; /cancel не стирает БД.
        from app.bot.handlers import handle_review_callback
        post, batch = await fixture()
        class FakeState:
            async def clear(self): pass
            async def set_state(self, state): pass
            async def update_data(self, **kwargs): pass
        class FakeMessage:
            async def edit_reply_markup(self, **kwargs): pass
            async def answer(self, *args, **kwargs): pass
        class FakeCallback:
            data = f'review:change:{post}'
            from_user = SimpleNamespace(id=REVIEWER)
            message = FakeMessage()
            async def answer(self, *args, **kwargs): pass
        await handle_review_callback(FakeCallback(), pool, FakeState())
        assert await pool.fetchval("SELECT count(*) FROM review_actions WHERE generated_post_id=$1 AND review_details->>'button_clicked'='change'", post) == 1
        assert await find_due_auto_post(pool, now=NOW) is None
        checks += 1

        # Конфигурация валидируется без подключения к Telegram.
        for invalid in ({'CHANNEL_PROD_VIP_ID':CHANNELS[0]}, {'CHANNEL_PROD_VIP_ID':None}, {'CHANNEL_PROD_ID':123}):
            kwargs = dict(APP_ENV='testing',TELEGRAM_BOT_USERNAME='test',TELEGRAM_BOT_TOKEN='123456:fake',TELEGRAM_CHANNEL_ID=-1009000000099, DB_NAME='test',DB_USER='test',DB_PASSWORD='test', AUTO_PUBLICATION_ENABLED=True,CHANNEL_PROD_ID=CHANNELS[0],CHANNEL_PROD_VIP_ID=CHANNELS[1])
            kwargs.update(invalid)
            try:
                Settings(_env_file=None, **kwargs)
            except ValidationError:
                pass
            else:
                raise AssertionError('Invalid config accepted')
        checks += 1
        await pool.execute('TRUNCATE daily_workflow_runs, publication_batches CASCADE')
    await close_database_pool(pool)
    print(f'PASS: {checks} PostgreSQL auto-publication scenarios; real Telegram/OpenAI calls=0')


if __name__ == '__main__':
    asyncio.run(main())
