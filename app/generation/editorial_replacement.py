"""Настоящая замена инфоповода: новый batch, новый текст и новая картинка.

Старый выпуск скрыт решением changes_required, но не портится при сбое.
Новый выпуск виден редактору только после завершения обоих этапов.
"""
from dataclasses import dataclass
import json

import asyncpg

from app.db.generation_selection import choose_next_generation_combination, load_generation_batch_selection, load_generation_combination
from app.generation.openai_pipeline import run_reserved_openai_generation
from app.generation.openai_image_factory import create_openai_image_runtime
from app.generation.openai_image_pipeline import run_reserved_openai_image_generation


@dataclass(frozen=True)
class EditorialReplacementResult:
    batch_id: int
    generated_post_id: int
    news_ids: tuple[int, int, int]
    completed: bool = True


async def _source(pool, review_action_id):
    row = await pool.fetchrow(
        """SELECT ra.*, p.batch_id, p.post_status, b.batch_status, b.ranking_run_id,
                  b.publication_date, b.target_telegram_chat_id, b.metadata
           FROM review_actions ra JOIN generated_posts p USING(generated_post_id)
           JOIN publication_batches b USING(batch_id) WHERE ra.review_action_id=$1""", review_action_id,
    )
    if row is None or row['reviewer_type'] != 'human' or row['decision'] != 'changes_required' or row['requested_action'] != 'replace_news':
        raise ValueError('Для замены нужно человеческое решение replace_news')
    return row


async def _prepare_plan(pool, *, source, positions):
    existing = await pool.fetchrow('SELECT * FROM editorial_replacement_requests WHERE review_action_id=$1', source['review_action_id'])
    if existing is not None:
        if tuple(existing['replacement_positions']) != positions:
            raise ValueError('Номера замены отличаются от сохранённого запроса')
        return existing
    if source['post_status'] != 'awaiting_review' or source['batch_status'] != 'awaiting_review':
        raise ValueError('Заменить можно только ещё не опубликованный черновик')
    current = await load_generation_batch_selection(pool, ranking_run_id=source['ranking_run_id'], batch_id=source['batch_id'])
    removed = [current.news_ids[p-1] for p in positions]
    metadata = json.loads(source['metadata']) if isinstance(source['metadata'], str) else source['metadata']
    excluded = set(metadata.get('editorial_excluded_news_ids', [])) | set(removed)
    # Запрещаем весь отвергнутый инфоповод, включая другой источник того же события.
    members = await pool.fetch(
        """SELECT m.news_id FROM ranking_event_members m WHERE m.ranking_run_id=$1
           AND m.ranking_event_id IN (SELECT ranking_event_id FROM news_scores
               WHERE ranking_run_id=$1 AND news_id=ANY($2::bigint[]))""", source['ranking_run_id'], removed,
    )
    excluded.update(r['news_id'] for r in members)
    used = await pool.fetch('SELECT combination_id FROM editorial_replacement_requests WHERE combination_id IN (SELECT combination_id FROM ranking_combinations WHERE ranking_run_id=$1)', source['ranking_run_id'])
    used_sets = await pool.fetch("SELECT metadata->'news_ids' AS ids FROM publication_batches WHERE ranking_run_id=$1", source['ranking_run_id'])
    excluded_sets = tuple(tuple(json.loads(row['ids']) if isinstance(row['ids'], str) else row['ids']) for row in used_sets if row['ids'])
    replacement = await choose_next_generation_combination(
        pool, ranking_run_id=source['ranking_run_id'], current_news_ids=current.news_ids,
        excluded_news_ids=tuple(sorted(excluded)), excluded_combination_ids=tuple(r['combination_id'] for r in used),
        excluded_news_sets=excluded_sets,
    )
    kept = set(current.news_ids) - set(removed)
    if replacement is None or not kept.issubset(replacement.selection.news_ids):
        raise ValueError('В сохранённом ранжировании нет подходящей замены с сохранением остальных новостей. Исходный выпуск не изменён.')
    return await pool.fetchrow(
        """INSERT INTO editorial_replacement_requests
           (review_action_id,source_generated_post_id,combination_id,source_news_ids,
            excluded_news_ids,replacement_positions)
           VALUES($1,$2,$3,$4,$5,$6) RETURNING *""", source['review_action_id'], source['generated_post_id'],
        replacement.combination.combination_id, list(current.news_ids), sorted(excluded), list(positions),
    )


async def _activate(pool, *, source, plan, post_id):
    async with pool.acquire() as connection:
        async with connection.transaction():
            old = await connection.fetchrow('SELECT p.post_status,b.batch_status FROM generated_posts p JOIN publication_batches b USING(batch_id) WHERE p.generated_post_id=$1 FOR UPDATE OF p,b', source['generated_post_id'])
            new = await connection.fetchrow('SELECT p.*,b.batch_status FROM generated_posts p JOIN publication_batches b USING(batch_id) WHERE p.generated_post_id=$1 FOR UPDATE OF p,b', post_id)
            if old['post_status'] != 'awaiting_review' or old['batch_status'] != 'awaiting_review':
                raise ValueError('Исходный выпуск изменился во время замены')
            if new['post_status'] != 'awaiting_review' or new['batch_status'] != 'awaiting_review' or not new['image_path'] or not new['image_sha256']:
                raise ValueError('Новый текст и изображение ещё не готовы')
            # Реальная человеческая реакция наследуется новым batch, не разрешая auto publish.
            await connection.execute("""INSERT INTO review_actions(generated_post_id,reviewer_type,reviewer_telegram_user_id,decision,comment_text,review_details)
                VALUES($1,'human',$2,'comment','Подборка заменена по замечанию редактора.',jsonb_build_object('source_review_action_id',$3::bigint))""",post_id,source['reviewer_telegram_user_id'],source['review_action_id'])
            await connection.execute("UPDATE generated_posts SET post_status='superseded' WHERE batch_id=$1 AND post_status='awaiting_review'",source['batch_id'])
            await connection.execute("UPDATE publication_batches SET batch_status='superseded' WHERE batch_id=$1",source['batch_id'])
            await connection.execute("UPDATE publication_batches SET metadata=jsonb_set(metadata,'{editorial_replacement_pending}','false'::jsonb) WHERE batch_id=$1",new['batch_id'])
            await connection.execute("UPDATE editorial_replacement_requests SET replacement_status='completed',generated_post_id=$2,completed_at=now(),error_message=NULL WHERE review_action_id=$1",source['review_action_id'],post_id)


async def run_editorial_replacement(
    pool: asyncpg.Pool, *, generator, settings, review_action_id: int, positions: tuple[int,...],
    text_runner=run_reserved_openai_generation, image_runner=run_reserved_openai_image_generation,
    image_runtime_factory=create_openai_image_runtime,
) -> EditorialReplacementResult:
    if not positions or any(p not in (1,2,3) for p in positions) or len(set(positions)) != len(positions):
        raise ValueError('Нужны уникальные номера новостей от 1 до 3')
    positions = tuple(sorted(positions))
    # Session lock переживает короткие транзакции, но освобождается при обрыве процесса.
    async with pool.acquire() as lock_connection:
        locked = await lock_connection.fetchval('SELECT pg_try_advisory_lock(21821,$1::integer)', review_action_id)
        if not locked:
            raise ValueError('Эта замена уже выполняется. Дождитесь результата.')
        try:
            source = await _source(pool, review_action_id)
            plan = await _prepare_plan(pool, source=source, positions=positions)
            selection = (await load_generation_combination(pool,ranking_run_id=source['ranking_run_id'],combination_id=plan['combination_id'])).selection
            if plan['replacement_status']=='completed':
                return EditorialReplacementResult(plan['new_batch_id'],plan['generated_post_id'],selection.news_ids)

            async def observe(reservation):
                if not reservation.created_new and plan['new_batch_id'] != reservation.batch_id:
                    raise ValueError('Текстовая reservation принадлежит другому запросу')
                async with pool.acquire() as connection, connection.transaction():
                    await connection.execute('UPDATE editorial_replacement_requests SET new_batch_id=$2 WHERE review_action_id=$1',review_action_id,reservation.batch_id)
                    data = json.dumps(dict(editorial_replacement_pending=True,source_review_action_id=review_action_id,editorial_excluded_news_ids=list(plan['excluded_news_ids'])))
                    await connection.execute('UPDATE publication_batches SET metadata=metadata || $2::jsonb WHERE batch_id=$1',reservation.batch_id,data)

            if plan['generated_post_id'] is None:
                text = await text_runner(pool,generator=generator,ranking_run_id=source['ranking_run_id'],combination_id=plan['combination_id'],publication_date=source['publication_date'],telegram_chat_id=source['target_telegram_chat_id'],reservation_observer=observe)
                post_id = text.generated_post_id
                if post_id is None:
                    # Рестарт после completion текста, до сохранения ссылки в плане.
                    post_id = await pool.fetchval("SELECT generated_post_id FROM generated_posts WHERE batch_id=$1 AND post_status='awaiting_review' ORDER BY version_number DESC LIMIT 1",text.batch_id)
                if post_id is None:
                    raise ValueError('Текстовая генерация не завершена; повторный платный вызов заблокирован')
                await pool.execute("UPDATE editorial_replacement_requests SET generated_post_id=$2,replacement_status='text_ready' WHERE review_action_id=$1",review_action_id,post_id)
            else:
                post_id = plan['generated_post_id']
            new_batch_id = await pool.fetchval('SELECT batch_id FROM generated_posts WHERE generated_post_id=$1',post_id)
            image_runtime = image_runtime_factory(settings)
            try:
                await image_runner(pool,generator=image_runtime.generator,selection=selection,batch_id=new_batch_id,generated_post_id=post_id)
            finally:
                await image_runtime.sdk_client.close()
            await _activate(pool,source=source,plan=plan,post_id=post_id)
            return EditorialReplacementResult(new_batch_id,post_id,selection.news_ids)
        except Exception as error:
            await pool.execute('UPDATE editorial_replacement_requests SET error_message=$2 WHERE review_action_id=$1',review_action_id,f'{type(error).__name__}: {error}')
            raise
        finally:
            await lock_connection.execute('SELECT pg_advisory_unlock(21821,$1::integer)',review_action_id)
