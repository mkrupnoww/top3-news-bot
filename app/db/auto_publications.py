"""Атомарный выбор выпуска и независимые попытки отправки в два канала."""

from datetime import UTC, datetime, time
import json

import asyncpg


def publication_deadline(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Нужно время с часовым поясом")
    return datetime.combine(now.astimezone(UTC).date(), time(12), UTC)


_DUE_POST_SQL = """
    SELECT p.*, b.batch_status, b.publication_date
    FROM daily_workflow_runs d
    JOIN generated_posts p ON p.generated_post_id=d.generated_post_id
    JOIN publication_batches b ON b.batch_id=p.batch_id AND b.batch_id=d.batch_id
    WHERE d.publication_date=$1 AND b.publication_date=$1
      AND d.workflow_status='awaiting_review'
      AND p.post_status='awaiting_review' AND b.batch_status='awaiting_review'
      AND p.created_at < $2
      AND EXISTS (SELECT 1 FROM review_delivery_attempts r
                  WHERE r.generated_post_id=p.generated_post_id
                    AND r.delivery_status='sent' AND r.sent_at < $2)
      AND NOT EXISTS (SELECT 1 FROM generated_posts bp
                      JOIN review_actions ra USING (generated_post_id)
                      WHERE bp.batch_id=b.batch_id AND ra.reviewer_type='human')
      AND NOT EXISTS (SELECT 1 FROM generated_posts bp
                      JOIN publication_attempts pa USING (generated_post_id)
                      WHERE bp.batch_id=b.batch_id)
"""


async def find_due_auto_post(pool: asyncpg.Pool, *, now: datetime):
    """Только сегодняшний готовый daily-пост, доставленный человеку до полудня."""
    deadline = publication_deadline(now)
    if now < deadline:
        return None
    async with pool.acquire() as connection:
        return await connection.fetchrow(_DUE_POST_SQL, deadline.date(), deadline)


async def get_auto_plan(pool: asyncpg.Pool, *, now: datetime):
    deadline = publication_deadline(now)
    if now < deadline:
        return None
    async with pool.acquire() as connection:
        return await connection.fetchrow(
            'SELECT * FROM daily_auto_publications WHERE publication_date=$1', deadline.date(),
        )


async def claim_auto_post(
    pool: asyncpg.Pool, *, now: datetime, source_post,
    target_chat_ids: tuple[int, int],
):
    """Блокировка p/b общая с кнопками и ручной публикацией; побеждает один."""
    if len(set(target_chat_ids)) != 2 or not all(str(c).startswith('-100') for c in target_chat_ids):
        raise ValueError("Нужны два разных полных ID каналов")
    deadline = publication_deadline(now)
    if now < deadline:
        return None
    async with pool.acquire() as connection:
        async with connection.transaction():
            post = await connection.fetchrow(
                _DUE_POST_SQL + ' FOR UPDATE OF p, b', deadline.date(), deadline,
            )
            if post is None:
                return None
            # Новый READ COMMITTED snapshot после получения блокировок:
            # конкурентный click мог записать comment, не меняя статусы p/b.
            touched = await connection.fetchval(
                """SELECT EXISTS (SELECT 1 FROM generated_posts bp
                   JOIN review_actions ra USING (generated_post_id)
                   WHERE bp.batch_id=$1 AND ra.reviewer_type='human')""", post['batch_id'],
            )
            if touched:
                return None
            # Исключаем отправку старой версии при изменении после preflight.
            for field in ('generated_post_id', 'post_text', 'text_format', 'image_path', 'image_sha256'):
                if post[field] != source_post[field]:
                    raise ValueError("Пост изменился после подготовки автопубликации")
            plan = await connection.fetchrow(
                """INSERT INTO daily_auto_publications
                   (generated_post_id, publication_date, target_chat_ids, deadline_at)
                   VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING RETURNING *""",
                post['generated_post_id'], deadline.date(), list(target_chat_ids), deadline,
            )
            if plan is None:
                return None
            await connection.execute(
                "UPDATE generated_posts SET post_status='approved' WHERE generated_post_id=$1",
                post['generated_post_id'],
            )
            await connection.execute(
                """UPDATE publication_batches SET batch_status='publishing',
                   approved_at=now(), approved_by_telegram_user_id=NULL, error_message=NULL
                   WHERE batch_id=$1""", post['batch_id'],
            )
            return plan


async def reserve_auto_delivery(
    pool: asyncpg.Pool, *, generated_post_id: int, chat_id: int, request_payload: dict,
):
    """started/unknown/published данного канала блокируют повторную отправку."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            post = await connection.fetchrow(
                'SELECT * FROM generated_posts WHERE generated_post_id=$1 FOR UPDATE', generated_post_id,
            )
            plan = await connection.fetchrow(
                'SELECT * FROM daily_auto_publications WHERE generated_post_id=$1', generated_post_id,
            )
            if plan is None or chat_id not in plan['target_chat_ids']:
                raise ValueError("Канал не входит в зафиксированный план")
            if post['post_text'] != request_payload['source_post_text'] or post['text_format'] != request_payload['source_text_format']:
                raise ValueError("Исходный текст или его формат изменился")
            if post['image_sha256'] != request_payload['photo']['sha256'] or post['image_path'] != request_payload['photo']['path']:
                raise ValueError("Изображение изменилось")
            blocking = await connection.fetchval(
                """SELECT publication_attempt_id FROM publication_attempts
                   WHERE generated_post_id=$1 AND telegram_chat_id=$2
                     AND attempt_status IN ('started','unknown','published') LIMIT 1""",
                generated_post_id, chat_id,
            )
            if blocking is not None:
                return None
            return await connection.fetchval(
                """INSERT INTO publication_attempts
                   (generated_post_id, attempt_number, attempt_status, telegram_chat_id, request_payload)
                   SELECT $1, COALESCE(MAX(attempt_number),0)+1, 'started', $2, $3::jsonb
                   FROM publication_attempts WHERE generated_post_id=$1
                   RETURNING publication_attempt_id""",
                generated_post_id, chat_id, json.dumps(request_payload, ensure_ascii=False),
            )


async def finish_auto_delivery(
    pool: asyncpg.Pool, *, attempt_id: int, status: str,
    message_id: int | None = None, response: dict | None = None, error: str | None = None,
) -> None:
    if status not in {'published', 'failed', 'unknown'}:
        raise ValueError("Неизвестный статус отправки")
    async with pool.acquire() as connection:
        updated = await connection.fetchval(
            """UPDATE publication_attempts SET attempt_status=$2, telegram_message_id=$3,
               response_payload=$4::jsonb, error_message=$5, finished_at=now()
               WHERE publication_attempt_id=$1 AND attempt_status='started'
               RETURNING publication_attempt_id""",
            attempt_id, status, message_id, json.dumps(response or {}), error,
        )
        if updated is None:
            raise ValueError("Попытка уже завершена или не найдена")


async def auto_delivery_statuses(pool: asyncpg.Pool, *, generated_post_id: int) -> dict[int, str]:
    async with pool.acquire() as connection:
        records = await connection.fetch(
            """SELECT DISTINCT ON (telegram_chat_id) telegram_chat_id, attempt_status
               FROM publication_attempts WHERE generated_post_id=$1
               ORDER BY telegram_chat_id, attempt_number DESC""", generated_post_id,
        )
    return {row['telegram_chat_id']: row['attempt_status'] for row in records}


async def finalize_auto_publication(pool: asyncpg.Pool, *, generated_post_id: int) -> bool:
    """Весь выпуск published только после двух подтверждённых отправок."""
    async with pool.acquire() as connection:
        async with connection.transaction():
            post = await connection.fetchrow(
                """SELECT p.batch_id FROM generated_posts p JOIN publication_batches b USING(batch_id)
                   WHERE p.generated_post_id=$1 FOR UPDATE OF p,b""", generated_post_id,
            )
            complete = await connection.fetchval(
                """SELECT NOT EXISTS (
                     SELECT 1 FROM daily_auto_publications a, unnest(a.target_chat_ids) AS target(chat_id)
                     WHERE a.generated_post_id=$1 AND NOT EXISTS (
                       SELECT 1 FROM publication_attempts pa WHERE pa.generated_post_id=$1
                       AND pa.telegram_chat_id=target.chat_id AND pa.attempt_status='published'))
                   AND EXISTS (SELECT 1 FROM daily_auto_publications WHERE generated_post_id=$1)""",
                generated_post_id,
            )
            if complete:
                await connection.execute("UPDATE generated_posts SET post_status='published' WHERE generated_post_id=$1", generated_post_id)
                await connection.execute("UPDATE publication_batches SET batch_status='published', published_at=now(), error_message=NULL WHERE batch_id=$1", post['batch_id'])
                await connection.execute("UPDATE daily_auto_publications SET published_at=COALESCE(published_at, now()) WHERE generated_post_id=$1", generated_post_id)
            else:
                await connection.execute("UPDATE publication_batches SET error_message='Автопубликация завершена не во всех каналах; проверьте publication_attempts' WHERE batch_id=$1", post['batch_id'])
            return bool(complete)
