"""Само нажатие кнопки сохраняется до FSM и любых платных операций."""

import asyncpg


async def record_review_button_interaction(
    pool: asyncpg.Pool, *, generated_post_id: int,
    telegram_user_id: int, action: str,
) -> None:
    if action not in {"approve", "change", "reject"}:
        raise ValueError("Неизвестная кнопка проверки")
    async with pool.acquire() as connection:
        async with connection.transaction():
            reviewer = await connection.fetchval(
                """SELECT telegram_user_id FROM bot_users
                   WHERE telegram_user_id=$1 AND is_active
                     AND user_role IN ('admin', 'reviewer') FOR SHARE""",
                telegram_user_id,
            )
            if reviewer is None:
                raise PermissionError("Недостаточно прав для проверки")
            post = await connection.fetchrow(
                """SELECT p.post_status, b.batch_status FROM generated_posts p
                   JOIN publication_batches b USING (batch_id)
                   WHERE p.generated_post_id=$1 FOR UPDATE OF p, b""",
                generated_post_id,
            )
            if post is None or post['post_status'] != 'awaiting_review' or post['batch_status'] != 'awaiting_review':
                raise ValueError("Пост уже не ожидает проверки; возможно, началась публикация")
            # comment не меняет прежнюю семантику approve/reject/changes_required.
            await connection.execute(
                """INSERT INTO review_actions
                   (generated_post_id, reviewer_type, reviewer_telegram_user_id,
                    decision, requires_human_review, comment_text, review_details)
                   VALUES ($1, 'human', $2, 'comment', false,
                           'Нажата inline-кнопка проверки; автопубликация отменена.',
                           jsonb_build_object('button_clicked', $3::text))""",
                generated_post_id, telegram_user_id, action,
            )
