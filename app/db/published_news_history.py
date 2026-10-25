"""Фактически опубликованные выпуски для семантической проверки фактов."""
from datetime import timedelta

import asyncpg


async def load_published_news_history(pool: asyncpg.Pool, *, as_of, days: int = 7) -> tuple[dict, ...]:
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """SELECT b.batch_id, b.publication_date, p.post_text,
                      bi.position, bi.news_id,
                      COALESCE(NULLIF(n.normalized_title,''),n.raw_title) AS title,
                      COALESCE(NULLIF(n.normalized_summary,''),n.raw_summary) AS summary,
                      n.source_url
               FROM publication_batches b
               JOIN LATERAL (SELECT gp.post_text FROM generated_posts gp
                    WHERE gp.batch_id=b.batch_id AND (gp.post_status='published' OR EXISTS (
                        SELECT 1 FROM publication_attempts pa WHERE pa.generated_post_id=gp.generated_post_id
                        AND pa.attempt_status='published'))
                    ORDER BY gp.version_number DESC LIMIT 1) p ON true
               JOIN batch_items bi USING(batch_id)
               JOIN news_items n USING(news_id)
               WHERE b.publication_date >= $1 AND b.publication_date < $2
                 AND COALESCE(b.metadata->>'technical_test','false') <> 'true'
                 AND (b.batch_status='published' OR EXISTS (
                    SELECT 1 FROM publication_attempts pa JOIN generated_posts gp USING(generated_post_id)
                    WHERE gp.batch_id=b.batch_id AND pa.attempt_status='published'))
               ORDER BY b.publication_date, b.batch_id, bi.position""",
            as_of.date()-timedelta(days=days), as_of.date(),
        )
    issues = {}
    for row in rows:
        issue = issues.setdefault(row['batch_id'], dict(
            batch_id=row['batch_id'], publication_date=row['publication_date'].isoformat(),
            published_text=row['post_text'], items=[],
        ))
        issue['items'].append({key: row[key] for key in ('news_id','position','title','summary','source_url')})
    return tuple(issues.values())


def published_news_ids(history) -> set[int]:
    return {item['news_id'] for issue in history for item in issue['items']}
