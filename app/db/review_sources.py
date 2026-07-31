from dataclasses import dataclass
from datetime import datetime

import asyncpg


@dataclass(frozen=True, slots=True)
class ReviewSourceItem:
    """Источник одной новости внутри выпуска TOP-3."""

    position: int
    news_id: int
    title: str
    source_name: str
    source_url: str
    source_published_at: datetime | None
    selection_reason: str | None
    primary_image_url: str | None
    image_credit: str | None


async def get_review_sources(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
) -> tuple[ReviewSourceItem, ...]:
    """
    Возвращает три новости, связанные с generated_post.

    Никаких изменений в PostgreSQL не выполняется.
    """

    query = """
        SELECT
            bi.position,
            n.news_id,
            COALESCE(
                NULLIF(n.normalized_title, ''),
                NULLIF(n.raw_title, ''),
                'Без заголовка'
            ) AS title,
            s.source_name,
            n.source_url,
            n.source_published_at,
            bi.selection_reason,
            n.primary_image_url,
            n.image_credit
        FROM generated_posts AS p
        JOIN batch_items AS bi
            ON bi.batch_id = p.batch_id
        JOIN news_items AS n
            ON n.news_id = bi.news_id
        JOIN sources AS s
            ON s.source_id = n.source_id
        WHERE p.generated_post_id = $1
        ORDER BY bi.position
    """

    async with pool.acquire() as connection:
        post_exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM generated_posts
                WHERE generated_post_id = $1
            )
            """,
            generated_post_id,
        )

        if not post_exists:
            raise LookupError(
                "Пост не найден: "
                f"generated_post_id={generated_post_id}"
            )

        records = await connection.fetch(
            query,
            generated_post_id,
        )

    if len(records) != 3:
        raise ValueError(
            "Для выпуска должно быть связано ровно "
            "три новости: "
            f"generated_post_id={generated_post_id}, "
            f"news_count={len(records)}"
        )

    positions = {
        record["position"]
        for record in records
    }

    if positions != {1, 2, 3}:
        raise ValueError(
            "У выпуска обнаружен некорректный набор "
            f"позиций: {sorted(positions)}"
        )

    return tuple(
        ReviewSourceItem(
            position=record["position"],
            news_id=record["news_id"],
            title=record["title"],
            source_name=record["source_name"],
            source_url=record["source_url"],
            source_published_at=(
                record["source_published_at"]
            ),
            selection_reason=(
                record["selection_reason"]
            ),
            primary_image_url=(
                record["primary_image_url"]
            ),
            image_credit=record["image_credit"],
        )
        for record in records
    )