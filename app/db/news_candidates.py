from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import asyncpg


@dataclass(frozen=True, slots=True)
class NewsCandidate:
    """Одна новость, попавшая во временное окно отбора."""

    news_id: int
    source_id: int
    source_code: str
    source_name: str
    collection_priority: int
    processing_status: str
    title: str
    summary: str | None
    author_name: str | None
    source_published_at: datetime
    age_hours: float
    source_url: str
    primary_image_url: str | None


@dataclass(frozen=True, slots=True)
class CandidateSelectionResult:
    """Результат выборки кандидатов из PostgreSQL."""

    window_start: datetime
    window_end: datetime
    window_hours: float
    candidates: tuple[NewsCandidate, ...]


def _normalize_datetime(
    value: datetime,
) -> datetime:
    """Приводит дату с часовым поясом к UTC."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Дата as_of должна содержать часовой пояс."
        )

    return value.astimezone(timezone.utc)


async def select_news_candidates(
    pool: asyncpg.Pool,
    *,
    as_of: datetime,
    window_hours: float = 24.0,
    limit: int = 500,
    source_codes: tuple[str, ...] | None = None,
) -> CandidateSelectionResult:
    """
    Выбирает новости из заданного временного окна.

    PostgreSQL изменён не будет.
    """

    normalized_as_of = _normalize_datetime(
        as_of
    )

    if window_hours <= 0:
        raise ValueError(
            "window_hours должен быть больше нуля."
        )

    if limit <= 0:
        raise ValueError(
            "limit должен быть больше нуля."
        )

    if limit > 5000:
        raise ValueError(
            "limit не может превышать 5000."
        )

    normalized_source_codes: tuple[str, ...] | None

    if source_codes:
        normalized_source_codes = tuple(
            sorted(
                {
                    source_code.strip().lower()
                    for source_code in source_codes
                    if source_code.strip()
                }
            )
        )

        if not normalized_source_codes:
            normalized_source_codes = None
    else:
        normalized_source_codes = None

    window_start = (
        normalized_as_of
        - timedelta(hours=window_hours)
    )

    query = """
        SELECT
            n.news_id,
            s.source_id,
            s.source_code,
            s.source_name,
            s.collection_priority,
            n.processing_status,
            COALESCE(
                NULLIF(n.normalized_title, ''),
                NULLIF(n.raw_title, ''),
                'Без заголовка'
            ) AS title,
            COALESCE(
                NULLIF(n.normalized_summary, ''),
                NULLIF(n.raw_summary, '')
            ) AS summary,
            n.author_name,
            n.source_published_at,
            (
                EXTRACT(
                    EPOCH FROM (
                        $2::timestamptz
                        - n.source_published_at
                    )
                )
                / 3600.0
            )::double precision AS age_hours,
            n.source_url,
            n.primary_image_url
        FROM news_items AS n
        JOIN sources AS s
            ON s.source_id = n.source_id
        WHERE s.is_active = true
          AND n.processing_status IN (
              'collected',
              'candidate'
          )
          AND n.source_published_at IS NOT NULL
          AND n.source_published_at >= $1
          AND n.source_published_at <= $2
          AND (
              $4::text[] IS NULL
              OR s.source_code = ANY($4::text[])
          )
        ORDER BY
            n.source_published_at DESC,
            s.collection_priority DESC,
            n.news_id DESC
        LIMIT $3
    """

    async with pool.acquire() as connection:
        records = await connection.fetch(
            query,
            window_start,
            normalized_as_of,
            limit,
            (
                list(normalized_source_codes)
                if normalized_source_codes
                else None
            ),
        )

    candidates = tuple(
        NewsCandidate(
            news_id=record["news_id"],
            source_id=record["source_id"],
            source_code=record["source_code"],
            source_name=record["source_name"],
            collection_priority=(
                record["collection_priority"]
            ),
            processing_status=(
                record["processing_status"]
            ),
            title=record["title"],
            summary=record["summary"],
            author_name=record["author_name"],
            source_published_at=(
                record["source_published_at"]
            ),
            age_hours=record["age_hours"],
            source_url=record["source_url"],
            primary_image_url=(
                record["primary_image_url"]
            ),
        )
        for record in records
    )

    return CandidateSelectionResult(
        window_start=window_start,
        window_end=normalized_as_of,
        window_hours=window_hours,
        candidates=candidates,
    )