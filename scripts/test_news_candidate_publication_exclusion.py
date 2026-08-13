import asyncio
from datetime import date, datetime, timezone

import asyncpg

from app.config import get_settings
from app.db.news_candidates import (
    select_news_candidates,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


TEST_PUBLICATION_DATE = date(
    2099,
    12,
    31,
)


class _SingleConnectionAcquire:
    """
    Async context manager для одной уже
    открытой asyncpg connection.
    """

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    async def __aenter__(
        self,
    ) -> asyncpg.Connection:
        return self._connection

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None


class _SingleConnectionPool:
    """
    Минимальный pool-like объект.

    Нужен, чтобы select_news_candidates()
    выполнялся в той же транзакции, где
    создаются тестовые publication_batches.
    """

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    def acquire(
        self,
    ) -> _SingleConnectionAcquire:
        return _SingleConnectionAcquire(
            self._connection
        )


async def _create_test_batch(
    connection: asyncpg.Connection,
    *,
    edition: int,
    batch_status: str,
    news_id: int,
) -> int:
    """Создаёт тестовый batch и один batch_item."""

    published_at = (
        datetime.now(timezone.utc)
        if batch_status == "published"
        else None
    )

    batch_id = await connection.fetchval(
        """
        INSERT INTO top3_news.publication_batches (
            publication_date,
            edition,
            batch_status,
            published_at,
            metadata
        )
        VALUES (
            $1,
            $2,
            $3,
            $4,
            '{}'::jsonb
        )
        RETURNING batch_id
        """,
        TEST_PUBLICATION_DATE,
        edition,
        batch_status,
        published_at,
    )

    if batch_id is None:
        raise AssertionError(
            "Не удалось создать тестовый "
            "publication_batch."
        )

    await connection.execute(
        """
        INSERT INTO top3_news.batch_items (
            batch_id,
            news_id,
            position
        )
        VALUES (
            $1,
            $2,
            1
        )
        """,
        batch_id,
        news_id,
    )

    return int(batch_id)


async def test_publication_exclusion(
    connection: asyncpg.Connection,
) -> None:
    """
    Проверяет защиту кандидатов:

    - published исключается;
    - failed остаётся;
    - rejected остаётся.
    """

    test_pool = _SingleConnectionPool(
        connection
    )

    as_of = datetime.now(timezone.utc)

    baseline = await select_news_candidates(
        test_pool,
        as_of=as_of,
        window_hours=168.0,
        limit=5000,
    )

    if len(baseline.candidates) < 3:
        raise AssertionError(
            "Для теста требуется минимум "
            "три неопубликованных кандидата "
            "за последние 168 часов."
        )

    published_candidate = (
        baseline.candidates[0]
    )
    failed_candidate = (
        baseline.candidates[1]
    )
    rejected_candidate = (
        baseline.candidates[2]
    )

    baseline_ids = {
        candidate.news_id
        for candidate in baseline.candidates
    }

    assert (
        published_candidate.news_id
        in baseline_ids
    )

    assert (
        failed_candidate.news_id
        in baseline_ids
    )

    assert (
        rejected_candidate.news_id
        in baseline_ids
    )

    maximum_edition = await connection.fetchval(
        """
        SELECT COALESCE(
            MAX(edition),
            0
        )
        FROM top3_news.publication_batches
        WHERE publication_date = $1
        """,
        TEST_PUBLICATION_DATE,
    )

    first_test_edition = (
        int(maximum_edition) + 1
    )

    if first_test_edition + 2 > 32767:
        raise AssertionError(
            "Недостаточно свободных edition "
            "для тестовой даты."
        )

    await _create_test_batch(
        connection,
        edition=first_test_edition,
        batch_status="published",
        news_id=published_candidate.news_id,
    )

    await _create_test_batch(
        connection,
        edition=first_test_edition + 1,
        batch_status="failed",
        news_id=failed_candidate.news_id,
    )

    await _create_test_batch(
        connection,
        edition=first_test_edition + 2,
        batch_status="rejected",
        news_id=rejected_candidate.news_id,
    )

    result = await select_news_candidates(
        test_pool,
        as_of=as_of,
        window_hours=168.0,
        limit=5000,
    )

    result_ids = {
        candidate.news_id
        for candidate in result.candidates
    }

    assert (
        published_candidate.news_id
        not in result_ids
    ), (
        "Опубликованная новость повторно "
        "попала в список кандидатов."
    )

    assert (
        failed_candidate.news_id
        in result_ids
    ), (
        "Новость из failed batch ошибочно "
        "исключена из кандидатов."
    )

    assert (
        rejected_candidate.news_id
        in result_ids
    ), (
        "Новость из rejected batch ошибочно "
        "исключена из кандидатов."
    )

    print("Published news exclusion: OK")
    print(
        "published_news_id="
        f"{published_candidate.news_id}"
    )

    print()
    print("Failed news remains candidate: OK")
    print(
        "failed_news_id="
        f"{failed_candidate.news_id}"
    )

    print()
    print(
        "Rejected news remains candidate: OK"
    )
    print(
        "rejected_news_id="
        f"{rejected_candidate.news_id}"
    )


async def main() -> int:
    """Запускает тест с полным rollback."""

    print(
        "News candidate publication "
        "exclusion test"
    )
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print()

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    try:
        async with pool.acquire() as connection:
            transaction = (
                connection.transaction()
            )

            await transaction.start()

            try:
                await test_publication_exclusion(
                    connection
                )
            finally:
                await transaction.rollback()
    finally:
        await close_database_pool(pool)

    print()
    print("Database changes=rolled_back")
    print("OpenAI requests=not_performed")
    print("Telegram requests=not_performed")
    print(
        "News candidate publication "
        "exclusion test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )