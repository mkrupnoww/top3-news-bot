import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.ranking_scores import (
    ManualNewsAssessment,
    persist_manual_ranking_test,
)


TEST_KEY_PREFIX = "variety_formula_sync_v3"

WINDOW_STARTED_AT = datetime(
    2026,
    7,
    30,
    11,
    21,
    tzinfo=timezone.utc,
)

WINDOW_FINISHED_AT = datetime(
    2026,
    7,
    31,
    11,
    21,
    tzinfo=timezone.utc,
)


def build_assessments(
    news_ids: tuple[int, ...],
) -> tuple[ManualNewsAssessment, ...]:
    """Создаёт оценки для временных news_items."""

    if len(news_ids) != 5:
        raise ValueError(
            "Для теста требуется пять news_id."
        )

    return (
        ManualNewsAssessment(
            news_id=news_ids[0],
            f_score="9.0",
            m_score="5.5",
            r_score="4.5",
            h_score="6.0",
            q_score="0.85",
            explanation=(
                "Свежая тематическая рецензия; "
                "умеренный масштаб и резонанс."
            ),
        ),
        ManualNewsAssessment(
            news_id=news_ids[1],
            f_score="8.8",
            m_score="5.0",
            r_score="4.2",
            h_score="5.8",
            q_score="0.80",
            explanation=(
                "Свежая рецензия с нишевой "
                "историко-кинематографической темой."
            ),
        ),
        ManualNewsAssessment(
            news_id=news_ids[2],
            f_score="8.5",
            m_score="7.5",
            r_score="6.5",
            h_score="7.0",
            q_score="0.90",
            explanation=(
                "Международный фестиваль, рекордные "
                "показатели и новая AI-категория."
            ),
        ),
        ManualNewsAssessment(
            news_id=news_ids[3],
            f_score="8.0",
            m_score="4.0",
            r_score="3.5",
            h_score="3.0",
            q_score="0.95",
            explanation=(
                "Свежая, но преимущественно "
                "корпоративная кадровая новость."
            ),
        ),
        ManualNewsAssessment(
            news_id=news_ids[4],
            f_score="6.5",
            m_score="8.0",
            r_score="7.0",
            h_score="6.5",
            q_score="0.95",
            explanation=(
                "Крупная студия, финансовые результаты "
                "и заметное падение выручки."
            ),
        ),
    )


async def create_test_news_items(
    pool: asyncpg.Pool,
) -> tuple[int, ...]:
    """Создаёт пять временных news_items."""

    fixture_token = uuid4().hex

    async with pool.acquire() as connection:
        source_id = await connection.fetchval(
            """
            SELECT source_id
            FROM top3_news.sources
            WHERE source_code = 'variety_film'
            """
        )

        if source_id is None:
            raise LookupError(
                "Тестовый источник variety_film не найден."
            )

        created_news_ids: list[int] = []

        async with connection.transaction():
            for position in range(1, 6):
                news_id = await connection.fetchval(
                    """
                    INSERT INTO top3_news.news_items (
                        source_id,
                        external_id,
                        source_url,
                        raw_title,
                        raw_summary,
                        author_name,
                        source_published_at,
                        processing_status,
                        metadata
                    )
                    VALUES (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        'Integration Test',
                        $6,
                        'collected',
                        jsonb_build_object(
                            'integration_test',
                            true,
                            'fixture_token',
                            $7::text
                        )
                    )
                    RETURNING news_id
                    """,
                    source_id,
                    (
                        "postgres-score-formula-"
                        f"{fixture_token}-{position}"
                    ),
                    (
                        "https://example.com/"
                        "postgres-score-formula/"
                        f"{fixture_token}/{position}"
                    ),
                    (
                        "Postgres score formula test news "
                        f"{position}"
                    ),
                    (
                        "Postgres score formula test summary "
                        f"{position}"
                    ),
                    (
                        WINDOW_FINISHED_AT
                        - timedelta(hours=position)
                    ),
                    fixture_token,
                )

                if news_id is None:
                    raise RuntimeError(
                        "Не удалось создать тестовый news_item."
                    )

                created_news_ids.append(int(news_id))

    return tuple(created_news_ids)


async def cleanup_test_data(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int | None,
    news_ids: tuple[int, ...],
) -> None:
    """Удаляет ranking fixture и временные news_items."""

    async with pool.acquire() as connection:
        if ranking_run_id is not None:
            result = await connection.execute(
                """
                DELETE FROM top3_news.ranking_runs
                WHERE ranking_run_id = $1
                """,
                ranking_run_id,
            )

            if result != "DELETE 1":
                raise RuntimeError(
                    "Не удалось удалить тестовый ranking_run: "
                    f"{result}"
                )

        if news_ids:
            result = await connection.execute(
                """
                DELETE FROM top3_news.news_items
                WHERE news_id = ANY($1::bigint[])
                """,
                list(news_ids),
            )

            expected = f"DELETE {len(news_ids)}"
            if result != expected:
                raise RuntimeError(
                    "Удалено неожиданное число news_items: "
                    f"expected={expected}, actual={result}"
                )

    print()
    print("Test data cleanup: OK")
    print(
        "temporary_ranking_run_id="
        + (
            str(ranking_run_id)
            if ranking_run_id is not None
            else "none"
        )
    )
    print(
        "temporary_news_ids="
        + ",".join(str(news_id) for news_id in news_ids)
    )
    print("temporary_test_data_deleted=true")



async def main() -> int:
    """Проверяет равенство формул Python и PostgreSQL."""

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    ranking_run_id: int | None = None
    created_news_ids: tuple[int, ...] = ()

    try:
        async with database_pool.acquire() as connection:
            column_state = await connection.fetchrow(
                """
                SELECT
                    is_generated,
                    is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'top3_news'
                  AND table_name = 'news_scores'
                  AND column_name = 'individual_score'
                """
            )

        if column_state is None:
            raise AssertionError(
                "news_scores.individual_score не найден."
            )

        assert column_state["is_generated"] == "NEVER"
        assert column_state["is_nullable"] == "NO"

        created_news_ids = await create_test_news_items(
            database_pool
        )
        assessments = build_assessments(
            created_news_ids
        )
        test_key = (
            f"{TEST_KEY_PREFIX}_{uuid4().hex}"
        )

        print("Temporary score fixture: OK")
        print(
            "temporary_news_ids="
            + ",".join(
                str(news_id)
                for news_id in created_news_ids
            )
        )

        result = await persist_manual_ranking_test(
            database_pool,
            test_key=test_key,
            window_started_at=WINDOW_STARTED_AT,
            window_finished_at=WINDOW_FINISHED_AT,
            assessments=assessments,
        )
        ranking_run_id = result.ranking_run_id

        print(
            "individual_score_storage="
            "versioned_persisted"
        )

        if result.already_persisted:
            raise AssertionError(
                "Уникальная тестовая фикстура неожиданно "
                "была распознана как уже сохранённая."
            )

        print(
            "Ranking formula test persisted successfully"
        )
        print("already_persisted=false")
        print(
            f"ranking_run_id="
            f"{result.ranking_run_id}"
        )
        print(
            f"run_status={result.run_status}"
        )
        print(
            "formula_version="
            f"{result.formula_version}"
        )
        print(
            f"candidate_count="
            f"{result.candidate_count}"
        )
        print(
            f"scored_count="
            f"{result.scored_count}"
        )
        print(
            f"eligible_count="
            f"{result.eligible_count}"
        )

        for score in result.scores:
            print()
            print(
                f"rank={score.rank_position}"
            )
            print(f"news_id={score.news_id}")
            print(f"score_id={score.score_id}")
            print(
                "python_individual_score="
                f"{score.python_individual_score}"
            )
            print(
                "postgres_individual_score="
                f"{score.postgres_individual_score}"
            )
            print(
                "scores_match="
                f"{str(score.scores_match).lower()}"
            )

        if not all(
            score.scores_match
            for score in result.scores
        ):
            raise AssertionError(
                "Python/PostgreSQL score mismatch."
            )

        print()
        print("OpenAI requests: not performed")
        print("Telegram publication: not performed")
        print(
            "Score persistence synchronization test: OK"
        )
    finally:
        try:
            await cleanup_test_data(
                database_pool,
                ranking_run_id=ranking_run_id,
                news_ids=created_news_ids,
            )
        finally:
            await close_database_pool(
                database_pool
            )


    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )