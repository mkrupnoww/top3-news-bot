import asyncio
from decimal import Decimal

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.score_formula import (
    FORMULA_VERSION,
    calculate_individual_score,
    create_score_components,
)


CASES = (
    ("10", "10", "10", "10", "1"),
    ("8", "6", "5", "7", "0.9"),
    ("0.310791", "0.387246", "4.723361", "1.729807", "0.678077"),
)


async def main() -> int:
    assert FORMULA_VERSION == "individual_score_v3"

    settings = get_settings()
    pool = await create_database_pool(settings)

    try:
        async with pool.acquire() as connection:
            migration = await connection.fetchrow(
                """
                SELECT version, description
                FROM top3_news.schema_migrations
                WHERE version = '018'
                """
            )

            if migration is None:
                raise AssertionError("Migration 018 is not applied.")

            mismatch_detected = False

            for index, values in enumerate(CASES, start=1):
                components = create_score_components(
                    f_score=values[0],
                    m_score=values[1],
                    r_score=values[2],
                    h_score=values[3],
                    q_score=values[4],
                )
                python_result = calculate_individual_score(components)

                postgres_score = await connection.fetchval(
                    """
                    SELECT top3_news.calculate_individual_score_v3(
                        $1::numeric,
                        $2::numeric,
                        $3::numeric,
                        $4::numeric,
                        $5::numeric
                    )
                    """,
                    components.f_score,
                    components.m_score,
                    components.r_score,
                    components.h_score,
                    components.q_score,
                )

                assert isinstance(postgres_score, Decimal)
                assert postgres_score == python_result.individual_score

                print(
                    f"case_{index}_score={postgres_score}; "
                    "python_postgres_match=true"
                )

                if index == 2:
                    tampered_python_score = (
                        python_result.individual_score
                        + Decimal("0.100000")
                    )
                    mismatch_detected = (
                        postgres_score
                        != tampered_python_score
                    )

            assert mismatch_detected is True

    finally:
        await close_database_pool(pool)

    print("Tampered Python score detection: OK")
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("PostgreSQL individual_score_v3 guard test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
