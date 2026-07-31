import asyncio
from datetime import datetime, timezone

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.ranking_scores import (
    ManualNewsAssessment,
    persist_manual_ranking_test,
)


TEST_KEY = "variety_formula_sync_v1"

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


ASSESSMENTS = (
    ManualNewsAssessment(
        news_id=7,
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
        news_id=8,
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
        news_id=9,
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
        news_id=10,
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
        news_id=11,
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


async def main() -> int:
    """Проверяет равенство формул Python и PostgreSQL."""

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    try:
        result = await persist_manual_ranking_test(
            database_pool,
            test_key=TEST_KEY,
            window_started_at=WINDOW_STARTED_AT,
            window_finished_at=WINDOW_FINISHED_AT,
            assessments=ASSESSMENTS,
        )
    finally:
        await close_database_pool(
            database_pool
        )

    if result.already_persisted:
        print(
            "Ranking formula test was already persisted"
        )
    else:
        print(
            "Ranking formula test persisted successfully"
        )

    print(
        "already_persisted="
        f"{str(result.already_persisted).lower()}"
    )
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
        print()
        print("Formula synchronization test: FAILED")
        return 1

    print()
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Formula synchronization test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )