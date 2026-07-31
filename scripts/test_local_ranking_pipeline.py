import asyncio
from datetime import datetime, timezone

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.local_pipeline import (
    FixtureScore,
    LocalFixtureEvaluator,
    run_local_ranking_pipeline,
)


TEST_KEY = "variety_local_pipeline_v1"

AS_OF = datetime(
    2026,
    7,
    31,
    11,
    21,
    tzinfo=timezone.utc,
)


FIXTURE_SCORES = {
    7: FixtureScore(
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
    8: FixtureScore(
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
    9: FixtureScore(
        f_score="8.5",
        m_score="7.5",
        r_score="6.5",
        h_score="7.0",
        q_score="0.90",
        explanation=(
            "Международный фестиваль, "
            "рекордные показатели и "
            "новая AI-категория."
        ),
    ),
    10: FixtureScore(
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
    11: FixtureScore(
        f_score="6.5",
        m_score="8.0",
        r_score="7.0",
        h_score="6.5",
        q_score="0.95",
        explanation=(
            "Крупная студия, финансовые "
            "результаты и заметное "
            "падение выручки."
        ),
    ),
}


async def main() -> int:
    """Проверяет полный локальный конвейер."""

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    evaluator = LocalFixtureEvaluator(
        FIXTURE_SCORES
    )

    try:
        result = (
            await run_local_ranking_pipeline(
                database_pool,
                as_of=AS_OF,
                window_hours=24.0,
                source_codes=(
                    "variety_film",
                ),
                candidate_limit=100,
                top_size=3,
                test_key=TEST_KEY,
                evaluator=evaluator,
            )
        )
    finally:
        await close_database_pool(
            database_pool
        )

    print(
        "Local ranking pipeline completed"
    )
    print(
        "evaluator_version="
        f"{evaluator.evaluator_version}"
    )
    print(
        "already_persisted="
        f"{str(result.already_persisted).lower()}"
    )
    print(
        "ranking_run_id="
        f"{result.ranking_run_id}"
    )
    print(
        f"run_status={result.run_status}"
    )
    print(
        "window_started_at="
        f"{result.window_started_at.isoformat()}"
    )
    print(
        "window_finished_at="
        f"{result.window_finished_at.isoformat()}"
    )
    print(
        "candidate_count="
        f"{result.candidate_count}"
    )
    print(
        f"scored_count={result.scored_count}"
    )
    print(
        "eligible_count="
        f"{result.eligible_count}"
    )

    print()
    print("Full ranking:")

    for candidate in (
        result.ranked_candidates
    ):
        print(
            f"{candidate.rank_position}. "
            f"news_id={candidate.news_id} "
            f"score={candidate.individual_score}"
        )
        print(
            f"   title={candidate.title}"
        )
        print(
            f"   source="
            f"{candidate.source_name} "
            f"[{candidate.source_code}]"
        )
        print(
            f"   source_url="
            f"{candidate.source_url}"
        )

    print()
    print("TOP-3:")

    for candidate in result.top_candidates:
        print(
            f"{candidate.rank_position}. "
            f"news_id={candidate.news_id} "
            f"score={candidate.individual_score}"
        )
        print(
            f"   title={candidate.title}"
        )

    expected_top_ids = (
        9,
        11,
        7,
    )

    actual_top_ids = tuple(
        candidate.news_id
        for candidate in result.top_candidates
    )

    if actual_top_ids != expected_top_ids:
        print()
        print(
            "Unexpected TOP-3: "
            f"expected={expected_top_ids}, "
            f"actual={actual_top_ids}"
        )
        return 1

    if result.run_status != "completed":
        print()
        print(
            "Unexpected run status: "
            f"{result.run_status}"
        )
        return 1

    print()
    print("Expected TOP-3: OK")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print(
        "Local ranking pipeline test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )