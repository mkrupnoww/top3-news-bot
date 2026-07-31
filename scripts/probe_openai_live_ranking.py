import argparse
import asyncio
from datetime import datetime, timezone
import inspect

from app.config import get_settings
from app.db.news_candidates import NewsCandidate
from app.ranking.openai_factory import (
    create_openai_ranking_runtime,
)
from app.ranking.score_formula import (
    calculate_individual_score,
    create_score_components,
)


def parse_arguments() -> argparse.Namespace:
    """Разбирает защитный флаг живого запроса."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет один контролируемый "
            "запрос к OpenAI для оценки "
            "одной тестовой киноновости."
        ),
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Подтверждает выполнение одного "
            "реального платного API-запроса."
        ),
    )

    return parser.parse_args()


def build_candidate() -> NewsCandidate:
    """Создаёт одну реалистичную тестовую новость."""

    return NewsCandidate(
        news_id=900001,
        source_id=900001,
        source_code="openai_live_probe",
        source_name="OpenAI Live Probe",
        collection_priority=100,
        processing_status="collected",
        title=(
            "Global Film Festival Adds "
            "AI-Made Film Competition "
            "After Record Submissions"
        ),
        summary=(
            "An Oscar-qualifying short film "
            "festival reported 3,340 submissions "
            "from more than 100 countries and "
            "added new competitions for AI-made "
            "films and screendance."
        ),
        author_name="Live Probe",
        source_published_at=datetime(
            2026,
            7,
            31,
            13,
            0,
            tzinfo=timezone.utc,
        ),
        age_hours=2.0,
        source_url=(
            "https://example.com/"
            "openai-live-ranking-probe"
        ),
        primary_image_url=None,
    )


async def close_sdk_client(
    sdk_client: object,
) -> None:
    """Закрывает AsyncOpenAI-клиент."""

    close_method = getattr(
        sdk_client,
        "close",
        None,
    )

    if close_method is None:
        return

    close_result = close_method()

    if inspect.isawaitable(close_result):
        await close_result


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет один живой запрос."""

    if not arguments.confirm_live_request:
        print("OpenAI live ranking probe refused")
        print(
            "Use --confirm-live-request "
            "to perform one paid API request."
        )
        print("OpenAI requests: not performed")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 2

    settings = get_settings()

    runtime = create_openai_ranking_runtime(
        settings
    )

    candidate = build_candidate()

    print("OpenAI live ranking probe started")
    print(
        f"model="
        f"{runtime.evaluator.metadata.model_name}"
    )
    print("candidate_count=1")
    print(f"news_id={candidate.news_id}")
    print(f"title={candidate.title}")

    try:
        assessments = (
            await runtime.evaluator.evaluate(
                (candidate,)
            )
        )
    except Exception as error:
        print()
        print("OpenAI live ranking probe: FAILED")
        print(
            f"error_type="
            f"{type(error).__name__}"
        )
        print(f"error={error}")
        print("Database changes: not performed")
        print("Telegram publication: not performed")
        return 1
    finally:
        await close_sdk_client(
            runtime.sdk_client
        )

    if len(assessments) != 1:
        print()
        print("OpenAI live ranking probe: FAILED")
        print(
            "Unexpected assessment count: "
            f"{len(assessments)}"
        )
        return 1

    assessment = assessments[0]

    if assessment.news_id != candidate.news_id:
        print()
        print("OpenAI live ranking probe: FAILED")
        print(
            "Unexpected news_id: "
            f"{assessment.news_id}"
        )
        return 1

    components = create_score_components(
        f_score=assessment.f_score,
        m_score=assessment.m_score,
        r_score=assessment.r_score,
        h_score=assessment.h_score,
        q_score=assessment.q_score,
    )

    calculated_score = (
        calculate_individual_score(
            components
        )
    )

    print()
    print("Assessment received")
    print(
        f"f_score={components.f_score}"
    )
    print(
        f"m_score={components.m_score}"
    )
    print(
        f"r_score={components.r_score}"
    )
    print(
        f"h_score={components.h_score}"
    )
    print(
        f"q_score={components.q_score}"
    )
    print(
        "individual_score="
        f"{calculated_score.individual_score}"
    )
    print(
        f"explanation="
        f"{assessment.explanation}"
    )

    print()
    print("OpenAI requests: performed=1")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("OpenAI live ranking probe: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(parse_arguments())
        )
    )