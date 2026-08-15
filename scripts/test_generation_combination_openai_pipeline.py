import asyncio
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import json

import asyncpg

from app.config import get_settings
from app.db.generation_selection import (
    load_generation_combination,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    OpenAITelegramPostGenerator,
)
from app.generation.openai_pipeline import (
    run_reserved_openai_generation,
)
from app.generation.official_trailer_enrichment import (
    OfficialTrailerEnrichmentResult,
)
from app.generation.request_key import (
    create_generation_request_key,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


RANKING_RUN_ID = 142
WINNER_COMBINATION_ID = 1844
REPLACEMENT_COMBINATION_ID = 1845

WINNER_NEWS_IDS = (
    1029,
    1037,
    986,
)

REPLACEMENT_NEWS_IDS = (
    1029,
    1030,
    986,
)

TEST_PUBLICATION_DATE = date(
    2600,
    8,
    15,
)

TEST_MODEL_NAME = (
    "test-generation-combination-pipeline-v1"
)


class _SingleConnectionAcquire:
    """Context manager одной asyncpg connection."""

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
    """Pool-like wrapper для полного rollback теста."""

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


@dataclass(frozen=True, slots=True)
class _Telemetry:
    """Синтетическая телеметрия fake model."""

    usage: OpenAITokenUsage
    cost: OpenAICostEstimate


def _build_telemetry(
    *,
    model_name: str,
) -> _Telemetry:
    """Возвращает согласованные usage/cost."""

    return _Telemetry(
        usage=OpenAITokenUsage(
            input_tokens=1000,
            cached_input_tokens=100,
            cache_write_tokens=50,
            output_tokens=200,
            reasoning_tokens=25,
            total_tokens=1200,
        ),
        cost=OpenAICostEstimate(
            model_name=model_name,
            pricing_version=(
                "test_generation_combination_"
                "pipeline_pricing_v1"
            ),
            regular_input_cost_usd=(
                Decimal("0.00100000")
            ),
            cached_input_cost_usd=(
                Decimal("0.00002000")
            ),
            cache_write_cost_usd=(
                Decimal("0.00010000")
            ),
            output_cost_usd=(
                Decimal("0.00200000")
            ),
            total_cost_usd=(
                Decimal("0.00312000")
            ),
        ),
    )


class FakeStructuredGenerationClient:
    """Fake Responses client без OpenAI API."""

    def __init__(self) -> None:
        self.requests: list[
            GenerationModelRequest
        ] = []

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Возвращает валидный primary/self-review JSON."""

        self.requests.append(request)

        payload = json.loads(
            request.input_text
        )

        task = payload.get("task")

        is_self_review = (
            task
            == (
                "self_review_russian_telegram_"
                "movie_news_top3"
            )
        )

        if task not in {
            (
                "generate_russian_telegram_"
                "movie_news_top3"
            ),
            (
                "self_review_russian_telegram_"
                "movie_news_top3"
            ),
        }:
            raise AssertionError(
                f"Неожиданный task: {task!r}"
            )

        if is_self_review:
            source_post_text = payload.get(
                "source_post_text"
            )

            if (
                not isinstance(
                    source_post_text,
                    str,
                )
                or not source_post_text.strip()
            ):
                raise AssertionError(
                    "Self-review не получил "
                    "source_post_text."
                )

        news = payload.get("news")

        if (
            not isinstance(news, list)
            or len(news) != 3
        ):
            raise AssertionError(
                "Fake model ожидает ровно "
                "три новости."
            )

        items: list[
            dict[str, object]
        ] = []

        sections: list[str] = []

        for expected_position, news_item in enumerate(
            news,
            start=1,
        ):
            if not isinstance(
                news_item,
                dict,
            ):
                raise AssertionError(
                    "news item должен быть object."
                )

            position = news_item.get(
                "position"
            )
            news_id = news_item.get(
                "news_id"
            )
            title = news_item.get(
                "title"
            )
            summary = news_item.get(
                "summary"
            )

            if position != expected_position:
                raise AssertionError(
                    "Нарушен порядок positions."
                )

            if (
                isinstance(news_id, bool)
                or not isinstance(news_id, int)
            ):
                raise AssertionError(
                    "news_id должен быть int."
                )

            if (
                not isinstance(title, str)
                or not title.strip()
            ):
                raise AssertionError(
                    "title должен быть непустым str."
                )

            if (
                not isinstance(summary, str)
                or not summary.strip()
            ):
                raise AssertionError(
                    "summary должен быть непустым str."
                )

            headline = title.strip()[:50]
            body = summary.strip()[:100]

            if is_self_review:
                headline = (
                    f"{headline} — проверено"
                )

            items.append(
                {
                    "position": position,
                    "news_id": news_id,
                    "headline": headline,
                    "body": body,
                }
            )

            sections.append(
                f"**{position}. {headline}**\n"
                f"{body}"
            )

        post_text = (
            "**TOP-3 киноновости дня**\n\n"
            + "\n\n".join(sections)
            + (
                "\n\n"
                "__Какую из новостей "
                "обсудим подробнее?__"
            )
        )

        if len(post_text) > 1000:
            raise AssertionError(
                "Fake post неожиданно "
                "превысил 1000 символов."
            )

        telemetry = _build_telemetry(
            model_name=request.model
        )

        return GenerationModelResponse(
            output_text=json.dumps(
                {
                    "post_text": post_text,
                    "items": items,
                },
                ensure_ascii=False,
            ),
            usage=telemetry.usage,
            cost_estimate=telemetry.cost,
            web_search_used=is_self_review,
            web_search_call_count=(
                1 if is_self_review else 0
            ),
            web_source_urls=(
                (
                    "https://example.test/"
                    "combination-self-review"
                ),
            )
            if is_self_review
            else (),
        )


class NoTrailerEnricher:
    """Fake trailer enrichment без HTTP."""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self,
        *,
        source_url: str,
        source_title: str,
        source_summary: str,
    ) -> OfficialTrailerEnrichmentResult:
        """Всегда сообщает, что trailer enrichment не нужен."""

        self.call_count += 1

        return OfficialTrailerEnrichmentResult(
            attempted=False,
            verified=False,
            official_trailer_url=None,
            reason="source_is_not_trailer_news",
            article_final_url=None,
            youtube_candidate_urls=(),
            checked_video_urls=(),
            verification_reasons=(),
            oembed_error_count=0,
            error_type=None,
        )


async def _load_target_chat_id(
    connection: asyncpg.Connection,
) -> int:
    """Берёт production channel ID из workflow ranking 142."""

    value = await connection.fetchval(
        """
        SELECT target_telegram_chat_id
        FROM top3_news.daily_workflow_runs
        WHERE ranking_run_id = $1
        """,
        RANKING_RUN_ID,
    )

    if value is None:
        raise AssertionError(
            "Не найден daily workflow "
            "для ranking_run_id=142."
        )

    return int(value)


async def main() -> int:
    """Проверяет explicit combination text pipeline в rollback."""

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    replacement_request_key: str | None = None
    batch_id: int | None = None

    try:
        async with database_pool.acquire() as connection:
            transaction = connection.transaction()
            await transaction.start()

            pool = _SingleConnectionPool(
                connection
            )

            try:
                telegram_chat_id = (
                    await _load_target_chat_id(
                        connection
                    )
                )

                winner = (
                    await load_generation_combination(
                        pool,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        combination_id=(
                            WINNER_COMBINATION_ID
                        ),
                    )
                )

                replacement = (
                    await load_generation_combination(
                        pool,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        combination_id=(
                            REPLACEMENT_COMBINATION_ID
                        ),
                    )
                )

                assert (
                    winner.news_ids
                    == WINNER_NEWS_IDS
                )

                assert (
                    replacement.news_ids
                    == REPLACEMENT_NEWS_IDS
                )

                print(
                    "Explicit combination selection: OK"
                )
                print(
                    "combination_id="
                    f"{REPLACEMENT_COMBINATION_ID}"
                )
                print(
                    "news_ids="
                    + ",".join(
                        str(news_id)
                        for news_id
                        in replacement.news_ids
                    )
                )

                fake_client = (
                    FakeStructuredGenerationClient()
                )

                trailer_enricher = (
                    NoTrailerEnricher()
                )

                generator = (
                    OpenAITelegramPostGenerator(
                        client=fake_client,
                        model_name=(
                            TEST_MODEL_NAME
                        ),
                    )
                )

                winner_model_request = (
                    generator.build_request(
                        winner.selection.items
                    )
                )

                winner_request_key = (
                    create_generation_request_key(
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        publication_date=(
                            TEST_PUBLICATION_DATE
                        ),
                        telegram_chat_id=(
                            telegram_chat_id
                        ),
                        metadata=(
                            generator.metadata
                        ),
                        model_request=(
                            winner_model_request
                        ),
                        items=(
                            winner.selection.items
                        ),
                    )
                )

                first_result = (
                    await run_reserved_openai_generation(
                        pool,
                        generator=generator,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        combination_id=(
                            REPLACEMENT_COMBINATION_ID
                        ),
                        publication_date=(
                            TEST_PUBLICATION_DATE
                        ),
                        telegram_chat_id=(
                            telegram_chat_id
                        ),
                        trailer_enricher=(
                            trailer_enricher
                        ),
                    )
                )

                replacement_request_key = (
                    first_result.request_key.value
                )

                batch_id = first_result.batch_id

                assert (
                    first_result.selection.news_ids
                    == REPLACEMENT_NEWS_IDS
                )

                assert (
                    first_result.request_key.value
                    != winner_request_key.value
                )

                assert (
                    first_result.reservation.created_new
                    is True
                )

                assert (
                    first_result.model_called
                    is True
                )

                assert (
                    first_result.completed
                    is True
                )

                assert (
                    first_result.batch_status
                    == "awaiting_review"
                )

                assert (
                    first_result.generated_post_id
                    is not None
                )

                assert len(
                    fake_client.requests
                ) == 2

                assert (
                    trailer_enricher.call_count
                    == 3
                )

                print(
                    "Replacement request key "
                    "differs from winner: OK"
                )
                print(
                    "Fake text generation "
                    "+ self-review: OK"
                )

                persisted = await connection.fetchrow(
                    """
                    SELECT
                        b.batch_id,
                        b.ranking_run_id,
                        b.batch_status,
                        ARRAY(
                            SELECT bi.news_id
                            FROM top3_news.batch_items AS bi
                            WHERE bi.batch_id = b.batch_id
                            ORDER BY bi.position
                        ) AS news_ids,
                        ARRAY(
                            SELECT bi.position
                            FROM top3_news.batch_items AS bi
                            WHERE bi.batch_id = b.batch_id
                            ORDER BY bi.position
                        ) AS positions,
                        (
                            SELECT COUNT(*)::integer
                            FROM top3_news.generated_posts AS gp
                            WHERE gp.batch_id = b.batch_id
                              AND gp.post_status =
                                  'awaiting_review'
                        ) AS awaiting_post_count
                    FROM top3_news.publication_batches AS b
                    WHERE b.batch_id = $1
                    """,
                    batch_id,
                )

                if persisted is None:
                    raise AssertionError(
                        "Replacement batch не найден."
                    )

                persisted_news_ids = tuple(
                    int(value)
                    for value
                    in persisted["news_ids"]
                )

                persisted_positions = tuple(
                    int(value)
                    for value
                    in persisted["positions"]
                )

                assert (
                    persisted_news_ids
                    == REPLACEMENT_NEWS_IDS
                )

                assert (
                    persisted_positions
                    == (1, 2, 3)
                )

                assert (
                    persisted["ranking_run_id"]
                    == RANKING_RUN_ID
                )

                assert (
                    persisted["batch_status"]
                    == "awaiting_review"
                )

                assert (
                    persisted["awaiting_post_count"]
                    == 1
                )

                print(
                    "Replacement batch_items: OK"
                )

                second_result = (
                    await run_reserved_openai_generation(
                        pool,
                        generator=generator,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        combination_id=(
                            REPLACEMENT_COMBINATION_ID
                        ),
                        publication_date=(
                            TEST_PUBLICATION_DATE
                        ),
                        telegram_chat_id=(
                            telegram_chat_id
                        ),
                        trailer_enricher=(
                            trailer_enricher
                        ),
                    )
                )

                assert (
                    second_result.batch_id
                    == first_result.batch_id
                )

                assert (
                    second_result.request_key.value
                    == first_result.request_key.value
                )

                assert (
                    second_result.reservation.created_new
                    is False
                )

                assert (
                    second_result.model_called
                    is False
                )

                assert (
                    second_result
                    .duplicate_request_blocked
                    is True
                )

                assert len(
                    fake_client.requests
                ) == 2

                assert (
                    trailer_enricher.call_count
                    == 3
                )

                generated_post_count = (
                    await connection.fetchval(
                        """
                        SELECT COUNT(*)::integer
                        FROM top3_news.generated_posts
                        WHERE batch_id = $1
                        """,
                        batch_id,
                    )
                )

                assert (
                    int(generated_post_count)
                    == 1
                )

                print(
                    "Repeat combination request "
                    "idempotency: OK"
                )
                print(
                    "fake_model_call_count=2"
                )

            finally:
                await transaction.rollback()

        if (
            replacement_request_key is None
            or batch_id is None
        ):
            raise AssertionError(
                "Тест не дошёл до "
                "replacement reservation."
            )

        async with database_pool.acquire() as connection:
            remaining = await connection.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM top3_news.publication_batches
                WHERE generation_request_key = $1
                """,
                replacement_request_key,
            )

        assert int(remaining) == 0

        print()
        print(
            "Database changes=rolled_back"
        )
        print(
            "OpenAI requests=fake_only"
        )
        print(
            "Telegram requests=not_performed"
        )
        print(
            "Generation combination text "
            "pipeline test: OK"
        )

        return 0

    finally:
        await close_database_pool(
            database_pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )