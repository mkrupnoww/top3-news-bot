import asyncio

from app.config import get_settings
from app.db.generation_selection import (
    choose_next_generation_combination,
    load_generation_combination,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


RANKING_RUN_ID = 142

WINNER_COMBINATION_ID = 1844
WINNER_NEWS_IDS = (
    1029,
    1037,
    986,
)

EXPECTED_FIRST_REPLACEMENT_ID = 1845
EXPECTED_FIRST_REPLACEMENT_NEWS_IDS = (
    1029,
    1030,
    986,
)

EXPECTED_SECOND_REPLACEMENT_ID = 1846
EXPECTED_SECOND_REPLACEMENT_NEWS_IDS = (
    1029,
    1034,
    986,
)


async def main() -> int:
    """Проверяет read-only replacement selection на ranking 142."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    try:
        winner = await load_generation_combination(
            pool,
            ranking_run_id=RANKING_RUN_ID,
            combination_id=WINNER_COMBINATION_ID,
        )

        assert winner.is_winner is True
        assert winner.combination_rank == 1
        assert winner.news_ids == WINNER_NEWS_IDS

        print(
            "Winner combination load: OK"
        )
        print(
            "winner_combination_id="
            f"{winner.combination_id}"
        )
        print(
            "winner_news_ids="
            + ",".join(
                str(news_id)
                for news_id in winner.news_ids
            )
        )

        first = (
            await choose_next_generation_combination(
                pool,
                ranking_run_id=RANKING_RUN_ID,
                current_news_ids=WINNER_NEWS_IDS,
                excluded_combination_ids=(
                    WINNER_COMBINATION_ID,
                ),
            )
        )

        if first is None:
            raise AssertionError(
                "Первая replacement combination "
                "не найдена."
            )

        assert (
            first.combination.combination_id
            == EXPECTED_FIRST_REPLACEMENT_ID
        )
        assert (
            first.combination.combination_rank
            == 2
        )
        assert (
            first.combination.news_ids
            == EXPECTED_FIRST_REPLACEMENT_NEWS_IDS
        )
        assert first.overlap_count == 2
        assert first.removed_news_ids == (
            1037,
        )
        assert first.added_news_ids == (
            1030,
        )

        print(
            "First overlap=2 replacement: OK"
        )
        print(
            "replacement_combination_id="
            f"{first.combination.combination_id}"
        )
        print(
            "replacement_news_ids="
            + ",".join(
                str(news_id)
                for news_id
                in first.combination.news_ids
            )
        )
        print(
            "removed_news_ids="
            + ",".join(
                str(news_id)
                for news_id
                in first.removed_news_ids
            )
        )
        print(
            "added_news_ids="
            + ",".join(
                str(news_id)
                for news_id
                in first.added_news_ids
            )
        )

        second = (
            await choose_next_generation_combination(
                pool,
                ranking_run_id=RANKING_RUN_ID,
                current_news_ids=WINNER_NEWS_IDS,
                excluded_combination_ids=(
                    WINNER_COMBINATION_ID,
                    EXPECTED_FIRST_REPLACEMENT_ID,
                ),
            )
        )

        if second is None:
            raise AssertionError(
                "Вторая replacement combination "
                "не найдена."
            )

        assert (
            second.combination.combination_id
            == EXPECTED_SECOND_REPLACEMENT_ID
        )
        assert (
            second.combination.combination_rank
            == 3
        )
        assert (
            second.combination.news_ids
            == EXPECTED_SECOND_REPLACEMENT_NEWS_IDS
        )
        assert second.overlap_count == 2
        assert second.removed_news_ids == (
            1037,
        )
        assert second.added_news_ids == (
            1034,
        )

        print(
            "Used combination exclusion: OK"
        )
        print(
            "second_combination_id="
            f"{second.combination.combination_id}"
        )

        print()
        print(
            "Database changes=not_performed"
        )
        print(
            "OpenAI requests=not_performed"
        )
        print(
            "Telegram requests=not_performed"
        )
        print(
            "Generation combination replacement "
            "selection test: OK"
        )

        return 0

    finally:
        await close_database_pool(
            pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )
