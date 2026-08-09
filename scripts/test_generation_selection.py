import asyncio

from app.config import get_settings
from app.db.generation_selection import (
    _resolve_saved_top3_mode,
    load_generation_top3,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)


LEGACY_RANKING_RUN_ID = 18
DIVERSITY_RANKING_RUN_ID = 104
SECOND_DIVERSITY_RANKING_RUN_ID = 98


def assert_positions(selection: object) -> None:
    """Проверяет канонические позиции 1, 2, 3."""

    items = getattr(selection, "items")

    positions = tuple(
        item.position
        for item in items
    )

    assert positions == (1, 2, 3)


def test_selection_mode_validation() -> None:
    """Проверяет новый режим, legacy и corruption."""

    assert (
        _resolve_saved_top3_mode(
            3,
            ranking_run_id=104,
        )
        is True
    )

    assert (
        _resolve_saved_top3_mode(
            0,
            ranking_run_id=18,
        )
        is False
    )

    for invalid_count in (1, 2, 4):
        try:
            _resolve_saved_top3_mode(
                invalid_count,
                ranking_run_id=999,
            )
        except ValueError as error:
            assert (
                "некорректное число строк"
                in str(error)
            )
        else:
            raise AssertionError(
                "Частично сохранённый "
                "selected_for_top3 должен "
                "блокировать генерацию."
            )

    print("Selection mode validation: OK")


async def test_legacy_selection(pool: object) -> None:
    """Проверяет fallback старого ranking_run."""

    selection = await load_generation_top3(
        pool,
        ranking_run_id=LEGACY_RANKING_RUN_ID,
    )

    assert selection.ranking_run_id == 18
    assert selection.run_status == "completed"
    assert selection.news_ids == (11, 9, 10)

    assert_positions(selection)

    print()
    print("Legacy generation selection: OK")
    print("ranking_run_id=18")
    print("selection_mode=legacy_rank_position")
    print("news_ids=11,9,10")


async def test_diversity_selection(
    pool: object,
) -> None:
    """Проверяет финальный diversity TOP-3."""

    selection = await load_generation_top3(
        pool,
        ranking_run_id=DIVERSITY_RANKING_RUN_ID,
    )

    assert selection.ranking_run_id == 104
    assert selection.run_status == "completed"

    assert (
        selection.news_ids
        == (131, 151, 145)
    )

    assert 163 not in selection.news_ids

    assert_positions(selection)

    print()
    print("Diversity generation selection: OK")
    print("ranking_run_id=104")
    print("selection_mode=selected_for_top3")
    print("news_ids=131,151,145")
    print("rank_position_3_not_used=163")


async def test_second_diversity_selection(
    pool: object,
) -> None:
    """Проверяет второй сохранённый diversity run."""

    selection = await load_generation_top3(
        pool,
        ranking_run_id=(
            SECOND_DIVERSITY_RANKING_RUN_ID
        ),
    )

    assert selection.ranking_run_id == 98
    assert selection.run_status == "completed"

    assert (
        selection.news_ids
        == (134, 142, 153)
    )

    assert 140 not in selection.news_ids

    assert_positions(selection)

    print()
    print(
        "Second diversity generation "
        "selection: OK"
    )
    print("ranking_run_id=98")
    print("selection_mode=selected_for_top3")
    print("news_ids=134,142,153")
    print("rank_position_3_not_used=140")


async def main() -> int:
    """Запускает read-only тест выбора TOP-3."""

    test_selection_mode_validation()

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    try:
        await test_legacy_selection(pool)
        await test_diversity_selection(pool)
        await test_second_diversity_selection(
            pool
        )
    finally:
        await close_database_pool(pool)

    print()
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Generation selection test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )