from datetime import datetime, timezone

from app.db.news_candidates import (
    CandidateSelectionResult,
    NewsCandidate,
)
from app.ranking.cinema_relevance import (
    evaluate_cinema_relevance,
    filter_cinema_relevance,
)


AS_OF = datetime(
    2026,
    8,
    3,
    12,
    0,
    tzinfo=timezone.utc,
)


def build_candidate(
    *,
    news_id: int,
    source_code: str = "hollywood_reporter",
    title: str,
    summary: str | None = None,
    source_url: str,
    categories: tuple[str, ...] = (),
    requires_filter: bool = True,
) -> NewsCandidate:
    """Создаёт одного тестового кандидата."""

    return NewsCandidate(
        news_id=news_id,
        source_id=100 + news_id,
        source_code=source_code,
        source_name=source_code,
        collection_priority=100,
        processing_status="collected",
        title=title,
        summary=summary,
        author_name=None,
        source_published_at=AS_OF,
        age_hours=0.0,
        source_url=source_url,
        primary_image_url=None,
        categories=categories,
        requires_cinema_relevance_filter=(
            requires_filter
        ),
        source_weight=3,
    )


def test_curated_source_passes() -> None:
    """Специализированный источник не фильтруется."""

    candidate = build_candidate(
        news_id=1,
        source_code="variety_film",
        title="Unrelated title",
        source_url=(
            "https://example.com/general/story"
        ),
        requires_filter=False,
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is True
    assert decision.signals == (
        "source_scope:curated",
    )


def test_category_signal() -> None:
    """Категория Movies является сильным сигналом."""

    candidate = build_candidate(
        news_id=2,
        title="Festival programme announced",
        source_url=(
            "https://example.com/news/story"
        ),
        categories=(
            "Movie News",
            "Movies",
        ),
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is True
    assert "category:Movie News" in (
        decision.signals
    )
    assert "category:Movies" in (
        decision.signals
    )


def test_url_signal() -> None:
    """Кинораздел URL является сильным сигналом."""

    candidate = build_candidate(
        news_id=3,
        title="New project announced",
        source_url=(
            "https://example.com/movies/"
            "movie-news/new-project"
        ),
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is True
    assert "url:/movies/" in (
        decision.signals
    )


def test_text_signal() -> None:
    """Кинотермин в заголовке или описании проходит."""

    candidate = build_candidate(
        news_id=4,
        title=(
            "Director unveils documentary "
            "at Locarno"
        ),
        source_url=(
            "https://example.com/general/story"
        ),
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is True
    assert "text:documentary" in (
        decision.signals
    )
    assert "text:locarno" in (
        decision.signals
    )


def test_business_industry_signal() -> None:
    """Студия плюс сделка распознаются как киноиндустрия."""

    candidate = build_candidate(
        news_id=5,
        title=(
            "Regulators examine Paramount "
            "and Warner Bros. merger"
        ),
        source_url=(
            "https://example.com/business/story"
        ),
        categories=(
            "Business",
            "Business News",
        ),
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is True
    assert any(
        signal.startswith(
            "industry_entity:"
        )
        for signal in decision.signals
    )
    assert any(
        signal.startswith(
            "industry_action:"
        )
        for signal in decision.signals
    )


def test_imax_signal() -> None:
    """IMAX распознаётся как сигнал кинопроката."""

    candidate = build_candidate(
        news_id=6,
        title=(
            "Imax shifts screens between "
            "two major releases"
        ),
        source_url=(
            "https://example.com/business/story"
        ),
        categories=(
            "Business",
            "Business News",
            "imax",
        ),
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is True
    assert "category:imax" in (
        decision.signals
    )


def test_irrelevant_item_is_excluded() -> None:
    """Музыкальная новость без кино-сигналов исключается."""

    candidate = build_candidate(
        news_id=7,
        title=(
            "Singer announces another "
            "world tour"
        ),
        summary=(
            "The album remains at the top "
            "of the charts."
        ),
        source_url=(
            "https://example.com/music/"
            "music-news/world-tour"
        ),
        categories=(
            "Music",
            "Music News",
        ),
    )

    decision = evaluate_cinema_relevance(
        candidate
    )

    assert decision.is_relevant is False
    assert decision.signals == ()


def test_selection_filter() -> None:
    """Фильтр сохраняет порядок и границы окна."""

    curated = build_candidate(
        news_id=10,
        source_code="deadline_film",
        title="Curated publication",
        source_url=(
            "https://example.com/story"
        ),
        requires_filter=False,
    )

    movie = build_candidate(
        news_id=11,
        title="Film festival reveals lineup",
        source_url=(
            "https://example.com/news/"
            "festival-lineup"
        ),
    )

    music = build_candidate(
        news_id=12,
        title="Singer releases a new album",
        source_url=(
            "https://example.com/music/"
            "new-album"
        ),
        categories=("Music",),
    )

    selection = CandidateSelectionResult(
        window_start=AS_OF,
        window_end=AS_OF,
        window_hours=24.0,
        candidates=(
            curated,
            movie,
            music,
        ),
    )

    result = filter_cinema_relevance(
        selection
    )

    assert result.input_count == 3
    assert result.included_count == 2
    assert result.excluded_count == 1
    assert result.excluded_news_ids == (12,)
    assert tuple(
        candidate.news_id
        for candidate
        in result.selection.candidates
    ) == (10, 11)
    assert (
        result.selection.window_start
        == selection.window_start
    )
    assert (
        result.selection.window_end
        == selection.window_end
    )
    assert (
        result.selection.window_hours
        == selection.window_hours
    )


def main() -> int:
    """Запускает тесты без внешних систем."""

    test_curated_source_passes()
    test_category_signal()
    test_url_signal()
    test_text_signal()
    test_business_industry_signal()
    test_imax_signal()
    test_irrelevant_item_is_excluded()
    test_selection_filter()

    print("Curated source pass-through: OK")
    print("Category relevance signal: OK")
    print("URL relevance signal: OK")
    print("Text relevance signal: OK")
    print("Industry business signal: OK")
    print("IMAX relevance signal: OK")
    print("Irrelevant item exclusion: OK")
    print("Selection filtering: OK")
    print()
    print("Network requests: not performed")
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Cinema relevance test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())