from pathlib import Path

from app.collectors.feed_parser import (
    parse_feed_document,
)


FIXTURES_DIR = Path("tests/fixtures")


def test_rss() -> None:
    """Проверяет локальный RSS 2.0 документ."""

    result = parse_feed_document(
        (
            FIXTURES_DIR
            / "sample_movie_rss.xml"
        ).read_bytes()
    )

    assert result.feed_type == "rss"
    assert result.feed_title == "Movie Wire RSS"
    assert len(result.entries) == 2
    assert result.skipped_count == 1

    first_entry = result.entries[0]

    assert (
        first_entry.external_id
        == "movie-rss-001"
    )
    assert (
        first_entry.source_url
        == (
            "https://example.com/movies/"
            "science-fiction-film"
        )
    )
    assert (
        first_entry.author_name
        == "Alex Reporter"
    )
    assert first_entry.source_published_at is not None
    assert (
        first_entry.source_published_at.isoformat()
        == "2026-07-31T07:15:00+00:00"
    )
    assert (
        first_entry.primary_image_url
        == (
            "https://example.com/images/"
            "science-fiction.jpg"
        )
    )
    assert first_entry.summary is not None
    assert "<b>" not in first_entry.summary

    print("RSS parser: OK")
    print(f"feed_title={result.feed_title}")
    print(f"entry_count={len(result.entries)}")
    print(f"skipped_count={result.skipped_count}")


def test_atom() -> None:
    """Проверяет локальный Atom документ."""

    result = parse_feed_document(
        (
            FIXTURES_DIR
            / "sample_movie_atom.xml"
        ).read_bytes()
    )

    assert result.feed_type == "atom"
    assert result.feed_title == "Cinema Atom"
    assert len(result.entries) == 2
    assert result.skipped_count == 0

    first_entry = result.entries[0]

    assert (
        first_entry.external_id
        == "tag:example.org,2026:movie-atom-001"
    )
    assert (
        first_entry.source_url
        == (
            "https://example.org/cinema/"
            "upcoming-thriller"
        )
    )
    assert (
        first_entry.author_name
        == "Maria Journalist"
    )
    assert first_entry.source_published_at is not None
    assert (
        first_entry.source_published_at.isoformat()
        == "2026-07-31T07:45:00+00:00"
    )
    assert (
        first_entry.primary_image_url
        == (
            "https://example.org/images/"
            "thriller.jpg"
        )
    )

    print()
    print("Atom parser: OK")
    print(f"feed_title={result.feed_title}")
    print(f"entry_count={len(result.entries)}")
    print(f"skipped_count={result.skipped_count}")


def main() -> int:
    """Запускает тесты без сети и PostgreSQL."""

    test_rss()
    test_atom()

    print()
    print("Network requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Feed parser test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())