from app.generation.official_trailer_verifier import (
    verify_official_trailer,
)
from app.generation.youtube_oembed import (
    YouTubeOEmbedMetadata,
)


def build_metadata(
    *,
    title: str = (
        "Primetime | Official Trailer HD | A24"
    ),
    author_name: str = "A24",
) -> YouTubeOEmbedMetadata:
    """Создаёт локальные тестовые metadata."""

    return YouTubeOEmbedMetadata(
        video_id="5fHXyqQOKL8",
        canonical_url=(
            "https://www.youtube.com/"
            "watch?v=5fHXyqQOKL8"
        ),
        title=title,
        author_name=author_name,
        author_url=(
            "https://www.youtube.com/@A24"
        ),
    )


def test_real_thr_case() -> None:
    """Проверяет реальный кейс Primetime / A24."""

    result = verify_official_trailer(
        build_metadata(),
        source_title=(
            "Jeff Zucker Disses Robert Pattinson "
            "in Riveting ‘Primetime’ Trailer: "
            "“I Don’t Like Your Show”"
        ),
        source_summary=(
            "The official trailer for A24's "
            "dramatic take on Chris Hansen's "
            "controversial 'To Catch a Predator' "
            "series."
        ),
    )

    assert result.verified is True

    assert result.official_trailer_url == (
        "https://www.youtube.com/"
        "watch?v=5fHXyqQOKL8"
    )

    assert (
        result.reason
        == "verified_official_trailer"
    )
    assert result.official_trailer_channel_name == "A24"

    print("Real THR/A24 trailer case: OK")
    print("verified=true")
    print(
        "official_trailer_url="
        f"{result.official_trailer_url}"
    )


def test_unrelated_video() -> None:
    """Проверяет ролик, который не является трейлером."""

    result = verify_official_trailer(
        build_metadata(
            title=(
                "Primetime Interview With Cast"
            ),
        ),
        source_title=(
            "Primetime official trailer released"
        ),
        source_summary=(
            "A24 released the official trailer."
        ),
    )

    assert result.verified is False
    assert (
        result.reason
        == "youtube_title_is_not_trailer"
    )

    print()
    print("Unrelated video blocking: OK")


def test_unconfirmed_channel() -> None:
    """Проверяет неизвестный YouTube-канал."""

    result = verify_official_trailer(
        build_metadata(
            author_name="Movie Uploads",
        ),
        source_title=(
            "Primetime trailer released"
        ),
        source_summary=(
            "The official trailer for A24's "
            "new film has been released."
        ),
    )

    assert result.verified is False

    assert result.reason == (
        "youtube_author_not_confirmed_by_source"
    )

    print()
    print("Unconfirmed channel blocking: OK")


def test_missing_official_marker() -> None:
    """Проверяет отсутствие подтверждения official."""

    result = verify_official_trailer(
        build_metadata(
            title="Primetime Trailer HD | A24",
        ),
        source_title=(
            "Primetime trailer arrives"
        ),
        source_summary=(
            "A24 released a new trailer."
        ),
    )

    assert result.verified is False

    assert (
        result.reason
        == "official_marker_not_confirmed"
    )

    print()
    print("Missing official marker blocking: OK")


def test_non_trailer_source() -> None:
    """Проверяет новость не о трейлере."""

    result = verify_official_trailer(
        build_metadata(),
        source_title=(
            "Primetime sets September release date"
        ),
        source_summary=(
            "A24 confirmed the film's release date."
        ),
    )

    assert result.verified is False

    assert (
        result.reason
        == "source_is_not_trailer_news"
    )

    print()
    print("Non-trailer source blocking: OK")


def test_channel_token_match() -> None:
    """Разрешает безопасное сокращённое упоминание студии в source."""

    result = verify_official_trailer(
        build_metadata(
            title="Film | Official Trailer | Amazon MGM Studios",
            author_name="Amazon MGM Studios",
        ),
        source_title="Amazon MGM releases official trailer for Film",
        source_summary="The new trailer is now online.",
    )

    assert result.verified is True
    assert result.official_trailer_channel_name == "Amazon MGM Studios"

    print()
    print("Channel token confirmation: OK")


def main() -> int:
    """Запускает pure-Python тест verifier."""

    test_real_thr_case()
    test_unrelated_video()
    test_unconfirmed_channel()
    test_missing_official_marker()
    test_non_trailer_source()
    test_channel_token_match()

    print()
    print("Network requests: not performed")
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Official trailer verifier test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )