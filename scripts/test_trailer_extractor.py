from app.generation.trailer_extractor import (
    build_youtube_watch_url,
    extract_youtube_iframe_urls,
    extract_youtube_video_id,
)


VIDEO_ID = "5fHXyqQOKL8"
WATCH_URL = (
    "https://www.youtube.com/watch?v="
    f"{VIDEO_ID}"
)


def test_video_id_extraction() -> None:
    """Проверяет основные формы YouTube URL."""

    cases = {
        (
            "https://www.youtube.com/"
            f"embed/{VIDEO_ID}?start=128"
        ): VIDEO_ID,
        (
            "https://www.youtube.com/"
            f"watch?v={VIDEO_ID}"
        ): VIDEO_ID,
        (
            "https://youtu.be/"
            f"{VIDEO_ID}?t=128"
        ): VIDEO_ID,
        (
            "https://www.youtube-nocookie.com/"
            f"embed/{VIDEO_ID}"
        ): VIDEO_ID,
        (
            "https://www.youtube.com/"
            f"shorts/{VIDEO_ID}"
        ): VIDEO_ID,
        (
            "https://example.com/"
            f"embed/{VIDEO_ID}"
        ): None,
        "javascript:alert(1)": None,
        "": None,
    }

    for url, expected_video_id in (
        cases.items()
    ):
        assert (
            extract_youtube_video_id(url)
            == expected_video_id
        )

    print("YouTube video ID extraction: OK")


def test_watch_url_builder() -> None:
    """Проверяет canonical watch URL."""

    assert (
        build_youtube_watch_url(VIDEO_ID)
        == WATCH_URL
    )

    print()
    print("YouTube watch URL builder: OK")
    print(f"watch_url={WATCH_URL}")


def test_thr_like_iframe_extraction() -> None:
    """Проверяет HTML, похожий на статью THR."""

    html_content = f"""
    <html>
      <body>
        <p>Article text before video.</p>
        <figure
          class="wp-block-embed
                 is-provider-youtube"
        >
          <div class="wp-block-embed__wrapper">
            <span class="embed-youtube">
              <iframe
                loading="lazy"
                class="youtube-player"
                width="640"
                height="360"
                src="https://www.youtube.com/embed/{VIDEO_ID}?version=3&amp;rel=1&amp;start=128"
                allowfullscreen="true"
              ></iframe>
            </span>
          </div>
        </figure>
        <p>Primetime is set to hit theaters.</p>
      </body>
    </html>
    """

    urls = extract_youtube_iframe_urls(
        html_content
    )

    assert urls == (WATCH_URL,)

    print()
    print("THR-like iframe extraction: OK")
    print(f"youtube_url={urls[0]}")


def test_duplicates_and_non_youtube() -> None:
    """Проверяет дедупликацию и чужие iframe."""

    html_content = f"""
    <iframe
      src="https://www.youtube.com/embed/{VIDEO_ID}"
    ></iframe>
    <iframe
      src="https://youtu.be/{VIDEO_ID}"
    ></iframe>
    <iframe
      src="https://player.vimeo.com/video/12345"
    ></iframe>
    """

    urls = extract_youtube_iframe_urls(
        html_content
    )

    assert urls == (WATCH_URL,)

    print()
    print("YouTube iframe deduplication: OK")
    print("youtube_count=1")
    print("non_youtube_ignored=true")


def test_empty_html() -> None:
    """Проверяет HTML без YouTube iframe."""

    urls = extract_youtube_iframe_urls(
        "<html><body>No video</body></html>"
    )

    assert urls == ()

    print()
    print("HTML without YouTube: OK")


def main() -> int:
    """Запускает pure-Python тест extractor."""

    test_video_id_extraction()
    test_watch_url_builder()
    test_thr_like_iframe_extraction()
    test_duplicates_and_non_youtube()
    test_empty_html()

    print()
    print("Network requests: not performed")
    print("Database changes: not performed")
    print("OpenAI requests: not performed")
    print("Telegram publication: not performed")
    print("Trailer extractor test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )