import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json

from app.db.generation_revision_selection import (
    _hydrate_trailer_metadata,
)
from app.db.generation_selection import (
    GenerationTop3Selection,
)
from app.generation.official_trailer_enrichment import (
    OfficialTrailerEnrichmentResult,
    preflight_generation_official_trailers,
    source_requires_official_trailer,
)
from app.generation.openai_generator import (
    GenerationNewsItem,
    OpenAIGeneratedNewsPayload,
    build_official_trailer_markdown,
    build_top3_post_text,
)
from app.generation.trailer_extractor import (
    extract_youtube_document_urls,
)


TRAILER_URL = "https://www.youtube.com/watch?v=B3tR6qQjbgI"


def _items() -> tuple[GenerationNewsItem, GenerationNewsItem, GenerationNewsItem]:
    common = dict(
        source_name="Trade",
        source_published_at=datetime(2026, 9, 4, 7, 0, tzinfo=timezone.utc),
        individual_score=Decimal("4.2"),
        selection_reason="test",
    )
    return (
        GenerationNewsItem(
            position=1,
            news_id=1,
            title="First film gets release date",
            summary="The studio dated the film.",
            source_url="https://example.com/1",
            **common,
        ),
        GenerationNewsItem(
            position=2,
            news_id=2,
            title="Netflix releases official trailer for Unabomber",
            summary="The new trailer stars Jacob Tremblay.",
            source_url="https://example.com/2",
            **common,
        ),
        GenerationNewsItem(
            position=3,
            news_id=3,
            title="Third film begins production",
            summary="Production has started.",
            source_url="https://example.com/3",
            **common,
        ),
    )


def test_markdown_builder() -> None:
    assert build_official_trailer_markdown(
        TRAILER_URL,
        "Netflix",
    ) == f"[▶️ Официальный трейлер Netflix]({TRAILER_URL})"

    assert build_official_trailer_markdown(
        TRAILER_URL,
        "bad[channel]",
    ) == f"[▶️ Официальный трейлер]({TRAILER_URL})"

    assert build_official_trailer_markdown(
        TRAILER_URL,
        "bad_channel*",
    ) == f"[▶️ Официальный трейлер]({TRAILER_URL})"

    payload_items = [
        OpenAIGeneratedNewsPayload(
            position=1,
            news_id=1,
            headline="Первая новость",
            body="Текст первой новости.",
        ),
        OpenAIGeneratedNewsPayload(
            position=2,
            news_id=2,
            headline="Netflix выпустил трейлер Unabomber",
            body=(
                "Netflix выпустил первый трейлер Unabomber. "
                "В фильме также снялись Рассел Кроу и Шейлин Вудли."
            ),
            official_trailer_url=TRAILER_URL,
            official_trailer_channel_name="Netflix",
        ),
        OpenAIGeneratedNewsPayload(
            position=3,
            news_id=3,
            headline="Третья новость",
            body="Текст третьей новости.",
        ),
    ]

    post = build_top3_post_text(payload_items)
    assert f"[▶️ Официальный трейлер Netflix]({TRAILER_URL})" in post
    assert post.count(TRAILER_URL) == 1
    assert "В фильме также снялись" in post

    print("Deterministic trailer Markdown: OK")
    print("Channel-name fallback: OK")


def test_document_extractor() -> None:
    html = r'''
    <a href="https://youtu.be/B3tR6qQjbgI">watch</a>
    <script>
      {"embed":"https:\/\/www.youtube.com\/embed\/B3tR6qQjbgI"}
    </script>
    '''
    assert extract_youtube_document_urls(html) == (TRAILER_URL,)
    print("YouTube document extraction: OK")


class _VerifiedEnricher:
    async def __call__(self, **kwargs):
        del kwargs
        return OfficialTrailerEnrichmentResult(
            attempted=True,
            verified=True,
            official_trailer_url=TRAILER_URL,
            reason="verified_official_trailer",
            article_final_url="https://example.com/2",
            youtube_candidate_urls=(TRAILER_URL,),
            checked_video_urls=(TRAILER_URL,),
            verification_reasons=("verified_official_trailer",),
            oembed_error_count=0,
            error_type=None,
            official_trailer_channel_name="Netflix",
        )


class _UnverifiedEnricher:
    async def __call__(self, **kwargs):
        del kwargs
        return OfficialTrailerEnrichmentResult(
            attempted=True,
            verified=False,
            official_trailer_url=None,
            reason="official_trailer_not_verified",
            article_final_url="https://example.com/2",
            youtube_candidate_urls=(),
            checked_video_urls=(),
            verification_reasons=(),
            oembed_error_count=0,
            error_type=None,
        )


async def test_preflight() -> None:
    items = _items()

    assert source_requires_official_trailer(
        items[1].title,
        items[1].summary,
    ) is True
    assert source_requires_official_trailer(
        items[0].title,
        items[0].summary,
    ) is False

    verified = await preflight_generation_official_trailers(
        items,
        trailer_enricher=_VerifiedEnricher(),
    )
    assert verified.ready is True
    assert verified.required_news_ids == (2,)
    assert verified.verified_news_ids == (2,)
    assert verified.items[1].official_trailer_url == TRAILER_URL
    assert verified.items[1].official_trailer_channel_name == "Netflix"

    unverified = await preflight_generation_official_trailers(
        items,
        trailer_enricher=_UnverifiedEnricher(),
    )
    assert unverified.ready is False
    assert unverified.unverified_required_news_ids == (2,)

    print("Trailer preflight verified path: OK")
    print("Trailer preflight non-fatal unverified path: OK")


def test_revision_metadata_hydration() -> None:
    items = _items()
    selection = GenerationTop3Selection(
        ranking_run_id=100,
        run_status="completed",
        eligible_count=3,
        score_ids=(11, 12, 13),
        items=items,
    )

    metadata_json = json.dumps(
        {
            "generated_items": [
                {"position": 1, "news_id": 1, "headline": "a", "body": "a"},
                {
                    "position": 2,
                    "news_id": 2,
                    "headline": "b",
                    "body": "b",
                    "official_trailer_url": TRAILER_URL,
                    "official_trailer_channel_name": "Netflix",
                },
                {"position": 3, "news_id": 3, "headline": "c", "body": "c"},
            ]
        }
    )

    hydrated = _hydrate_trailer_metadata(selection, metadata_json)
    assert hydrated.items[1].official_trailer_url == TRAILER_URL
    assert hydrated.items[1].official_trailer_channel_name == "Netflix"

    # Historical metadata without a channel remains safe: the renderer will
    # use the generic visible label instead of failing the revision.
    legacy = json.dumps(
        {
            "generated_items": [
                {"position": 2, "news_id": 2, "official_trailer_url": TRAILER_URL}
            ]
        }
    )
    legacy_hydrated = _hydrate_trailer_metadata(selection, legacy)
    assert legacy_hydrated.items[1].official_trailer_channel_name is None

    print("Revision trailer metadata hydration: OK")
    print("Historical generic-label fallback: OK")


async def main() -> int:
    print("Trailer contract isolated test")
    print("database_connections=not_performed")
    print("openai_requests=not_performed")
    print("telegram_requests=not_performed")
    print()

    test_markdown_builder()
    test_document_extractor()
    await test_preflight()
    test_revision_metadata_hydration()

    print()
    print("Trailer contract test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
