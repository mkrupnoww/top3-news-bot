from dataclasses import dataclass
import logging

import asyncpg

from app.collectors.feed_http import (
    download_feed_document,
)
from app.collectors.feed_parser import (
    parse_feed_document,
)
from app.db.feed_collection import (
    mark_feed_collection_failed,
    persist_feed_collection,
    start_feed_collection_run,
)


logger = logging.getLogger(__name__)


COLLECTOR_NAME = "rss_atom_http"
COLLECTOR_VERSION = "1"


@dataclass(frozen=True, slots=True)
class FeedCollectionResult:
    """Итог полного запуска RSS/Atom-сборщика."""

    source_id: int
    collection_run_id: int
    run_status: str
    feed_title: str | None
    feed_type: str
    requested_url: str
    final_url: str
    bytes_downloaded: int
    redirect_count: int
    fetched_count: int
    inserted_count: int
    duplicate_count: int
    rejected_count: int
    news_ids: tuple[int, ...]


async def collect_feed(
    pool: asyncpg.Pool,
    *,
    source_code: str,
    source_name: str,
    feed_url: str,
    base_url: str | None,
    language_code: str,
    collection_priority: int = 100,
    max_entries: int = 100,
    timeout_seconds: float = 15.0,
    max_response_bytes: int = 2_000_000,
) -> FeedCollectionResult:
    """Загружает, разбирает и сохраняет RSS/Atom-ленту."""

    run = await start_feed_collection_run(
        pool,
        source_code=source_code,
        source_name=source_name,
        feed_url=feed_url,
        base_url=base_url,
        language_code=language_code,
        collection_priority=collection_priority,
        collector_name=COLLECTOR_NAME,
        collector_version=COLLECTOR_VERSION,
    )

    current_stage = "http_download"

    try:
        download_result = (
            await download_feed_document(
                feed_url,
                timeout_seconds=timeout_seconds,
                max_response_bytes=(
                    max_response_bytes
                ),
            )
        )

        current_stage = "xml_parsing"

        parse_result = parse_feed_document(
            download_result.content,
            max_entries=max_entries,
            max_document_bytes=(
                max_response_bytes
            ),
        )

        current_stage = "database_persistence"

        persistence_result = (
            await persist_feed_collection(
                pool,
                run=run,
                language_code=language_code,
                download_result=download_result,
                parse_result=parse_result,
            )
        )

    except Exception as error:
        error_message = (
            f"{type(error).__name__}: {error}"
        )

        try:
            await mark_feed_collection_failed(
                pool,
                collection_run_id=(
                    run.collection_run_id
                ),
                error_message=error_message,
                failed_stage=current_stage,
            )
        except Exception:
            logger.exception(
                "Не удалось сохранить статус failed: "
                "collection_run_id=%s",
                run.collection_run_id,
            )

        raise

    return FeedCollectionResult(
        source_id=run.source_id,
        collection_run_id=(
            run.collection_run_id
        ),
        run_status=(
            persistence_result.run_status
        ),
        feed_title=parse_result.feed_title,
        feed_type=parse_result.feed_type,
        requested_url=(
            download_result.requested_url
        ),
        final_url=download_result.final_url,
        bytes_downloaded=(
            download_result.bytes_downloaded
        ),
        redirect_count=(
            download_result.redirect_count
        ),
        fetched_count=(
            persistence_result.fetched_count
        ),
        inserted_count=(
            persistence_result.inserted_count
        ),
        duplicate_count=(
            persistence_result.duplicate_count
        ),
        rejected_count=(
            persistence_result.rejected_count
        ),
        news_ids=persistence_result.news_ids,
    )