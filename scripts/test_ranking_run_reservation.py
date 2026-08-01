import asyncio
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

import asyncpg

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.ranking_run_reservation import (
    reserve_ranking_run,
)
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_evaluator import (
    OPENAI_EVALUATOR_VERSION,
    OPENAI_PROMPT_VERSION,
)
from app.ranking.request_key import (
    REQUEST_KEY_VERSION,
    RankingRequestKey,
)
from app.ranking.score_formula import (
    FORMULA_VERSION,
)


TEST_NEWS_IDS = (
    7,
    8,
    9,
)


def build_metadata() -> RankingEvaluatorMetadata:
    """Создаёт метаданные OpenAI-оценщика."""

    return RankingEvaluatorMetadata(
        run_mode="openai_ranking",
        evaluator_name=(
            "OpenAIRankingEvaluator"
        ),
        evaluator_version=(
            OPENAI_EVALUATOR_VERSION
        ),
        prompt_version=(
            OPENAI_PROMPT_VERSION
        ),
        model_name="gpt-5.6-terra",
    )


def build_test_request_key() -> RankingRequestKey:
    """Создаёт уникальный ключ тестового запуска."""

    payload = {
        "test": (
            "ranking_run_reservation"
        ),
        "nonce": uuid4().hex,
    }

    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    value = sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

    return RankingRequestKey(
        value=value,
        version=REQUEST_KEY_VERSION,
        canonical_json=canonical_json,
    )


def decode_jsonb_array(
    value: Any,
) -> list[int]:
    """
    Преобразует значение jsonb в список int.

    asyncpg по умолчанию может вернуть jsonb
    как строку JSON, а при пользовательском
    кодеке — уже как Python-объект.
    """

    decoded_value: Any

    if isinstance(value, str):
        try:
            decoded_value = json.loads(value)
        except json.JSONDecodeError as error:
            raise AssertionError(
                "Поле jsonb содержит "
                "некорректный JSON."
            ) from error
    else:
        decoded_value = value

    if not isinstance(decoded_value, list):
        raise AssertionError(
            "Поле jsonb news_ids "
            "не является массивом."
        )

    result: list[int] = []

    for item in decoded_value:
        if isinstance(item, bool):
            raise AssertionError(
                "news_ids содержит bool."
            )

        if not isinstance(item, int):
            raise AssertionError(
                "news_ids содержит значение "
                "не типа int."
            )

        result.append(item)

    return result


async def delete_test_run(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> None:
    """Удаляет временный тестовый запуск."""

    async with pool.acquire() as connection:
        result = await connection.execute(
            """
            DELETE FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            ranking_run_id,
        )

    if result != "DELETE 1":
        raise RuntimeError(
            "Не удалось удалить тестовый "
            f"ranking_run: {result}"
        )


async def assert_test_run_deleted(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> None:
    """Проверяет удаление тестовой записи."""

    async with pool.acquire() as connection:
        exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM top3_news.ranking_runs
                WHERE ranking_run_id = $1
            )
            """,
            ranking_run_id,
        )

    if exists:
        raise AssertionError(
            "Тестовый ranking_run "
            "остался в базе данных."
        )


async def test_reservation(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Проверяет резервирование запуска."""

    request_key = build_test_request_key()
    metadata = build_metadata()

    window_started_at = datetime(
        2026,
        7,
        30,
        11,
        21,
        tzinfo=timezone.utc,
    )

    window_finished_at = datetime(
        2026,
        7,
        31,
        11,
        21,
        tzinfo=timezone.utc,
    )

    first_reservation = (
        await reserve_ranking_run(
            pool,
            request_key=request_key,
            formula_version=FORMULA_VERSION,
            metadata=metadata,
            window_started_at=(
                window_started_at
            ),
            window_finished_at=(
                window_finished_at
            ),
            news_ids=TEST_NEWS_IDS,
        )
    )

    # Сохраняем ID немедленно, чтобы finally
    # удалил запись при любом последующем сбое.
    created_run_ids.add(
        first_reservation.ranking_run_id
    )

    assert first_reservation.created_new is True

    assert (
        first_reservation.should_call_model
        is True
    )

    assert (
        first_reservation.run_status
        == "running"
    )

    assert (
        first_reservation.request_key
        == request_key.value
    )

    assert (
        first_reservation.formula_version
        == FORMULA_VERSION
    )

    assert (
        first_reservation.candidate_count
        == len(TEST_NEWS_IDS)
    )

    print("Initial reservation: OK")
    print(
        "ranking_run_id="
        f"{first_reservation.ranking_run_id}"
    )
    print(
        "request_key="
        f"{first_reservation.request_key}"
    )
    print(
        "formula_version="
        f"{first_reservation.formula_version}"
    )
    print("created_new=true")
    print("should_call_model=true")
    print("run_status=running")

    second_reservation = (
        await reserve_ranking_run(
            pool,
            request_key=request_key,
            formula_version=FORMULA_VERSION,
            metadata=metadata,
            window_started_at=(
                window_started_at
            ),
            window_finished_at=(
                window_finished_at
            ),
            news_ids=TEST_NEWS_IDS,
        )
    )

    assert (
        second_reservation.ranking_run_id
        == first_reservation.ranking_run_id
    )

    assert (
        second_reservation.created_new
        is False
    )

    assert (
        second_reservation.should_call_model
        is False
    )

    assert (
        second_reservation.run_status
        == "running"
    )

    print()
    print("Repeated reservation: OK")
    print(
        "ranking_run_id="
        f"{second_reservation.ranking_run_id}"
    )
    print("created_new=false")
    print("should_call_model=false")
    print(
        "Duplicate paid request: blocked"
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                ranking_run_id,
                request_key,
                run_status,
                formula_version,
                model_name,
                prompt_version,
                candidate_count,
                scored_count,
                eligible_count,
                parameters->>'run_mode'
                    AS run_mode,
                parameters->>'evaluator_name'
                    AS evaluator_name,
                parameters->>'evaluator_version'
                    AS evaluator_version,
                parameters->>'request_key_version'
                    AS request_key_version,
                parameters->'news_ids'
                    AS news_ids
            FROM top3_news.ranking_runs
            WHERE ranking_run_id = $1
            """,
            first_reservation.ranking_run_id,
        )

    if record is None:
        raise AssertionError(
            "Зарезервированный ranking_run "
            "не найден."
        )

    persisted_news_ids = decode_jsonb_array(
        record["news_ids"]
    )

    assert (
        record["request_key"]
        == request_key.value
    )

    assert record["run_status"] == "running"

    assert (
        record["formula_version"]
        == FORMULA_VERSION
    )

    assert (
        record["model_name"]
        == metadata.model_name
    )

    assert (
        record["prompt_version"]
        == OPENAI_PROMPT_VERSION
    )

    assert (
        record["candidate_count"]
        == len(TEST_NEWS_IDS)
    )

    assert record["scored_count"] == 0
    assert record["eligible_count"] == 0

    assert (
        record["run_mode"]
        == metadata.run_mode
    )

    assert (
        record["evaluator_name"]
        == metadata.evaluator_name
    )

    assert (
        record["evaluator_version"]
        == OPENAI_EVALUATOR_VERSION
    )

    assert (
        record["request_key_version"]
        == REQUEST_KEY_VERSION
    )

    assert (
        persisted_news_ids
        == list(TEST_NEWS_IDS)
    )

    print()
    print("Persisted reservation data: OK")
    print(
        "formula_version="
        f"{record['formula_version']}"
    )
    print(
        "prompt_version="
        f"{record['prompt_version']}"
    )
    print(
        "evaluator_version="
        f"{record['evaluator_version']}"
    )
    print(
        "candidate_count="
        f"{record['candidate_count']}"
    )
    print(
        "scored_count="
        f"{record['scored_count']}"
    )
    print(
        "eligible_count="
        f"{record['eligible_count']}"
    )
    print(
        "request_key_version="
        f"{record['request_key_version']}"
    )
    print(
        "news_ids="
        + ",".join(
            str(news_id)
            for news_id in persisted_news_ids
        )
    )

    try:
        await reserve_ranking_run(
            pool,
            request_key=request_key,
            formula_version=(
                "different_formula_version"
            ),
            metadata=metadata,
            window_started_at=(
                window_started_at
            ),
            window_finished_at=(
                window_finished_at
            ),
            news_ids=TEST_NEWS_IDS,
        )
    except ValueError as error:
        assert (
            "request_key уже существует "
            "с другими параметрами"
            in str(error)
        )

        print()
        print(
            "Conflicting reservation "
            "blocking: OK"
        )
    else:
        raise AssertionError(
            "Конфликт параметров "
            "не был заблокирован."
        )


async def cleanup_test_runs(
    pool: asyncpg.Pool,
    *,
    created_run_ids: set[int],
) -> None:
    """Удаляет все созданные тестом запуски."""

    for ranking_run_id in sorted(
        created_run_ids
    ):
        await delete_test_run(
            pool,
            ranking_run_id=ranking_run_id,
        )

        await assert_test_run_deleted(
            pool,
            ranking_run_id=ranking_run_id,
        )

        print()
        print("Test data cleanup: OK")
        print(
            "temporary_ranking_run_id="
            f"{ranking_run_id}"
        )
        print(
            "temporary_ranking_run_deleted=true"
        )


async def main() -> int:
    """Запускает интеграционный тест."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    created_run_ids: set[int] = set()

    try:
        await test_reservation(
            pool,
            created_run_ids=created_run_ids,
        )
    finally:
        try:
            await cleanup_test_runs(
                pool,
                created_run_ids=created_run_ids,
            )
        finally:
            await close_database_pool(pool)

    print()
    print("OpenAI requests: not performed")
    print(
        "Database changes: "
        "temporary row inserted and deleted"
    )
    print("Telegram publication: not performed")
    print(
        "Ranking run reservation test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )