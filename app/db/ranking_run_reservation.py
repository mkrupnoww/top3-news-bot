from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Mapping

import asyncpg

from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.request_key import (
    RankingRequestKey,
)


@dataclass(frozen=True, slots=True)
class RankingRunReservation:
    """Результат резервирования ranking run."""

    ranking_run_id: int
    request_key: str
    run_status: str
    formula_version: str
    candidate_count: int
    created_new: bool

    @property
    def should_call_model(self) -> bool:
        """
        Показывает, разрешён ли вызов модели.

        Только процесс, создавший новую запись,
        может выполнить платный API-запрос.
        """

        return (
            self.created_new
            and self.run_status == "running"
        )


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный текст."""

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_datetime(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    """Приводит дату с часовым поясом к UTC."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} должен содержать "
            "часовой пояс."
        )

    return value.astimezone(
        timezone.utc
    )


def _normalize_news_ids(
    news_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Проверяет идентификаторы кандидатов."""

    if not news_ids:
        raise ValueError(
            "news_ids не может быть пустым."
        )

    for news_id in news_ids:
        if isinstance(news_id, bool):
            raise TypeError(
                "news_id не может быть bool."
            )

        if not isinstance(news_id, int):
            raise TypeError(
                "Каждый news_id должен быть int."
            )

        if news_id <= 0:
            raise ValueError(
                "Каждый news_id должен быть "
                "больше нуля."
            )

    if len(set(news_ids)) != len(news_ids):
        raise ValueError(
            "news_ids содержит дубликаты."
        )

    return news_ids


def _normalize_metadata(
    metadata: RankingEvaluatorMetadata,
) -> RankingEvaluatorMetadata:
    """Проверяет метаданные оценщика."""

    model_name = metadata.model_name

    if model_name is None:
        raise ValueError(
            "metadata.model_name обязателен "
            "для OpenAI-запуска."
        )

    return RankingEvaluatorMetadata(
        run_mode=_normalize_required_text(
            metadata.run_mode,
            field_name="metadata.run_mode",
        ),
        evaluator_name=_normalize_required_text(
            metadata.evaluator_name,
            field_name="metadata.evaluator_name",
        ),
        evaluator_version=(
            _normalize_required_text(
                metadata.evaluator_version,
                field_name=(
                    "metadata.evaluator_version"
                ),
            )
        ),
        prompt_version=_normalize_required_text(
            metadata.prompt_version,
            field_name=(
                "metadata.prompt_version"
            ),
        ),
        model_name=_normalize_required_text(
            model_name,
            field_name="metadata.model_name",
        ),
    )


async def _validate_news_items(
    connection: asyncpg.Connection,
    *,
    news_ids: tuple[int, ...],
) -> None:
    """Проверяет новости перед резервированием."""

    records = await connection.fetch(
        """
        SELECT
            news_id,
            processing_status
        FROM news_items
        WHERE news_id = ANY($1::bigint[])
        ORDER BY news_id
        """,
        list(news_ids),
    )

    found_ids = {
        record["news_id"]
        for record in records
    }

    missing_ids = sorted(
        set(news_ids) - found_ids
    )

    if missing_ids:
        raise LookupError(
            "Не найдены новости: "
            + ",".join(
                str(news_id)
                for news_id in missing_ids
            )
        )

    invalid_records = [
        record
        for record in records
        if record["processing_status"]
        not in {
            "collected",
            "candidate",
        }
    ]

    if invalid_records:
        details = ", ".join(
            (
                f"{record['news_id']}:"
                f"{record['processing_status']}"
            )
            for record in invalid_records
        )

        raise ValueError(
            "Новости имеют неподходящий статус: "
            f"{details}"
        )


def _validate_existing_run(
    record: asyncpg.Record,
    *,
    request_key: RankingRequestKey,
    formula_version: str,
    metadata: RankingEvaluatorMetadata,
    window_started_at: datetime,
    window_finished_at: datetime,
    news_ids: tuple[int, ...],
) -> None:
    """Проверяет найденное резервирование."""

    differences: list[str] = []

    expected_values = {
        "request_key": request_key.value,
        "formula_version": formula_version,
        "model_name": metadata.model_name,
        "prompt_version": (
            metadata.prompt_version
        ),
        "window_started_at": (
            window_started_at
        ),
        "window_finished_at": (
            window_finished_at
        ),
        "candidate_count": len(news_ids),
        "run_mode": metadata.run_mode,
        "evaluator_name": (
            metadata.evaluator_name
        ),
        "evaluator_version": (
            metadata.evaluator_version
        ),
        "request_key_version": (
            request_key.version
        ),
    }

    for field_name, expected_value in (
        expected_values.items()
    ):
        actual_value = record[field_name]

        if actual_value != expected_value:
            differences.append(
                f"{field_name}: "
                f"expected={expected_value!r}, "
                f"actual={actual_value!r}"
            )

    if record["news_ids_match"] is not True:
        differences.append(
            "news_ids: "
            f"expected={news_ids!r}, "
            "actual parameters differ"
        )

    if differences:
        raise ValueError(
            "request_key уже существует "
            "с другими параметрами: "
            + "; ".join(differences)
        )


def _build_reservation(
    record: asyncpg.Record,
    *,
    created_new: bool,
) -> RankingRunReservation:
    """Создаёт объект результата."""

    return RankingRunReservation(
        ranking_run_id=int(
            record["ranking_run_id"]
        ),
        request_key=record["request_key"],
        run_status=record["run_status"],
        formula_version=(
            record["formula_version"]
        ),
        candidate_count=int(
            record["candidate_count"]
        ),
        created_new=created_new,
    )


async def reserve_ranking_run(
    pool: asyncpg.Pool,
    *,
    request_key: RankingRequestKey,
    formula_version: str,
    metadata: RankingEvaluatorMetadata,
    window_started_at: datetime,
    window_finished_at: datetime,
    news_ids: tuple[int, ...],
) -> RankingRunReservation:
    """
    Резервирует запуск до вызова OpenAI.

    Уникальный request_key гарантирует, что
    только один процесс создаст новую запись
    и получит should_call_model=True.

    Повторный процесс получит существующую
    запись и не должен обращаться к модели.
    """

    normalized_formula_version = (
        _normalize_required_text(
            formula_version,
            field_name="formula_version",
        )
    )

    normalized_metadata = (
        _normalize_metadata(metadata)
    )

    normalized_window_start = (
        _normalize_datetime(
            window_started_at,
            field_name="window_started_at",
        )
    )

    normalized_window_finish = (
        _normalize_datetime(
            window_finished_at,
            field_name="window_finished_at",
        )
    )

    if (
        normalized_window_finish
        <= normalized_window_start
    ):
        raise ValueError(
            "window_finished_at должен быть "
            "позже window_started_at."
        )

    normalized_news_ids = (
        _normalize_news_ids(news_ids)
    )

    parameters = _encode_json(
        {
            "mode": (
                normalized_metadata.run_mode
            ),
            "run_mode": (
                normalized_metadata.run_mode
            ),
            "evaluator_name": (
                normalized_metadata
                .evaluator_name
            ),
            "evaluator_version": (
                normalized_metadata
                .evaluator_version
            ),
            "prompt_version": (
                normalized_metadata
                .prompt_version
            ),
            "model_name": (
                normalized_metadata.model_name
            ),
            "request_key_version": (
                request_key.version
            ),
            "news_ids": list(
                normalized_news_ids
            ),
            "idempotency_reserved": True,
        }
    )

    encoded_news_ids = _encode_json(
        {
            "news_ids": list(
                normalized_news_ids
            )
        }
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            await _validate_news_items(
                connection,
                news_ids=normalized_news_ids,
            )

            inserted_record = (
                await connection.fetchrow(
                    """
                    INSERT INTO ranking_runs (
                        run_status,
                        formula_version,
                        model_name,
                        prompt_version,
                        window_started_at,
                        window_finished_at,
                        candidate_count,
                        scored_count,
                        eligible_count,
                        parameters,
                        request_key
                    )
                    VALUES (
                        'running',
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        0,
                        0,
                        $7::jsonb,
                        $8
                    )
                    ON CONFLICT (request_key)
                    WHERE request_key IS NOT NULL
                    DO NOTHING
                    RETURNING
                        ranking_run_id,
                        request_key,
                        run_status,
                        formula_version,
                        candidate_count
                    """,
                    normalized_formula_version,
                    normalized_metadata.model_name,
                    normalized_metadata.prompt_version,
                    normalized_window_start,
                    normalized_window_finish,
                    len(normalized_news_ids),
                    parameters,
                    request_key.value,
                )
            )

            if inserted_record is not None:
                return _build_reservation(
                    inserted_record,
                    created_new=True,
                )

            existing_record = (
                await connection.fetchrow(
                    """
                    SELECT
                        ranking_run_id,
                        request_key,
                        run_status,
                        formula_version,
                        model_name,
                        prompt_version,
                        window_started_at,
                        window_finished_at,
                        candidate_count,
                        parameters->>'run_mode'
                            AS run_mode,
                        parameters->>'evaluator_name'
                            AS evaluator_name,
                        parameters->>'evaluator_version'
                            AS evaluator_version,
                        parameters->>'request_key_version'
                            AS request_key_version,
                        (
                            parameters->'news_ids'
                            =
                            (
                                $2::jsonb
                                -> 'news_ids'
                            )
                        ) AS news_ids_match
                    FROM ranking_runs
                    WHERE request_key = $1
                    FOR UPDATE
                    """,
                    request_key.value,
                    encoded_news_ids,
                )
            )

            if existing_record is None:
                raise RuntimeError(
                    "Не удалось получить "
                    "существующий ranking_run "
                    "после конфликта request_key."
                )

            _validate_existing_run(
                existing_record,
                request_key=request_key,
                formula_version=(
                    normalized_formula_version
                ),
                metadata=normalized_metadata,
                window_started_at=(
                    normalized_window_start
                ),
                window_finished_at=(
                    normalized_window_finish
                ),
                news_ids=normalized_news_ids,
            )

            return _build_reservation(
                existing_record,
                created_new=False,
            )