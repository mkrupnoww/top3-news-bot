from datetime import timedelta

import asyncpg

from app.db.daily_workflow import (
    DailyWorkflowRun,
    load_daily_workflow,
)


_RUNNING_STAGE_ORDER = {
    "reserved": 0,
    "ranking": 1,
    "generation": 2,
    "image": 3,
    "review_delivery": 4,
}

_SHA256_ALPHABET = frozenset(
    "0123456789abcdef"
)


class DailyWorkflowRecoveryAmbiguousError(
    RuntimeError
):
    """Recovery нашёл несколько логически подходящих записей."""


def _positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительный integer."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} должен быть больше нуля."
        )

    return value


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный непустой текст."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть str."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized


def _normalize_sha256(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет lowercase SHA-256."""

    normalized = _normalize_required_text(
        value,
        field_name=field_name,
    )

    if (
        len(normalized) != 64
        or any(
            character not in _SHA256_ALPHABET
            for character in normalized
        )
    ):
        raise ValueError(
            f"{field_name} должен быть "
            "lowercase SHA-256."
        )

    return normalized


def _advance_stage(
    current_stage: str,
    target_stage: str,
) -> str:
    """Не позволяет откатывать running workflow назад."""

    if current_stage not in _RUNNING_STAGE_ORDER:
        raise ValueError(
            "Некорректная running stage: "
            f"{current_stage}"
        )

    if target_stage not in _RUNNING_STAGE_ORDER:
        raise ValueError(
            "Некорректная target stage: "
            f"{target_stage}"
        )

    if (
        _RUNNING_STAGE_ORDER[target_stage]
        > _RUNNING_STAGE_ORDER[current_stage]
    ):
        return target_stage

    return current_stage


async def _load_workflow_for_update(
    connection: asyncpg.Connection,
    *,
    daily_workflow_run_id: int,
) -> asyncpg.Record:
    """Читает workflow с блокировкой строки."""

    record = await connection.fetchrow(
        """
        SELECT
            daily_workflow_run_id,
            publication_date,
            workflow_version,
            workflow_status,
            current_stage,
            as_of,
            window_hours,
            target_telegram_chat_id,
            ranking_run_id,
            batch_id,
            generated_post_id,
            image_generation_id
        FROM daily_workflow_runs
        WHERE daily_workflow_run_id = $1
        FOR UPDATE
        """,
        daily_workflow_run_id,
    )

    if record is None:
        raise LookupError(
            "daily_workflow_run не найден: "
            f"daily_workflow_run_id="
            f"{daily_workflow_run_id}"
        )

    return record


def _require_running(
    record: asyncpg.Record,
) -> None:
    """Запрещает checkpoint terminal workflow."""

    if record["workflow_status"] != "running":
        raise ValueError(
            "Checkpoint разрешён только для "
            "running daily workflow: "
            f"workflow_status="
            f"{record['workflow_status']}"
        )


async def checkpoint_ranking_reservation(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    ranking_run_id: int,
) -> DailyWorkflowRun:
    """
    Закрепляет ranking reservation за daily workflow.

    Допускаются running/completed/failed ranking,
    потому что checkpoint фиксирует identity этапа,
    а не его успешность.
    """

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    ranking_id = _positive_integer(
        ranking_run_id,
        field_name="ranking_run_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await _load_workflow_for_update(
                connection,
                daily_workflow_run_id=workflow_id,
            )

            _require_running(workflow)

            ranking = await connection.fetchrow(
                """
                SELECT
                    ranking_run_id,
                    run_status,
                    window_started_at,
                    window_finished_at
                FROM ranking_runs
                WHERE ranking_run_id = $1
                """,
                ranking_id,
            )

            if ranking is None:
                raise LookupError(
                    "ranking_run не найден: "
                    f"ranking_run_id={ranking_id}"
                )

            expected_start = (
                workflow["as_of"]
                - timedelta(
                    hours=int(
                        workflow["window_hours"]
                    )
                )
            )

            if (
                ranking["window_finished_at"]
                != workflow["as_of"]
            ):
                raise ValueError(
                    "ranking window_finished_at "
                    "не совпадает с workflow as_of."
                )

            if (
                ranking["window_started_at"]
                != expected_start
            ):
                raise ValueError(
                    "ranking window_started_at "
                    "не совпадает с workflow window."
                )

            existing_id = workflow[
                "ranking_run_id"
            ]

            if (
                existing_id is not None
                and int(existing_id) != ranking_id
            ):
                raise ValueError(
                    "Daily workflow уже связан "
                    "с другим ranking_run_id: "
                    f"existing={existing_id}, "
                    f"new={ranking_id}"
                )

            target_stage = _advance_stage(
                workflow["current_stage"],
                "ranking",
            )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET
                    ranking_run_id = $2,
                    current_stage = $3
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                ranking_id,
                target_stage,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def checkpoint_generation_reservation(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    batch_id: int,
) -> DailyWorkflowRun:
    """Закрепляет publication batch до внешних generation calls."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_batch_id = _positive_integer(
        batch_id,
        field_name="batch_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await _load_workflow_for_update(
                connection,
                daily_workflow_run_id=workflow_id,
            )

            _require_running(workflow)

            if workflow["ranking_run_id"] is None:
                raise ValueError(
                    "Перед generation reservation "
                    "должен быть закреплён ranking_run."
                )

            batch = await connection.fetchrow(
                """
                SELECT
                    batch_id,
                    publication_date,
                    ranking_run_id,
                    batch_status,
                    target_telegram_chat_id
                FROM publication_batches
                WHERE batch_id = $1
                """,
                normalized_batch_id,
            )

            if batch is None:
                raise LookupError(
                    "publication_batch не найден: "
                    f"batch_id={normalized_batch_id}"
                )

            if (
                batch["ranking_run_id"]
                != workflow["ranking_run_id"]
            ):
                raise ValueError(
                    "Batch связан с другим "
                    "ranking_run_id."
                )

            if (
                batch["publication_date"]
                != workflow["publication_date"]
            ):
                raise ValueError(
                    "Batch publication_date "
                    "не совпадает с daily workflow."
                )

            if (
                batch["target_telegram_chat_id"]
                != workflow[
                    "target_telegram_chat_id"
                ]
            ):
                raise ValueError(
                    "Batch target Telegram chat "
                    "не совпадает с daily workflow."
                )

            existing_batch_id = workflow[
                "batch_id"
            ]

            if (
                existing_batch_id is not None
                and int(existing_batch_id)
                != normalized_batch_id
            ):
                raise ValueError(
                    "Daily workflow уже связан "
                    "с другим batch_id: "
                    f"existing={existing_batch_id}, "
                    f"new={normalized_batch_id}"
                )

            target_stage = _advance_stage(
                workflow["current_stage"],
                "generation",
            )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET
                    batch_id = $2,
                    current_stage = $3
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                normalized_batch_id,
                target_stage,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def checkpoint_generated_post(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    generated_post_id: int,
) -> DailyWorkflowRun:
    """Закрепляет уже сохранённый generated_post."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_post_id = _positive_integer(
        generated_post_id,
        field_name="generated_post_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await _load_workflow_for_update(
                connection,
                daily_workflow_run_id=workflow_id,
            )

            _require_running(workflow)

            if workflow["batch_id"] is None:
                raise ValueError(
                    "Перед generated_post должен "
                    "быть закреплён batch_id."
                )

            post = await connection.fetchrow(
                """
                SELECT
                    generated_post_id,
                    batch_id,
                    post_status
                FROM generated_posts
                WHERE generated_post_id = $1
                """,
                normalized_post_id,
            )

            if post is None:
                raise LookupError(
                    "generated_post не найден: "
                    f"generated_post_id="
                    f"{normalized_post_id}"
                )

            if (
                post["batch_id"]
                != workflow["batch_id"]
            ):
                raise ValueError(
                    "generated_post принадлежит "
                    "другому batch."
                )

            existing_post_id = workflow[
                "generated_post_id"
            ]

            if (
                existing_post_id is not None
                and int(existing_post_id)
                != normalized_post_id
            ):
                raise ValueError(
                    "Daily workflow уже связан "
                    "с другим generated_post_id: "
                    f"existing={existing_post_id}, "
                    f"new={normalized_post_id}"
                )

            target_stage = _advance_stage(
                workflow["current_stage"],
                "generation",
            )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET
                    generated_post_id = $2,
                    current_stage = $3
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                normalized_post_id,
                target_stage,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def checkpoint_image_reservation(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    image_generation_id: int,
) -> DailyWorkflowRun:
    """Закрепляет initial image reservation до Image API."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_image_id = _positive_integer(
        image_generation_id,
        field_name="image_generation_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await _load_workflow_for_update(
                connection,
                daily_workflow_run_id=workflow_id,
            )

            _require_running(workflow)

            if (
                workflow["batch_id"] is None
                or workflow["generated_post_id"] is None
            ):
                raise ValueError(
                    "Перед image reservation должны "
                    "быть закреплены batch/post."
                )

            image = await connection.fetchrow(
                """
                SELECT
                    image_generation_id,
                    batch_id,
                    generated_post_id,
                    request_kind,
                    image_status
                FROM image_generation_requests
                WHERE image_generation_id = $1
                """,
                normalized_image_id,
            )

            if image is None:
                raise LookupError(
                    "image_generation_request "
                    "не найден: "
                    f"image_generation_id="
                    f"{normalized_image_id}"
                )

            if image["request_kind"] != "initial":
                raise ValueError(
                    "Daily workflow может закреплять "
                    "только initial image request."
                )

            if (
                image["batch_id"]
                != workflow["batch_id"]
            ):
                raise ValueError(
                    "Image request связан "
                    "с другим batch."
                )

            if (
                image["generated_post_id"]
                != workflow["generated_post_id"]
            ):
                raise ValueError(
                    "Image request связан "
                    "с другим generated_post."
                )

            existing_image_id = workflow[
                "image_generation_id"
            ]

            if (
                existing_image_id is not None
                and int(existing_image_id)
                != normalized_image_id
            ):
                existing_image = await connection.fetchrow(
                    """
                    SELECT
                        image_generation_id,
                        batch_id,
                        generated_post_id,
                        request_kind,
                        image_status
                    FROM image_generation_requests
                    WHERE image_generation_id = $1
                    FOR UPDATE
                    """,
                    int(existing_image_id),
                )

                if existing_image is None:
                    raise ValueError(
                        "Daily workflow связан "
                        "с отсутствующим previous "
                        "image_generation_id: "
                        f"{existing_image_id}"
                    )

                replace_allowed = (
                    existing_image["request_kind"]
                    == "initial"
                    and existing_image["image_status"]
                    == "failed"
                    and existing_image["batch_id"]
                    == workflow["batch_id"]
                    and existing_image[
                        "generated_post_id"
                    ]
                    == workflow["generated_post_id"]
                )

                if not replace_allowed:
                    raise ValueError(
                        "Daily workflow уже связан "
                        "с другим active/non-failed "
                        "image_generation_id: "
                        f"existing={existing_image_id}, "
                        f"new={normalized_image_id}"
                    )

            target_stage = _advance_stage(
                workflow["current_stage"],
                "image",
            )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET
                    image_generation_id = $2,
                    current_stage = $3
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                normalized_image_id,
                target_stage,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def recover_ranking_run_id(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    formula_version: str,
    model_name: str,
    prompt_version: str,
    run_mode: str,
    evaluator_name: str,
    evaluator_version: str,
    request_key_version: str,
) -> int | None:
    """
    Восстанавливает ranking reservation
    после crash-gap между reserve_ranking_run()
    и checkpoint observer.

    Workflow cutoff/window, полная identity
    текущего ranking runtime и короткое окно
    после workflow.started_at используются
    для поиска совместимой reservation.

    Candidate set намеренно не вычисляется
    повторно: после reservation в БД могли
    появиться late-arriving RSS items.

    Если найдено несколько совместимых runs,
    recovery считается неоднозначным и
    автоматически продолжаться не может.
    """

    workflow = await load_daily_workflow(
        pool,
        daily_workflow_run_id=(
            daily_workflow_run_id
        ),
    )

    if workflow.ranking_run_id is not None:
        return workflow.ranking_run_id

    normalized_formula_version = (
        _normalize_required_text(
            formula_version,
            field_name="formula_version",
        )
    )

    normalized_model_name = (
        _normalize_required_text(
            model_name,
            field_name="model_name",
        )
    )

    normalized_prompt_version = (
        _normalize_required_text(
            prompt_version,
            field_name="prompt_version",
        )
    )

    normalized_run_mode = (
        _normalize_required_text(
            run_mode,
            field_name="run_mode",
        )
    )

    normalized_evaluator_name = (
        _normalize_required_text(
            evaluator_name,
            field_name="evaluator_name",
        )
    )

    normalized_evaluator_version = (
        _normalize_required_text(
            evaluator_version,
            field_name="evaluator_version",
        )
    )

    normalized_request_key_version = (
        _normalize_required_text(
            request_key_version,
            field_name="request_key_version",
        )
    )

    expected_start = (
        workflow.as_of
        - timedelta(
            hours=workflow.window_hours
        )
    )

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                rr.ranking_run_id,
                rr.request_key,
                rr.run_status
            FROM ranking_runs AS rr
            JOIN daily_workflow_runs AS dw
              ON dw.daily_workflow_run_id = $10
            WHERE rr.window_started_at = $1
              AND rr.window_finished_at = $2
              AND rr.formula_version = $3
              AND rr.model_name = $4
              AND rr.prompt_version = $5
              AND rr.parameters->>'run_mode' = $6
              AND rr.parameters->>'evaluator_name' = $7
              AND rr.parameters->>'evaluator_version' = $8
              AND rr.parameters->>'request_key_version' = $9
              AND rr.created_at >= dw.started_at
              AND rr.created_at <= (
                  dw.started_at
                  + INTERVAL '15 minutes'
              )
            ORDER BY rr.ranking_run_id
            """,
            expected_start,
            workflow.as_of,
            normalized_formula_version,
            normalized_model_name,
            normalized_prompt_version,
            normalized_run_mode,
            normalized_evaluator_name,
            normalized_evaluator_version,
            normalized_request_key_version,
            workflow.daily_workflow_run_id,
        )

    if not records:
        return None

    if len(records) > 1:
        details = tuple(
            (
                int(record["ranking_run_id"]),
                record["request_key"],
                record["run_status"],
            )
            for record in records
        )

        raise DailyWorkflowRecoveryAmbiguousError(
            "Найдено несколько совместимых "
            "ranking_run для daily workflow: "
            f"{details}"
        )

    return int(
        records[0]["ranking_run_id"]
    )


async def recover_batch_id(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    generation_request_key: str,
) -> int | None:
    """
    Восстанавливает publication batch
    по точному generation_request_key.

    Именно generation_request_key определяет
    identity защищённого generation-запуска.
    Ranking/date/chat используются как
    дополнительная проверка принадлежности
    найденной reservation daily workflow.
    """

    workflow = await load_daily_workflow(
        pool,
        daily_workflow_run_id=(
            daily_workflow_run_id
        ),
    )

    if workflow.batch_id is not None:
        return workflow.batch_id

    if workflow.ranking_run_id is None:
        return None

    normalized_request_key = _normalize_sha256(
        generation_request_key,
        field_name="generation_request_key",
    )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                batch_id,
                ranking_run_id,
                publication_date,
                target_telegram_chat_id,
                generation_request_key
            FROM publication_batches
            WHERE generation_request_key = $1
            """,
            normalized_request_key,
        )

    if record is None:
        return None

    differences: list[str] = []

    if (
        record["ranking_run_id"]
        != workflow.ranking_run_id
    ):
        differences.append(
            "ranking_run_id: "
            f"expected={workflow.ranking_run_id!r}, "
            f"actual={record['ranking_run_id']!r}"
        )

    if (
        record["publication_date"]
        != workflow.publication_date
    ):
        differences.append(
            "publication_date: "
            f"expected="
            f"{workflow.publication_date!r}, "
            f"actual="
            f"{record['publication_date']!r}"
        )

    if (
        record["target_telegram_chat_id"]
        != workflow.target_telegram_chat_id
    ):
        differences.append(
            "target_telegram_chat_id: "
            f"expected="
            f"{workflow.target_telegram_chat_id!r}, "
            f"actual="
            f"{record['target_telegram_chat_id']!r}"
        )

    if (
        record["generation_request_key"]
        != normalized_request_key
    ):
        differences.append(
            "generation_request_key differs"
        )

    if differences:
        raise ValueError(
            "Найденный publication batch "
            "не принадлежит daily workflow: "
            + "; ".join(differences)
        )

    return int(
        record["batch_id"]
    )


async def recover_generated_post_id(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> int | None:
    """Восстанавливает generated_post после generation completion."""

    workflow = await load_daily_workflow(
        pool,
        daily_workflow_run_id=(
            daily_workflow_run_id
        ),
    )

    if workflow.generated_post_id is not None:
        return workflow.generated_post_id

    if workflow.batch_id is None:
        return None

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT generated_post_id
            FROM generated_posts
            WHERE batch_id = $1
            ORDER BY
                version_number,
                generated_post_id
            """,
            workflow.batch_id,
        )

    if not records:
        return None

    if len(records) > 1:
        ids = tuple(
            int(record["generated_post_id"])
            for record in records
        )

        raise DailyWorkflowRecoveryAmbiguousError(
            "Для batch найдено несколько "
            "generated_posts до завершения "
            "daily workflow: "
            f"{ids}"
        )

    return int(
        records[0]["generated_post_id"]
    )


async def recover_image_generation_id(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> int | None:
    """Восстанавливает initial image request."""

    workflow = await load_daily_workflow(
        pool,
        daily_workflow_run_id=(
            daily_workflow_run_id
        ),
    )

    if workflow.image_generation_id is not None:
        return workflow.image_generation_id

    if (
        workflow.batch_id is None
        or workflow.generated_post_id is None
    ):
        return None

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                image_generation_id,
                image_status
            FROM image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
            ORDER BY
                CASE image_status
                    WHEN 'completed' THEN 0
                    WHEN 'reserved' THEN 1
                    ELSE 2
                END,
                image_generation_id DESC
            """,
            workflow.batch_id,
            workflow.generated_post_id,
        )

    if not records:
        return None

    active_records = [
        record
        for record in records
        if record["image_status"]
        in {"reserved", "completed"}
    ]

    if len(active_records) > 1:
        ids = tuple(
            int(record["image_generation_id"])
            for record in active_records
        )

        raise DailyWorkflowRecoveryAmbiguousError(
            "Найдено несколько active initial "
            "image requests: "
            f"{ids}"
        )

    if active_records:
        return int(
            active_records[0][
                "image_generation_id"
            ]
        )

    # Если все попытки failed, фиксируем последнюю.
    # Orchestrator увидит failed и не станет
    # выполнять неизвестный повтор автоматически.
    return int(
        records[0]["image_generation_id"]
    )