from dataclasses import dataclass
from datetime import date, datetime, timezone

import asyncpg


DAILY_WORKFLOW_VERSION = "daily_workflow_v1"
MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION = 2


class DailyWorkflowImageModerationRetryNotAllowedError(
    ValueError
):
    """Workflow не удовлетворяет строгим условиям image moderation retry."""


_RUNNING_STAGES = (
    "reserved",
    "ranking",
    "generation",
    "image",
    "review_delivery",
)

_STAGE_ORDER = {
    "reserved": 0,
    "ranking": 1,
    "generation": 2,
    "image": 3,
    "review_delivery": 4,
}


@dataclass(frozen=True, slots=True)
class DailyWorkflowRun:
    """Состояние одного ежедневного production workflow."""

    daily_workflow_run_id: int
    publication_date: date
    workflow_version: str
    workflow_status: str
    current_stage: str
    as_of: datetime
    window_hours: int
    target_telegram_chat_id: int

    ranking_run_id: int | None
    batch_id: int | None
    generated_post_id: int | None
    image_generation_id: int | None

    created_new: bool

    @property
    def running(self) -> bool:
        return self.workflow_status == "running"

    @property
    def awaiting_review(self) -> bool:
        return (
            self.workflow_status == "awaiting_review"
            and self.current_stage == "awaiting_review"
        )

    @property
    def failed(self) -> bool:
        return self.workflow_status == "failed"


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


def _normalize_publication_date(
    value: date,
) -> date:
    """Проверяет logical publication date."""

    if isinstance(value, datetime):
        raise TypeError(
            "publication_date должен быть date, "
            "а не datetime."
        )

    if not isinstance(value, date):
        raise TypeError(
            "publication_date должен быть date."
        )

    return value


def _normalize_as_of(
    value: datetime,
) -> datetime:
    """Нормализует cutoff в UTC."""

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "as_of должен содержать часовой пояс."
        )

    return value.astimezone(
        timezone.utc
    )


def _normalize_workflow_version(
    value: str,
) -> str:
    """Проверяет workflow version."""

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "workflow_version не может быть пустым."
        )

    return normalized


def _normalize_channel_id(
    value: int,
) -> int:
    """Проверяет полный Telegram channel ID."""

    if isinstance(value, bool):
        raise TypeError(
            "target_telegram_chat_id "
            "не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            "target_telegram_chat_id "
            "должен быть int."
        )

    if not str(value).startswith("-100"):
        raise ValueError(
            "target_telegram_chat_id должен "
            "начинаться с -100."
        )

    return value


def _build_workflow(
    record: asyncpg.Record,
    *,
    created_new: bool,
) -> DailyWorkflowRun:
    """Строит dataclass из PostgreSQL record."""

    return DailyWorkflowRun(
        daily_workflow_run_id=int(
            record["daily_workflow_run_id"]
        ),
        publication_date=(
            record["publication_date"]
        ),
        workflow_version=(
            record["workflow_version"]
        ),
        workflow_status=(
            record["workflow_status"]
        ),
        current_stage=(
            record["current_stage"]
        ),
        as_of=record["as_of"],
        window_hours=int(
            record["window_hours"]
        ),
        target_telegram_chat_id=int(
            record["target_telegram_chat_id"]
        ),
        ranking_run_id=(
            int(record["ranking_run_id"])
            if record["ranking_run_id"] is not None
            else None
        ),
        batch_id=(
            int(record["batch_id"])
            if record["batch_id"] is not None
            else None
        ),
        generated_post_id=(
            int(record["generated_post_id"])
            if record["generated_post_id"] is not None
            else None
        ),
        image_generation_id=(
            int(record["image_generation_id"])
            if record["image_generation_id"] is not None
            else None
        ),
        created_new=created_new,
    )


def _normalize_image_prompt_version(
    value: str,
) -> str:
    """Проверяет prompt_version Image API."""

    if not isinstance(value, str):
        raise TypeError(
            "prompt_version должен быть str."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "prompt_version не может быть пустым."
        )

    return normalized


def _is_definitive_image_moderation_failure(
    record: asyncpg.Record,
) -> bool:
    """Проверяет доказанный BadRequestError/moderation_blocked."""

    if (
        record["image_status"] != "failed"
        or record["failed_at"] is None
    ):
        return False

    error_type = str(
        record["error_type"] or ""
    ).strip()

    error_message = str(
        record["error_message"] or ""
    ).strip().lower()

    return (
        error_type == "BadRequestError"
        and "moderation_blocked" in error_message
    )


async def _require_image_moderation_retry_locked(
    connection: asyncpg.Connection,
    *,
    workflow: asyncpg.Record,
    prompt_version: str,
) -> int:
    """
    Проверяет право на следующий initial Image API call
    для конкретной версии промпта.

    Для одной prompt_version разрешено максимум две попытки.
    Новая prompt_version получает собственный лимит.

    При этом новая попытка допустима только после доказанного
    moderation_blocked предыдущего linked image request. Любые
    reserved/unknown/completed initial requests блокируют новый call.
    """

    normalized_prompt_version = (
        _normalize_image_prompt_version(
            prompt_version
        )
    )

    required_ids = (
        "ranking_run_id",
        "batch_id",
        "generated_post_id",
        "image_generation_id",
    )

    missing = [
        field_name
        for field_name in required_ids
        if workflow[field_name] is None
    ]

    if missing:
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "не закреплены "
                + ", ".join(missing)
            )
        )

    batch_id = int(workflow["batch_id"])
    post_id = int(workflow["generated_post_id"])
    linked_image_id = int(
        workflow["image_generation_id"]
    )

    generation = await connection.fetchrow(
        """
        SELECT
            b.batch_id,
            b.ranking_run_id,
            b.batch_status,
            p.generated_post_id,
            p.post_status,
            p.image_path,
            p.image_sha256
        FROM publication_batches AS b
        JOIN generated_posts AS p
          ON p.batch_id = b.batch_id
        WHERE b.batch_id = $1
          AND p.generated_post_id = $2
        """,
        batch_id,
        post_id,
    )

    if generation is None:
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "batch/generated_post не найден."
            )
        )

    if (
        int(generation["ranking_run_id"])
        != int(workflow["ranking_run_id"])
    ):
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "batch связан с другим ranking_run."
            )
        )

    if (
        generation["batch_status"] != "awaiting_review"
        or generation["post_status"] != "awaiting_review"
    ):
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry разрешён "
                "только для awaiting_review batch/post."
            )
        )

    if (
        generation["image_path"] is not None
        or generation["image_sha256"] is not None
    ):
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "у generated_post уже есть image artifact."
            )
        )

    linked_image = await connection.fetchrow(
        """
        SELECT
            image_generation_id,
            batch_id,
            generated_post_id,
            image_status,
            request_kind,
            prompt_version,
            error_type,
            error_message,
            failed_at
        FROM image_generation_requests
        WHERE image_generation_id = $1
        FOR UPDATE
        """,
        linked_image_id,
    )

    if linked_image is None:
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "linked image request не найден."
            )
        )

    if (
        linked_image["batch_id"] != batch_id
        or linked_image["generated_post_id"] != post_id
        or linked_image["request_kind"] != "initial"
    ):
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "linked image request не соответствует "
                "workflow batch/post/initial."
            )
        )

    if not _is_definitive_image_moderation_failure(
        linked_image
    ):
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "linked image request не является "
                "доказанным moderation_blocked."
            )
        )

    active_or_completed = await connection.fetch(
        """
        SELECT
            image_generation_id,
            image_status,
            prompt_version
        FROM image_generation_requests
        WHERE batch_id = $1
          AND generated_post_id = $2
          AND request_kind = 'initial'
          AND image_status IN (
              'reserved',
              'completed'
          )
        ORDER BY image_generation_id
        FOR UPDATE
        """,
        batch_id,
        post_id,
    )

    if active_or_completed:
        details = ", ".join(
            (
                f"{row['image_generation_id']}:"
                f"{row['image_status']}:"
                f"{row['prompt_version']}"
            )
            for row in active_or_completed
        )

        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "существует active/completed initial request: "
                + details
            )
        )

    version_attempts = await connection.fetch(
        """
        SELECT
            image_generation_id,
            image_status,
            prompt_version,
            error_type,
            error_message,
            failed_at
        FROM image_generation_requests
        WHERE batch_id = $1
          AND generated_post_id = $2
          AND request_kind = 'initial'
          AND prompt_version = $3
        ORDER BY image_generation_id
        FOR UPDATE
        """,
        batch_id,
        post_id,
        normalized_prompt_version,
    )

    for attempt in version_attempts:
        if not _is_definitive_image_moderation_failure(
            attempt
        ):
            raise (
                DailyWorkflowImageModerationRetryNotAllowedError(
                    "Image moderation retry запрещён: "
                    "для текущей prompt_version есть "
                    "неоднозначная/не-moderation failed attempt: "
                    f"image_generation_id="
                    f"{attempt['image_generation_id']}"
                )
            )

    attempt_count = len(
        version_attempts
    )

    if (
        attempt_count
        >= MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION
    ):
        raise (
            DailyWorkflowImageModerationRetryNotAllowedError(
                "Image moderation retry запрещён: "
                "лимит попыток для prompt_version исчерпан: "
                f"prompt_version={normalized_prompt_version}, "
                f"attempts={attempt_count}, "
                f"limit="
                f"{MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION}"
            )
        )

    return attempt_count


async def require_daily_workflow_image_moderation_retry(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    prompt_version: str,
) -> int:
    """
    Возвращает число уже использованных попыток
    для текущей Image API prompt_version.

    PostgreSQL не изменяется.
    """

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_prompt_version = (
        _normalize_image_prompt_version(
            prompt_version
        )
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await connection.fetchrow(
                """
                SELECT
                    daily_workflow_run_id,
                    workflow_status,
                    current_stage,
                    ranking_run_id,
                    batch_id,
                    generated_post_id,
                    image_generation_id
                FROM daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                workflow_id,
            )

            if workflow is None:
                raise LookupError(
                    "daily_workflow_run не найден: "
                    f"daily_workflow_run_id={workflow_id}"
                )

            if workflow["workflow_status"] not in {
                "running",
                "failed",
            }:
                raise (
                    DailyWorkflowImageModerationRetryNotAllowedError(
                        "Image moderation retry запрещён "
                        "для workflow_status="
                        f"{workflow['workflow_status']}."
                    )
                )

            if (
                workflow["workflow_status"] == "running"
                and workflow["current_stage"] != "image"
            ):
                raise (
                    DailyWorkflowImageModerationRetryNotAllowedError(
                        "Running workflow может retry image "
                        "только на current_stage=image."
                    )
                )

            if (
                workflow["workflow_status"] == "failed"
                and workflow["current_stage"] != "failed"
            ):
                raise (
                    DailyWorkflowImageModerationRetryNotAllowedError(
                        "Failed workflow имеет некорректную "
                        f"current_stage={workflow['current_stage']}."
                    )
                )

            return await _require_image_moderation_retry_locked(
                connection,
                workflow=workflow,
                prompt_version=(
                    normalized_prompt_version
                ),
            )


async def reopen_daily_workflow_for_image_moderation_retry(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    prompt_version: str,
) -> DailyWorkflowRun:
    """
    Reopen failed workflow для следующей попытки
    текущей Image API prompt_version.

    Исторический failed image_generation_id остаётся
    закреплён до reservation нового запроса.
    """

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_prompt_version = (
        _normalize_image_prompt_version(
            prompt_version
        )
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await connection.fetchrow(
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
                workflow_id,
            )

            if workflow is None:
                raise LookupError(
                    "daily_workflow_run не найден: "
                    f"daily_workflow_run_id={workflow_id}"
                )

            if (
                workflow["workflow_status"] != "failed"
                or workflow["current_stage"] != "failed"
            ):
                raise (
                    DailyWorkflowImageModerationRetryNotAllowedError(
                        "Reopen разрешён только для "
                        "failed/failed daily workflow."
                    )
                )

            await _require_image_moderation_retry_locked(
                connection,
                workflow=workflow,
                prompt_version=(
                    normalized_prompt_version
                ),
            )

            updated = await connection.fetchrow(
                """
                UPDATE daily_workflow_runs
                SET
                    workflow_status = 'running',
                    current_stage = 'image',
                    error_type = NULL,
                    error_message = NULL,
                    finished_at = NULL
                WHERE daily_workflow_run_id = $1
                RETURNING
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
                """,
                workflow_id,
            )

    if updated is None:
        raise RuntimeError(
            "Не удалось reopen daily workflow "
            "для image moderation retry."
        )

    return _build_workflow(
        updated,
        created_new=False,
    )


async def load_daily_workflow(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> DailyWorkflowRun:
    """Загружает workflow по ID."""

    normalized_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    async with pool.acquire() as connection:
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
            """,
            normalized_id,
        )

    if record is None:
        raise LookupError(
            "daily_workflow_run не найден: "
            f"daily_workflow_run_id={normalized_id}"
        )

    return _build_workflow(
        record,
        created_new=False,
    )


async def reserve_daily_workflow(
    pool: asyncpg.Pool,
    *,
    publication_date: date,
    as_of: datetime,
    target_telegram_chat_id: int,
    workflow_version: str = DAILY_WORKFLOW_VERSION,
    window_hours: int = 24,
) -> DailyWorkflowRun:
    """
    Создаёт или возвращает workflow на дату.

    Повторный вызов обязан передать ровно тот же
    as_of, channel, window и workflow version.
    """

    normalized_date = (
        _normalize_publication_date(
            publication_date
        )
    )

    normalized_as_of = (
        _normalize_as_of(
            as_of
        )
    )

    normalized_chat_id = (
        _normalize_channel_id(
            target_telegram_chat_id
        )
    )

    normalized_version = (
        _normalize_workflow_version(
            workflow_version
        )
    )

    if window_hours != 24:
        raise ValueError(
            "Daily workflow требует "
            "строго window_hours=24."
        )

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    $1::bigint
                )
                """,
                normalized_date.toordinal(),
            )

            inserted = await connection.fetchrow(
                """
                INSERT INTO daily_workflow_runs (
                    publication_date,
                    workflow_version,
                    workflow_status,
                    current_stage,
                    as_of,
                    window_hours,
                    target_telegram_chat_id
                )
                VALUES (
                    $1,
                    $2,
                    'running',
                    'reserved',
                    $3,
                    24,
                    $4
                )
                ON CONFLICT (publication_date)
                DO NOTHING
                RETURNING
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
                """,
                normalized_date,
                normalized_version,
                normalized_as_of,
                normalized_chat_id,
            )

            if inserted is not None:
                return _build_workflow(
                    inserted,
                    created_new=True,
                )

            existing = await connection.fetchrow(
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
                WHERE publication_date = $1
                FOR UPDATE
                """,
                normalized_date,
            )

            if existing is None:
                raise RuntimeError(
                    "Не удалось получить "
                    "daily workflow после "
                    "конфликта publication_date."
                )

            differences: list[str] = []

            expected = {
                "workflow_version": (
                    normalized_version
                ),
                "as_of": normalized_as_of,
                "window_hours": 24,
                "target_telegram_chat_id": (
                    normalized_chat_id
                ),
            }

            for field_name, expected_value in (
                expected.items()
            ):
                if existing[field_name] != expected_value:
                    differences.append(
                        f"{field_name}: "
                        f"expected={expected_value!r}, "
                        f"actual="
                        f"{existing[field_name]!r}"
                    )

            if differences:
                raise ValueError(
                    "Daily workflow на эту дату "
                    "уже существует с другими "
                    "параметрами: "
                    + "; ".join(differences)
                )

            return _build_workflow(
                existing,
                created_new=False,
            )


async def mark_daily_workflow_stage(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    stage: str,
) -> DailyWorkflowRun:
    """Продвигает running workflow на следующую стадию."""

    normalized_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_stage = stage.strip()

    if normalized_stage not in _STAGE_ORDER:
        raise ValueError(
            "Неподдерживаемая running stage: "
            f"{stage}"
        )

    async with pool.acquire() as connection:
        async with connection.transaction():
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
                normalized_id,
            )

            if record is None:
                raise LookupError(
                    "daily_workflow_run не найден: "
                    f"daily_workflow_run_id="
                    f"{normalized_id}"
                )

            if record["workflow_status"] != "running":
                raise ValueError(
                    "Стадию можно менять только "
                    "у running workflow: "
                    f"workflow_status="
                    f"{record['workflow_status']}"
                )

            current_stage = record[
                "current_stage"
            ]

            if current_stage not in _STAGE_ORDER:
                raise ValueError(
                    "Некорректная текущая running "
                    f"stage: {current_stage}"
                )

            if (
                _STAGE_ORDER[normalized_stage]
                < _STAGE_ORDER[current_stage]
            ):
                raise ValueError(
                    "Daily workflow нельзя "
                    "перевести назад: "
                    f"{current_stage} -> "
                    f"{normalized_stage}"
                )

            if normalized_stage == current_stage:
                return _build_workflow(
                    record,
                    created_new=False,
                )

            updated = await connection.fetchrow(
                """
                UPDATE daily_workflow_runs
                SET current_stage = $2
                WHERE daily_workflow_run_id = $1
                RETURNING
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
                """,
                normalized_id,
                normalized_stage,
            )

    if updated is None:
        raise RuntimeError(
            "Не удалось обновить daily workflow stage."
        )

    return _build_workflow(
        updated,
        created_new=False,
    )


async def attach_daily_workflow_ranking(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    ranking_run_id: int,
) -> DailyWorkflowRun:
    """Связывает workflow с completed ranking run."""

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
            workflow = await connection.fetchrow(
                """
                SELECT
                    daily_workflow_run_id,
                    workflow_status,
                    as_of,
                    ranking_run_id
                FROM daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                workflow_id,
            )

            if workflow is None:
                raise LookupError(
                    "daily_workflow_run не найден."
                )

            if workflow["workflow_status"] != "running":
                raise ValueError(
                    "Ranking можно прикрепить "
                    "только к running workflow."
                )

            ranking = await connection.fetchrow(
                """
                SELECT
                    ranking_run_id,
                    run_status,
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

            if ranking["run_status"] != "completed":
                raise ValueError(
                    "Для daily workflow требуется "
                    "completed ranking_run: "
                    f"run_status="
                    f"{ranking['run_status']}"
                )

            if (
                ranking["window_finished_at"]
                != workflow["as_of"]
            ):
                raise ValueError(
                    "ranking window_finished_at "
                    "не совпадает с workflow as_of."
                )

            existing_id = workflow[
                "ranking_run_id"
            ]

            if (
                existing_id is not None
                and int(existing_id) != ranking_id
            ):
                raise ValueError(
                    "Workflow уже связан "
                    "с другим ranking_run_id."
                )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET ranking_run_id = $2
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                ranking_id,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def attach_daily_workflow_generation(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    batch_id: int,
    generated_post_id: int,
) -> DailyWorkflowRun:
    """Связывает workflow с готовой text generation."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_batch_id = _positive_integer(
        batch_id,
        field_name="batch_id",
    )

    normalized_post_id = _positive_integer(
        generated_post_id,
        field_name="generated_post_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await connection.fetchrow(
                """
                SELECT
                    workflow_status,
                    publication_date,
                    target_telegram_chat_id,
                    ranking_run_id,
                    batch_id,
                    generated_post_id
                FROM daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                workflow_id,
            )

            if workflow is None:
                raise LookupError(
                    "daily_workflow_run не найден."
                )

            if workflow["workflow_status"] != "running":
                raise ValueError(
                    "Generation можно прикрепить "
                    "только к running workflow."
                )

            generation = await connection.fetchrow(
                """
                SELECT
                    b.batch_id,
                    b.publication_date,
                    b.ranking_run_id,
                    b.batch_status,
                    b.target_telegram_chat_id,
                    p.generated_post_id,
                    p.post_status
                FROM publication_batches AS b
                JOIN generated_posts AS p
                  ON p.batch_id = b.batch_id
                WHERE b.batch_id = $1
                  AND p.generated_post_id = $2
                """,
                normalized_batch_id,
                normalized_post_id,
            )

            if generation is None:
                raise LookupError(
                    "Связанный batch/post не найден."
                )

            if (
                generation["batch_status"]
                != "awaiting_review"
                or generation["post_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Daily workflow требует "
                    "awaiting_review batch/post."
                )

            expected_ranking = workflow[
                "ranking_run_id"
            ]

            if expected_ranking is None:
                raise ValueError(
                    "Сначала нужно прикрепить ranking_run."
                )

            if (
                generation["ranking_run_id"]
                != expected_ranking
            ):
                raise ValueError(
                    "Batch связан с другим "
                    "ranking_run_id."
                )

            if (
                generation["publication_date"]
                != workflow["publication_date"]
            ):
                raise ValueError(
                    "Batch publication_date "
                    "не совпадает с workflow."
                )

            if (
                generation[
                    "target_telegram_chat_id"
                ]
                != workflow[
                    "target_telegram_chat_id"
                ]
            ):
                raise ValueError(
                    "Batch Telegram channel "
                    "не совпадает с workflow."
                )

            existing_batch = workflow["batch_id"]

            if (
                existing_batch is not None
                and int(existing_batch)
                != normalized_batch_id
            ):
                raise ValueError(
                    "Workflow уже связан "
                    "с другим batch_id."
                )

            existing_post = workflow[
                "generated_post_id"
            ]

            if (
                existing_post is not None
                and int(existing_post)
                != normalized_post_id
            ):
                raise ValueError(
                    "Workflow уже связан с другим "
                    "generated_post_id."
                )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET
                    batch_id = $2,
                    generated_post_id = $3
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                normalized_batch_id,
                normalized_post_id,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def attach_daily_workflow_image(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    image_generation_id: int,
) -> DailyWorkflowRun:
    """Связывает workflow с completed image request."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    image_id = _positive_integer(
        image_generation_id,
        field_name="image_generation_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await connection.fetchrow(
                """
                SELECT
                    workflow_status,
                    ranking_run_id,
                    batch_id,
                    generated_post_id,
                    image_generation_id
                FROM daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                workflow_id,
            )

            if workflow is None:
                raise LookupError(
                    "daily_workflow_run не найден."
                )

            if workflow["workflow_status"] != "running":
                raise ValueError(
                    "Image можно прикрепить "
                    "только к running workflow."
                )

            if (
                workflow["ranking_run_id"] is None
                or workflow["batch_id"] is None
                or workflow["generated_post_id"] is None
            ):
                raise ValueError(
                    "Перед image должны быть "
                    "прикреплены ranking/batch/post."
                )

            image = await connection.fetchrow(
                """
                SELECT
                    igr.image_generation_id,
                    b.ranking_run_id,
                    igr.batch_id,
                    igr.generated_post_id,
                    igr.image_status,
                    igr.request_kind
                FROM image_generation_requests AS igr
                JOIN publication_batches AS b
                ON b.batch_id = igr.batch_id
                WHERE igr.image_generation_id = $1
                """,
                image_id,
            )

            if image is None:
                raise LookupError(
                    "image_generation_request "
                    "не найден."
                )

            if image["image_status"] != "completed":
                raise ValueError(
                    "Для workflow требуется "
                    "completed image request."
                )

            if image["request_kind"] != "initial":
               raise ValueError(
                   "Daily workflow может использовать "
                   "только initial image request."
               )

            for field_name in (
                "ranking_run_id",
                "batch_id",
                "generated_post_id",
            ):
                if (
                    image[field_name]
                    != workflow[field_name]
                ):
                    raise ValueError(
                        "Image request не совпадает "
                        "с workflow: "
                        f"field={field_name}"
                    )

            existing_image = workflow[
                "image_generation_id"
            ]

            if (
                existing_image is not None
                and int(existing_image) != image_id
            ):
                raise ValueError(
                    "Workflow уже связан "
                    "с другим image_generation_id."
                )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET image_generation_id = $2
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
                image_id,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def complete_daily_workflow(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> DailyWorkflowRun:
    """Переводит workflow в awaiting_review."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await connection.fetchrow(
                """
                SELECT
                    workflow_status,
                    ranking_run_id,
                    batch_id,
                    generated_post_id,
                    image_generation_id
                FROM daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                workflow_id,
            )

            if record is None:
                raise LookupError(
                    "daily_workflow_run не найден."
                )

            if record["workflow_status"] == (
                "awaiting_review"
            ):
                raise ValueError(
                    "Daily workflow уже завершён "
                    "и ожидает review."
                )

            if record["workflow_status"] != "running":
                raise ValueError(
                    "Завершить можно только "
                    "running workflow."
                )

            required_fields = (
                "ranking_run_id",
                "batch_id",
                "generated_post_id",
                "image_generation_id",
            )

            missing = [
                field_name
                for field_name in required_fields
                if record[field_name] is None
            ]

            if missing:
                raise ValueError(
                    "Workflow нельзя завершить: "
                    "не заполнены "
                    + ", ".join(missing)
                )

            await connection.execute(
                """
                UPDATE daily_workflow_runs
                SET
                    workflow_status =
                        'awaiting_review',
                    current_stage =
                        'awaiting_review',
                    finished_at = now(),
                    error_type = NULL,
                    error_message = NULL
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
            )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )


async def fail_daily_workflow(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    error_type: str,
    error_message: str,
) -> DailyWorkflowRun:
    """Фиксирует terminal failure верхнего workflow."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_error_type = (
        error_type.strip()[:500]
    )

    normalized_error_message = (
        error_message.strip()[:8000]
    )

    if not normalized_error_type:
        raise ValueError(
            "error_type не может быть пустым."
        )

    if not normalized_error_message:
        raise ValueError(
            "error_message не может быть пустым."
        )

    async with pool.acquire() as connection:
        await connection.execute(
            """
            UPDATE daily_workflow_runs
            SET
                workflow_status = 'failed',
                current_stage = 'failed',
                error_type = $2,
                error_message = $3,
                finished_at = now()
            WHERE daily_workflow_run_id = $1
              AND workflow_status = 'running'
            """,
            workflow_id,
            normalized_error_type,
            normalized_error_message,
        )

    return await load_daily_workflow(
        pool,
        daily_workflow_run_id=workflow_id,
    )