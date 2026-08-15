from dataclasses import dataclass
from datetime import datetime

import asyncpg


class DailyWorkflowSelectionError(ValueError):
    """Некорректный переход selection state machine."""


@dataclass(frozen=True, slots=True)
class DailyWorkflowSelectionAttempt:
    """Одна ranking combination, использованная daily workflow."""

    selection_attempt_id: int
    daily_workflow_run_id: int
    ranking_run_id: int
    combination_id: int
    combination_rank: int
    attempt_number: int
    selection_kind: str
    selection_status: str
    source_selection_attempt_id: int | None
    batch_id: int | None
    generated_post_id: int | None
    image_generation_id: int | None
    ended_at: datetime | None
    created_at: datetime
    updated_at: datetime
    created_new: bool

    @property
    def active(self) -> bool:
        return self.selection_status == "active"

    @property
    def moderation_blocked(self) -> bool:
        return (
            self.selection_status
            == "moderation_blocked"
        )


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


def _optional_integer(
    value: int | None,
    *,
    field_name: str,
) -> int | None:
    """Проверяет optional положительный integer."""

    if value is None:
        return None

    return _positive_integer(
        value,
        field_name=field_name,
    )


def _build_attempt(
    record: asyncpg.Record,
    *,
    created_new: bool,
) -> DailyWorkflowSelectionAttempt:
    """Строит dataclass из PostgreSQL record."""

    return DailyWorkflowSelectionAttempt(
        selection_attempt_id=int(
            record["selection_attempt_id"]
        ),
        daily_workflow_run_id=int(
            record["daily_workflow_run_id"]
        ),
        ranking_run_id=int(
            record["ranking_run_id"]
        ),
        combination_id=int(
            record["combination_id"]
        ),
        combination_rank=int(
            record["combination_rank"]
        ),
        attempt_number=int(
            record["attempt_number"]
        ),
        selection_kind=record[
            "selection_kind"
        ],
        selection_status=record[
            "selection_status"
        ],
        source_selection_attempt_id=(
            int(
                record[
                    "source_selection_attempt_id"
                ]
            )
            if (
                record[
                    "source_selection_attempt_id"
                ]
                is not None
            )
            else None
        ),
        batch_id=(
            int(record["batch_id"])
            if record["batch_id"] is not None
            else None
        ),
        generated_post_id=(
            int(record["generated_post_id"])
            if (
                record["generated_post_id"]
                is not None
            )
            else None
        ),
        image_generation_id=(
            int(record["image_generation_id"])
            if (
                record["image_generation_id"]
                is not None
            )
            else None
        ),
        ended_at=record["ended_at"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        created_new=created_new,
    )


_ATTEMPT_SELECT = """
    SELECT
        dws.selection_attempt_id,
        dws.daily_workflow_run_id,
        dws.ranking_run_id,
        dws.combination_id,
        rc.combination_rank,
        dws.attempt_number,
        dws.selection_kind,
        dws.selection_status,
        dws.source_selection_attempt_id,
        dws.batch_id,
        dws.generated_post_id,
        dws.image_generation_id,
        dws.ended_at,
        dws.created_at,
        dws.updated_at
    FROM
        top3_news.daily_workflow_selection_attempts
        AS dws
    JOIN top3_news.ranking_combinations AS rc
      ON rc.combination_id = dws.combination_id
     AND rc.ranking_run_id = dws.ranking_run_id
"""


async def _lock_workflow(
    connection: asyncpg.Connection,
    *,
    daily_workflow_run_id: int,
) -> asyncpg.Record:
    """Блокирует workflow и возвращает его ranking identity."""

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
        FROM top3_news.daily_workflow_runs
        WHERE daily_workflow_run_id = $1
        FOR UPDATE
        """,
        daily_workflow_run_id,
    )

    if workflow is None:
        raise LookupError(
            "daily_workflow_run не найден: "
            f"daily_workflow_run_id="
            f"{daily_workflow_run_id}"
        )

    if workflow["ranking_run_id"] is None:
        raise DailyWorkflowSelectionError(
            "Selection history требует "
            "прикреплённый ranking_run_id."
        )

    ranking = await connection.fetchrow(
        """
        SELECT
            ranking_run_id,
            run_status
        FROM top3_news.ranking_runs
        WHERE ranking_run_id = $1
        """,
        int(workflow["ranking_run_id"]),
    )

    if ranking is None:
        raise DailyWorkflowSelectionError(
            "Связанный ranking_run не найден."
        )

    if ranking["run_status"] != "completed":
        raise DailyWorkflowSelectionError(
            "Selection history требует "
            "completed ranking_run: "
            f"run_status={ranking['run_status']}"
        )

    return workflow


async def _load_combination(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    combination_id: int,
) -> asyncpg.Record:
    """Читает combination внутри конкретного ranking_run."""

    combination = await connection.fetchrow(
        """
        SELECT
            combination_id,
            ranking_run_id,
            combination_rank,
            is_winner
        FROM top3_news.ranking_combinations
        WHERE ranking_run_id = $1
          AND combination_id = $2
        """,
        ranking_run_id,
        combination_id,
    )

    if combination is None:
        raise DailyWorkflowSelectionError(
            "ranking combination не принадлежит "
            "workflow ranking_run: "
            f"ranking_run_id={ranking_run_id}, "
            f"combination_id={combination_id}"
        )

    return combination


async def _load_attempt_locked(
    connection: asyncpg.Connection,
    *,
    daily_workflow_run_id: int,
    selection_attempt_id: int,
) -> asyncpg.Record | None:
    """Читает selection attempt с row lock."""

    return await connection.fetchrow(
        """
        SELECT
            selection_attempt_id,
            daily_workflow_run_id,
            ranking_run_id,
            combination_id,
            attempt_number,
            selection_kind,
            selection_status,
            source_selection_attempt_id,
            batch_id,
            generated_post_id,
            image_generation_id,
            ended_at,
            created_at,
            updated_at
        FROM top3_news.daily_workflow_selection_attempts
        WHERE daily_workflow_run_id = $1
          AND selection_attempt_id = $2
        FOR UPDATE
        """,
        daily_workflow_run_id,
        selection_attempt_id,
    )


async def _load_attempt_for_return(
    connection: asyncpg.Connection,
    *,
    selection_attempt_id: int,
    created_new: bool,
) -> DailyWorkflowSelectionAttempt:
    """Повторно читает attempt вместе с combination_rank."""

    record = await connection.fetchrow(
        _ATTEMPT_SELECT
        + """
        WHERE dws.selection_attempt_id = $1
        """,
        selection_attempt_id,
    )

    if record is None:
        raise RuntimeError(
            "Не удалось перечитать "
            "daily workflow selection attempt."
        )

    return _build_attempt(
        record,
        created_new=created_new,
    )


async def ensure_initial_daily_workflow_selection(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    combination_id: int,
) -> DailyWorkflowSelectionAttempt:
    """
    Создаёт исходную winner selection или возвращает уже существующую.

    Первый attempt всегда обязан быть сохранённой winner combination.
    Повторный вызов с тем же combination_id идемпотентен.
    """

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    normalized_combination_id = (
        _positive_integer(
            combination_id,
            field_name="combination_id",
        )
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await _lock_workflow(
                connection,
                daily_workflow_run_id=(
                    workflow_id
                ),
            )

            ranking_run_id = int(
                workflow["ranking_run_id"]
            )

            combination = await _load_combination(
                connection,
                ranking_run_id=ranking_run_id,
                combination_id=(
                    normalized_combination_id
                ),
            )

            if not bool(combination["is_winner"]):
                raise DailyWorkflowSelectionError(
                    "Первый selection attempt "
                    "должен использовать winner "
                    "ranking combination."
                )

            existing_rows = await connection.fetch(
                """
                SELECT
                    selection_attempt_id,
                    combination_id,
                    attempt_number
                FROM
                    top3_news
                    .daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                ORDER BY attempt_number
                FOR UPDATE
                """,
                workflow_id,
            )

            if existing_rows:
                first = existing_rows[0]

                if (
                    int(first["attempt_number"]) != 1
                    or int(first["combination_id"])
                    != normalized_combination_id
                ):
                    raise DailyWorkflowSelectionError(
                        "Workflow уже имеет другую "
                        "selection history."
                    )

                return await _load_attempt_for_return(
                    connection,
                    selection_attempt_id=int(
                        first["selection_attempt_id"]
                    ),
                    created_new=False,
                )

            inserted_id = await connection.fetchval(
                """
                INSERT INTO
                    top3_news
                    .daily_workflow_selection_attempts (
                        daily_workflow_run_id,
                        ranking_run_id,
                        combination_id,
                        attempt_number,
                        selection_kind,
                        selection_status
                    )
                VALUES (
                    $1,
                    $2,
                    $3,
                    1,
                    'winner',
                    'active'
                )
                RETURNING selection_attempt_id
                """,
                workflow_id,
                ranking_run_id,
                normalized_combination_id,
            )

            if inserted_id is None:
                raise RuntimeError(
                    "Не удалось создать initial "
                    "daily workflow selection."
                )

            return await _load_attempt_for_return(
                connection,
                selection_attempt_id=int(
                    inserted_id
                ),
                created_new=True,
            )


async def replace_daily_workflow_selection_after_moderation(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    current_selection_attempt_id: int,
    replacement_combination_id: int,
    failed_image_generation_id: int,
) -> DailyWorkflowSelectionAttempt:
    """
    Атомарно закрывает active selection как moderation_blocked
    и создаёт следующую replacement selection.

    Повтор того же уже завершённого перехода идемпотентно возвращает
    созданного child attempt и не создаёт новую строку.
    """

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    current_attempt_id = _positive_integer(
        current_selection_attempt_id,
        field_name="current_selection_attempt_id",
    )

    replacement_id = _positive_integer(
        replacement_combination_id,
        field_name="replacement_combination_id",
    )

    failed_image_id = _positive_integer(
        failed_image_generation_id,
        field_name="failed_image_generation_id",
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            workflow = await _lock_workflow(
                connection,
                daily_workflow_run_id=(
                    workflow_id
                ),
            )

            ranking_run_id = int(
                workflow["ranking_run_id"]
            )

            current = await _load_attempt_locked(
                connection,
                daily_workflow_run_id=(
                    workflow_id
                ),
                selection_attempt_id=(
                    current_attempt_id
                ),
            )

            if current is None:
                raise LookupError(
                    "Текущий selection attempt "
                    "не найден."
                )

            existing_child = await connection.fetchrow(
                """
                SELECT
                    selection_attempt_id,
                    combination_id
                FROM
                    top3_news
                    .daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                  AND source_selection_attempt_id = $2
                FOR UPDATE
                """,
                workflow_id,
                current_attempt_id,
            )

            if existing_child is not None:
                if (
                    int(existing_child["combination_id"])
                    != replacement_id
                ):
                    raise DailyWorkflowSelectionError(
                        "Текущий selection attempt "
                        "уже имеет другой replacement child."
                    )

                if (
                    current["selection_status"]
                    != "moderation_blocked"
                    or current["image_generation_id"]
                    is None
                    or int(
                        current[
                            "image_generation_id"
                        ]
                    )
                    != failed_image_id
                ):
                    raise DailyWorkflowSelectionError(
                        "Существующий replacement "
                        "не согласован с blocking image."
                    )

                return await _load_attempt_for_return(
                    connection,
                    selection_attempt_id=int(
                        existing_child[
                            "selection_attempt_id"
                        ]
                    ),
                    created_new=False,
                )

            if current["selection_status"] != "active":
                raise DailyWorkflowSelectionError(
                    "Replacement разрешён только "
                    "из active selection."
                )

            if (
                int(current["ranking_run_id"])
                != ranking_run_id
            ):
                raise DailyWorkflowSelectionError(
                    "Selection attempt связан "
                    "с другим ranking_run."
                )

            if (
                int(current["combination_id"])
                == replacement_id
            ):
                raise DailyWorkflowSelectionError(
                    "Replacement combination должна "
                    "отличаться от текущей."
                )

            replacement = await _load_combination(
                connection,
                ranking_run_id=ranking_run_id,
                combination_id=replacement_id,
            )

            if bool(replacement["is_winner"]):
                raise DailyWorkflowSelectionError(
                    "Replacement не может снова "
                    "использовать winner combination."
                )

            already_used = await connection.fetchval(
                """
                SELECT selection_attempt_id
                FROM
                    top3_news
                    .daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                  AND combination_id = $2
                """,
                workflow_id,
                replacement_id,
            )

            if already_used is not None:
                raise DailyWorkflowSelectionError(
                    "Replacement combination уже "
                    "использовалась этим workflow."
                )

            image = await connection.fetchrow(
                """
                SELECT
                    igr.image_generation_id,
                    igr.batch_id,
                    igr.generated_post_id,
                    igr.image_status,
                    igr.request_kind,
                    igr.error_type,
                    igr.error_message,
                    igr.failed_at,
                    b.ranking_run_id
                FROM
                    top3_news
                    .image_generation_requests AS igr
                JOIN
                    top3_news.publication_batches AS b
                  ON b.batch_id = igr.batch_id
                WHERE igr.image_generation_id = $1
                FOR UPDATE
                """,
                failed_image_id,
            )

            if image is None:
                raise DailyWorkflowSelectionError(
                    "Blocking image request не найден."
                )

            if (
                int(image["ranking_run_id"])
                != ranking_run_id
            ):
                raise DailyWorkflowSelectionError(
                    "Blocking image относится "
                    "к другому ranking_run."
                )

            if image["request_kind"] != "initial":
                raise DailyWorkflowSelectionError(
                    "Replacement разрешён только "
                    "после initial image request."
                )

            if (
                image["image_status"] != "failed"
                or image["failed_at"] is None
            ):
                raise DailyWorkflowSelectionError(
                    "Blocking image не имеет "
                    "definitive failed status."
                )

            error_type = str(
                image["error_type"] or ""
            ).strip()

            error_message = str(
                image["error_message"] or ""
            ).strip().lower()

            if (
                error_type != "BadRequestError"
                or "moderation_blocked"
                not in error_message
            ):
                raise DailyWorkflowSelectionError(
                    "Replacement разрешён только "
                    "после доказанного "
                    "BadRequestError/moderation_blocked."
                )

            batch_id = int(image["batch_id"])
            generated_post_id = int(
                image["generated_post_id"]
            )

            workflow_batch_id = _optional_integer(
                workflow["batch_id"],
                field_name="workflow.batch_id",
            )
            workflow_post_id = _optional_integer(
                workflow["generated_post_id"],
                field_name=(
                    "workflow.generated_post_id"
                ),
            )

            if (
                workflow_batch_id is not None
                and workflow_batch_id != batch_id
            ):
                raise DailyWorkflowSelectionError(
                    "Blocking image batch_id "
                    "не совпадает с workflow."
                )

            if (
                workflow_post_id is not None
                and workflow_post_id
                != generated_post_id
            ):
                raise DailyWorkflowSelectionError(
                    "Blocking image generated_post_id "
                    "не совпадает с workflow."
                )

            max_attempt_number = (
                await connection.fetchval(
                    """
                    SELECT
                        COALESCE(
                            MAX(attempt_number),
                            0
                        )::integer
                    FROM
                        top3_news
                        .daily_workflow_selection_attempts
                    WHERE daily_workflow_run_id = $1
                    """,
                    workflow_id,
                )
            )

            if (
                int(max_attempt_number)
                != int(current["attempt_number"])
            ):
                raise DailyWorkflowSelectionError(
                    "Active selection не является "
                    "последним attempt workflow."
                )

            update_result = await connection.execute(
                """
                UPDATE
                    top3_news
                    .daily_workflow_selection_attempts
                SET
                    selection_status =
                        'moderation_blocked',
                    batch_id = $3,
                    generated_post_id = $4,
                    image_generation_id = $5,
                    ended_at = now()
                WHERE daily_workflow_run_id = $1
                  AND selection_attempt_id = $2
                  AND selection_status = 'active'
                """,
                workflow_id,
                current_attempt_id,
                batch_id,
                generated_post_id,
                failed_image_id,
            )

            if update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось закрыть active "
                    "selection attempt."
                )

            inserted_id = await connection.fetchval(
                """
                INSERT INTO
                    top3_news
                    .daily_workflow_selection_attempts (
                        daily_workflow_run_id,
                        ranking_run_id,
                        combination_id,
                        attempt_number,
                        selection_kind,
                        selection_status,
                        source_selection_attempt_id
                    )
                VALUES (
                    $1,
                    $2,
                    $3,
                    $4,
                    'replacement',
                    'active',
                    $5
                )
                RETURNING selection_attempt_id
                """,
                workflow_id,
                ranking_run_id,
                replacement_id,
                int(current["attempt_number"]) + 1,
                current_attempt_id,
            )

            if inserted_id is None:
                raise RuntimeError(
                    "Не удалось создать replacement "
                    "selection attempt."
                )

            return await _load_attempt_for_return(
                connection,
                selection_attempt_id=int(
                    inserted_id
                ),
                created_new=True,
            )


async def load_active_daily_workflow_selection(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> DailyWorkflowSelectionAttempt | None:
    """Возвращает единственную active selection workflow."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    async with pool.acquire() as connection:
        records = await connection.fetch(
            _ATTEMPT_SELECT
            + """
            WHERE dws.daily_workflow_run_id = $1
              AND dws.selection_status = 'active'
            ORDER BY dws.attempt_number
            """,
            workflow_id,
        )

    if not records:
        return None

    if len(records) != 1:
        raise RuntimeError(
            "Найдено несколько active "
            "daily workflow selections."
        )

    return _build_attempt(
        records[0],
        created_new=False,
    )


async def load_daily_workflow_selection_attempts(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> tuple[DailyWorkflowSelectionAttempt, ...]:
    """Возвращает всю selection history в порядке attempts."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    async with pool.acquire() as connection:
        records = await connection.fetch(
            _ATTEMPT_SELECT
            + """
            WHERE dws.daily_workflow_run_id = $1
            ORDER BY dws.attempt_number
            """,
            workflow_id,
        )

    return tuple(
        _build_attempt(
            record,
            created_new=False,
        )
        for record in records
    )


async def load_used_daily_workflow_combination_ids(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
) -> tuple[int, ...]:
    """Возвращает все уже использованные combination_id."""

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT combination_id
            FROM
                top3_news
                .daily_workflow_selection_attempts
            WHERE daily_workflow_run_id = $1
            ORDER BY attempt_number
            """,
            workflow_id,
        )

    return tuple(
        int(record["combination_id"])
        for record in records
    )