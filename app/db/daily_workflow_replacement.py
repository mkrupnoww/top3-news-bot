from dataclasses import dataclass

import asyncpg

from app.db.daily_workflow_selection_attempts import (
    DailyWorkflowSelectionAttempt,
    _load_attempt_for_return,
    _positive_integer,
)


class DailyWorkflowReplacementNotAllowedError(
    ValueError
):
    """Replacement daily workflow запрещён текущим состоянием."""


@dataclass(frozen=True, slots=True)
class DailyWorkflowReplacementResult:
    """Результат атомарного перехода на другую TOP-3 combination."""

    daily_workflow_run_id: int
    ranking_run_id: int

    source_selection_attempt_id: int
    source_combination_id: int

    replacement_selection: (
        DailyWorkflowSelectionAttempt
    )

    superseded_batch_id: int
    superseded_generated_post_id: int
    failed_image_generation_id: int

    created_new: bool

    @property
    def replacement_selection_attempt_id(
        self,
    ) -> int:
        """Возвращает ID нового selection attempt."""

        return (
            self.replacement_selection
            .selection_attempt_id
        )

    @property
    def replacement_combination_id(
        self,
    ) -> int:
        """Возвращает новую ranking combination."""

        return (
            self.replacement_selection
            .combination_id
        )


async def _lock_workflow(
    connection: asyncpg.Connection,
    *,
    daily_workflow_run_id: int,
) -> asyncpg.Record:
    """Блокирует daily workflow."""

    workflow = await connection.fetchrow(
        """
        SELECT
            daily_workflow_run_id,
            publication_date,
            workflow_status,
            current_stage,
            as_of,
            window_hours,
            target_telegram_chat_id,
            ranking_run_id,
            batch_id,
            generated_post_id,
            image_generation_id,
            error_type,
            error_message,
            finished_at
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

    return workflow


async def _lock_selection_attempt(
    connection: asyncpg.Connection,
    *,
    daily_workflow_run_id: int,
    selection_attempt_id: int,
) -> asyncpg.Record:
    """Блокирует source selection attempt."""

    record = await connection.fetchrow(
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
            ended_at
        FROM
            top3_news
            .daily_workflow_selection_attempts
        WHERE daily_workflow_run_id = $1
          AND selection_attempt_id = $2
        FOR UPDATE
        """,
        daily_workflow_run_id,
        selection_attempt_id,
    )

    if record is None:
        raise LookupError(
            "daily workflow selection attempt "
            "не найден: "
            f"daily_workflow_run_id="
            f"{daily_workflow_run_id}, "
            f"selection_attempt_id="
            f"{selection_attempt_id}"
        )

    return record


async def _load_existing_child(
    connection: asyncpg.Connection,
    *,
    daily_workflow_run_id: int,
    source_selection_attempt_id: int,
) -> asyncpg.Record | None:
    """Читает уже созданный replacement child."""

    return await connection.fetchrow(
        """
        SELECT
            selection_attempt_id,
            combination_id,
            selection_status
        FROM
            top3_news
            .daily_workflow_selection_attempts
        WHERE daily_workflow_run_id = $1
          AND source_selection_attempt_id = $2
        FOR UPDATE
        """,
        daily_workflow_run_id,
        source_selection_attempt_id,
    )


async def _validate_idempotent_transition(
    connection: asyncpg.Connection,
    *,
    workflow: asyncpg.Record,
    source: asyncpg.Record,
    child: asyncpg.Record,
    replacement_combination_id: int,
    failed_image_generation_id: int,
) -> None:
    """
    Проверяет, что повтор вызова относится к уже завершённому
    точно такому же replacement transition.
    """

    if (
        int(child["combination_id"])
        != replacement_combination_id
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source selection уже имеет другой "
            "replacement child."
        )

    if (
        source["selection_status"]
        != "moderation_blocked"
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Существующий replacement child "
            "не согласован со статусом source selection."
        )

    source_image_id = source[
        "image_generation_id"
    ]

    if (
        source_image_id is None
        or int(source_image_id)
        != failed_image_generation_id
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Существующий replacement child "
            "не согласован с blocking image."
        )

    if (
        source["batch_id"] is None
        or source["generated_post_id"] is None
        or source["ended_at"] is None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Завершённая source selection "
            "не содержит artifact provenance."
        )

    if (
        workflow["workflow_status"] != "running"
        or workflow["current_stage"]
        != "generation"
        or workflow["batch_id"] is not None
        or workflow["generated_post_id"]
        is not None
        or workflow["image_generation_id"]
        is not None
        or workflow["finished_at"] is not None
        or workflow["error_type"] is not None
        or workflow["error_message"] is not None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Workflow не находится в ожидаемом "
            "post-replacement состоянии."
        )

    superseded_batch = await connection.fetchrow(
        """
        SELECT
            batch_status
        FROM top3_news.publication_batches
        WHERE batch_id = $1
        FOR UPDATE
        """,
        int(source["batch_id"]),
    )

    if (
        superseded_batch is None
        or superseded_batch["batch_status"]
        != "superseded"
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source publication_batch не имеет "
            "статус superseded."
        )

    superseded_post = await connection.fetchrow(
        """
        SELECT
            post_status
        FROM top3_news.generated_posts
        WHERE generated_post_id = $1
        FOR UPDATE
        """,
        int(source["generated_post_id"]),
    )

    if (
        superseded_post is None
        or superseded_post["post_status"]
        != "superseded"
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source generated_post не имеет "
            "статус superseded."
        )


async def _require_new_transition_state(
    connection: asyncpg.Connection,
    *,
    workflow: asyncpg.Record,
    source: asyncpg.Record,
    replacement_combination_id: int,
    failed_image_generation_id: int,
) -> tuple[
    int,
    int,
    int,
]:
    """
    Проверяет все условия нового replacement transition.

    Возвращает:
    (ranking_run_id, batch_id, generated_post_id).
    """

    if (
        workflow["workflow_status"] != "running"
        or workflow["current_stage"] != "image"
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Новый replacement разрешён только "
            "для running workflow на стадии image."
        )

    if (
        workflow["finished_at"] is not None
        or workflow["error_type"] is not None
        or workflow["error_message"] is not None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Running image workflow имеет "
            "terminal/error поля."
        )

    required_workflow_ids = (
        "ranking_run_id",
        "batch_id",
        "generated_post_id",
        "image_generation_id",
    )

    missing = [
        field_name
        for field_name in required_workflow_ids
        if workflow[field_name] is None
    ]

    if missing:
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement запрещён: workflow "
            "не содержит "
            + ", ".join(missing)
        )

    ranking_run_id = int(
        workflow["ranking_run_id"]
    )

    batch_id = int(
        workflow["batch_id"]
    )

    generated_post_id = int(
        workflow["generated_post_id"]
    )

    workflow_image_id = int(
        workflow["image_generation_id"]
    )

    if (
        workflow_image_id
        != failed_image_generation_id
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "failed_image_generation_id "
            "не является текущим image pointer workflow."
        )

    ranking = await connection.fetchrow(
        """
        SELECT
            ranking_run_id,
            run_status
        FROM top3_news.ranking_runs
        WHERE ranking_run_id = $1
        """,
        ranking_run_id,
    )

    if (
        ranking is None
        or ranking["run_status"] != "completed"
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement требует completed ranking_run."
        )

    if (
        source["selection_status"] != "active"
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source selection должна быть active."
        )

    if (
        int(source["ranking_run_id"])
        != ranking_run_id
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source selection связана "
            "с другим ranking_run."
        )

    if (
        source["batch_id"] is not None
        or source["generated_post_id"]
        is not None
        or source["image_generation_id"]
        is not None
        or source["ended_at"] is not None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Active source selection уже содержит "
            "terminal artifact provenance."
        )

    max_attempt_number = await connection.fetchval(
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
        int(workflow["daily_workflow_run_id"]),
    )

    if (
        int(max_attempt_number)
        != int(source["attempt_number"])
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source selection не является "
            "последним attempt workflow."
        )

    if (
        int(source["combination_id"])
        == replacement_combination_id
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement combination должна "
            "отличаться от source combination."
        )

    replacement = await connection.fetchrow(
        """
        SELECT
            combination_id,
            ranking_run_id
        FROM top3_news.ranking_combinations
        WHERE ranking_run_id = $1
          AND combination_id = $2
        """,
        ranking_run_id,
        replacement_combination_id,
    )

    if replacement is None:
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement combination не принадлежит "
            "workflow ranking_run."
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
        int(workflow["daily_workflow_run_id"]),
        replacement_combination_id,
    )

    if already_used is not None:
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement combination уже "
            "использовалась этим workflow."
        )

    batch = await connection.fetchrow(
        """
        SELECT
            batch_id,
            publication_date,
            ranking_run_id,
            batch_status,
            target_telegram_chat_id,
            approved_at,
            published_at,
            approved_by_telegram_user_id
        FROM top3_news.publication_batches
        WHERE batch_id = $1
        FOR UPDATE
        """,
        batch_id,
    )

    if batch is None:
        raise DailyWorkflowReplacementNotAllowedError(
            "Source publication_batch не найден."
        )

    if (
        batch["ranking_run_id"] != ranking_run_id
        or batch["publication_date"]
        != workflow["publication_date"]
        or batch["target_telegram_chat_id"]
        != workflow["target_telegram_chat_id"]
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Source publication_batch "
            "не совпадает с daily workflow identity."
        )

    if batch["batch_status"] != "awaiting_review":
        raise DailyWorkflowReplacementNotAllowedError(
            "Source publication_batch должен "
            "иметь status awaiting_review."
        )

    if (
        batch["approved_at"] is not None
        or batch["published_at"] is not None
        or batch[
            "approved_by_telegram_user_id"
        ] is not None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Уже approved/published batch "
            "нельзя заменять автоматически."
        )

    post = await connection.fetchrow(
        """
        SELECT
            generated_post_id,
            batch_id,
            post_status,
            image_path,
            image_sha256
        FROM top3_news.generated_posts
        WHERE generated_post_id = $1
        FOR UPDATE
        """,
        generated_post_id,
    )

    if post is None:
        raise DailyWorkflowReplacementNotAllowedError(
            "Source generated_post не найден."
        )

    if int(post["batch_id"]) != batch_id:
        raise DailyWorkflowReplacementNotAllowedError(
            "Source generated_post принадлежит "
            "другому batch."
        )

    if post["post_status"] != "awaiting_review":
        raise DailyWorkflowReplacementNotAllowedError(
            "Source generated_post должен иметь "
            "status awaiting_review."
        )

    if (
        post["image_path"] is not None
        or post["image_sha256"] is not None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Пост уже содержит сохранённое "
            "изображение и не может быть заменён "
            "по старому moderation block."
        )

    image = await connection.fetchrow(
        """
        SELECT
            image_generation_id,
            batch_id,
            generated_post_id,
            request_kind,
            image_status,
            error_type,
            error_message,
            failed_at
        FROM top3_news.image_generation_requests
        WHERE image_generation_id = $1
        FOR UPDATE
        """,
        failed_image_generation_id,
    )

    if image is None:
        raise DailyWorkflowReplacementNotAllowedError(
            "Blocking image request не найден."
        )

    if (
        int(image["batch_id"]) != batch_id
        or int(image["generated_post_id"])
        != generated_post_id
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Blocking image request связан "
            "с другим batch/post."
        )

    if image["request_kind"] != "initial":
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement разрешён только после "
            "initial image request."
        )

    if (
        image["image_status"] != "failed"
        or image["failed_at"] is None
    ):
        raise DailyWorkflowReplacementNotAllowedError(
            "Blocking image request не имеет "
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
        raise DailyWorkflowReplacementNotAllowedError(
            "Replacement разрешён только после "
            "доказанного "
            "BadRequestError/moderation_blocked."
        )

    conflicting_image_count = (
        await connection.fetchval(
            """
            SELECT COUNT(*)::integer
            FROM top3_news.image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
              AND image_generation_id <> $3
              AND image_status IN (
                    'reserved',
                    'completed'
              )
            """,
            batch_id,
            generated_post_id,
            failed_image_generation_id,
        )
    )

    if int(conflicting_image_count) != 0:
        raise DailyWorkflowReplacementNotAllowedError(
            "Найден другой reserved/completed "
            "initial image request для source post."
        )

    combination_match = await connection.fetchval(
        """
        WITH batch_selection AS (
            SELECT
                bi.position,
                bi.news_id,
                bi.score_id
            FROM top3_news.batch_items AS bi
            WHERE bi.batch_id = $1
        ),
        ranking_selection AS (
            SELECT
                rci.position,
                ns.news_id,
                rci.score_id
            FROM
                top3_news.ranking_combination_items
                AS rci
            JOIN top3_news.news_scores AS ns
              ON ns.score_id = rci.score_id
             AND ns.ranking_run_id =
                 rci.ranking_run_id
            WHERE rci.ranking_run_id = $2
              AND rci.combination_id = $3
        )
        SELECT (
            (SELECT COUNT(*) FROM batch_selection) = 3
            AND
            (SELECT COUNT(*) FROM ranking_selection) = 3
            AND
            NOT EXISTS (
                (
                    SELECT
                        position,
                        news_id,
                        score_id
                    FROM batch_selection
                )
                EXCEPT
                (
                    SELECT
                        position,
                        news_id,
                        score_id
                    FROM ranking_selection
                )
            )
            AND
            NOT EXISTS (
                (
                    SELECT
                        position,
                        news_id,
                        score_id
                    FROM ranking_selection
                )
                EXCEPT
                (
                    SELECT
                        position,
                        news_id,
                        score_id
                    FROM batch_selection
                )
            )
        )
        """,
        batch_id,
        ranking_run_id,
        int(source["combination_id"]),
    )

    if combination_match is not True:
        raise DailyWorkflowReplacementNotAllowedError(
            "Source batch_items не соответствуют "
            "source ranking combination."
        )

    return (
        ranking_run_id,
        batch_id,
        generated_post_id,
    )


async def replace_daily_workflow_after_image_moderation(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    current_selection_attempt_id: int,
    replacement_combination_id: int,
    failed_image_generation_id: int,
) -> DailyWorkflowReplacementResult:
    """
    Атомарно переключает daily workflow на новую TOP-3 combination.

    Новый переход разрешён только если:
    - workflow running/image;
    - текущая selection active;
    - linked initial Image API request definitively failed;
    - ошибка является BadRequestError/moderation_blocked;
    - source post ещё не имеет сохранённого изображения;
    - отсутствует другой reserved/completed initial image request;
    - source batch/post соответствуют source ranking combination.

    В одной транзакции:
    1. source generated_post -> superseded;
    2. source publication_batch -> superseded;
    3. source selection -> moderation_blocked;
    4. создаётся active replacement selection;
    5. workflow возвращается в running/generation;
    6. batch/post/image pointers workflow очищаются.

    Повтор точно такого же уже завершённого перехода идемпотентен.
    """

    workflow_id = _positive_integer(
        daily_workflow_run_id,
        field_name="daily_workflow_run_id",
    )

    source_attempt_id = _positive_integer(
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

            source = await _lock_selection_attempt(
                connection,
                daily_workflow_run_id=(
                    workflow_id
                ),
                selection_attempt_id=(
                    source_attempt_id
                ),
            )

            child = await _load_existing_child(
                connection,
                daily_workflow_run_id=(
                    workflow_id
                ),
                source_selection_attempt_id=(
                    source_attempt_id
                ),
            )

            if child is not None:
                await _validate_idempotent_transition(
                    connection,
                    workflow=workflow,
                    source=source,
                    child=child,
                    replacement_combination_id=(
                        replacement_id
                    ),
                    failed_image_generation_id=(
                        failed_image_id
                    ),
                )

                replacement_selection = (
                    await _load_attempt_for_return(
                        connection,
                        selection_attempt_id=int(
                            child[
                                "selection_attempt_id"
                            ]
                        ),
                        created_new=False,
                    )
                )

                return DailyWorkflowReplacementResult(
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                    ranking_run_id=int(
                        source["ranking_run_id"]
                    ),
                    source_selection_attempt_id=(
                        source_attempt_id
                    ),
                    source_combination_id=int(
                        source["combination_id"]
                    ),
                    replacement_selection=(
                        replacement_selection
                    ),
                    superseded_batch_id=int(
                        source["batch_id"]
                    ),
                    superseded_generated_post_id=int(
                        source[
                            "generated_post_id"
                        ]
                    ),
                    failed_image_generation_id=(
                        failed_image_id
                    ),
                    created_new=False,
                )

            (
                ranking_run_id,
                batch_id,
                generated_post_id,
            ) = await _require_new_transition_state(
                connection,
                workflow=workflow,
                source=source,
                replacement_combination_id=(
                    replacement_id
                ),
                failed_image_generation_id=(
                    failed_image_id
                ),
            )

            post_update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.generated_posts
                    SET
                        post_status = 'superseded'
                    WHERE generated_post_id = $1
                      AND batch_id = $2
                      AND post_status =
                          'awaiting_review'
                      AND image_path IS NULL
                      AND image_sha256 IS NULL
                    """,
                    generated_post_id,
                    batch_id,
                )
            )

            if post_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось перевести source "
                    "generated_post в superseded."
                )

            batch_update_result = (
                await connection.execute(
                    """
                    UPDATE
                        top3_news.publication_batches
                    SET
                        batch_status = 'superseded',
                        error_message = NULL
                    WHERE batch_id = $1
                      AND batch_status =
                          'awaiting_review'
                      AND approved_at IS NULL
                      AND published_at IS NULL
                      AND approved_by_telegram_user_id
                          IS NULL
                    """,
                    batch_id,
                )
            )

            if batch_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось перевести source "
                    "publication_batch в superseded."
                )

            source_update_result = (
                await connection.execute(
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
                    source_attempt_id,
                    batch_id,
                    generated_post_id,
                    failed_image_id,
                )
            )

            if source_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось закрыть source "
                    "selection attempt."
                )

            replacement_attempt_id = (
                await connection.fetchval(
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
                    int(source["attempt_number"]) + 1,
                    source_attempt_id,
                )
            )

            if replacement_attempt_id is None:
                raise RuntimeError(
                    "Не удалось создать replacement "
                    "selection attempt."
                )

            workflow_update_result = (
                await connection.execute(
                    """
                    UPDATE top3_news.daily_workflow_runs
                    SET
                        workflow_status = 'running',
                        current_stage = 'generation',
                        batch_id = NULL,
                        generated_post_id = NULL,
                        image_generation_id = NULL,
                        error_type = NULL,
                        error_message = NULL,
                        finished_at = NULL
                    WHERE daily_workflow_run_id = $1
                      AND workflow_status = 'running'
                      AND current_stage = 'image'
                      AND ranking_run_id = $2
                      AND batch_id = $3
                      AND generated_post_id = $4
                      AND image_generation_id = $5
                    """,
                    workflow_id,
                    ranking_run_id,
                    batch_id,
                    generated_post_id,
                    failed_image_id,
                )
            )

            if workflow_update_result != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось вернуть daily workflow "
                    "на стадию generation."
                )

            replacement_selection = (
                await _load_attempt_for_return(
                    connection,
                    selection_attempt_id=int(
                        replacement_attempt_id
                    ),
                    created_new=True,
                )
            )

            return DailyWorkflowReplacementResult(
                daily_workflow_run_id=workflow_id,
                ranking_run_id=ranking_run_id,
                source_selection_attempt_id=(
                    source_attempt_id
                ),
                source_combination_id=int(
                    source["combination_id"]
                ),
                replacement_selection=(
                    replacement_selection
                ),
                superseded_batch_id=batch_id,
                superseded_generated_post_id=(
                    generated_post_id
                ),
                failed_image_generation_id=(
                    failed_image_id
                ),
                created_new=True,
            )