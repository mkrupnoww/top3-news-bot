from dataclasses import dataclass

import asyncpg


TRAILER_REJECTION_REASON = "official_trailer_not_verified"


class DailyWorkflowTrailerReplacementError(ValueError):
    """Некорректный transition после trailer preflight."""


@dataclass(frozen=True, slots=True)
class DailyWorkflowTrailerReplacementResult:
    """Результат атомарной замены TOP-3 до text generation."""

    source_selection_attempt_id: int
    replacement_selection_attempt_id: int
    source_combination_id: int
    replacement_combination_id: int
    rejected_news_ids: tuple[int, ...]
    created_new: bool


def _positive_integer(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} должен быть int.")
    if value <= 0:
        raise ValueError(f"{field_name} должен быть больше нуля.")
    return value


def _normalize_news_ids(values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError("rejected_news_ids должен быть tuple.")
    if not values:
        raise ValueError("rejected_news_ids не может быть пустым.")

    normalized = tuple(
        _positive_integer(value, field_name=f"rejected_news_ids[{index}]")
        for index, value in enumerate(values, start=1)
    )

    if len(set(normalized)) != len(normalized):
        raise ValueError("rejected_news_ids содержит дубли.")

    return normalized


async def replace_daily_workflow_after_trailer_unverified(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    current_selection_attempt_id: int,
    replacement_combination_id: int,
    rejected_news_ids: tuple[int, ...],
) -> DailyWorkflowTrailerReplacementResult:
    """
    Закрывает active selection как trailer_unverified и создаёт replacement.

    Переход выполняется ДО publication_batch reservation, поэтому source
    selection не имеет batch/post/image artifacts. Повтор того же transition
    идемпотентно возвращает уже созданный child attempt.
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
    rejected_ids = _normalize_news_ids(rejected_news_ids)

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
                FROM top3_news.daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                workflow_id,
            )

            if workflow is None:
                raise LookupError(
                    f"Daily workflow не найден: {workflow_id}"
                )

            if (
                workflow["workflow_status"] != "running"
                or workflow["current_stage"] != "generation"
            ):
                raise DailyWorkflowTrailerReplacementError(
                    "Trailer replacement разрешён только для "
                    "workflow running/generation."
                )

            if any(
                workflow[field] is not None
                for field in (
                    "batch_id",
                    "generated_post_id",
                    "image_generation_id",
                )
            ):
                raise DailyWorkflowTrailerReplacementError(
                    "Trailer replacement должен выполняться до "
                    "generation reservation."
                )

            ranking_run_id = _positive_integer(
                int(workflow["ranking_run_id"]),
                field_name="ranking_run_id",
            )

            source = await connection.fetchrow(
                """
                SELECT
                    selection_attempt_id,
                    ranking_run_id,
                    combination_id,
                    attempt_number,
                    selection_status,
                    rejection_reason,
                    rejected_news_ids
                FROM top3_news.daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                  AND selection_attempt_id = $2
                FOR UPDATE
                """,
                workflow_id,
                source_attempt_id,
            )

            if source is None:
                raise LookupError("Source selection attempt не найден.")

            existing_child = await connection.fetchrow(
                """
                SELECT selection_attempt_id, combination_id
                FROM top3_news.daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                  AND source_selection_attempt_id = $2
                FOR UPDATE
                """,
                workflow_id,
                source_attempt_id,
            )

            if existing_child is not None:
                persisted_rejected = tuple(
                    int(value)
                    for value in (source["rejected_news_ids"] or ())
                )
                if (
                    source["selection_status"] != "trailer_unverified"
                    or source["rejection_reason"] != TRAILER_REJECTION_REASON
                    or persisted_rejected != rejected_ids
                    or int(existing_child["combination_id"]) != replacement_id
                ):
                    raise DailyWorkflowTrailerReplacementError(
                        "Существующий trailer replacement не совпадает "
                        "с повторным запросом."
                    )

                return DailyWorkflowTrailerReplacementResult(
                    source_selection_attempt_id=source_attempt_id,
                    replacement_selection_attempt_id=int(
                        existing_child["selection_attempt_id"]
                    ),
                    source_combination_id=int(source["combination_id"]),
                    replacement_combination_id=replacement_id,
                    rejected_news_ids=rejected_ids,
                    created_new=False,
                )

            if source["selection_status"] != "active":
                raise DailyWorkflowTrailerReplacementError(
                    "Trailer replacement разрешён только из active selection."
                )

            if int(source["ranking_run_id"]) != ranking_run_id:
                raise DailyWorkflowTrailerReplacementError(
                    "Source selection относится к другому ranking_run."
                )

            source_combination_id = int(source["combination_id"])
            if source_combination_id == replacement_id:
                raise DailyWorkflowTrailerReplacementError(
                    "Replacement combination должна отличаться от source."
                )

            current_news = await connection.fetch(
                """
                SELECT ns.news_id
                FROM top3_news.ranking_combination_items AS rci
                JOIN top3_news.news_scores AS ns
                  ON ns.score_id = rci.score_id
                 AND ns.ranking_run_id = rci.ranking_run_id
                WHERE rci.ranking_run_id = $1
                  AND rci.combination_id = $2
                ORDER BY rci.position
                """,
                ranking_run_id,
                source_combination_id,
            )
            current_news_ids = {int(row["news_id"]) for row in current_news}

            if not set(rejected_ids).issubset(current_news_ids):
                raise DailyWorkflowTrailerReplacementError(
                    "rejected_news_ids не являются частью source combination."
                )

            replacement_news = await connection.fetch(
                """
                SELECT ns.news_id
                FROM top3_news.ranking_combination_items AS rci
                JOIN top3_news.news_scores AS ns
                  ON ns.score_id = rci.score_id
                 AND ns.ranking_run_id = rci.ranking_run_id
                WHERE rci.ranking_run_id = $1
                  AND rci.combination_id = $2
                ORDER BY rci.position
                """,
                ranking_run_id,
                replacement_id,
            )

            if len(replacement_news) != 3:
                raise DailyWorkflowTrailerReplacementError(
                    "Replacement ranking combination не найдена или неполна."
                )

            replacement_news_ids = {
                int(row["news_id"])
                for row in replacement_news
            }
            if replacement_news_ids & set(rejected_ids):
                raise DailyWorkflowTrailerReplacementError(
                    "Replacement combination всё ещё содержит "
                    "trailer-unverified news_id."
                )

            latest_attempt = await connection.fetchval(
                """
                SELECT COALESCE(MAX(attempt_number), 0)::integer
                FROM top3_news.daily_workflow_selection_attempts
                WHERE daily_workflow_run_id = $1
                """,
                workflow_id,
            )
            if int(latest_attempt) != int(source["attempt_number"]):
                raise DailyWorkflowTrailerReplacementError(
                    "Source selection не является последним attempt workflow."
                )

            update_result = await connection.execute(
                """
                UPDATE top3_news.daily_workflow_selection_attempts
                SET
                    selection_status = 'trailer_unverified',
                    rejection_reason = $3,
                    rejected_news_ids = $4::bigint[],
                    ended_at = now()
                WHERE daily_workflow_run_id = $1
                  AND selection_attempt_id = $2
                  AND selection_status = 'active'
                """,
                workflow_id,
                source_attempt_id,
                TRAILER_REJECTION_REASON,
                list(rejected_ids),
            )
            if update_result != "UPDATE 1":
                raise RuntimeError("Не удалось закрыть source selection.")

            child_id = await connection.fetchval(
                """
                INSERT INTO top3_news.daily_workflow_selection_attempts (
                    daily_workflow_run_id,
                    ranking_run_id,
                    combination_id,
                    attempt_number,
                    selection_kind,
                    selection_status,
                    source_selection_attempt_id
                )
                VALUES ($1, $2, $3, $4, 'replacement', 'active', $5)
                RETURNING selection_attempt_id
                """,
                workflow_id,
                ranking_run_id,
                replacement_id,
                int(source["attempt_number"]) + 1,
                source_attempt_id,
            )

            if child_id is None:
                raise RuntimeError("Не удалось создать trailer replacement.")

            return DailyWorkflowTrailerReplacementResult(
                source_selection_attempt_id=source_attempt_id,
                replacement_selection_attempt_id=int(child_id),
                source_combination_id=source_combination_id,
                replacement_combination_id=replacement_id,
                rejected_news_ids=rejected_ids,
                created_new=True,
            )
