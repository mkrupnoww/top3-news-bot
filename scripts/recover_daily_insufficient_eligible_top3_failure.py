from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.ranking.full_formula import (
    ELIGIBILITY_FALLBACK_THRESHOLD,
    FULL_FORMULA_VERSION,
)
from app.workflows.daily_production import (
    run_daily_production_workflow,
)


EXPECTED_WORKFLOW_VERSION = "daily_workflow_v1"
EXPECTED_ERROR_TYPE = "ValueError"
EXPECTED_HISTORICAL_FORMULA_VERSION = "top3_cinema_v4"
EXPECTED_RECOVERY_FORMULA_VERSION = "top3_cinema_v5"
EXPECTED_EVALUATOR_VERSION = "event_ranking_evaluator_v8"
EXPECTED_REQUEST_KEY_VERSION = "event_ranking_request_key_v1"
EXPECTED_RUN_MODE = "openai_event_ranking"
EXPECTED_FAILURE_VERSION = (
    "reserved_event_ranking_diagnostic_failure_v1"
)
EXPECTED_FAILURE_STAGE = "top3_selection"
EXPECTED_EXCLUSION_REASON = "individual_score_below_3_5"
EXPECTED_FALLBACK_THRESHOLD = "3.000000"

INSUFFICIENT_TOP3_ERROR_PATTERN = re.compile(
    r"^Для выбора TOP-3 требуется минимум три допустимых "
    r"инфоповода: eligible_count=(?P<count>[0-2])$"
)


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    daily_workflow_run_id: int
    publication_date: date
    workflow_version: str
    as_of: datetime
    window_hours: int
    historical_failed_ranking_run_id: int
    historical_formula_version: str
    historical_model_name: str
    historical_prompt_version: str
    historical_evaluator_version: str
    historical_candidate_count: int
    historical_scored_count: int
    historical_strict_eligible_count: int
    fallback_promotable_news_ids: tuple[int, ...]
    expected_effective_eligible_count: int
    historical_request_key: str


def _progress(message: str) -> None:
    print(message, flush=True)


def _json_object(
    value,
    *,
    field_name: str,
) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field_name} содержит невалидный JSON."
            ) from exc

        if isinstance(decoded, dict):
            return decoded

    raise ValueError(
        f"{field_name} не является JSON-объектом."
    )


def _required_text(
    value,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} должен быть строкой."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} не должен быть пустым."
        )

    return normalized


def _parse_insufficient_top3_error(
    value,
    *,
    field_name: str,
) -> int:
    message = _required_text(
        value,
        field_name=field_name,
    )

    match = INSUFFICIENT_TOP3_ERROR_PATTERN.fullmatch(
        message
    )

    if match is None:
        raise ValueError(
            "Recovery запрещён: ошибка не соответствует "
            "historical insufficient TOP-3 failure: "
            f"{message!r}"
        )

    return int(match.group("count"))


def _validate_runtime_policy() -> None:
    if FULL_FORMULA_VERSION != (
        EXPECTED_RECOVERY_FORMULA_VERSION
    ):
        raise ValueError(
            "Recovery запрещён: текущая formula_version "
            "не является top3_cinema_v5: "
            f"{FULL_FORMULA_VERSION!r}"
        )

    if str(ELIGIBILITY_FALLBACK_THRESHOLD) != (
        EXPECTED_FALLBACK_THRESHOLD
    ):
        raise ValueError(
            "Recovery запрещён: неожиданный "
            "eligibility fallback threshold: "
            f"{ELIGIBILITY_FALLBACK_THRESHOLD}"
        )


async def _reopen_failed_workflow(
    pool,
    *,
    daily_workflow_run_id: int,
    dry_run: bool,
) -> RecoveryContext:
    """
    Проверяет historical v4 insufficient-TOP3 failure.

    При dry_run=False атомарно reopen'ит только daily workflow.
    Historical failed ranking_run и его diagnostic scores не меняются.
    """

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
                    ranking_run_id,
                    batch_id,
                    generated_post_id,
                    image_generation_id,
                    error_type,
                    error_message
                FROM top3_news.daily_workflow_runs
                WHERE daily_workflow_run_id = $1
                FOR UPDATE
                """,
                daily_workflow_run_id,
            )

            if workflow is None:
                raise LookupError(
                    "daily_workflow_run не найден."
                )

            workflow_version = _required_text(
                workflow["workflow_version"],
                field_name="workflow.workflow_version",
            )

            if workflow_version != EXPECTED_WORKFLOW_VERSION:
                raise ValueError(
                    "Recovery запрещён: неизвестный "
                    "workflow_version: "
                    f"{workflow_version!r}"
                )

            if (
                workflow["workflow_status"] != "failed"
                or workflow["current_stage"] != "failed"
            ):
                raise ValueError(
                    "Recovery разрешён только для "
                    "failed/failed workflow."
                )

            if (
                str(workflow["error_type"] or "").strip()
                != EXPECTED_ERROR_TYPE
            ):
                raise ValueError(
                    "Recovery запрещён: workflow error_type "
                    "не является ValueError."
                )

            workflow_eligible_count = (
                _parse_insufficient_top3_error(
                    workflow["error_message"],
                    field_name="workflow.error_message",
                )
            )

            if workflow["ranking_run_id"] is None:
                raise ValueError(
                    "Workflow не содержит ranking_run_id."
                )

            for field_name in (
                "batch_id",
                "generated_post_id",
                "image_generation_id",
            ):
                if workflow[field_name] is not None:
                    raise ValueError(
                        "Recovery запрещён: downstream state "
                        "уже существует: "
                        f"{field_name}={workflow[field_name]}"
                    )

            as_of = workflow["as_of"]

            if not isinstance(as_of, datetime):
                raise ValueError(
                    "Workflow содержит некорректный as_of."
                )

            window_hours = int(workflow["window_hours"])

            if window_hours != 24:
                raise ValueError(
                    "Recovery разрешён только для строгого "
                    "24-часового окна."
                )

            failed_ranking_run_id = int(
                workflow["ranking_run_id"]
            )

            ranking = await connection.fetchrow(
                """
                SELECT
                    ranking_run_id,
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
                    error_message,
                    request_key
                FROM top3_news.ranking_runs
                WHERE ranking_run_id = $1
                FOR UPDATE
                """,
                failed_ranking_run_id,
            )

            if ranking is None:
                raise LookupError(
                    "Связанный ranking_run не найден."
                )

            if ranking["run_status"] != "failed":
                raise ValueError(
                    "Recovery запрещён: ranking_run не failed."
                )

            historical_formula_version = _required_text(
                ranking["formula_version"],
                field_name="ranking.formula_version",
            )

            if historical_formula_version != (
                EXPECTED_HISTORICAL_FORMULA_VERSION
            ):
                raise ValueError(
                    "Recovery разрешён только для historical "
                    "top3_cinema_v4 failure: "
                    f"{historical_formula_version!r}"
                )

            ranking_eligible_count = (
                _parse_insufficient_top3_error(
                    ranking["error_message"],
                    field_name="ranking.error_message",
                )
            )

            if ranking_eligible_count != workflow_eligible_count:
                raise ValueError(
                    "Workflow и ranking_run содержат разные "
                    "eligible_count в ошибке."
                )

            if int(ranking["eligible_count"]) != (
                ranking_eligible_count
            ):
                raise ValueError(
                    "ranking_runs.eligible_count не совпадает "
                    "с diagnostic error."
                )

            candidate_count = int(
                ranking["candidate_count"]
            )
            scored_count = int(
                ranking["scored_count"]
            )

            if candidate_count <= 0 or scored_count <= 0:
                raise ValueError(
                    "Recovery запрещён: historical ranking "
                    "не содержит diagnostic scores."
                )

            parameters = _json_object(
                ranking["parameters"],
                field_name="ranking.parameters",
            )

            evaluator_version = str(
                parameters.get("evaluator_version", "")
                or ""
            ).strip()

            if evaluator_version != EXPECTED_EVALUATOR_VERSION:
                raise ValueError(
                    "Recovery запрещён: unexpected "
                    "historical evaluator_version: "
                    f"{evaluator_version!r}"
                )

            if str(
                parameters.get("request_key_version", "")
                or ""
            ).strip() != EXPECTED_REQUEST_KEY_VERSION:
                raise ValueError(
                    "Recovery запрещён: unexpected "
                    "request_key_version."
                )

            if str(
                parameters.get("run_mode", "")
                or ""
            ).strip() != EXPECTED_RUN_MODE:
                raise ValueError(
                    "Recovery запрещён: unexpected run_mode."
                )

            if str(
                parameters.get("failure_version", "")
                or ""
            ).strip() != EXPECTED_FAILURE_VERSION:
                raise ValueError(
                    "Recovery запрещён: unexpected "
                    "failure_version."
                )

            if parameters.get(
                "diagnostic_scores_persisted"
            ) is not True:
                raise ValueError(
                    "Recovery запрещён: diagnostic scores "
                    "не подтверждены как persisted."
                )

            if int(
                parameters.get("combination_count", -1)
            ) != 0:
                raise ValueError(
                    "Recovery запрещён: historical failure "
                    "уже содержит ranking combinations."
                )

            winner_news_ids = parameters.get(
                "winner_news_ids"
            )

            if winner_news_ids != []:
                raise ValueError(
                    "Recovery запрещён: historical failure "
                    "содержит winner_news_ids."
                )

            failure = _json_object(
                parameters.get("failure"),
                field_name="ranking.parameters.failure",
            )

            if str(
                failure.get("stage", "")
                or ""
            ).strip() != EXPECTED_FAILURE_STAGE:
                raise ValueError(
                    "Recovery запрещён: failure.stage "
                    "не top3_selection."
                )

            if str(
                failure.get("error_type", "")
                or ""
            ).strip() != EXPECTED_ERROR_TYPE:
                raise ValueError(
                    "Recovery запрещён: failure.error_type "
                    "не ValueError."
                )

            failure_eligible_count = (
                _parse_insufficient_top3_error(
                    failure.get("error_message"),
                    field_name=(
                        "ranking.parameters.failure.error_message"
                    ),
                )
            )

            if failure_eligible_count != ranking_eligible_count:
                raise ValueError(
                    "parameters.failure eligible_count "
                    "не совпадает с ranking_run."
                )

            score_counts = await connection.fetchrow(
                """
                SELECT
                    COUNT(*)::integer AS score_count,
                    COUNT(*) FILTER (
                        WHERE is_eligible = true
                    )::integer AS strict_eligible_count,
                    COUNT(*) FILTER (
                        WHERE
                            is_eligible = false
                            AND exclusion_reason = $2
                            AND q_score > 0
                            AND individual_score >= $3::numeric
                    )::integer AS promotable_count
                FROM top3_news.news_scores
                WHERE ranking_run_id = $1
                """,
                failed_ranking_run_id,
                EXPECTED_EXCLUSION_REASON,
                EXPECTED_FALLBACK_THRESHOLD,
            )

            if score_counts is None:
                raise AssertionError(
                    "Не удалось прочитать historical news_scores."
                )

            if int(score_counts["score_count"]) != scored_count:
                raise ValueError(
                    "Количество historical news_scores "
                    "не совпадает со scored_count."
                )

            strict_eligible_count = int(
                score_counts["strict_eligible_count"]
            )

            if strict_eligible_count != ranking_eligible_count:
                raise ValueError(
                    "Historical strict eligible count "
                    "не совпадает с ranking_runs.eligible_count."
                )

            promotable_rows = await connection.fetch(
                """
                SELECT
                    news_id,
                    individual_score
                FROM top3_news.news_scores
                WHERE ranking_run_id = $1
                  AND is_eligible = false
                  AND exclusion_reason = $2
                  AND q_score > 0
                  AND individual_score >= $3::numeric
                ORDER BY
                    individual_score DESC,
                    news_id
                """,
                failed_ranking_run_id,
                EXPECTED_EXCLUSION_REASON,
                EXPECTED_FALLBACK_THRESHOLD,
            )

            promotable_news_ids = tuple(
                int(row["news_id"])
                for row in promotable_rows
            )

            if len(promotable_news_ids) != int(
                score_counts["promotable_count"]
            ):
                raise RuntimeError(
                    "Promotable count изменился между запросами."
                )

            expected_effective_eligible_count = (
                strict_eligible_count
                + len(promotable_news_ids)
            )

            if expected_effective_eligible_count < 3:
                raise ValueError(
                    "Recovery бессмысленен: даже top3_cinema_v5 "
                    "fallback floor 3.0 не даст три события: "
                    f"strict={strict_eligible_count}, "
                    f"promotable={len(promotable_news_ids)}"
                )

            expected_window_started_at = (
                as_of
                - timedelta(hours=window_hours)
            )

            if ranking["window_started_at"] != (
                expected_window_started_at
            ):
                raise ValueError(
                    "ranking window_started_at не совпадает "
                    "с workflow window."
                )

            if ranking["window_finished_at"] != as_of:
                raise ValueError(
                    "ranking window_finished_at не совпадает "
                    "с workflow as_of."
                )

            model_name = _required_text(
                ranking["model_name"],
                field_name="ranking.model_name",
            )
            prompt_version = _required_text(
                ranking["prompt_version"],
                field_name="ranking.prompt_version",
            )
            request_key = _required_text(
                ranking["request_key"],
                field_name="ranking.request_key",
            )

            if not re.fullmatch(
                r"[0-9a-f]{64}",
                request_key,
            ):
                raise ValueError(
                    "Historical request_key не является "
                    "lowercase SHA-256."
                )

            context = RecoveryContext(
                daily_workflow_run_id=int(
                    workflow["daily_workflow_run_id"]
                ),
                publication_date=workflow["publication_date"],
                workflow_version=workflow_version,
                as_of=as_of,
                window_hours=window_hours,
                historical_failed_ranking_run_id=(
                    failed_ranking_run_id
                ),
                historical_formula_version=(
                    historical_formula_version
                ),
                historical_model_name=model_name,
                historical_prompt_version=prompt_version,
                historical_evaluator_version=(
                    evaluator_version
                ),
                historical_candidate_count=(
                    candidate_count
                ),
                historical_scored_count=scored_count,
                historical_strict_eligible_count=(
                    strict_eligible_count
                ),
                fallback_promotable_news_ids=(
                    promotable_news_ids
                ),
                expected_effective_eligible_count=(
                    expected_effective_eligible_count
                ),
                historical_request_key=request_key,
            )

            if dry_run:
                return context

            updated = await connection.execute(
                """
                UPDATE top3_news.daily_workflow_runs
                SET
                    workflow_status = 'running',
                    current_stage = 'ranking',
                    ranking_run_id = NULL,
                    batch_id = NULL,
                    generated_post_id = NULL,
                    image_generation_id = NULL,
                    error_type = NULL,
                    error_message = NULL,
                    finished_at = NULL
                WHERE daily_workflow_run_id = $1
                  AND workflow_status = 'failed'
                  AND current_stage = 'failed'
                  AND ranking_run_id = $2
                """,
                daily_workflow_run_id,
                failed_ranking_run_id,
            )

            if updated != "UPDATE 1":
                raise RuntimeError(
                    "Не удалось reopen daily workflow."
                )

            return context


def _print_context(
    context: RecoveryContext,
) -> None:
    _progress(
        "daily_workflow_run_id="
        f"{context.daily_workflow_run_id}"
    )
    _progress(
        "publication_date="
        f"{context.publication_date}"
    )
    _progress(
        f"as_of={context.as_of.isoformat()}"
    )
    _progress(
        f"window_hours={context.window_hours}"
    )
    _progress(
        "historical_failed_ranking_run_id="
        f"{context.historical_failed_ranking_run_id}"
    )
    _progress(
        "historical_formula_version="
        f"{context.historical_formula_version}"
    )
    _progress(
        "recovery_formula_version="
        f"{FULL_FORMULA_VERSION}"
    )
    _progress(
        "historical_model_name="
        f"{context.historical_model_name}"
    )
    _progress(
        "historical_prompt_version="
        f"{context.historical_prompt_version}"
    )
    _progress(
        "historical_evaluator_version="
        f"{context.historical_evaluator_version}"
    )
    _progress(
        "historical_candidate_count="
        f"{context.historical_candidate_count}"
    )
    _progress(
        "historical_scored_count="
        f"{context.historical_scored_count}"
    )
    _progress(
        "historical_strict_eligible_count="
        f"{context.historical_strict_eligible_count}"
    )
    _progress(
        "fallback_promotable_news_ids="
        + ",".join(
            str(news_id)
            for news_id in (
                context.fallback_promotable_news_ids
            )
        )
    )
    _progress(
        "expected_effective_eligible_count="
        f"{context.expected_effective_eligible_count}"
    )
    _progress(
        "historical_request_key="
        f"{context.historical_request_key}"
    )


async def async_main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a daily workflow failed under "
            "top3_cinema_v4 because strict eligible_count < 3, "
            "after deploying top3_cinema_v5 score-floor fallback."
        )
    )

    parser.add_argument(
        "--daily-workflow-run-id",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate historical failure without changing "
            "the database or running production."
        ),
    )

    args = parser.parse_args()

    if args.daily_workflow_run_id <= 0:
        raise ValueError(
            "daily-workflow-run-id должен быть больше нуля."
        )

    if args.candidate_limit <= 0:
        raise ValueError(
            "candidate-limit должен быть больше нуля."
        )

    _validate_runtime_policy()

    settings = get_settings()
    pool = await create_database_pool(settings)

    try:
        context = await _reopen_failed_workflow(
            pool,
            daily_workflow_run_id=(
                args.daily_workflow_run_id
            ),
            dry_run=args.dry_run,
        )

        if args.dry_run:
            _progress("Dry-run validation: OK")
            _print_context(context)
            _progress("database_changes=none")
            _progress("production_workflow_started=false")
            return 0

        _progress(
            "Insufficient TOP-3 workflow reopened"
        )
        _print_context(context)

        result = await run_daily_production_workflow(
            pool,
            settings=settings,
            publication_date=context.publication_date,
            as_of=context.as_of,
            candidate_limit=args.candidate_limit,
            progress=_progress,
        )

        _progress("Recovery workflow: OK")
        _progress(
            "result_status="
            f"{result.workflow_status}"
        )
        _progress(
            "new_ranking_run_id="
            f"{result.ranking_run_id}"
        )
        _progress(
            "generated_post_id="
            f"{result.generated_post_id}"
        )
        _progress(
            "image_generation_id="
            f"{result.image_generation_id}"
        )

        return 0
    finally:
        await close_database_pool(pool)


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
