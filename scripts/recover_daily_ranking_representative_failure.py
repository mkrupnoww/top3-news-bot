from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.workflows.daily_production import (
    run_daily_production_workflow,
)


EXPECTED_WORKFLOW_VERSION = "daily_workflow_v1"

EXPECTED_ERROR_TYPE = "ValueError"

EXPECTED_ERROR_MESSAGE = (
    "is_representative=true должен соответствовать "
    "representative_news_id."
)

EXPECTED_EVALUATOR_VERSION = (
    "event_ranking_evaluator_v7"
)

EXPECTED_REQUEST_KEY_VERSION = (
    "event_ranking_request_key_v1"
)

EXPECTED_RUN_MODE = "openai_event_ranking"

EXPECTED_FAILURE_VERSION = (
    "reserved_ranking_failure_v1"
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
    historical_request_key: str


def _progress(message: str) -> None:
    print(message, flush=True)


def _json_object(
    value,
    *,
    field_name: str,
) -> dict:
    """
    Нормализует json/jsonb в dict.

    Поддерживает как уже декодированный объект,
    так и строковое представление JSON.
    """

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


async def _reopen_failed_ranking_workflow(
    pool,
    *,
    daily_workflow_run_id: int,
    dry_run: bool,
) -> RecoveryContext:
    """
    Проверяет и при dry_run=False атомарно reopen'ит
    только workflow, упавший на старой проверке
    is_representative evaluator v7.

    Исторический failed ranking_run не изменяется.

    При dry_run=True выполняются те же SELECT,
    блокировки и проверки, но UPDATE отсутствует.
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

            if (
                workflow_version
                != EXPECTED_WORKFLOW_VERSION
            ):
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

            workflow_error_type = str(
                workflow["error_type"] or ""
            ).strip()

            workflow_error_message = str(
                workflow["error_message"] or ""
            ).strip()

            if (
                workflow_error_type
                != EXPECTED_ERROR_TYPE
            ):
                raise ValueError(
                    "Recovery запрещён: workflow "
                    "error_type не соответствует "
                    "старой representative-ошибке: "
                    f"{workflow_error_type!r}"
                )

            if (
                workflow_error_message
                != EXPECTED_ERROR_MESSAGE
            ):
                raise ValueError(
                    "Recovery запрещён: workflow "
                    "error_message не соответствует "
                    "старой representative-ошибке."
                )

            if workflow["ranking_run_id"] is None:
                raise ValueError(
                    "Workflow не содержит "
                    "ranking_run_id."
                )

            if workflow["batch_id"] is not None:
                raise ValueError(
                    "Recovery запрещён: batch_id "
                    "уже существует."
                )

            if (
                workflow["generated_post_id"]
                is not None
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "generated_post_id уже существует."
                )

            if (
                workflow["image_generation_id"]
                is not None
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "image_generation_id уже существует."
                )

            as_of = workflow["as_of"]

            if not isinstance(as_of, datetime):
                raise ValueError(
                    "Workflow содержит некорректный "
                    "as_of."
                )

            window_hours = int(
                workflow["window_hours"]
            )

            if window_hours <= 0:
                raise ValueError(
                    "Workflow содержит некорректный "
                    "window_hours."
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

            if (
                int(ranking["ranking_run_id"])
                != failed_ranking_run_id
            ):
                raise RuntimeError(
                    "Получен неожиданный ranking_run."
                )

            if ranking["run_status"] != "failed":
                raise ValueError(
                    "Recovery запрещён: связанный "
                    "ranking_run не failed."
                )

            ranking_error_message = str(
                ranking["error_message"] or ""
            ).strip()

            if (
                ranking_error_message
                != EXPECTED_ERROR_MESSAGE
            ):
                raise ValueError(
                    "Recovery запрещён: ranking "
                    "error_message не соответствует "
                    "старой representative-ошибке."
                )

            parameters = _json_object(
                ranking["parameters"],
                field_name="ranking.parameters",
            )

            evaluator_version = str(
                parameters.get(
                    "evaluator_version",
                    "",
                )
                or ""
            ).strip()

            if (
                evaluator_version
                != EXPECTED_EVALUATOR_VERSION
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "evaluator_version не является "
                    "старым v7: "
                    f"{evaluator_version!r}"
                )

            request_key_version = str(
                parameters.get(
                    "request_key_version",
                    "",
                )
                or ""
            ).strip()

            if (
                request_key_version
                != EXPECTED_REQUEST_KEY_VERSION
            ):
                raise ValueError(
                    "Recovery запрещён: неизвестный "
                    "request_key_version: "
                    f"{request_key_version!r}"
                )

            run_mode = str(
                parameters.get(
                    "run_mode",
                    "",
                )
                or ""
            ).strip()

            if run_mode != EXPECTED_RUN_MODE:
                raise ValueError(
                    "Recovery запрещён: run_mode "
                    "не соответствует event ranking: "
                    f"{run_mode!r}"
                )

            failure_version = str(
                parameters.get(
                    "failure_version",
                    "",
                )
                or ""
            ).strip()

            if (
                failure_version
                != EXPECTED_FAILURE_VERSION
            ):
                raise ValueError(
                    "Recovery запрещён: неизвестный "
                    "failure_version: "
                    f"{failure_version!r}"
                )

            if (
                parameters.get(
                    "idempotency_reserved"
                )
                is not True
            ):
                raise ValueError(
                    "Recovery запрещён: historical "
                    "ranking не подтверждает "
                    "idempotency_reserved=true."
                )

            failure = _json_object(
                parameters.get("failure"),
                field_name=(
                    "ranking.parameters.failure"
                ),
            )

            failure_error_type = str(
                failure.get(
                    "error_type",
                    "",
                )
                or ""
            ).strip()

            failure_error_message = str(
                failure.get(
                    "error_message",
                    "",
                )
                or ""
            ).strip()

            if (
                failure_error_type
                != EXPECTED_ERROR_TYPE
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "parameters.failure.error_type "
                    "не соответствует ожидаемому."
                )

            if (
                failure_error_message
                != EXPECTED_ERROR_MESSAGE
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "parameters.failure.error_message "
                    "не соответствует ожидаемому."
                )

            candidate_count = int(
                ranking["candidate_count"]
            )

            if candidate_count <= 0:
                raise ValueError(
                    "Recovery запрещён: historical "
                    "ranking не содержит кандидатов."
                )

            if int(ranking["scored_count"]) != 0:
                raise ValueError(
                    "Recovery запрещён: failed ranking "
                    "уже содержит scored results."
                )

            if int(ranking["eligible_count"]) != 0:
                raise ValueError(
                    "Recovery запрещён: failed ranking "
                    "уже содержит eligible results."
                )

            news_ids = parameters.get("news_ids")

            if not isinstance(news_ids, list):
                raise ValueError(
                    "Recovery запрещён: parameters.news_ids "
                    "отсутствует или имеет неверный тип."
                )

            if len(news_ids) != candidate_count:
                raise ValueError(
                    "Recovery запрещён: количество "
                    "parameters.news_ids не совпадает "
                    "с candidate_count."
                )

            expected_window_started_at = (
                as_of
                - timedelta(
                    hours=window_hours
                )
            )

            if (
                ranking["window_started_at"]
                != expected_window_started_at
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "window_started_at ranking_run "
                    "не совпадает с workflow window."
                )

            if (
                ranking["window_finished_at"]
                != as_of
            ):
                raise ValueError(
                    "Recovery запрещён: "
                    "window_finished_at ranking_run "
                    "не совпадает с workflow as_of."
                )

            formula_version = _required_text(
                ranking["formula_version"],
                field_name=(
                    "ranking.formula_version"
                ),
            )

            model_name = _required_text(
                ranking["model_name"],
                field_name="ranking.model_name",
            )

            prompt_version = _required_text(
                ranking["prompt_version"],
                field_name=(
                    "ranking.prompt_version"
                ),
            )

            request_key = _required_text(
                ranking["request_key"],
                field_name="ranking.request_key",
            )

            if len(request_key) != 64:
                raise ValueError(
                    "Recovery запрещён: historical "
                    "request_key имеет неверную длину."
                )

            if any(
                character
                not in "0123456789abcdef"
                for character in request_key
            ):
                raise ValueError(
                    "Recovery запрещён: historical "
                    "request_key не является "
                    "lowercase SHA-256."
                )

            context = RecoveryContext(
                daily_workflow_run_id=int(
                    workflow[
                        "daily_workflow_run_id"
                    ]
                ),
                publication_date=workflow[
                    "publication_date"
                ],
                workflow_version=(
                    workflow_version
                ),
                as_of=as_of,
                window_hours=window_hours,
                historical_failed_ranking_run_id=(
                    failed_ranking_run_id
                ),
                historical_formula_version=(
                    formula_version
                ),
                historical_model_name=model_name,
                historical_prompt_version=(
                    prompt_version
                ),
                historical_evaluator_version=(
                    evaluator_version
                ),
                historical_candidate_count=(
                    candidate_count
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
        "workflow_version="
        f"{context.workflow_version}"
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
        "historical_request_key="
        f"{context.historical_request_key}"
    )


async def async_main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a daily workflow failed "
            "specifically by the old evaluator v7 "
            "is_representative validation."
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
            "Validate the historical failure "
            "without changing the database or "
            "running production."
        ),
    )

    args = parser.parse_args()

    if args.daily_workflow_run_id <= 0:
        raise ValueError(
            "daily-workflow-run-id должен быть "
            "больше нуля."
        )

    if args.candidate_limit <= 0:
        raise ValueError(
            "candidate-limit должен быть больше нуля."
        )

    settings = get_settings()
    pool = await create_database_pool(settings)

    try:
        context = (
            await _reopen_failed_ranking_workflow(
                pool,
                daily_workflow_run_id=(
                    args.daily_workflow_run_id
                ),
                dry_run=args.dry_run,
            )
        )

        if args.dry_run:
            _progress(
                "Dry-run validation: OK"
            )
            _print_context(context)
            _progress(
                "database_changes=none"
            )
            _progress(
                "production_workflow_started=false"
            )
            return 0

        _progress(
            "Representative ranking workflow reopened"
        )
        _print_context(context)

        result = await run_daily_production_workflow(
            pool,
            settings=settings,
            publication_date=(
                context.publication_date
            ),
            as_of=context.as_of,
            candidate_limit=args.candidate_limit,
            progress=_progress,
        )

        _progress("Recovery workflow: OK")
        _progress(
            "result_status="
            f"{result.workflow_status}"
        )

        return 0
    finally:
        await close_database_pool(pool)


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())