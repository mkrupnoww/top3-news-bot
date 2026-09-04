import inspect
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import AsyncIterator, Callable

import asyncpg
from aiogram import Bot

from app.bot.handlers import build_review_keyboard
from app.bot.review_delivery_service import (
    ReviewDeliveryResult,
    deliver_generated_post_to_reviewers,
)
from app.config import Settings
from app.db.daily_workflow import (
    DAILY_WORKFLOW_VERSION,
    MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION,
    DailyWorkflowImageModerationRetryNotAllowedError,
    DailyWorkflowRun,
    complete_daily_workflow,
    fail_daily_workflow,
    load_daily_workflow,
    mark_daily_workflow_stage,
    reopen_daily_workflow_for_image_moderation_retry,
    require_daily_workflow_image_moderation_retry,
    reserve_daily_workflow,
)
from app.db.daily_workflow_replacement import (
    replace_daily_workflow_after_image_moderation,
)
from app.db.daily_workflow_trailer_replacement import (
    replace_daily_workflow_after_trailer_unverified,
)
from app.db.daily_workflow_selection_attempts import (
    DailyWorkflowSelectionAttempt,
    ensure_initial_daily_workflow_selection,
    load_active_daily_workflow_selection,
    load_used_daily_workflow_combination_ids,
)
from app.db.daily_workflow_checkpoints import (
    checkpoint_generated_post,
    checkpoint_generation_reservation,
    checkpoint_image_reservation,
    checkpoint_ranking_reservation,
    recover_batch_id,
    recover_image_generation_id,
    recover_ranking_run_id,
)
from app.db.daily_workflow_state import (
    load_generation_workflow_state,
    load_ranking_workflow_state,
)
from app.db.generation_selection import (
    GenerationTop3Selection,
    choose_next_generation_combination,
    load_generation_combination,
)
from app.generation.openai_factory import (
    OpenAIGenerationRuntime,
    create_openai_generation_runtime,
)
from app.generation.image_generator import (
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    OPENAI_IMAGE_PROMPT_VERSION,
)
from app.generation.openai_image_factory import (
    OpenAIImageGenerationRuntime,
    create_openai_image_generation_runtime,
)
from app.generation.openai_image_pipeline import (
    run_reserved_openai_image_generation,
)
from app.generation.openai_pipeline import (
    run_reserved_openai_generation,
)
from app.generation.official_trailer_enrichment import (
    preflight_generation_official_trailers,
)
from app.generation.request_key import (
    GenerationRequestKey,
    create_generation_request_key,
)
from app.ranking.event_request_key import (
    EVENT_REQUEST_KEY_VERSION,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)
from app.ranking.openai_event_factory import (
    OpenAIEventRankingRuntime,
    create_openai_event_ranking_runtime,
)
from app.ranking.openai_event_pipeline import (
    run_reserved_openai_event_ranking,
)


logger = logging.getLogger(__name__)

ProgressReporter = Callable[[str], None]


def _report_progress(
    progress: ProgressReporter | None,
    message: str,
) -> None:
    """Best-effort сообщает текущий этап вызывающему коду."""

    if progress is not None:
        progress(message)


_DAILY_WORKFLOW_LOCK_BASE = (
    730_000_000_000_000_000
)

MAX_TOP3_REPLACEMENTS = 3


class DailyProductionWorkflowError(RuntimeError):
    """Базовая ошибка production daily workflow."""


class DailyProductionWorkflowBusyError(
    DailyProductionWorkflowError
):
    """Другой процесс уже выполняет этот daily workflow."""


class DailyProductionWorkflowTerminalError(
    DailyProductionWorkflowError
):
    """Daily workflow уже находится в terminal failure."""


class DailyProductionWorkflowOrphanError(
    DailyProductionWorkflowError
):
    """
    Найдена незавершённая child reservation.

    Автоматический повтор внешнего запроса
    запрещён, потому что его фактический outcome
    нельзя надёжно доказать.
    """


@dataclass(frozen=True, slots=True)
class DailyProductionResult:
    """Финальный результат одного daily workflow."""

    daily_workflow_run_id: int
    publication_date: date
    as_of: datetime
    workflow_status: str

    ranking_run_id: int
    batch_id: int
    generated_post_id: int
    image_generation_id: int

    ranking_model_called: bool
    generation_model_called: bool
    image_model_called: bool

    reviewer_count: int
    review_sent_count: int
    review_failed_count: int
    review_unknown_count: int
    review_skipped_count: int


def _positive_integer(
    value: int | None,
    *,
    field_name: str,
) -> int:
    """Проверяет обязательный положительный ID."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise DailyProductionWorkflowError(
            f"{field_name} должен быть "
            "положительным int: "
            f"value={value!r}"
        )

    return value


def _required_text(
    value: str | None,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный текст identity."""

    if not isinstance(value, str):
        raise DailyProductionWorkflowError(
            f"{field_name} должен быть str."
        )

    normalized = value.strip()

    if not normalized:
        raise DailyProductionWorkflowError(
            f"{field_name} не может быть пустым."
        )

    return normalized


def _lock_key(
    publication_date: date,
) -> int:
    """Строит отдельный session advisory-lock key."""

    return (
        _DAILY_WORKFLOW_LOCK_BASE
        + publication_date.toordinal()
    )


@asynccontextmanager
async def _daily_workflow_lock(
    pool: asyncpg.Pool,
    *,
    publication_date: date,
) -> AsyncIterator[None]:
    """Не допускает два одновременных orchestrator процесса."""

    key = _lock_key(
        publication_date
    )

    async with pool.acquire() as connection:
        acquired = await connection.fetchval(
            """
            SELECT pg_try_advisory_lock(
                $1::bigint
            )
            """,
            key,
        )

        if acquired is not True:
            raise DailyProductionWorkflowBusyError(
                "Daily workflow уже выполняется "
                "другим процессом: "
                f"publication_date="
                f"{publication_date}"
            )

        try:
            yield
        finally:
            released = await connection.fetchval(
                """
                SELECT pg_advisory_unlock(
                    $1::bigint
                )
                """,
                key,
            )

            if released is not True:
                logger.error(
                    "Не удалось подтвердить release "
                    "daily workflow advisory lock: "
                    "publication_date=%s key=%s",
                    publication_date,
                    key,
                )


async def _close_sdk_client(
    sdk_client: object,
) -> None:
    """Best-effort закрывает совместимый SDK client."""

    close_method = getattr(
        sdk_client,
        "close",
        None,
    )

    if close_method is None:
        return

    try:
        result = close_method()

        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception(
            "Не удалось корректно закрыть "
            "OpenAI SDK client."
        )


async def _close_bot(
    bot: Bot | None,
) -> None:
    """Best-effort закрывает Telegram HTTP session."""

    if bot is None:
        return

    try:
        await bot.session.close()
    except Exception:
        logger.exception(
            "Не удалось корректно закрыть "
            "Telegram Bot session."
        )


async def _fail_workflow_after_error(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    error: BaseException,
) -> None:
    """Best-effort переводит только running workflow в failed."""

    try:
        workflow = await load_daily_workflow(
            pool,
            daily_workflow_run_id=(
                daily_workflow_run_id
            ),
        )

        if not workflow.running:
            return

        await fail_daily_workflow(
            pool,
            daily_workflow_run_id=(
                daily_workflow_run_id
            ),
            error_type=type(error).__name__,
            error_message=(
                str(error).strip()
                or type(error).__name__
            ),
        )
    except Exception as failure_error:
        try:
            error.add_note(
                "Дополнительно не удалось "
                "зафиксировать daily workflow "
                "в failed: "
                f"{type(failure_error).__name__}: "
                f"{failure_error}"
            )
        except Exception:
            pass


def _ranking_recovery_identity(
    runtime: OpenAIEventRankingRuntime,
) -> dict[str, str]:
    """Возвращает identity текущего ranking runtime."""

    metadata = runtime.evaluator.metadata

    return {
        "formula_version": (
            _required_text(
                FULL_FORMULA_VERSION,
                field_name="formula_version",
            )
        ),
        "model_name": (
            _required_text(
                metadata.model_name,
                field_name="model_name",
            )
        ),
        "prompt_version": (
            _required_text(
                metadata.prompt_version,
                field_name="prompt_version",
            )
        ),
        "run_mode": (
            _required_text(
                metadata.run_mode,
                field_name="run_mode",
            )
        ),
        "evaluator_name": (
            _required_text(
                metadata.evaluator_name,
                field_name="evaluator_name",
            )
        ),
        "evaluator_version": (
            _required_text(
                metadata.evaluator_version,
                field_name="evaluator_version",
            )
        ),
        "request_key_version": (
            _required_text(
                EVENT_REQUEST_KEY_VERSION,
                field_name="request_key_version",
            )
        ),
    }


def _generation_request_key(
    *,
    runtime: OpenAIGenerationRuntime,
    selection: GenerationTop3Selection,
    workflow: DailyWorkflowRun,
) -> GenerationRequestKey:
    """Строит тот же base request key, что text pipeline."""

    model_request = (
        runtime.generator.build_request(
            selection.items
        )
    )

    return create_generation_request_key(
        ranking_run_id=(
            selection.ranking_run_id
        ),
        publication_date=(
            workflow.publication_date
        ),
        telegram_chat_id=(
            workflow.target_telegram_chat_id
        ),
        metadata=runtime.generator.metadata,
        model_request=model_request,
        items=selection.items,
    )


async def _load_image_request_status(
    pool: asyncpg.Pool,
    *,
    image_generation_id: int,
    expected_batch_id: int,
    expected_generated_post_id: int,
) -> str:
    """Проверяет identity и статус initial image request."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
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
            image_generation_id,
        )

    if record is None:
        raise LookupError(
            "image_generation_request "
            "не найден: "
            f"image_generation_id="
            f"{image_generation_id}"
        )

    if int(record["batch_id"]) != expected_batch_id:
        raise DailyProductionWorkflowError(
            "Image request относится "
            "к другому batch."
        )

    if (
        int(record["generated_post_id"])
        != expected_generated_post_id
    ):
        raise DailyProductionWorkflowError(
            "Image request относится "
            "к другому generated_post."
        )

    if record["request_kind"] != "initial":
        raise DailyProductionWorkflowError(
            "Production daily workflow "
            "ожидает initial image request."
        )

    return _required_text(
        record["image_status"],
        field_name="image_status",
    )



@dataclass(frozen=True, slots=True)
class _ImageRequestState:
    """Минимальное доказуемое состояние initial image request."""

    image_generation_id: int
    batch_id: int
    generated_post_id: int
    image_status: str
    prompt_version: str
    definitive_moderation_block: bool


async def _load_image_request_state(
    pool: asyncpg.Pool,
    *,
    image_generation_id: int,
    expected_batch_id: int,
    expected_generated_post_id: int,
) -> _ImageRequestState:
    """Проверяет identity и возвращает доказуемое состояние image request."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                image_generation_id,
                batch_id,
                generated_post_id,
                request_kind,
                image_status,
                prompt_version,
                error_type,
                error_message,
                failed_at
            FROM image_generation_requests
            WHERE image_generation_id = $1
            """,
            image_generation_id,
        )

    if record is None:
        raise LookupError(
            "image_generation_request "
            "не найден: "
            f"image_generation_id="
            f"{image_generation_id}"
        )

    if int(record["batch_id"]) != expected_batch_id:
        raise DailyProductionWorkflowError(
            "Image request относится "
            "к другому batch."
        )

    if (
        int(record["generated_post_id"])
        != expected_generated_post_id
    ):
        raise DailyProductionWorkflowError(
            "Image request относится "
            "к другому generated_post."
        )

    if record["request_kind"] != "initial":
        raise DailyProductionWorkflowError(
            "Production daily workflow "
            "ожидает initial image request."
        )

    image_status = _required_text(
        record["image_status"],
        field_name="image_status",
    )

    prompt_version = _required_text(
        record["prompt_version"],
        field_name="prompt_version",
    )

    error_type = str(
        record["error_type"] or ""
    ).strip()

    error_message = str(
        record["error_message"] or ""
    ).strip().lower()

    definitive_moderation_block = (
        image_status == "failed"
        and record["failed_at"] is not None
        and error_type == "BadRequestError"
        and "moderation_blocked"
        in error_message
    )

    return _ImageRequestState(
        image_generation_id=int(
            record["image_generation_id"]
        ),
        batch_id=int(record["batch_id"]),
        generated_post_id=int(
            record["generated_post_id"]
        ),
        image_status=image_status,
        prompt_version=prompt_version,
        definitive_moderation_block=(
            definitive_moderation_block
        ),
    )


async def _fallback_budget_is_exhausted(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
    generated_post_id: int,
) -> bool:
    """
    Проверяет именно исчерпание fallback budget.

    True допустим только когда:
    - нет reserved/completed initial requests;
    - все fallback attempts являются definitive moderation_blocked;
    - число fallback attempts достигло лимита.
    """

    async with pool.acquire() as connection:
        active_count = await connection.fetchval(
            """
            SELECT COUNT(*)::integer
            FROM image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
              AND image_status IN (
                    'reserved',
                    'completed'
              )
            """,
            batch_id,
            generated_post_id,
        )

        if int(active_count) != 0:
            return False

        records = await connection.fetch(
            """
            SELECT
                image_status,
                error_type,
                error_message,
                failed_at
            FROM image_generation_requests
            WHERE batch_id = $1
              AND generated_post_id = $2
              AND request_kind = 'initial'
              AND prompt_version = $3
            ORDER BY image_generation_id
            """,
            batch_id,
            generated_post_id,
            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
        )

    if (
        len(records)
        < MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION
    ):
        return False

    for record in records:
        error_type = str(
            record["error_type"] or ""
        ).strip()

        error_message = str(
            record["error_message"] or ""
        ).strip().lower()

        if (
            record["image_status"] != "failed"
            or record["failed_at"] is None
            or error_type != "BadRequestError"
            or "moderation_blocked"
            not in error_message
        ):
            return False

    return True


async def _load_winner_combination_id(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> int:
    """Возвращает единственную winner ranking combination."""

    async with pool.acquire() as connection:
        records = await connection.fetch(
            """
            SELECT
                combination_id,
                combination_rank
            FROM ranking_combinations
            WHERE ranking_run_id = $1
              AND is_winner = true
            ORDER BY combination_rank
            """,
            ranking_run_id,
        )

    if len(records) != 1:
        raise DailyProductionWorkflowError(
            "Ranking должен содержать ровно одну "
            "winner combination: "
            f"ranking_run_id={ranking_run_id}, "
            f"found={len(records)}"
        )

    if int(records[0]["combination_rank"]) != 1:
        raise DailyProductionWorkflowError(
            "Winner combination должна иметь "
            "combination_rank=1: "
            f"ranking_run_id={ranking_run_id}, "
            f"combination_rank="
            f"{records[0]['combination_rank']}"
        )

    return _positive_integer(
        int(records[0]["combination_id"]),
        field_name="winner_combination_id",
    )


async def _resolve_active_selection(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    ranking_run_id: int,
) -> tuple[
    DailyWorkflowSelectionAttempt,
    GenerationTop3Selection,
]:
    """
    Возвращает restart-safe active TOP-3 selection.

    Для нового workflow создаёт attempt #1 по winner combination.
    После replacement загружает уже сохранённый active child.
    """

    active = (
        await load_active_daily_workflow_selection(
            pool,
            daily_workflow_run_id=(
                daily_workflow_run_id
            ),
        )
    )

    if active is None:
        winner_combination_id = (
            await _load_winner_combination_id(
                pool,
                ranking_run_id=(
                    ranking_run_id
                ),
            )
        )

        active = (
            await
            ensure_initial_daily_workflow_selection(
                pool,
                daily_workflow_run_id=(
                    daily_workflow_run_id
                ),
                combination_id=(
                    winner_combination_id
                ),
            )
        )

    if active.ranking_run_id != ranking_run_id:
        raise DailyProductionWorkflowError(
            "Active selection относится "
            "к другому ranking_run: "
            f"selection_ranking_run_id="
            f"{active.ranking_run_id}, "
            f"workflow_ranking_run_id="
            f"{ranking_run_id}"
        )

    ranked_combination = (
        await load_generation_combination(
            pool,
            ranking_run_id=ranking_run_id,
            combination_id=(
                active.combination_id
            ),
        )
    )

    if (
        ranked_combination.combination_rank
        != active.combination_rank
    ):
        raise DailyProductionWorkflowError(
            "Active selection combination_rank "
            "не совпадает с ranking combination."
        )

    return (
        active,
        ranked_combination.selection,
    )


async def _choose_workflow_replacement(
    pool: asyncpg.Pool,
    *,
    daily_workflow_run_id: int,
    ranking_run_id: int,
    active_selection: DailyWorkflowSelectionAttempt,
    current_selection: GenerationTop3Selection,
    excluded_news_ids: tuple[int, ...] = (),
):
    """Выбирает следующий replacement с лимитом workflow."""

    replacements_used = (
        active_selection.attempt_number - 1
    )

    if replacements_used >= MAX_TOP3_REPLACEMENTS:
        return None

    used_combination_ids = (
        await load_used_daily_workflow_combination_ids(
            pool,
            daily_workflow_run_id=(
                daily_workflow_run_id
            ),
        )
    )

    return (
        await choose_next_generation_combination(
            pool,
            ranking_run_id=ranking_run_id,
            current_news_ids=(
                current_selection.news_ids
            ),
            excluded_combination_ids=(
                used_combination_ids
            ),
            excluded_news_ids=excluded_news_ids,
        )
    )


def _validate_review_delivery(
    result: ReviewDeliveryResult,
) -> None:
    """Требует однозначную доставку всем reviewer-ам."""

    if result.reviewer_count <= 0:
        raise DailyProductionWorkflowError(
            "В bot_users нет активных "
            "admin/reviewer для review delivery."
        )

    if result.unknown_count > 0:
        raise DailyProductionWorkflowOrphanError(
            "Telegram review delivery имеет "
            "unknown outcome. Автоматический "
            "повтор запрещён."
        )

    if result.failed_count > 0:
        raise DailyProductionWorkflowError(
            "Telegram отклонил review delivery "
            "для одного или нескольких reviewer-ов: "
            f"failed_count={result.failed_count}"
        )

    if result.sent_count != result.reviewer_count:
        raise DailyProductionWorkflowOrphanError(
            "Не все review delivery имеют "
            "доказанный status=sent: "
            f"reviewer_count={result.reviewer_count}, "
            f"sent_count={result.sent_count}, "
            f"skipped_count={result.skipped_count}"
        )


def _terminal_result(
    workflow: DailyWorkflowRun,
) -> DailyProductionResult:
    """Возвращает уже завершённый workflow без повторов."""

    return DailyProductionResult(
        daily_workflow_run_id=(
            workflow.daily_workflow_run_id
        ),
        publication_date=(
            workflow.publication_date
        ),
        as_of=workflow.as_of,
        workflow_status=(
            workflow.workflow_status
        ),
        ranking_run_id=(
            _positive_integer(
                workflow.ranking_run_id,
                field_name="ranking_run_id",
            )
        ),
        batch_id=(
            _positive_integer(
                workflow.batch_id,
                field_name="batch_id",
            )
        ),
        generated_post_id=(
            _positive_integer(
                workflow.generated_post_id,
                field_name="generated_post_id",
            )
        ),
        image_generation_id=(
            _positive_integer(
                workflow.image_generation_id,
                field_name="image_generation_id",
            )
        ),
        ranking_model_called=False,
        generation_model_called=False,
        image_model_called=False,
        reviewer_count=0,
        review_sent_count=0,
        review_failed_count=0,
        review_unknown_count=0,
        review_skipped_count=0,
    )


async def run_daily_production_workflow(
    pool: asyncpg.Pool,
    *,
    settings: Settings,
    publication_date: date,
    as_of: datetime,
    candidate_limit: int = 500,
    progress: ProgressReporter | None = None,
) -> DailyProductionResult:
    """
    Выполняет production pipeline до Telegram review.

    Последовательность:

    daily workflow reservation
    -> ranking reservation/recovery
    -> OpenAI event ranking
    -> saved TOP-3
    -> official trailer preflight / optional TOP-3 replacement
    -> text generation + self-review
    -> image generation
    -> native Telegram photo+caption
    -> review buttons
    -> daily workflow awaiting_review

    Уже завершённые child stages повторно
    OpenAI/Telegram не вызывают.

    Failed и orphan/uncertain child states
    автоматически не переигрываются, кроме
    доказанного Image API moderation_blocked.

    Trailer/teaser-news проверяется до платной text generation.
    Если официальный URL не подтверждён, workflow пробует
    следующую ranking combination без этой новости. Если
    replacement недоступен, текущий TOP-3 публикуется в
    degraded режиме без trailer-ссылки: daily post не падает.

    Для каждой active TOP-3 combination:
    - normal image выполняется один раз;
    - moderation-safe fallback сохраняет собственный
      prompt_version/request key и текущий retry budget;
    - после исчерпания fallback budget выбирается следующая
      сохранённая ranking combination с максимальным overlap;
    - старые batch/post атомарно переводятся в superseded;
    - новый combination получает новый batch/post/image budget.

    На этом этапе разрешено максимум три replacement TOP-3
    после исходного winner. Emergency title-poster/local PNG
    добавляются отдельным следующим слоем.
    """

    if (
        isinstance(candidate_limit, bool)
        or not isinstance(candidate_limit, int)
        or candidate_limit <= 0
    ):
        raise ValueError(
            "candidate_limit должен быть "
            "положительным int."
        )

    ranking_runtime: (
        OpenAIEventRankingRuntime
        | None
    ) = None

    generation_runtime: (
        OpenAIGenerationRuntime
        | None
    ) = None

    image_runtime: (
        OpenAIImageGenerationRuntime
        | None
    ) = None

    bot: Bot | None = None

    ranking_model_called = False
    generation_model_called = False
    image_model_called = False

    async with _daily_workflow_lock(
        pool,
        publication_date=publication_date,
    ):
        _report_progress(
            progress,
            "[daily] reserve/load workflow",
        )

        workflow = await reserve_daily_workflow(
            pool,
            publication_date=publication_date,
            as_of=as_of,
            target_telegram_chat_id=(
                settings.telegram_channel_id
            ),
            workflow_version=(
                DAILY_WORKFLOW_VERSION
            ),
            window_hours=24,
        )

        if workflow.awaiting_review:
            return _terminal_result(
                workflow
            )

        image_recovery_from_failed_request = False
        image_use_moderation_safe_fallback = False
        image_fallback_attempts_used = 0

        if workflow.failed:
            try:
                workflow = (
                    await
                    reopen_daily_workflow_for_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            workflow.daily_workflow_run_id
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                        ),
                    )
                )

                image_fallback_attempts_used = (
                    await
                    require_daily_workflow_image_moderation_retry(
                        pool,
                        daily_workflow_run_id=(
                            workflow.daily_workflow_run_id
                        ),
                        prompt_version=(
                            OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                        ),
                    )
                )
            except (
                DailyWorkflowImageModerationRetryNotAllowedError
            ) as retry_error:
                raise (
                    DailyProductionWorkflowTerminalError(
                        "Daily workflow уже failed "
                        "и не допускает moderation-safe "
                        "image fallback: "
                        f"daily_workflow_run_id="
                        f"{workflow.daily_workflow_run_id}; "
                        f"fallback_prompt_version="
                        f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}; "
                        f"reason={retry_error}"
                    )
                ) from retry_error

            image_recovery_from_failed_request = True
            image_use_moderation_safe_fallback = True

            _report_progress(
                progress,
                "[daily] reopen failed workflow "
                "for moderation-safe image fallback "
                f"prompt_version="
                f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}; "
                f"attempts_used="
                f"{image_fallback_attempts_used}",
            )

        if not workflow.running:
            raise DailyProductionWorkflowError(
                "Неподдерживаемый daily workflow "
                "status: "
                f"{workflow.workflow_status}"
            )

        workflow_id = (
            workflow.daily_workflow_run_id
        )

        try:
            # ============================================================
            # Ranking
            # ============================================================

            _report_progress(
                progress,
                "[ranking] resolve reservation/state",
            )

            ranking_runtime = (
                create_openai_event_ranking_runtime(
                    settings
                )
            )

            ranking_identity = (
                _ranking_recovery_identity(
                    ranking_runtime
                )
            )

            ranking_run_id = (
                workflow.ranking_run_id
            )

            if ranking_run_id is None:
                ranking_run_id = (
                    await recover_ranking_run_id(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        **ranking_identity,
                    )
                )

                if ranking_run_id is not None:
                    await checkpoint_ranking_reservation(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        ranking_run_id=(
                            ranking_run_id
                        ),
                    )

            if ranking_run_id is None:

                async def ranking_observer(
                    reservation,
                ) -> None:
                    await checkpoint_ranking_reservation(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        ranking_run_id=(
                            reservation
                            .ranking_run_id
                        ),
                    )

                _report_progress(
                    progress,
                    "[ranking] run protected OpenAI ranking",
                )

                ranking_result = (
                    await run_reserved_openai_event_ranking(
                        pool,
                        evaluator=(
                            ranking_runtime.evaluator
                        ),
                        as_of=as_of,
                        window_hours=24.0,
                        limit=candidate_limit,
                        reservation_observer=(
                            ranking_observer
                        ),
                    )
                )

                ranking_model_called = (
                    ranking_result.model_called
                )

                ranking_run_id = (
                    ranking_result.ranking_run_id
                )

            ranking_run_id = _positive_integer(
                ranking_run_id,
                field_name="ranking_run_id",
            )

            ranking_state = (
                await load_ranking_workflow_state(
                    pool,
                    ranking_run_id=(
                        ranking_run_id
                    ),
                )
            )

            if ranking_state.failed:
                raise DailyProductionWorkflowError(
                    "Ranking child stage failed: "
                    f"ranking_run_id="
                    f"{ranking_run_id}, "
                    f"run_status="
                    f"{ranking_state.run_status}"
                )

            if ranking_state.in_progress:
                raise DailyProductionWorkflowOrphanError(
                    "Ranking reservation осталась "
                    "running. Автоматический "
                    "повтор OpenAI запрещён: "
                    f"ranking_run_id="
                    f"{ranking_run_id}"
                )

            if not ranking_state.ready_for_generation:
                raise DailyProductionWorkflowError(
                    "Ranking не готов к generation: "
                    f"ranking_run_id="
                    f"{ranking_run_id}, "
                    f"run_status="
                    f"{ranking_state.run_status}, "
                    f"top3_news_ids="
                    f"{ranking_state.top3_news_ids!r}"
                )

            _report_progress(
                progress,
                f"[ranking] completed ranking_run_id={ranking_run_id}",
            )

            generation_runtime = (
                create_openai_generation_runtime(
                    settings
                )
            )

            # ============================================================
            # Restart-safe TOP-3 selection + replacement cascade
            # ============================================================

            while True:
                (
                    active_selection,
                    selection,
                ) = await _resolve_active_selection(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                    ranking_run_id=(
                        ranking_run_id
                    ),
                )

                combination_id = (
                    active_selection.combination_id
                )

                _report_progress(
                    progress,
                    "[selection] active "
                    f"attempt="
                    f"{active_selection.attempt_number}; "
                    f"combination_id="
                    f"{combination_id}; "
                    f"combination_rank="
                    f"{active_selection.combination_rank}; "
                    f"news_ids="
                    f"{selection.news_ids!r}",
                )

                # ========================================================
                # Text generation + automatic self-review
                # ========================================================

                _report_progress(
                    progress,
                    "[generation] resolve reservation/state",
                )

                workflow = await load_daily_workflow(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                )

                generation_key = (
                    _generation_request_key(
                        runtime=(
                            generation_runtime
                        ),
                        selection=selection,
                        workflow=workflow,
                    )
                )

                batch_id = workflow.batch_id

                if batch_id is None:
                    batch_id = await recover_batch_id(
                        pool,
                        daily_workflow_run_id=(
                            workflow_id
                        ),
                        generation_request_key=(
                            generation_key.value
                        ),
                    )

                    if batch_id is not None:
                        await checkpoint_generation_reservation(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                            batch_id=batch_id,
                        )

                prepared_generation_items = None

                if batch_id is None:
                    trailer_preflight = (
                        await preflight_generation_official_trailers(
                            selection.items
                        )
                    )

                    if not trailer_preflight.ready:
                        rejected_news_ids = (
                            trailer_preflight
                            .unverified_required_news_ids
                        )

                        replacement_candidate = (
                            await _choose_workflow_replacement(
                                pool,
                                daily_workflow_run_id=workflow_id,
                                ranking_run_id=ranking_run_id,
                                active_selection=active_selection,
                                current_selection=selection,
                                excluded_news_ids=rejected_news_ids,
                            )
                        )

                        if replacement_candidate is not None:
                            replacement_result = (
                                await replace_daily_workflow_after_trailer_unverified(
                                    pool,
                                    daily_workflow_run_id=workflow_id,
                                    current_selection_attempt_id=(
                                        active_selection.selection_attempt_id
                                    ),
                                    replacement_combination_id=(
                                        replacement_candidate
                                        .combination
                                        .combination_id
                                    ),
                                    rejected_news_ids=rejected_news_ids,
                                )
                            )

                            _report_progress(
                                progress,
                                "[selection] official trailer not verified; "
                                "replace TOP-3 "
                                f"source_combination_id={combination_id}; "
                                f"replacement_combination_id="
                                f"{replacement_result.replacement_combination_id}; "
                                f"rejected_news_ids={rejected_news_ids!r}",
                            )
                            continue

                        # Availability wins over trailer completeness. This is
                        # the final safety net requested for production: never
                        # lose the whole daily post only because a verified
                        # official trailer URL could not be found.
                        _report_progress(
                            progress,
                            "[selection] official trailer not verified and "
                            "replacement unavailable; continue degraded "
                            f"without trailer link news_ids={rejected_news_ids!r}",
                        )

                    prepared_generation_items = trailer_preflight.items

                if batch_id is None:

                    async def generation_observer(
                        reservation,
                    ) -> None:
                        await checkpoint_generation_reservation(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                            batch_id=(
                                reservation.batch_id
                            ),
                        )

                    _report_progress(
                        progress,
                        "[generation] run text + self-review "
                        f"combination_id="
                        f"{combination_id}",
                    )

                    generation_result = (
                        await
                        run_reserved_openai_generation(
                            pool,
                            generator=(
                                generation_runtime
                                .generator
                            ),
                            ranking_run_id=(
                                ranking_run_id
                            ),
                            combination_id=(
                                combination_id
                            ),
                            publication_date=(
                                workflow
                                .publication_date
                            ),
                            telegram_chat_id=(
                                workflow
                                .target_telegram_chat_id
                            ),
                            prepared_items=(
                                prepared_generation_items
                            ),
                            reservation_observer=(
                                generation_observer
                            ),
                        )
                    )

                    generation_model_called = (
                        generation_model_called
                        or generation_result.model_called
                    )

                    batch_id = (
                        generation_result.batch_id
                    )

                batch_id = _positive_integer(
                    batch_id,
                    field_name="batch_id",
                )

                generation_state = (
                    await load_generation_workflow_state(
                        pool,
                        batch_id=batch_id,
                    )
                )

                if generation_state.failed:
                    raise DailyProductionWorkflowError(
                        "Generation child stage failed: "
                        f"batch_id={batch_id}, "
                        f"batch_status="
                        f"{generation_state.batch_status}"
                    )

                if (
                    generation_state
                    .generation_in_progress
                ):
                    raise (
                        DailyProductionWorkflowOrphanError(
                            "Generation reservation осталась "
                            "in-progress. Автоматический "
                            "повтор HTTP/OpenAI запрещён: "
                            f"batch_id={batch_id}"
                        )
                    )

                if (
                    generation_state
                    .human_review_already_progressed
                ):
                    raise (
                        DailyProductionWorkflowOrphanError(
                            "Human review уже изменил "
                            "состояние batch до завершения "
                            "daily workflow. Требуется "
                            "ручная сверка: "
                            f"batch_id={batch_id}"
                        )
                    )

                generated_post_id = (
                    _positive_integer(
                        generation_state
                        .generated_post_id,
                        field_name=(
                            "generated_post_id"
                        ),
                    )
                )

                _report_progress(
                    progress,
                    "[generation] ready "
                    f"generated_post_id="
                    f"{generated_post_id}; "
                    f"combination_id="
                    f"{combination_id}",
                )

                await checkpoint_generated_post(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                    generated_post_id=(
                        generated_post_id
                    ),
                )

                # ========================================================
                # Image attempts for current TOP-3 selection
                # ========================================================

                _report_progress(
                    progress,
                    "[image] resolve reservation/state "
                    f"combination_id="
                    f"{combination_id}",
                )

                workflow = await load_daily_workflow(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                )

                image_generation_id = (
                    workflow.image_generation_id
                )

                if image_generation_id is None:
                    image_generation_id = (
                        await recover_image_generation_id(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                        )
                    )

                    if (
                        image_generation_id
                        is not None
                    ):
                        await checkpoint_image_reservation(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                            image_generation_id=(
                                image_generation_id
                            ),
                        )

                async def run_initial_image_once(
                    *,
                    recovery_from_failed_request: bool,
                    use_moderation_safe_fallback: bool,
                ) -> int:
                    """Выполняет ровно одну protected initial image attempt."""

                    nonlocal image_runtime
                    nonlocal image_model_called

                    if (
                        not
                        recovery_from_failed_request
                    ):
                        current_generation_state = (
                            await
                            load_generation_workflow_state(
                                pool,
                                batch_id=batch_id,
                            )
                        )

                        if (
                            current_generation_state
                            .image_state_inconsistent
                        ):
                            raise (
                                DailyProductionWorkflowError(
                                    "Generation/image state "
                                    "несогласован до "
                                    "Image API: "
                                    f"batch_id="
                                    f"{batch_id}"
                                )
                            )

                        if (
                            not
                            current_generation_state
                            .ready_for_image
                        ):
                            raise (
                                DailyProductionWorkflowError(
                                    "Post не готов "
                                    "к initial image "
                                    "generation: "
                                    f"batch_id="
                                    f"{batch_id}, "
                                    f"generated_post_id="
                                    f"{generated_post_id}"
                                )
                            )

                    if image_runtime is None:
                        image_runtime = (
                            create_openai_image_generation_runtime(
                                settings
                            )
                        )

                    image_runtime.generator.set_moderation_safe_editorial_fallback(
                        use_moderation_safe_fallback
                    )

                    expected_prompt_version = (
                        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                        if use_moderation_safe_fallback
                        else OPENAI_IMAGE_PROMPT_VERSION
                    )

                    actual_prompt_version = (
                        image_runtime
                        .generator
                        .metadata
                        .prompt_version
                    )

                    if (
                        actual_prompt_version
                        != expected_prompt_version
                    ):
                        raise DailyProductionWorkflowError(
                            "Image runtime "
                            "prompt_version "
                            "не совпадает "
                            "с выбранным режимом: "
                            f"runtime="
                            f"{actual_prompt_version}, "
                            f"expected="
                            f"{expected_prompt_version}"
                        )

                    async def image_observer(
                        reservation,
                    ) -> None:
                        await checkpoint_image_reservation(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                            image_generation_id=(
                                reservation
                                .image_generation_id
                            ),
                        )

                    _report_progress(
                        progress,
                        "[image] run protected Image API "
                        f"combination_id="
                        f"{combination_id}; "
                        f"prompt_version="
                        f"{expected_prompt_version}; "
                        f"moderation_safe_fallback="
                        f"{use_moderation_safe_fallback}",
                    )

                    image_result = (
                        await
                        run_reserved_openai_image_generation(
                            pool,
                            generator=(
                                image_runtime.generator
                            ),
                            selection=selection,
                            batch_id=batch_id,
                            generated_post_id=(
                                generated_post_id
                            ),
                            request_kind="initial",
                            reservation_observer=(
                                image_observer
                            ),
                        )
                    )

                    image_model_called = (
                        image_model_called
                        or image_result.model_called
                    )

                    result_image_id = (
                        _positive_integer(
                            image_result
                            .image_generation_id,
                            field_name=(
                                "image_generation_id"
                            ),
                        )
                    )

                    result_state = (
                        await _load_image_request_state(
                            pool,
                            image_generation_id=(
                                result_image_id
                            ),
                            expected_batch_id=(
                                batch_id
                            ),
                            expected_generated_post_id=(
                                generated_post_id
                            ),
                        )
                    )

                    if (
                        result_state.image_status
                        != "completed"
                    ):
                        raise (
                            DailyProductionWorkflowError(
                                "Image pipeline "
                                "не завершился "
                                "в completed: "
                                f"image_generation_id="
                                f"{result_image_id}, "
                                f"image_status="
                                f"{result_state.image_status}"
                            )
                        )

                    return result_image_id

                moderation_exhausted = False
                failed_image_generation_id: (
                    int | None
                ) = None

                use_fallback = False
                recovery_from_failed = False
                fallback_attempts_used = 0

                if image_generation_id is not None:
                    image_generation_id = (
                        _positive_integer(
                            image_generation_id,
                            field_name=(
                                "image_generation_id"
                            ),
                        )
                    )

                    existing_image_state = (
                        await _load_image_request_state(
                            pool,
                            image_generation_id=(
                                image_generation_id
                            ),
                            expected_batch_id=(
                                batch_id
                            ),
                            expected_generated_post_id=(
                                generated_post_id
                            ),
                        )
                    )

                    if (
                        existing_image_state
                        .image_status
                        == "failed"
                    ):
                        if (
                            not
                            existing_image_state
                            .definitive_moderation_block
                        ):
                            raise (
                                DailyProductionWorkflowError(
                                    "Initial image "
                                    "child stage failed "
                                    "не из-за доказанного "
                                    "moderation_blocked: "
                                    f"image_generation_id="
                                    f"{image_generation_id}"
                                )
                            )

                        try:
                            fallback_attempts_used = (
                                await
                                require_daily_workflow_image_moderation_retry(
                                    pool,
                                    daily_workflow_run_id=(
                                        workflow_id
                                    ),
                                    prompt_version=(
                                        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                                    ),
                                )
                            )
                        except (
                            DailyWorkflowImageModerationRetryNotAllowedError
                        ) as retry_error:
                            exhausted = (
                                await
                                _fallback_budget_is_exhausted(
                                    pool,
                                    batch_id=batch_id,
                                    generated_post_id=(
                                        generated_post_id
                                    ),
                                )
                            )

                            if not exhausted:
                                raise (
                                    DailyProductionWorkflowError(
                                        "Initial image "
                                        "moderation recovery "
                                        "запрещён по причине, "
                                        "отличной от исчерпания "
                                        "fallback budget: "
                                        f"{retry_error}"
                                    )
                                ) from retry_error

                            moderation_exhausted = True
                            failed_image_generation_id = (
                                image_generation_id
                            )
                        else:
                            use_fallback = True
                            recovery_from_failed = True
                            image_generation_id = None

                            _report_progress(
                                progress,
                                "[image] resume moderation-safe "
                                "fallback "
                                f"combination_id="
                                f"{combination_id}; "
                                f"attempts_used="
                                f"{fallback_attempts_used}",
                            )

                    elif (
                        existing_image_state
                        .image_status
                        == "reserved"
                    ):
                        raise (
                            DailyProductionWorkflowOrphanError(
                                "Initial image reservation "
                                "осталась reserved. "
                                "Автоматический повтор "
                                "Image API запрещён: "
                                f"image_generation_id="
                                f"{image_generation_id}"
                            )
                        )

                    elif (
                        existing_image_state
                        .image_status
                        != "completed"
                    ):
                        raise DailyProductionWorkflowError(
                            "Неподдерживаемый "
                            "image_status: "
                            f"{existing_image_state.image_status!r}"
                        )

                while (
                    image_generation_id is None
                    and not moderation_exhausted
                ):
                    workflow_before_call = (
                        await load_daily_workflow(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                        )
                    )

                    linked_image_before_call = (
                        workflow_before_call
                        .image_generation_id
                    )

                    try:
                        image_generation_id = (
                            await run_initial_image_once(
                                recovery_from_failed_request=(
                                    recovery_from_failed
                                ),
                                use_moderation_safe_fallback=(
                                    use_fallback
                                ),
                            )
                        )
                    except Exception as image_error:
                        workflow_after_error = (
                            await load_daily_workflow(
                                pool,
                                daily_workflow_run_id=(
                                    workflow_id
                                ),
                            )
                        )

                        failed_image_id = (
                            workflow_after_error
                            .image_generation_id
                        )

                        if (
                            failed_image_id is None
                            or failed_image_id
                            == linked_image_before_call
                        ):
                            raise

                        failed_state = (
                            await _load_image_request_state(
                                pool,
                                image_generation_id=(
                                    failed_image_id
                                ),
                                expected_batch_id=(
                                    batch_id
                                ),
                                expected_generated_post_id=(
                                    generated_post_id
                                ),
                            )
                        )

                        if (
                            not
                            failed_state
                            .definitive_moderation_block
                        ):
                            raise image_error

                        image_model_called = True
                        failed_image_generation_id = (
                            failed_image_id
                        )

                        if use_fallback:
                            fallback_attempts_used += 1

                            if (
                                fallback_attempts_used
                                >=
                                MAX_IMAGE_ATTEMPTS_PER_PROMPT_VERSION
                            ):
                                moderation_exhausted = True
                                break

                        try:
                            fallback_attempts_used = (
                                await
                                require_daily_workflow_image_moderation_retry(
                                    pool,
                                    daily_workflow_run_id=(
                                        workflow_id
                                    ),
                                    prompt_version=(
                                        OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                                    ),
                                )
                            )
                        except (
                            DailyWorkflowImageModerationRetryNotAllowedError
                        ) as retry_error:
                            exhausted = (
                                await
                                _fallback_budget_is_exhausted(
                                    pool,
                                    batch_id=batch_id,
                                    generated_post_id=(
                                        generated_post_id
                                    ),
                                )
                            )

                            if not exhausted:
                                raise image_error from retry_error

                            moderation_exhausted = True
                            break

                        recovery_from_failed = True
                        use_fallback = True
                        image_generation_id = None

                        _report_progress(
                            progress,
                            "[image] moderation_blocked; "
                            "continue moderation-safe fallback "
                            f"combination_id="
                            f"{combination_id}; "
                            f"attempts_used="
                            f"{fallback_attempts_used}",
                        )

                if moderation_exhausted:
                    failed_image_generation_id = (
                        _positive_integer(
                            failed_image_generation_id,
                            field_name=(
                                "failed_image_generation_id"
                            ),
                        )
                    )

                    replacement_candidate = (
                        await _choose_workflow_replacement(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                            ranking_run_id=(
                                ranking_run_id
                            ),
                            active_selection=(
                                active_selection
                            ),
                            current_selection=(
                                selection
                            ),
                        )
                    )

                    if replacement_candidate is None:
                        raise DailyProductionWorkflowError(
                            "Image moderation fallback "
                            "исчерпан и replacement limit "
                            "достигнут либо подходящих "
                            "ranking combinations больше нет: "
                            f"combination_id="
                            f"{combination_id}, "
                            f"selection_attempt="
                            f"{active_selection.attempt_number}, "
                            f"max_replacements="
                            f"{MAX_TOP3_REPLACEMENTS}"
                        )

                    replacement_result = (
                        await
                        replace_daily_workflow_after_image_moderation(
                            pool,
                            daily_workflow_run_id=(
                                workflow_id
                            ),
                            current_selection_attempt_id=(
                                active_selection
                                .selection_attempt_id
                            ),
                            replacement_combination_id=(
                                replacement_candidate
                                .combination
                                .combination_id
                            ),
                            failed_image_generation_id=(
                                failed_image_generation_id
                            ),
                        )
                    )

                    _report_progress(
                        progress,
                        "[selection] moderation exhausted; "
                        "replace TOP-3 "
                        f"source_combination_id="
                        f"{combination_id}; "
                        f"replacement_combination_id="
                        f"{replacement_result.replacement_combination_id}; "
                        f"removed_news_ids="
                        f"{replacement_candidate.removed_news_ids!r}; "
                        f"added_news_ids="
                        f"{replacement_candidate.added_news_ids!r}",
                    )

                    # Atomic replacement уже:
                    # - superseded старые batch/post;
                    # - создал новую active selection;
                    # - очистил workflow batch/post/image;
                    # - вернул workflow в running/generation.
                    #
                    # Следующая итерация заново строит text request key
                    # и получает новый image budget для нового batch/post.
                    continue

                image_generation_id = (
                    _positive_integer(
                        image_generation_id,
                        field_name=(
                            "image_generation_id"
                        ),
                    )
                )

                _report_progress(
                    progress,
                    "[image] completed "
                    f"image_generation_id="
                    f"{image_generation_id}; "
                    f"combination_id="
                    f"{combination_id}",
                )

                generation_state = (
                    await load_generation_workflow_state(
                        pool,
                        batch_id=batch_id,
                    )
                )

                if (
                    generation_state
                    .image_state_inconsistent
                ):
                    raise DailyProductionWorkflowError(
                        "После image completion "
                        "обнаружено несогласованное "
                        "image state: "
                        f"batch_id={batch_id}"
                    )

                if (
                    not
                    generation_state
                    .ready_for_review_delivery
                ):
                    raise DailyProductionWorkflowError(
                        "Post не готов к Telegram "
                        "review delivery после image: "
                        f"batch_id={batch_id}, "
                        f"generated_post_id="
                        f"{generated_post_id}"
                    )

                # Успешная selection становится финальной
                # для Telegram review.
                break

            # ============================================================
            # Native Telegram review delivery
            # ============================================================

            _report_progress(
                progress,
                "[telegram] deliver native photo review",
            )

            await mark_daily_workflow_stage(
                pool,
                daily_workflow_run_id=(
                    workflow_id
                ),
                stage="review_delivery",
            )

            bot = Bot(
                token=(
                    settings
                    .telegram_bot_token
                    .get_secret_value()
                )
            )

            delivery_result = (
                await deliver_generated_post_to_reviewers(
                    pool,
                    bot=bot,
                    generated_post_id=(
                        generated_post_id
                    ),
                    reply_markup=(
                        build_review_keyboard(
                            generated_post_id
                        )
                    ),
                )
            )

            _validate_review_delivery(
                delivery_result
            )

            _report_progress(
                progress,
                "[telegram] delivery state verified",
            )

            workflow = (
                await complete_daily_workflow(
                    pool,
                    daily_workflow_run_id=(
                        workflow_id
                    ),
                )
            )

            if not workflow.awaiting_review:
                raise DailyProductionWorkflowError(
                    "Daily workflow не перешёл "
                    "в awaiting_review."
                )

            _report_progress(
                progress,
                "[daily] workflow awaiting_review",
            )

            return DailyProductionResult(
                daily_workflow_run_id=(
                    workflow_id
                ),
                publication_date=(
                    workflow.publication_date
                ),
                as_of=workflow.as_of,
                workflow_status=(
                    workflow.workflow_status
                ),
                ranking_run_id=(
                    ranking_run_id
                ),
                batch_id=batch_id,
                generated_post_id=(
                    generated_post_id
                ),
                image_generation_id=(
                    image_generation_id
                ),
                ranking_model_called=(
                    ranking_model_called
                ),
                generation_model_called=(
                    generation_model_called
                ),
                image_model_called=(
                    image_model_called
                ),
                reviewer_count=(
                    delivery_result
                    .reviewer_count
                ),
                review_sent_count=(
                    delivery_result.sent_count
                ),
                review_failed_count=(
                    delivery_result.failed_count
                ),
                review_unknown_count=(
                    delivery_result.unknown_count
                ),
                review_skipped_count=(
                    delivery_result.skipped_count
                ),
            )

        except Exception as error:
            await _fail_workflow_after_error(
                pool,
                daily_workflow_run_id=(
                    workflow_id
                ),
                error=error,
            )
            raise

        finally:
            await _close_bot(
                bot
            )

            if image_runtime is not None:
                await _close_sdk_client(
                    image_runtime.sdk_client
                )

            if generation_runtime is not None:
                await _close_sdk_client(
                    generation_runtime.sdk_client
                )

            if ranking_runtime is not None:
                await _close_sdk_client(
                    ranking_runtime.sdk_client
                )