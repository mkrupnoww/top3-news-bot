from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import asyncpg

from app.db.event_ranking_run_completion import (
    EventRankingRunCompletionResult,
    complete_reserved_event_ranking_run,
)
from app.db.news_candidates import (
    CandidateSelectionResult,
    select_news_candidates,
)
from app.db.ranking_run_completion import (
    fail_reserved_ranking_run,
)
from app.db.ranking_run_reservation import (
    RankingRunReservation,
    reserve_ranking_run,
)
from app.ranking.event_evaluator import (
    EventRankingEvaluationResult,
    EventRankingModelRequest,
)
from app.ranking.event_formula_pipeline import (
    EventAudienceMetrics,
    EventFormulaCalculationResult,
    calculate_event_formula,
)
from app.ranking.event_request_key import (
    create_event_ranking_request_key,
)
from app.ranking.full_formula import (
    FULL_FORMULA_VERSION,
)
from app.ranking.openai_event_evaluator import (
    OpenAIEventRankingEvaluator,
)
from app.ranking.request_key import (
    RankingRequestKey,
)


REQUIRED_WINDOW_HOURS = Decimal("24")


@dataclass(frozen=True, slots=True)
class ReservedOpenAIEventRankingResult:
    """Результат защищённого event-level конвейера."""

    candidate_selection: (
        CandidateSelectionResult
    )

    model_request: EventRankingModelRequest

    request_key: RankingRequestKey

    reservation: RankingRunReservation

    model_called: bool

    evaluation: (
        EventRankingEvaluationResult
        | None
    )

    calculation: (
        EventFormulaCalculationResult
        | None
    )

    completion: (
        EventRankingRunCompletionResult
        | None
    )

    @property
    def ranking_run_id(self) -> int:
        """Возвращает идентификатор запуска."""

        return self.reservation.ranking_run_id

    @property
    def run_status(self) -> str:
        """Возвращает итоговый известный статус."""

        if self.completion is not None:
            return self.completion.run_status

        return self.reservation.run_status

    @property
    def duplicate_request_blocked(
        self,
    ) -> bool:
        """Показывает блокировку повтора."""

        return not self.reservation.created_new

    @property
    def completed(self) -> bool:
        """Показывает успешное завершение."""

        return (
            self.completion is not None
            and self.completion.run_status
            == "completed"
        )


def _validate_candidate_selection(
    selection: CandidateSelectionResult,
) -> None:
    """Проверяет выборку до платного запроса."""

    if not selection.candidates:
        raise ValueError(
            "В заданном временном окне "
            "нет кандидатов для ранжирования."
        )

    if len(selection.candidates) < 3:
        raise ValueError(
            "Для TOP-3 требуется минимум "
            "три публикации-кандидата."
        )

    if (
        selection.window_end
        <= selection.window_start
    ):
        raise ValueError(
            "Некорректное временное окно "
            "отбора кандидатов."
        )

    normalized_window_hours = Decimal(
        str(selection.window_hours)
    )

    if normalized_window_hours != (
        REQUIRED_WINDOW_HOURS
    ):
        raise ValueError(
            "Полная формула v2 требует "
            "строгое окно 24 часа: "
            f"window_hours={selection.window_hours}"
        )


def _require_evaluation_telemetry(
    evaluation: EventRankingEvaluationResult,
) -> None:
    """Проверяет наличие usage и стоимости."""

    if evaluation.model_response.usage is None:
        raise ValueError(
            "OpenAI event-оценщик не вернул "
            "token usage."
        )

    if (
        evaluation
        .model_response
        .cost_estimate
        is None
    ):
        raise ValueError(
            "OpenAI event-оценщик не вернул "
            "расчёт стоимости."
        )


async def _record_pipeline_failure(
    pool: asyncpg.Pool,
    *,
    reservation: RankingRunReservation,
    error: Exception,
) -> None:
    """Фиксирует ошибку event-конвейера."""

    try:
        await fail_reserved_ranking_run(
            pool,
            ranking_run_id=(
                reservation.ranking_run_id
            ),
            request_key=(
                reservation.request_key
            ),
            error_message=str(error),
            error_type=(
                type(error).__name__
            ),
        )
    except Exception as failure_error:
        error.add_note(
            "Дополнительно не удалось "
            "зафиксировать статус failed: "
            f"{type(failure_error).__name__}: "
            f"{failure_error}"
        )


async def run_reserved_openai_event_ranking(
    pool: asyncpg.Pool,
    *,
    evaluator: OpenAIEventRankingEvaluator,
    as_of: datetime,
    window_hours: float = 24.0,
    limit: int = 500,
    source_codes: (
        tuple[str, ...]
        | None
    ) = None,
    audience_metrics: tuple[
        EventAudienceMetrics,
        ...,
    ] = (),
) -> ReservedOpenAIEventRankingResult:
    """
    Выполняет защищённое полное ранжирование v2.

    Последовательность:

    1. Выбирает публикации за строгие 24 часа.
    2. Формирует event-level запрос модели.
    3. Создаёт request_key с audience-снимком.
    4. Резервирует ranking_run.
    5. Блокирует повторный платный запрос.
    6. Получает от OpenAI инфоповоды и оценки.
    7. Рассчитывает F/U/M/R/H/B и TOP(S).
    8. Атомарно сохраняет полный результат.
    9. При ошибке переводит запуск в failed.

    Жизненным циклом PostgreSQL pool и OpenAI
    SDK-клиента управляет вызывающий код.
    """

    candidate_selection = (
        await select_news_candidates(
            pool,
            as_of=as_of,
            window_hours=window_hours,
            limit=limit,
            source_codes=source_codes,
        )
    )

    _validate_candidate_selection(
        candidate_selection
    )

    model_request = evaluator.build_request(
        candidate_selection
    )

    news_ids = tuple(
        candidate.news_id
        for candidate
        in candidate_selection.candidates
    )

    request_key = (
        create_event_ranking_request_key(
            formula_version=(
                FULL_FORMULA_VERSION
            ),
            metadata=evaluator.metadata,
            model_request=model_request,
            window_started_at=(
                candidate_selection
                .window_start
            ),
            window_finished_at=(
                candidate_selection
                .window_end
            ),
            news_ids=news_ids,
            audience_metrics=audience_metrics,
        )
    )

    reservation = await reserve_ranking_run(
        pool,
        request_key=request_key,
        formula_version=(
            FULL_FORMULA_VERSION
        ),
        metadata=evaluator.metadata,
        window_started_at=(
            candidate_selection.window_start
        ),
        window_finished_at=(
            candidate_selection.window_end
        ),
        news_ids=news_ids,
    )

    if not reservation.should_call_model:
        return ReservedOpenAIEventRankingResult(
            candidate_selection=(
                candidate_selection
            ),
            model_request=model_request,
            request_key=request_key,
            reservation=reservation,
            model_called=False,
            evaluation=None,
            calculation=None,
            completion=None,
        )

    try:
        evaluation = (
            await evaluator
            .evaluate_prepared_request(
                candidate_selection,
                model_request,
            )
        )

        _require_evaluation_telemetry(
            evaluation
        )

        calculation = calculate_event_formula(
            selection=candidate_selection,
            events=evaluation.events,
            audience_metrics=audience_metrics,
        )

        usage = (
            evaluation.model_response.usage
        )

        cost_estimate = (
            evaluation
            .model_response
            .cost_estimate
        )

        if usage is None:
            raise RuntimeError(
                "OpenAI usage неожиданно "
                "отсутствует после проверки."
            )

        if cost_estimate is None:
            raise RuntimeError(
                "Расчёт стоимости неожиданно "
                "отсутствует после проверки."
            )

        completion = (
            await complete_reserved_event_ranking_run(
                pool,
                ranking_run_id=(
                    reservation.ranking_run_id
                ),
                request_key=(
                    reservation.request_key
                ),
                metadata=evaluator.metadata,
                candidate_news_ids=news_ids,
                calculation=calculation,
                usage=usage,
                cost_estimate=cost_estimate,
            )
        )

    except Exception as error:
        await _record_pipeline_failure(
            pool,
            reservation=reservation,
            error=error,
        )

        raise

    return ReservedOpenAIEventRankingResult(
        candidate_selection=(
            candidate_selection
        ),
        model_request=model_request,
        request_key=request_key,
        reservation=reservation,
        model_called=True,
        evaluation=evaluation,
        calculation=calculation,
        completion=completion,
    )