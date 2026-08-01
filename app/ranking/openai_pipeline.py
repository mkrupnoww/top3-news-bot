from dataclasses import dataclass
from datetime import datetime

import asyncpg

from app.db.news_candidates import (
    CandidateSelectionResult,
    select_news_candidates,
)
from app.db.ranking_run_completion import (
    RankingRunCompletionResult,
    complete_reserved_ranking_run,
    fail_reserved_ranking_run,
)
from app.db.ranking_run_reservation import (
    RankingRunReservation,
    reserve_ranking_run,
)
from app.ranking.openai_evaluator import (
    OpenAIRankingEvaluationResult,
    OpenAIRankingEvaluator,
    RankingModelRequest,
)
from app.ranking.request_key import (
    RankingRequestKey,
    create_ranking_request_key,
)
from app.ranking.score_formula import (
    FORMULA_VERSION,
)


@dataclass(frozen=True, slots=True)
class ReservedOpenAIRankingResult:
    """Результат защищённого OpenAI-конвейера."""

    candidate_selection: (
        CandidateSelectionResult
    )

    model_request: RankingModelRequest

    request_key: RankingRequestKey

    reservation: RankingRunReservation

    model_called: bool

    evaluation: (
        OpenAIRankingEvaluationResult
        | None
    )

    completion: (
        RankingRunCompletionResult
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
    """Проверяет результат отбора кандидатов."""

    if not selection.candidates:
        raise ValueError(
            "В заданном временном окне "
            "нет кандидатов для ранжирования."
        )

    if (
        selection.window_end
        <= selection.window_start
    ):
        raise ValueError(
            "Некорректное временное окно "
            "отбора кандидатов."
        )

    if selection.window_hours <= 0:
        raise ValueError(
            "window_hours должен быть "
            "больше нуля."
        )


def _require_evaluation_telemetry(
    evaluation: (
        OpenAIRankingEvaluationResult
    ),
) -> None:
    """Проверяет наличие usage и стоимости."""

    if evaluation.model_response.usage is None:
        raise ValueError(
            "OpenAI-оценщик не вернул "
            "token usage."
        )

    if (
        evaluation
        .model_response
        .cost_estimate
        is None
    ):
        raise ValueError(
            "OpenAI-оценщик не вернул "
            "расчёт стоимости."
        )


async def _record_pipeline_failure(
    pool: asyncpg.Pool,
    *,
    reservation: RankingRunReservation,
    error: Exception,
) -> None:
    """
    Фиксирует ошибку в ranking_runs.

    Ошибка фиксации добавляется примечанием
    к исходному исключению, но не заменяет его.
    """

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


async def run_reserved_openai_ranking(
    pool: asyncpg.Pool,
    *,
    evaluator: OpenAIRankingEvaluator,
    as_of: datetime,
    window_hours: float = 24.0,
    limit: int = 500,
    source_codes: (
        tuple[str, ...]
        | None
    ) = None,
) -> ReservedOpenAIRankingResult:
    """
    Запускает защищённое ранжирование.

    Последовательность:

    1. Выбирает кандидатов из PostgreSQL.
    2. Формирует точный запрос модели.
    3. Вычисляет детерминированный request_key.
    4. Резервирует ranking_run.
    5. Блокирует повторный платный запрос.
    6. При новом запуске вызывает OpenAI.
    7. Сохраняет оценки, токены и стоимость.
    8. При ошибке переводит запуск в failed.

    Функция не управляет жизненным циклом
    пула PostgreSQL или OpenAI SDK-клиента.
    Их должен закрывать вызывающий код.
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

    candidates = (
        candidate_selection.candidates
    )

    model_request = evaluator.build_request(
        candidates
    )

    news_ids = tuple(
        candidate.news_id
        for candidate in candidates
    )

    request_key = (
        create_ranking_request_key(
            formula_version=FORMULA_VERSION,
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
        )
    )

    reservation = await reserve_ranking_run(
        pool,
        request_key=request_key,
        formula_version=FORMULA_VERSION,
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
        return ReservedOpenAIRankingResult(
            candidate_selection=(
                candidate_selection
            ),
            model_request=model_request,
            request_key=request_key,
            reservation=reservation,
            model_called=False,
            evaluation=None,
            completion=None,
        )

    try:
        evaluation = (
            await evaluator
            .evaluate_prepared_request(
                candidates,
                model_request,
            )
        )

        _require_evaluation_telemetry(
            evaluation
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
            await complete_reserved_ranking_run(
                pool,
                ranking_run_id=(
                    reservation.ranking_run_id
                ),
                request_key=(
                    reservation.request_key
                ),
                metadata=evaluator.metadata,
                assessments=(
                    evaluation.assessments
                ),
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

    return ReservedOpenAIRankingResult(
        candidate_selection=(
            candidate_selection
        ),
        model_request=model_request,
        request_key=request_key,
        reservation=reservation,
        model_called=True,
        evaluation=evaluation,
        completion=completion,
    )