from dataclasses import dataclass
from datetime import date

import asyncpg

from app.db.generation_completion import (
    GenerationCompletionResult,
    complete_reserved_generation,
    fail_reserved_generation,
)
from app.db.generation_reservation import (
    GenerationReservation,
    reserve_generation,
)
from app.db.generation_selection import (
    GenerationTop3Selection,
    load_generation_top3,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    OpenAIPostGenerationResult,
    OpenAITelegramPostGenerator,
)
from app.generation.request_key import (
    GenerationRequestKey,
    create_generation_request_key,
)


@dataclass(frozen=True, slots=True)
class ReservedOpenAIGenerationResult:
    """Результат защищённого конвейера генерации."""

    selection: GenerationTop3Selection

    model_request: GenerationModelRequest

    request_key: GenerationRequestKey

    reservation: GenerationReservation

    model_called: bool

    generation: (
        OpenAIPostGenerationResult
        | None
    )

    completion: (
        GenerationCompletionResult
        | None
    )

    @property
    def batch_id(self) -> int:
        """Возвращает идентификатор выпуска."""

        return self.reservation.batch_id

    @property
    def batch_status(self) -> str:
        """Возвращает итоговый известный статус."""

        if self.completion is not None:
            return self.completion.batch_status

        return self.reservation.batch_status

    @property
    def generated_post_id(
        self,
    ) -> int | None:
        """Возвращает ID сохранённого поста."""

        if self.completion is None:
            return None

        return (
            self.completion.generated_post_id
        )

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
            and self.completion.batch_status
            == "awaiting_review"
            and self.completion.post_status
            == "awaiting_review"
        )


def _require_generation_telemetry(
    generation: OpenAIPostGenerationResult,
) -> None:
    """Проверяет наличие usage и стоимости."""

    if generation.model_response.usage is None:
        raise ValueError(
            "OpenAI-генератор не вернул "
            "token usage."
        )

    if (
        generation
        .model_response
        .cost_estimate
        is None
    ):
        raise ValueError(
            "OpenAI-генератор не вернул "
            "расчёт стоимости."
        )


async def _record_pipeline_failure(
    pool: asyncpg.Pool,
    *,
    reservation: GenerationReservation,
    error: Exception,
) -> None:
    """
    Фиксирует ошибку в publication_batches.

    Ошибка фиксации добавляется примечанием
    к исходному исключению, но не заменяет его.
    """

    error_message = str(error).strip()

    if not error_message:
        error_message = repr(error)

    try:
        await fail_reserved_generation(
            pool,
            batch_id=reservation.batch_id,
            request_key=(
                reservation.request_key
            ),
            error_message=error_message,
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


async def run_reserved_openai_generation(
    pool: asyncpg.Pool,
    *,
    generator: OpenAITelegramPostGenerator,
    ranking_run_id: int,
    publication_date: date,
    telegram_chat_id: int,
) -> ReservedOpenAIGenerationResult:
    """
    Запускает защищённую генерацию поста.

    Последовательность:

    1. Читает сохранённый TOP-3 из PostgreSQL.
    2. Формирует точный запрос модели.
    3. Вычисляет детерминированный request_key.
    4. Резервирует publication_batch.
    5. Блокирует повторный платный запрос.
    6. При новом выпуске вызывает OpenAI.
    7. Сохраняет generated_post, usage
       и оценку стоимости.
    8. Переводит выпуск в awaiting_review.
    9. При ошибке переводит выпуск в failed.

    Функция не управляет жизненным циклом
    пула PostgreSQL или OpenAI SDK-клиента.
    Их должен закрывать вызывающий код.

    Telegram не вызывается.
    """

    selection = await load_generation_top3(
        pool,
        ranking_run_id=ranking_run_id,
    )

    model_request = generator.build_request(
        selection.items
    )

    request_key = (
        create_generation_request_key(
            ranking_run_id=(
                selection.ranking_run_id
            ),
            publication_date=(
                publication_date
            ),
            telegram_chat_id=(
                telegram_chat_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=selection.items,
        )
    )

    reservation = await reserve_generation(
        pool,
        request_key=request_key,
        selection=selection,
        publication_date=publication_date,
        telegram_chat_id=telegram_chat_id,
        metadata=generator.metadata,
        model_request=model_request,
    )

    if not reservation.should_call_model:
        return ReservedOpenAIGenerationResult(
            selection=selection,
            model_request=model_request,
            request_key=request_key,
            reservation=reservation,
            model_called=False,
            generation=None,
            completion=None,
        )

    try:
        generation = (
            await generator
            .generate_prepared_request(
                selection.items,
                model_request,
            )
        )

        _require_generation_telemetry(
            generation
        )

        completion = (
            await complete_reserved_generation(
                pool,
                batch_id=reservation.batch_id,
                request_key=(
                    reservation.request_key
                ),
                metadata=generator.metadata,
                result=generation,
            )
        )

    except Exception as error:
        await _record_pipeline_failure(
            pool,
            reservation=reservation,
            error=error,
        )

        raise

    return ReservedOpenAIGenerationResult(
        selection=selection,
        model_request=model_request,
        request_key=request_key,
        reservation=reservation,
        model_called=True,
        generation=generation,
        completion=completion,
    )