from dataclasses import dataclass

import asyncpg

from app.db.generation_revision_completion import (
    GenerationRevisionCompletionResult,
    complete_reserved_generation_revision,
    fail_reserved_generation_revision,
)
from app.db.generation_revision_reservation import (
    GenerationRevisionReservation,
    reserve_generation_revision,
)
from app.db.generation_revision_selection import (
    GenerationRevisionSelection,
    load_generation_revision_selection,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    OpenAIPostGenerationResult,
    OpenAITelegramPostGenerator,
)
from app.generation.revision_request_key import (
    GenerationRevisionRequestKey,
    create_generation_revision_request_key,
)


@dataclass(frozen=True, slots=True)
class ReservedOpenAIGenerationRevisionResult:
    """Результат защищённого конвейера ревизии."""

    revision_selection: GenerationRevisionSelection

    model_request: GenerationModelRequest

    request_key: GenerationRevisionRequestKey

    reservation: GenerationRevisionReservation

    model_called: bool

    generation: (
        OpenAIPostGenerationResult
        | None
    )

    completion: (
        GenerationRevisionCompletionResult
        | None
    )

    @property
    def generation_revision_id(self) -> int:
        """Возвращает ID revision reservation."""

        return (
            self.reservation
            .generation_revision_id
        )

    @property
    def batch_id(self) -> int:
        """Возвращает идентификатор выпуска."""

        return self.reservation.batch_id

    @property
    def source_generated_post_id(
        self,
    ) -> int:
        """Возвращает ID исходной версии."""

        return (
            self.reservation
            .source_generated_post_id
        )

    @property
    def review_action_id(self) -> int:
        """Возвращает ID редакционного решения."""

        return (
            self.reservation
            .review_action_id
        )

    @property
    def target_version_number(self) -> int:
        """Возвращает номер целевой версии."""

        return (
            self.reservation
            .target_version_number
        )

    @property
    def revision_status(self) -> str:
        """Возвращает итоговый известный статус."""

        if self.completion is not None:
            return self.completion.revision_status

        return self.reservation.revision_status

    @property
    def generated_post_id(
        self,
    ) -> int | None:
        """Возвращает ID созданной версии."""

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

        if self.completion is not None:
            return (
                self.completion.revision_status
                == "completed"
                and self.completion.source_post_status
                == "superseded"
                and self.completion.post_status
                == "awaiting_review"
            )

        return (
            self.reservation.revision_status
            == "completed"
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


async def _record_revision_pipeline_failure(
    pool: asyncpg.Pool,
    *,
    reservation: GenerationRevisionReservation,
    error: Exception,
) -> None:
    """
    Фиксирует ошибку конкретной revision reservation.

    Ошибка фиксации добавляется примечанием
    к исходному исключению, но не заменяет его.
    """

    error_message = str(error).strip()

    if not error_message:
        error_message = repr(error)

    try:
        await fail_reserved_generation_revision(
            pool,
            generation_revision_id=(
                reservation
                .generation_revision_id
            ),
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
            "зафиксировать revision_status "
            "failed: "
            f"{type(failure_error).__name__}: "
            f"{failure_error}"
        )


async def run_reserved_openai_generation_revision(
    pool: asyncpg.Pool,
    *,
    generator: OpenAITelegramPostGenerator,
    review_action_id: int,
) -> ReservedOpenAIGenerationRevisionResult:
    """
    Запускает защищённую редакционную ревизию.

    Последовательность:

    1. Загружает review_action, исходный пост
       и сохранённый TOP-3 из PostgreSQL.
    2. Формирует точный revision-запрос модели.
    3. Вычисляет детерминированный request_key.
    4. Резервирует generation revision.
    5. Блокирует повторный платный запрос.
    6. При новой reservation вызывает OpenAI.
    7. Сохраняет новую generated_posts version N+1,
       usage и оценку стоимости.
    8. Переводит исходную версию в superseded,
       новую оставляет awaiting_review.
    9. publication_batch оставляет awaiting_review.
    10. При ошибке переводит только revision request
        в failed; batch и исходный пост не меняет.

    Функция не управляет жизненным циклом
    пула PostgreSQL или OpenAI SDK-клиента.
    Их должен закрывать вызывающий код.

    Telegram не вызывается.
    """

    revision_selection = (
        await load_generation_revision_selection(
            pool,
            review_action_id=review_action_id,
        )
    )

    model_request = (
        generator.build_revision_request(
            revision_selection.items,
            source_post_text=(
                revision_selection
                .source_post_text
            ),
            editorial_comment=(
                revision_selection
                .editorial_comment
            ),
            issues=(
                revision_selection.issues
            ),
        )
    )

    request_key = (
        create_generation_revision_request_key(
            batch_id=(
                revision_selection.batch_id
            ),
            source_generated_post_id=(
                revision_selection
                .source_generated_post_id
            ),
            review_action_id=(
                revision_selection
                .review_action_id
            ),
            target_version_number=(
                revision_selection
                .target_version_number
            ),
            source_post_text=(
                revision_selection
                .source_post_text
            ),
            editorial_comment=(
                revision_selection
                .editorial_comment
            ),
            issues=(
                revision_selection.issues
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=revision_selection.items,
        )
    )

    reservation = (
        await reserve_generation_revision(
            pool,
            request_key=request_key,
            batch_id=(
                revision_selection.batch_id
            ),
            source_generated_post_id=(
                revision_selection
                .source_generated_post_id
            ),
            review_action_id=(
                revision_selection
                .review_action_id
            ),
            target_version_number=(
                revision_selection
                .target_version_number
            ),
            source_post_text=(
                revision_selection
                .source_post_text
            ),
            editorial_comment=(
                revision_selection
                .editorial_comment
            ),
            issues=(
                revision_selection.issues
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=revision_selection.items,
        )
    )

    if not reservation.should_call_model:
        return (
            ReservedOpenAIGenerationRevisionResult(
                revision_selection=(
                    revision_selection
                ),
                model_request=model_request,
                request_key=request_key,
                reservation=reservation,
                model_called=False,
                generation=None,
                completion=None,
            )
        )

    try:
        generation = (
            await generator
            .generate_prepared_revision_request(
                revision_selection.items,
                model_request,
                source_post_text=(
                    revision_selection
                    .source_post_text
                ),
                editorial_comment=(
                    revision_selection
                    .editorial_comment
                ),
                issues=(
                    revision_selection.issues
                ),
            )
        )

        _require_generation_telemetry(
            generation
        )

        completion = (
            await complete_reserved_generation_revision(
                pool,
                generation_revision_id=(
                    reservation
                    .generation_revision_id
                ),
                request_key=(
                    reservation.request_key
                ),
                metadata=generator.metadata,
                result=generation,
            )
        )

    except Exception as error:
        await _record_revision_pipeline_failure(
            pool,
            reservation=reservation,
            error=error,
        )

        raise

    return ReservedOpenAIGenerationRevisionResult(
        revision_selection=(
            revision_selection
        ),
        model_request=model_request,
        request_key=request_key,
        reservation=reservation,
        model_called=True,
        generation=generation,
        completion=completion,
    )