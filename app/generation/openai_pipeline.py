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
    GenerationModelResponse,
    OpenAIPostGenerationResult,
    OpenAITelegramPostGenerator,
)
from app.generation.request_key import (
    GenerationRequestKey,
    create_generation_request_key,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
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


def _combine_generation_results(
    *,
    primary: OpenAIPostGenerationResult,
    self_review: OpenAIPostGenerationResult,
) -> OpenAIPostGenerationResult:
    """
    Объединяет два прохода в один итоговый результат.

    Финальный payload берётся из self-review.

    Usage и стоимость модели суммируются по двум
    Responses API вызовам. Стоимость самого
    web_search здесь отдельно не включается:
    текущий OpenAICostEstimate описывает только
    стоимость токенов модели.

    Телеметрия web_search сохраняется в итоговом
    GenerationModelResponse.
    """

    _require_generation_telemetry(primary)
    _require_generation_telemetry(self_review)

    primary_usage = (
        primary.model_response.usage
    )

    self_review_usage = (
        self_review.model_response.usage
    )

    primary_cost = (
        primary.model_response.cost_estimate
    )

    self_review_cost = (
        self_review.model_response.cost_estimate
    )

    if primary_usage is None:
        raise ValueError(
            "Primary generation usage отсутствует."
        )

    if self_review_usage is None:
        raise ValueError(
            "Self-review usage отсутствует."
        )

    if primary_cost is None:
        raise ValueError(
            "Primary generation cost отсутствует."
        )

    if self_review_cost is None:
        raise ValueError(
            "Self-review cost отсутствует."
        )

    if (
        primary_cost.model_name
        != self_review_cost.model_name
    ):
        raise ValueError(
            "Модель primary generation и "
            "self-review не совпадает."
        )

    if (
        primary_cost.pricing_version
        != self_review_cost.pricing_version
    ):
        raise ValueError(
            "Версия тарифа primary generation "
            "и self-review не совпадает."
        )

    combined_usage = OpenAITokenUsage(
        input_tokens=(
            primary_usage.input_tokens
            + self_review_usage.input_tokens
        ),
        cached_input_tokens=(
            primary_usage.cached_input_tokens
            + self_review_usage.cached_input_tokens
        ),
        cache_write_tokens=(
            primary_usage.cache_write_tokens
            + self_review_usage.cache_write_tokens
        ),
        output_tokens=(
            primary_usage.output_tokens
            + self_review_usage.output_tokens
        ),
        reasoning_tokens=(
            primary_usage.reasoning_tokens
            + self_review_usage.reasoning_tokens
        ),
        total_tokens=(
            primary_usage.total_tokens
            + self_review_usage.total_tokens
        ),
    )

    combined_cost = OpenAICostEstimate(
        model_name=primary_cost.model_name,
        pricing_version=(
            primary_cost.pricing_version
        ),
        regular_input_cost_usd=(
            primary_cost.regular_input_cost_usd
            + self_review_cost.regular_input_cost_usd
        ),
        cached_input_cost_usd=(
            primary_cost.cached_input_cost_usd
            + self_review_cost.cached_input_cost_usd
        ),
        cache_write_cost_usd=(
            primary_cost.cache_write_cost_usd
            + self_review_cost.cache_write_cost_usd
        ),
        output_cost_usd=(
            primary_cost.output_cost_usd
            + self_review_cost.output_cost_usd
        ),
        total_cost_usd=(
            primary_cost.total_cost_usd
            + self_review_cost.total_cost_usd
        ),
    )

    web_source_urls = tuple(
        dict.fromkeys(
            (
                *primary.model_response.web_source_urls,
                *self_review.model_response.web_source_urls,
            )
        )
    )

    combined_model_response = (
        GenerationModelResponse(
            output_text=(
                self_review
                .model_response
                .output_text
            ),
            usage=combined_usage,
            cost_estimate=combined_cost,
            web_search_used=(
                primary
                .model_response
                .web_search_used
                or self_review
                .model_response
                .web_search_used
            ),
            web_search_call_count=(
                primary
                .model_response
                .web_search_call_count
                + self_review
                .model_response
                .web_search_call_count
            ),
            web_source_urls=web_source_urls,
        )
    )

    return OpenAIPostGenerationResult(
        payload=self_review.payload,
        model_response=combined_model_response,
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
    2. Формирует точный запрос первичной модели.
    3. Вычисляет детерминированный request_key.
    4. Резервирует publication_batch.
    5. Блокирует повторный платный запуск.
    6. При новом выпуске выполняет первичную
       генерацию Telegram-поста.
    7. Выполняет автоматический self-review
       того же поста. Self-review сам решает,
       нужен ли web_search.
    8. Финальный текст берёт из self-review,
       а usage и стоимость модели суммирует
       по обоим Responses API вызовам.
    9. Сохраняет только финальный generated_post.
    10. Переводит выпуск в awaiting_review.
    11. При ошибке переводит выпуск в failed.

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
        primary_generation = (
            await generator
            .generate_prepared_request(
                selection.items,
                model_request,
            )
        )

        _require_generation_telemetry(
            primary_generation
        )

        self_review_generation = (
            await generator
            .generate_self_review_detailed(
                selection.items,
                source_post_text=(
                    primary_generation
                    .payload
                    .post_text
                ),
            )
        )

        _require_generation_telemetry(
            self_review_generation
        )

        generation = (
            _combine_generation_results(
                primary=primary_generation,
                self_review=(
                    self_review_generation
                ),
            )
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