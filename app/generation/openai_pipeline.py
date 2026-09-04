from __future__ import annotations

from collections.abc import (
    Awaitable,
    Callable,
    Sequence,
)
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
import logging

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
    load_generation_combination,
    load_generation_top3,
)
from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    GenerationNewsItem,
    OpenAIGeneratedPostPayload,
    OpenAIPostGenerationResult,
    OpenAITelegramPostGenerator,
)
from app.generation.official_trailer_enrichment import (
    OfficialTrailerEnrichmentResult,
    enrich_official_trailer,
)
from app.generation.post_integrity import (
    build_deterministic_integrity_fallback,
    build_post_integrity_editorial_comment,
    validate_generated_post_integrity,
)
from app.generation.request_key import (
    GenerationRequestKey,
    create_generation_request_key,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


MAX_GENERATION_INTEGRITY_REVISIONS = 2

logger = logging.getLogger(__name__)


OfficialTrailerEnricher = Callable[
    ...,
    Awaitable[OfficialTrailerEnrichmentResult],
]

GenerationReservationObserver = Callable[
    [GenerationReservation],
    Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class IntegrityRepairOutcome:
    """Итог bounded text-integrity recovery."""

    model_generations: tuple[
        OpenAIPostGenerationResult,
        ...,
    ]

    final_payload: OpenAIGeneratedPostPayload

    used_deterministic_fallback: bool


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
    results: Sequence[
        OpenAIPostGenerationResult
    ],
    *,
    final_payload: (
        OpenAIGeneratedPostPayload
        | None
    ) = None,
) -> OpenAIPostGenerationResult:
    """
    Объединяет несколько проходов генерации
    в один итоговый результат.

    Финальный payload берётся из последнего
    прохода.

    Usage и стоимость модели суммируются
    по всем Responses API вызовам.

    Телеметрия web_search агрегируется
    по всем проходам.
    """

    normalized_results = tuple(results)

    if not normalized_results:
        raise ValueError(
            "Для объединения должен быть "
            "передан хотя бы один результат."
        )

    for result in normalized_results:
        _require_generation_telemetry(
            result
        )

    first_response = (
        normalized_results[0]
        .model_response
    )

    first_usage = first_response.usage
    first_cost = (
        first_response.cost_estimate
    )

    if first_usage is None:
        raise ValueError(
            "Первый результат не содержит "
            "usage."
        )

    if first_cost is None:
        raise ValueError(
            "Первый результат не содержит "
            "cost_estimate."
        )

    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_cache_write_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    total_tokens = 0

    total_regular_input_cost = Decimal("0")
    total_cached_input_cost = Decimal("0")
    total_cache_write_cost = Decimal("0")
    total_output_cost = Decimal("0")
    total_cost = Decimal("0")

    web_search_used = False
    web_search_call_count = 0
    web_source_urls: list[str] = []

    for result in normalized_results:
        response = result.model_response
        usage = response.usage
        cost = response.cost_estimate

        if usage is None:
            raise ValueError(
                "Один из результатов не "
                "содержит usage."
            )

        if cost is None:
            raise ValueError(
                "Один из результатов не "
                "содержит cost_estimate."
            )

        if (
            cost.model_name
            != first_cost.model_name
        ):
            raise ValueError(
                "Модель в разных проходах "
                "генерации не совпадает."
            )

        if (
            cost.pricing_version
            != first_cost.pricing_version
        ):
            raise ValueError(
                "Версия тарифа в разных "
                "проходах генерации "
                "не совпадает."
            )

        total_input_tokens += (
            usage.input_tokens
        )
        total_cached_input_tokens += (
            usage.cached_input_tokens
        )
        total_cache_write_tokens += (
            usage.cache_write_tokens
        )
        total_output_tokens += (
            usage.output_tokens
        )
        total_reasoning_tokens += (
            usage.reasoning_tokens
        )
        total_tokens += usage.total_tokens

        total_regular_input_cost += (
            cost.regular_input_cost_usd
        )
        total_cached_input_cost += (
            cost.cached_input_cost_usd
        )
        total_cache_write_cost += (
            cost.cache_write_cost_usd
        )
        total_output_cost += (
            cost.output_cost_usd
        )
        total_cost += (
            cost.total_cost_usd
        )

        web_search_used = (
            web_search_used
            or response.web_search_used
        )

        web_search_call_count += (
            response.web_search_call_count
        )

        for url in response.web_source_urls:
            if url not in web_source_urls:
                web_source_urls.append(url)

    combined_usage = OpenAITokenUsage(
        input_tokens=total_input_tokens,
        cached_input_tokens=(
            total_cached_input_tokens
        ),
        cache_write_tokens=(
            total_cache_write_tokens
        ),
        output_tokens=total_output_tokens,
        reasoning_tokens=(
            total_reasoning_tokens
        ),
        total_tokens=total_tokens,
    )

    combined_cost = OpenAICostEstimate(
        model_name=first_cost.model_name,
        pricing_version=(
            first_cost.pricing_version
        ),
        regular_input_cost_usd=(
            total_regular_input_cost
        ),
        cached_input_cost_usd=(
            total_cached_input_cost
        ),
        cache_write_cost_usd=(
            total_cache_write_cost
        ),
        output_cost_usd=(
            total_output_cost
        ),
        total_cost_usd=total_cost,
    )

    final_result = normalized_results[-1]

    combined_model_response = (
        GenerationModelResponse(
            output_text=(
                final_result
                .model_response
                .output_text
            ),
            usage=combined_usage,
            cost_estimate=combined_cost,
            web_search_used=(
                web_search_used
            ),
            web_search_call_count=(
                web_search_call_count
            ),
            web_source_urls=tuple(
                web_source_urls
            ),
        )
    )

    return OpenAIPostGenerationResult(
        payload=(
            final_result.payload
            if final_payload is None
            else final_payload
        ),
        model_response=(
            combined_model_response
        ),
    )


async def _run_integrity_repairs_if_needed(
    generator: OpenAITelegramPostGenerator,
    *,
    items: tuple[
        GenerationNewsItem,
        GenerationNewsItem,
        GenerationNewsItem,
    ],
    initial_generation: (
        OpenAIPostGenerationResult
    ),
    primary_generation: (
        OpenAIPostGenerationResult
        | None
    ) = None,
    max_revision_attempts: int = (
        MAX_GENERATION_INTEGRITY_REVISIONS
    ),
) -> IntegrityRepairOutcome:
    """
    Выполняет bounded text-integrity recovery.

    Сначала проверяет self-review. Если обнаружен
    вероятно обрезанный длинный body, выполняет
    до max_revision_attempts модельных revision.

    Если модельные revisions не устранили только
    эвристическую проблему завершения текста,
    применяется локальный deterministic fail-safe.
    Сама эта эвристика больше не должна делать
    ежедневный workflow terminal failed.
    """

    if max_revision_attempts < 0:
        raise ValueError(
            "max_revision_attempts не может "
            "быть отрицательным."
        )

    _require_generation_telemetry(
        initial_generation
    )

    if primary_generation is not None:
        _require_generation_telemetry(
            primary_generation
        )

    generations: list[
        OpenAIPostGenerationResult
    ] = [initial_generation]

    current_generation = (
        initial_generation
    )

    issues = (
        validate_generated_post_integrity(
            current_generation.payload
        )
    )

    revision_attempts_used = 0

    while (
        issues
        and revision_attempts_used
        < max_revision_attempts
    ):
        editorial_comment = (
            build_post_integrity_editorial_comment(
                issues
            )
        )

        revision_request = (
            generator.build_revision_request(
                items,
                source_post_text=(
                    current_generation
                    .payload
                    .post_text
                ),
                editorial_comment=(
                    editorial_comment
                ),
                issues=issues,
            )
        )

        revised_generation = (
            await generator
            .generate_prepared_revision_request(
                items,
                revision_request,
                source_post_text=(
                    current_generation
                    .payload
                    .post_text
                ),
                editorial_comment=(
                    editorial_comment
                ),
                issues=issues,
            )
        )

        _require_generation_telemetry(
            revised_generation
        )

        generations.append(
            revised_generation
        )

        current_generation = (
            revised_generation
        )

        issues = (
            validate_generated_post_integrity(
                current_generation.payload
            )
        )

        revision_attempts_used += 1

    if not issues:
        return IntegrityRepairOutcome(
            model_generations=tuple(
                generations
            ),
            final_payload=(
                current_generation.payload
            ),
            used_deterministic_fallback=False,
        )

    deterministic_payload = (
        build_deterministic_integrity_fallback(
            current_generation.payload,
            fallback_payload=(
                primary_generation.payload
                if primary_generation is not None
                else None
            ),
        )
    )

    logger.warning(
        "Text integrity model repairs exhausted; "
        "deterministic local fail-safe applied. "
        "remaining_issues=%s",
        "; ".join(issues),
    )

    return IntegrityRepairOutcome(
        model_generations=tuple(
            generations
        ),
        final_payload=deterministic_payload,
        used_deterministic_fallback=True,
    )


async def _enrich_generation_items(
    items: tuple[
        GenerationNewsItem,
        GenerationNewsItem,
        GenerationNewsItem,
    ],
    *,
    trailer_enricher: OfficialTrailerEnricher,
) -> tuple[
    GenerationNewsItem,
    GenerationNewsItem,
    GenerationNewsItem,
]:
    """
    Добавляет verified official trailer URL
    только во временные items OpenAI-запроса.

    Исходный сохранённый TOP-3 не изменяется.
    Если официальный трейлер не подтверждён,
    соответствующий GenerationNewsItem остаётся
    без изменений.
    """

    enriched_items: list[
        GenerationNewsItem
    ] = []

    for item in items:
        enrichment = await trailer_enricher(
            source_url=item.source_url,
            source_title=item.title,
            source_summary=item.summary,
        )

        if (
            enrichment.verified
            and enrichment.official_trailer_url
            is not None
        ):
            enriched_items.append(
                replace(
                    item,
                    official_trailer_url=(
                        enrichment.official_trailer_url
                    ),
                    official_trailer_channel_name=(
                        enrichment.official_trailer_channel_name
                    ),
                )
            )
        else:
            enriched_items.append(item)

    return (
        enriched_items[0],
        enriched_items[1],
        enriched_items[2],
    )


def _validate_prepared_generation_items(
    selection: GenerationTop3Selection,
    prepared_items: tuple[
        GenerationNewsItem,
        GenerationNewsItem,
        GenerationNewsItem,
    ],
) -> tuple[
    GenerationNewsItem,
    GenerationNewsItem,
    GenerationNewsItem,
]:
    """Разрешает preflight менять только verified trailer metadata."""

    if len(prepared_items) != 3:
        raise ValueError(
            "prepared_items должен содержать ровно три новости."
        )

    for source_item, prepared_item in zip(
        selection.items,
        prepared_items,
        strict=True,
    ):
        expected = replace(
            source_item,
            official_trailer_url=prepared_item.official_trailer_url,
            official_trailer_channel_name=(
                prepared_item.official_trailer_channel_name
            ),
        )

        if prepared_item != expected:
            raise ValueError(
                "prepared_items могут отличаться от сохранённого TOP-3 "
                "только official trailer metadata: "
                f"news_id={source_item.news_id}"
            )

    return prepared_items


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
    combination_id: int | None = None,
    publication_date: date,
    telegram_chat_id: int,
    trailer_enricher: OfficialTrailerEnricher = (
        enrich_official_trailer
    ),
    prepared_items: tuple[
        GenerationNewsItem,
        GenerationNewsItem,
        GenerationNewsItem,
    ] | None = None,
    reservation_observer: (
        GenerationReservationObserver
        | None
    ) = None,
) -> ReservedOpenAIGenerationResult:
    """
    Запускает защищённую генерацию поста.

    Последовательность:

    1. Читает winner TOP-3 либо явно указанную
       сохранённую ranking combination.
    2. Формирует базовый запрос без внешнего
       trailer enrichment.
    3. Вычисляет детерминированный request_key.
    4. Резервирует publication_batch.
    5. Блокирует повторный запуск до любых
       HTTP-запросов enrichment и OpenAI.
    6. Для нового выпуска использует уже проверенные
       preflight items, если их передал orchestrator;
       иначе выполняет прежний best-effort trailer enrichment.
    7. Формирует фактический OpenAI-запрос
       по временным enriched items.
    8. Выполняет первичную генерацию
       Telegram-поста.
    9. Выполняет автоматический self-review
       того же поста. Self-review сам решает,
       нужен ли web_search.
    10. Проверяет self-review через
        deterministic text integrity gate.
    11. При необходимости выполняет до двух
        bounded revision-проходов.
    12. Если revisions не устранили вероятно
        обрезанный хвост, применяет локальный
        deterministic fail-safe без новых фактов.
    13. Usage и стоимость суммируются только по
        фактическим Responses API вызовам.
    14. Сохраняет только финальный generated_post.
    15. Переводит выпуск в awaiting_review.
    16. Реальные ошибки pipeline по-прежнему
        переводят выпуск в failed.

    Функция не управляет жизненным циклом
    пула PostgreSQL или OpenAI SDK-клиента.
    Их должен закрывать вызывающий код.

    Telegram не вызывается.
    """

    if combination_id is None:
        selection = await load_generation_top3(
            pool,
            ranking_run_id=ranking_run_id,
        )
    else:
        ranked_combination = (
            await load_generation_combination(
                pool,
                ranking_run_id=ranking_run_id,
                combination_id=combination_id,
            )
        )

        selection = (
            ranked_combination.selection
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
        combination_id=combination_id,
        publication_date=publication_date,
        telegram_chat_id=telegram_chat_id,
        metadata=generator.metadata,
        model_request=model_request,
    )

    if reservation_observer is not None:
        await reservation_observer(
            reservation
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
        if prepared_items is None:
            generation_items = (
                await _enrich_generation_items(
                    selection.items,
                    trailer_enricher=trailer_enricher,
                )
            )
        else:
            generation_items = (
                _validate_prepared_generation_items(
                    selection,
                    prepared_items,
                )
            )

        generation_model_request = (
            generator.build_request(
                generation_items
            )
        )

        primary_generation = (
            await generator
            .generate_prepared_request(
                generation_items,
                generation_model_request,
            )
        )

        _require_generation_telemetry(
            primary_generation
        )

        self_review_generation = (
            await generator
            .generate_self_review_detailed(
                generation_items,
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

        integrity_outcome = (
            await _run_integrity_repairs_if_needed(
                generator,
                items=generation_items,
                initial_generation=(
                    self_review_generation
                ),
                primary_generation=(
                    primary_generation
                ),
            )
        )

        generation = (
            _combine_generation_results(
                (
                    primary_generation,
                    *integrity_outcome
                    .model_generations,
                ),
                final_payload=(
                    integrity_outcome
                    .final_payload
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
