from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import asyncpg

from app.db.generation_selection import (
    GenerationTop3Selection,
)
from app.db.image_generation_completion import (
    ImageGenerationCompletionResult,
    fail_reserved_image_generation,
    complete_reserved_image_generation,
)
from app.db.image_generation_reservation import (
    ImageGenerationReservation,
    reserve_image_generation,
)
from app.generation.image_generator import (
    ImageGenerationNewsItem,
    ImageModelRequest,
    OpenAIImageGenerationResult,
    OpenAIImageUsage,
    OpenAIMovieNewsImageGenerator,
)
from app.generation.image_openai_usage import (
    build_openai_image_cost_payload,
)
from app.generation.image_request_key import (
    ImageRequestKey,
    create_image_request_key,
)
from app.generation.image_storage import (
    DEFAULT_IMAGE_OUTPUT_DIR,
    StoredImageArtifact,
    store_png_image,
)


ImageRequestKind = Literal[
    "initial",
    "regenerate",
]

ImageCostEstimator = Callable[
    [str, OpenAIImageUsage],
    Mapping[str, Any],
]

ImageGenerationReservationObserver = Callable[
    [ImageGenerationReservation],
    Awaitable[None],
]


@dataclass(frozen=True, slots=True)
class ReservedOpenAIImageGenerationResult:
    """Результат защищённого image-конвейера."""

    selection: GenerationTop3Selection

    items: tuple[
        ImageGenerationNewsItem,
        ImageGenerationNewsItem,
        ImageGenerationNewsItem,
    ]

    model_request: ImageModelRequest

    request_key: ImageRequestKey

    reservation: ImageGenerationReservation

    model_called: bool

    generation: (
        OpenAIImageGenerationResult
        | None
    )

    artifact: (
        StoredImageArtifact
        | None
    )

    completion: (
        ImageGenerationCompletionResult
        | None
    )

    @property
    def image_generation_id(self) -> int:
        """Возвращает ID image reservation."""

        return (
            self.reservation
            .image_generation_id
        )

    @property
    def batch_id(self) -> int:
        """Возвращает ID publication batch."""

        return self.reservation.batch_id

    @property
    def generated_post_id(self) -> int:
        """Возвращает ID изменяемого поста."""

        return (
            self.reservation
            .generated_post_id
        )

    @property
    def request_kind(self) -> str:
        """Возвращает initial/regenerate."""

        return (
            self.reservation
            .request_kind
        )

    @property
    def image_status(self) -> str:
        """Возвращает итоговый известный статус."""

        if self.completion is not None:
            return (
                self.completion
                .image_status
            )

        return (
            self.reservation
            .image_status
        )

    @property
    def duplicate_request_blocked(
        self,
    ) -> bool:
        """Показывает блокировку повторного API-call."""

        return not self.reservation.created_new

    @property
    def completed(self) -> bool:
        """Показывает успешное завершение."""

        if self.completion is not None:
            return (
                self.completion.image_status
                == "completed"
                and self.completion.batch_status
                == "awaiting_review"
                and self.completion.post_status
                == "awaiting_review"
            )

        return (
            self.reservation.image_status
            == "completed"
        )


def _normalize_request_kind(
    value: str,
) -> ImageRequestKind:
    """Проверяет тип image-запроса."""

    if value not in {
        "initial",
        "regenerate",
    }:
        raise ValueError(
            "request_kind должен быть "
            "initial или regenerate."
        )

    return value


def _normalize_optional_text(
    value: str | None,
    *,
    field_name: str,
) -> str | None:
    """Нормализует необязательный текст."""

    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть "
            "строкой или None."
        )

    normalized_value = value.strip()

    if not normalized_value:
        return None

    return normalized_value


def _normalize_issues(
    issues: tuple[str, ...],
) -> tuple[str, ...]:
    """Проверяет редакционные image issues."""

    if not isinstance(issues, tuple):
        raise TypeError(
            "issues должен быть tuple[str, ...]."
        )

    normalized_issues: list[str] = []

    for index, issue in enumerate(
        issues,
        start=1,
    ):
        if not isinstance(issue, str):
            raise TypeError(
                "Каждый issues item должен "
                f"быть строкой: index={index}."
            )

        normalized_issue = issue.strip()

        if not normalized_issue:
            raise ValueError(
                "issues не может содержать "
                f"пустой текст: index={index}."
            )

        normalized_issues.append(
            normalized_issue
        )

    return tuple(
        normalized_issues
    )


def _build_image_items(
    selection: GenerationTop3Selection,
) -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """Строит factual input из сохранённого TOP-3."""

    if not isinstance(
        selection,
        GenerationTop3Selection,
    ):
        raise TypeError(
            "selection должен быть "
            "GenerationTop3Selection."
        )

    if len(selection.items) != 3:
        raise ValueError(
            "Для image-generation требуется "
            "ровно три новости."
        )

    items = tuple(
        ImageGenerationNewsItem(
            position=item.position,
            news_id=item.news_id,
            title=item.title,
            summary=item.summary,
        )
        for item in selection.items
    )

    positions = tuple(
        item.position
        for item in items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Позиции image items должны "
            "быть 1, 2, 3."
        )

    news_ids = tuple(
        item.news_id
        for item in items
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Image items содержат "
            "дублирующиеся news_id."
        )

    return (
        items[0],
        items[1],
        items[2],
    )


def _validate_request_context(
    *,
    request_kind: ImageRequestKind,
    review_action_id: int | None,
    editorial_comment: str | None,
    issues: tuple[str, ...],
) -> None:
    """Проверяет initial/regenerate контекст."""

    if request_kind == "initial":
        if review_action_id is not None:
            raise ValueError(
                "initial image request не должен "
                "содержать review_action_id."
            )

        if editorial_comment is not None:
            raise ValueError(
                "initial image request не должен "
                "содержать editorial_comment."
            )

        if issues:
            raise ValueError(
                "initial image request не должен "
                "содержать issues."
            )

        return

    if isinstance(review_action_id, bool):
        raise TypeError(
            "review_action_id не может быть bool."
        )

    if not isinstance(
        review_action_id,
        int,
    ):
        raise TypeError(
            "Для regenerate требуется "
            "review_action_id int."
        )

    if review_action_id <= 0:
        raise ValueError(
            "review_action_id должен быть "
            "больше нуля."
        )

    if editorial_comment is None:
        raise ValueError(
            "Для regenerate требуется "
            "editorial_comment."
        )

    if not issues:
        raise ValueError(
            "Для regenerate требуется "
            "хотя бы один issue."
        )


def _build_response_metadata(
    generation: OpenAIImageGenerationResult,
    *,
    artifact: StoredImageArtifact | None = None,
) -> dict[str, Any]:
    """Формирует JSON metadata ответа Image API."""

    response = generation.model_response

    payload: dict[str, Any] = {
        "created": response.created,
        "output_format": (
            response.output_format
        ),
        "quality": response.quality,
        "size": response.size,
        "background": response.background,
        "revised_prompt": (
            response.revised_prompt
        ),
    }

    if artifact is not None:
        payload["storage"] = {
            "image_path": artifact.image_path,
            "image_sha256": (
                artifact.image_sha256
            ),
            "width": artifact.width,
            "height": artifact.height,
            "byte_count": (
                artifact.byte_count
            ),
            "already_stored": (
                artifact.already_stored
            ),
        }

    return payload


def _estimate_cost(
    generation: OpenAIImageGenerationResult,
    *,
    cost_estimator: (
        ImageCostEstimator
        | None
    ),
) -> Mapping[str, Any] | None:
    """Вычисляет cost payload при наличии usage."""

    usage = generation.model_response.usage

    if usage is None:
        return None

    if cost_estimator is None:
        raise ValueError(
            "Image API вернул usage, но "
            "ImageCostEstimator не настроен."
        )

    cost_payload = cost_estimator(
        generation.model_request.model,
        usage,
    )

    if not isinstance(
        cost_payload,
        Mapping,
    ):
        raise TypeError(
            "ImageCostEstimator должен "
            "вернуть Mapping."
        )

    return cost_payload


async def _record_image_pipeline_failure(
    pool: asyncpg.Pool,
    *,
    reservation: ImageGenerationReservation,
    error: Exception,
    generation: (
        OpenAIImageGenerationResult
        | None
    ),
    cost_payload: (
        Mapping[str, Any]
        | None
    ),
) -> None:
    """
    Фиксирует image failure, не повреждая текст.

    Ошибка фиксации добавляется к исходному
    исключению и не заменяет его.
    """

    error_message = str(error).strip()

    if not error_message:
        error_message = repr(error)

    response_metadata: (
        dict[str, Any]
        | None
    ) = None

    usage: OpenAIImageUsage | None = None
    failure_cost_payload: (
        Mapping[str, Any]
        | None
    ) = None

    if generation is not None:
        response_metadata = (
            _build_response_metadata(
                generation
            )
        )

        usage = (
            generation
            .model_response
            .usage
        )

        if (
            usage is not None
            and cost_payload is not None
        ):
            failure_cost_payload = (
                cost_payload
            )
        else:
            usage = None

    try:
        await fail_reserved_image_generation(
            pool,
            image_generation_id=(
                reservation
                .image_generation_id
            ),
            request_key=(
                reservation.request_key
            ),
            error_message=error_message,
            error_type=(
                type(error).__name__
            ),
            response_metadata=(
                response_metadata
            ),
            usage=usage,
            cost_payload=(
                failure_cost_payload
            ),
        )
    except Exception as failure_error:
        error.add_note(
            "Дополнительно не удалось "
            "зафиксировать image_status=failed: "
            f"{type(failure_error).__name__}: "
            f"{failure_error}"
        )


async def run_reserved_openai_image_generation(
    pool: asyncpg.Pool,
    *,
    generator: OpenAIMovieNewsImageGenerator,
    selection: GenerationTop3Selection,
    batch_id: int,
    generated_post_id: int,
    request_kind: str = "initial",
    review_action_id: int | None = None,
    editorial_comment: str | None = None,
    issues: tuple[str, ...] = (),
    output_dir: str | Path = (
        DEFAULT_IMAGE_OUTPUT_DIR
    ),
    cost_estimator: (
        ImageCostEstimator
        | None
    ) = build_openai_image_cost_payload,
    reservation_observer: (
        ImageGenerationReservationObserver
        | None
    ) = None,
) -> ReservedOpenAIImageGenerationResult:
    """
    Запускает защищённый image-generation pipeline.

    Последовательность:

    1. Получает уже сохранённый TOP-3.
    2. Строит factual image items.
    3. Формирует точный Image API request.
    4. Вычисляет детерминированный request_key.
    5. Резервирует image-generation в PostgreSQL.
    6. Блокирует повторный платный API-call.
    7. Для новой reservation вызывает Image API.
    8. Проверяет и атомарно сохраняет PNG.
    9. Записывает image_path/SHA и response metadata.
    10. Оставляет batch/post в awaiting_review.
    11. При ошибке переводит только image request
        в failed; текстовый пост не повреждается.

    Функция не закрывает PostgreSQL pool или
    OpenAI SDK client.

    Telegram не вызывается.
    """

    normalized_request_kind = (
        _normalize_request_kind(
            request_kind
        )
    )

    normalized_editorial_comment = (
        _normalize_optional_text(
            editorial_comment,
            field_name="editorial_comment",
        )
    )

    normalized_issues = (
        _normalize_issues(
            issues
        )
    )

    _validate_request_context(
        request_kind=(
            normalized_request_kind
        ),
        review_action_id=review_action_id,
        editorial_comment=(
            normalized_editorial_comment
        ),
        issues=normalized_issues,
    )

    items = _build_image_items(
        selection
    )

    model_request = generator.build_request(
        items=items,
        editorial_comment=(
            normalized_editorial_comment
        ),
        issues=normalized_issues,
    )

    request_key = (
        create_image_request_key(
            batch_id=batch_id,
            ranking_run_id=(
                selection.ranking_run_id
            ),
            request_kind=(
                normalized_request_kind
            ),
            review_action_id=(
                review_action_id
            ),
            metadata=generator.metadata,
            model_request=model_request,
            items=items,
        )
    )

    reservation = await reserve_image_generation(
        pool,
        request_key=request_key,
        batch_id=batch_id,
        generated_post_id=(
            generated_post_id
        ),
        ranking_run_id=(
            selection.ranking_run_id
        ),
        request_kind=(
            normalized_request_kind
        ),
        review_action_id=(
            review_action_id
        ),
        editorial_comment=(
            normalized_editorial_comment
        ),
        issues=normalized_issues,
        metadata=generator.metadata,
        model_request=model_request,
        items=items,
    )

    if reservation_observer is not None:
        await reservation_observer(
            reservation
        )

    if not reservation.should_call_model:
        return (
            ReservedOpenAIImageGenerationResult(
                selection=selection,
                items=items,
                model_request=model_request,
                request_key=request_key,
                reservation=reservation,
                model_called=False,
                generation=None,
                artifact=None,
                completion=None,
            )
        )

    generation: (
        OpenAIImageGenerationResult
        | None
    ) = None

    cost_payload: (
        Mapping[str, Any]
        | None
    ) = None

    try:
        generation = await generator.generate(
            items=items,
            editorial_comment=(
                normalized_editorial_comment
            ),
            issues=normalized_issues,
        )

        if (
            generation.model_request
            != model_request
        ):
            raise RuntimeError(
                "Фактический Image API request "
                "не совпадает с request, "
                "использованным для reservation."
            )

        cost_payload = _estimate_cost(
            generation,
            cost_estimator=(
                cost_estimator
            ),
        )

        artifact = store_png_image(
            generation.model_response.image_bytes,
            image_generation_id=(
                reservation
                .image_generation_id
            ),
            expected_size=(
                model_request.size
            ),
            output_dir=output_dir,
        )

        response_metadata = (
            _build_response_metadata(
                generation,
                artifact=artifact,
            )
        )

        completion = (
            await complete_reserved_image_generation(
                pool,
                image_generation_id=(
                    reservation
                    .image_generation_id
                ),
                request_key=(
                    reservation.request_key
                ),
                image_path=(
                    artifact.image_path
                ),
                image_sha256=(
                    artifact.image_sha256
                ),
                response_metadata=(
                    response_metadata
                ),
                usage=(
                    generation
                    .model_response
                    .usage
                ),
                cost_payload=(
                    cost_payload
                ),
            )
        )

    except Exception as error:
        await _record_image_pipeline_failure(
            pool,
            reservation=reservation,
            error=error,
            generation=generation,
            cost_payload=cost_payload,
        )

        raise

    return ReservedOpenAIImageGenerationResult(
        selection=selection,
        items=items,
        model_request=model_request,
        request_key=request_key,
        reservation=reservation,
        model_called=True,
        generation=generation,
        artifact=artifact,
        completion=completion,
    )