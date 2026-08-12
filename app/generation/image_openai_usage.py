from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from app.generation.image_generator import (
    OpenAIImageUsage,
)


TOKENS_PER_MILLION = Decimal("1000000")
USD_QUANTUM = Decimal("0.00000001")

GPT_IMAGE_2_PRICING_VERSION = (
    "2026-08-12"
)

IMAGE_PRICING_BASIS = (
    "reported_modality_tokens_standard_rates"
)


@dataclass(frozen=True, slots=True)
class OpenAIImageModelPricing:
    """Тариф Image API за один миллион токенов."""

    model_name: str

    text_input_usd_per_million: Decimal

    text_cached_input_usd_per_million: Decimal

    image_input_usd_per_million: Decimal

    image_cached_input_usd_per_million: Decimal

    image_output_usd_per_million: Decimal

    pricing_version: str

    def __post_init__(self) -> None:
        """Проверяет тариф Image API."""

        if not isinstance(
            self.model_name,
            str,
        ):
            raise TypeError(
                "model_name должен быть строкой."
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name не может быть пустым."
            )

        if not isinstance(
            self.pricing_version,
            str,
        ):
            raise TypeError(
                "pricing_version должен быть "
                "строкой."
            )

        if not self.pricing_version.strip():
            raise ValueError(
                "pricing_version не может "
                "быть пустым."
            )

        monetary_values = {
            "text_input_usd_per_million": (
                self
                .text_input_usd_per_million
            ),
            "text_cached_input_usd_per_million": (
                self
                .text_cached_input_usd_per_million
            ),
            "image_input_usd_per_million": (
                self
                .image_input_usd_per_million
            ),
            "image_cached_input_usd_per_million": (
                self
                .image_cached_input_usd_per_million
            ),
            "image_output_usd_per_million": (
                self
                .image_output_usd_per_million
            ),
        }

        for field_name, value in (
            monetary_values.items()
        ):
            if not isinstance(
                value,
                Decimal,
            ):
                raise TypeError(
                    f"{field_name} должен быть "
                    "Decimal."
                )

            if not value.is_finite():
                raise ValueError(
                    f"{field_name} должен быть "
                    "конечным числом."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} не может "
                    "быть отрицательным."
                )


@dataclass(frozen=True, slots=True)
class OpenAIImageCostEstimate:
    """Оценка стоимости одного Image API запроса."""

    model_name: str

    pricing_version: str

    pricing_basis: str

    input_text_cost_usd: Decimal

    input_image_cost_usd: Decimal

    output_text_cost_usd: Decimal

    output_image_cost_usd: Decimal

    total_cost_usd: Decimal

    def __post_init__(self) -> None:
        """Проверяет согласованность стоимости."""

        for field_name, value in {
            "model_name": self.model_name,
            "pricing_version": (
                self.pricing_version
            ),
            "pricing_basis": (
                self.pricing_basis
            ),
        }.items():
            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name} должен быть "
                    "строкой."
                )

            if not value.strip():
                raise ValueError(
                    f"{field_name} не может "
                    "быть пустым."
                )

        monetary_values = {
            "input_text_cost_usd": (
                self.input_text_cost_usd
            ),
            "input_image_cost_usd": (
                self.input_image_cost_usd
            ),
            "output_text_cost_usd": (
                self.output_text_cost_usd
            ),
            "output_image_cost_usd": (
                self.output_image_cost_usd
            ),
            "total_cost_usd": (
                self.total_cost_usd
            ),
        }

        for field_name, value in (
            monetary_values.items()
        ):
            if not isinstance(
                value,
                Decimal,
            ):
                raise TypeError(
                    f"{field_name} должен быть "
                    "Decimal."
                )

            if not value.is_finite():
                raise ValueError(
                    f"{field_name} должен быть "
                    "конечным числом."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} не может "
                    "быть отрицательным."
                )

        component_total = (
            self.input_text_cost_usd
            + self.input_image_cost_usd
            + self.output_text_cost_usd
            + self.output_image_cost_usd
        ).quantize(
            USD_QUANTUM,
            rounding=ROUND_HALF_UP,
        )

        if (
            component_total
            != self.total_cost_usd
        ):
            raise ValueError(
                "total_cost_usd не совпадает "
                "с суммой компонентов."
            )


GPT_IMAGE_2_PRICING = (
    OpenAIImageModelPricing(
        model_name="gpt-image-2",
        text_input_usd_per_million=(
            Decimal("5.00")
        ),
        text_cached_input_usd_per_million=(
            Decimal("1.25")
        ),
        image_input_usd_per_million=(
            Decimal("8.00")
        ),
        image_cached_input_usd_per_million=(
            Decimal("2.00")
        ),
        image_output_usd_per_million=(
            Decimal("30.00")
        ),
        pricing_version=(
            GPT_IMAGE_2_PRICING_VERSION
        ),
    )
)


IMAGE_MODEL_PRICING: dict[
    str,
    OpenAIImageModelPricing,
] = {
    GPT_IMAGE_2_PRICING.model_name: (
        GPT_IMAGE_2_PRICING
    ),
}


def get_image_model_pricing(
    model_name: str,
) -> OpenAIImageModelPricing:
    """Возвращает известный тариф Image API."""

    if not isinstance(
        model_name,
        str,
    ):
        raise TypeError(
            "model_name должен быть строкой."
        )

    normalized_model_name = (
        model_name.strip()
    )

    if not normalized_model_name:
        raise ValueError(
            "model_name не может быть пустым."
        )

    pricing = IMAGE_MODEL_PRICING.get(
        normalized_model_name
    )

    if pricing is None:
        raise LookupError(
            "Для Image API модели "
            "не настроен тариф: "
            f"{normalized_model_name}"
        )

    return pricing


def _calculate_component_cost(
    *,
    token_count: int,
    usd_per_million: Decimal,
) -> Decimal:
    """Рассчитывает стоимость токенов."""

    if isinstance(token_count, bool):
        raise TypeError(
            "token_count не может быть bool."
        )

    if not isinstance(
        token_count,
        int,
    ):
        raise TypeError(
            "token_count должен быть int."
        )

    if token_count < 0:
        raise ValueError(
            "token_count не может быть "
            "отрицательным."
        )

    return (
        Decimal(token_count)
        / TOKENS_PER_MILLION
        * usd_per_million
    ).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _validate_usage_breakdown(
    usage: OpenAIImageUsage,
) -> None:
    """Проверяет модальную разбивку Image usage."""

    if not isinstance(
        usage,
        OpenAIImageUsage,
    ):
        raise TypeError(
            "usage должен быть OpenAIImageUsage."
        )

    expected_input_tokens = (
        usage.input_text_tokens
        + usage.input_image_tokens
    )

    if (
        usage.input_tokens
        != expected_input_tokens
    ):
        raise ValueError(
            "input_tokens не совпадает "
            "с input_text_tokens + "
            "input_image_tokens."
        )

    expected_output_tokens = (
        usage.output_text_tokens
        + usage.output_image_tokens
    )

    if (
        usage.output_tokens
        != expected_output_tokens
    ):
        raise ValueError(
            "output_tokens не совпадает "
            "с output_text_tokens + "
            "output_image_tokens."
        )

    expected_total_tokens = (
        usage.input_tokens
        + usage.output_tokens
    )

    if (
        usage.total_tokens
        != expected_total_tokens
    ):
        raise ValueError(
            "total_tokens не совпадает "
            "с input_tokens + output_tokens."
        )


def calculate_openai_image_cost(
    usage: OpenAIImageUsage,
    pricing: OpenAIImageModelPricing,
) -> OpenAIImageCostEstimate:
    """
    Рассчитывает оценочную стоимость Image API.

    Текущий OpenAIImageUsage содержит разбивку
    text/image, но не выделяет cached input.
    Поэтому input оценивается по стандартным
    uncached rates. Cached rates сохраняются
    в тарифе для явной фиксации актуального
    прайсинга и будущего расширения telemetry.
    """

    _validate_usage_breakdown(
        usage
    )

    if not isinstance(
        pricing,
        OpenAIImageModelPricing,
    ):
        raise TypeError(
            "pricing должен быть "
            "OpenAIImageModelPricing."
        )

    if usage.output_text_tokens != 0:
        raise ValueError(
            "Для текущего gpt-image-2 "
            "Image API ожидается только "
            "image output; "
            "output_text_tokens должен быть 0."
        )

    input_text_cost = (
        _calculate_component_cost(
            token_count=(
                usage.input_text_tokens
            ),
            usd_per_million=(
                pricing
                .text_input_usd_per_million
            ),
        )
    )

    input_image_cost = (
        _calculate_component_cost(
            token_count=(
                usage.input_image_tokens
            ),
            usd_per_million=(
                pricing
                .image_input_usd_per_million
            ),
        )
    )

    output_text_cost = Decimal(
        "0"
    ).quantize(
        USD_QUANTUM
    )

    output_image_cost = (
        _calculate_component_cost(
            token_count=(
                usage.output_image_tokens
            ),
            usd_per_million=(
                pricing
                .image_output_usd_per_million
            ),
        )
    )

    total_cost = (
        input_text_cost
        + input_image_cost
        + output_text_cost
        + output_image_cost
    ).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )

    return OpenAIImageCostEstimate(
        model_name=pricing.model_name,
        pricing_version=(
            pricing.pricing_version
        ),
        pricing_basis=IMAGE_PRICING_BASIS,
        input_text_cost_usd=(
            input_text_cost
        ),
        input_image_cost_usd=(
            input_image_cost
        ),
        output_text_cost_usd=(
            output_text_cost
        ),
        output_image_cost_usd=(
            output_image_cost
        ),
        total_cost_usd=total_cost,
    )


def build_openai_image_cost_payload(
    model_name: str,
    usage: OpenAIImageUsage,
) -> dict[str, Any]:
    """
    Возвращает JSON-совместимый cost payload.

    Эта функция соответствует ImageCostEstimator
    из openai_image_pipeline.py.
    """

    pricing = get_image_model_pricing(
        model_name
    )

    cost_estimate = (
        calculate_openai_image_cost(
            usage,
            pricing,
        )
    )

    return {
        "model_name": (
            cost_estimate.model_name
        ),
        "pricing_version": (
            cost_estimate.pricing_version
        ),
        "pricing_basis": (
            cost_estimate.pricing_basis
        ),
        "input_text_cost_usd": str(
            cost_estimate
            .input_text_cost_usd
        ),
        "input_image_cost_usd": str(
            cost_estimate
            .input_image_cost_usd
        ),
        "output_text_cost_usd": str(
            cost_estimate
            .output_text_cost_usd
        ),
        "output_image_cost_usd": str(
            cost_estimate
            .output_image_cost_usd
        ),
        "total_cost_usd": str(
            cost_estimate.total_cost_usd
        ),
    }