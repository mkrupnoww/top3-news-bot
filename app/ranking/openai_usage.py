from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


TOKENS_PER_MILLION = Decimal("1000000")
USD_QUANTUM = Decimal("0.00000001")

GPT_5_6_TERRA_PRICING_VERSION = (
    "2026-07-31"
)


@dataclass(frozen=True, slots=True)
class OpenAITokenUsage:
    """Фактическое потребление токенов запроса."""

    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        """Проверяет согласованность счётчиков."""

        values = {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": (
                self.cached_input_tokens
            ),
            "cache_write_tokens": (
                self.cache_write_tokens
            ),
            "output_tokens": self.output_tokens,
            "reasoning_tokens": (
                self.reasoning_tokens
            ),
            "total_tokens": self.total_tokens,
        }

        for field_name, value in values.items():
            if not isinstance(value, int):
                raise TypeError(
                    f"{field_name} должен быть int."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} не может "
                    "быть отрицательным."
                )

        input_breakdown = (
            self.cached_input_tokens
            + self.cache_write_tokens
        )

        if input_breakdown > self.input_tokens:
            raise ValueError(
                "Сумма cached_input_tokens и "
                "cache_write_tokens превышает "
                "input_tokens."
            )

        if (
            self.reasoning_tokens
            > self.output_tokens
        ):
            raise ValueError(
                "reasoning_tokens не может "
                "превышать output_tokens."
            )

        expected_total = (
            self.input_tokens
            + self.output_tokens
        )

        if self.total_tokens != expected_total:
            raise ValueError(
                "total_tokens не совпадает "
                "с input_tokens + output_tokens: "
                f"total={self.total_tokens}, "
                f"expected={expected_total}"
            )

    @property
    def regular_input_tokens(self) -> int:
        """Возвращает обычные входные токены."""

        return (
            self.input_tokens
            - self.cached_input_tokens
            - self.cache_write_tokens
        )


@dataclass(frozen=True, slots=True)
class OpenAIModelPricing:
    """Тариф модели за один миллион токенов."""

    model_name: str
    input_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    cache_write_multiplier: Decimal
    pricing_version: str

    def __post_init__(self) -> None:
        """Проверяет тариф модели."""

        if not self.model_name.strip():
            raise ValueError(
                "model_name не может быть пустым."
            )

        if not self.pricing_version.strip():
            raise ValueError(
                "pricing_version не может "
                "быть пустым."
            )

        monetary_values = {
            "input_usd_per_million": (
                self.input_usd_per_million
            ),
            "cached_input_usd_per_million": (
                self.cached_input_usd_per_million
            ),
            "output_usd_per_million": (
                self.output_usd_per_million
            ),
            "cache_write_multiplier": (
                self.cache_write_multiplier
            ),
        }

        for field_name, value in (
            monetary_values.items()
        ):
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
class OpenAICostEstimate:
    """Расчёт стоимости одного запроса."""

    model_name: str
    pricing_version: str
    regular_input_cost_usd: Decimal
    cached_input_cost_usd: Decimal
    cache_write_cost_usd: Decimal
    output_cost_usd: Decimal
    total_cost_usd: Decimal


GPT_5_6_TERRA_PRICING = OpenAIModelPricing(
    model_name="gpt-5.6-terra",
    input_usd_per_million=Decimal("2.00"),
    cached_input_usd_per_million=(
        Decimal("0.20")
    ),
    output_usd_per_million=Decimal("12.00"),
    cache_write_multiplier=Decimal("1.25"),
    pricing_version=(
        GPT_5_6_TERRA_PRICING_VERSION
    ),
)


MODEL_PRICING: dict[
    str,
    OpenAIModelPricing,
] = {
    GPT_5_6_TERRA_PRICING.model_name: (
        GPT_5_6_TERRA_PRICING
    ),
}


def _read_integer(
    source: object,
    attribute_name: str,
    *,
    default: int | None = None,
) -> int:
    """Безопасно читает целочисленное поле."""

    value = getattr(
        source,
        attribute_name,
        default,
    )

    if value is None:
        raise ValueError(
            "В usage отсутствует поле "
            f"{attribute_name}."
        )

    if isinstance(value, bool):
        raise TypeError(
            f"{attribute_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{attribute_name} должен быть int."
        )

    return value


def extract_response_usage(
    response: object,
) -> OpenAITokenUsage:
    """Извлекает usage из ответа OpenAI SDK."""

    usage = getattr(
        response,
        "usage",
        None,
    )

    if usage is None:
        raise ValueError(
            "Ответ OpenAI не содержит usage."
        )

    input_details = getattr(
        usage,
        "input_tokens_details",
        None,
    )

    output_details = getattr(
        usage,
        "output_tokens_details",
        None,
    )

    cached_input_tokens = 0
    cache_write_tokens = 0
    reasoning_tokens = 0

    if input_details is not None:
        cached_input_tokens = _read_integer(
            input_details,
            "cached_tokens",
            default=0,
        )

        cache_write_tokens = _read_integer(
            input_details,
            "cache_write_tokens",
            default=0,
        )

    if output_details is not None:
        reasoning_tokens = _read_integer(
            output_details,
            "reasoning_tokens",
            default=0,
        )

    return OpenAITokenUsage(
        input_tokens=_read_integer(
            usage,
            "input_tokens",
        ),
        cached_input_tokens=(
            cached_input_tokens
        ),
        cache_write_tokens=(
            cache_write_tokens
        ),
        output_tokens=_read_integer(
            usage,
            "output_tokens",
        ),
        reasoning_tokens=reasoning_tokens,
        total_tokens=_read_integer(
            usage,
            "total_tokens",
        ),
    )


def get_model_pricing(
    model_name: str,
) -> OpenAIModelPricing:
    """Возвращает известный тариф модели."""

    normalized_model_name = (
        model_name.strip()
    )

    if not normalized_model_name:
        raise ValueError(
            "model_name не может быть пустым."
        )

    pricing = MODEL_PRICING.get(
        normalized_model_name
    )

    if pricing is None:
        raise LookupError(
            "Для модели не настроен тариф: "
            f"{normalized_model_name}"
        )

    return pricing


def _calculate_component_cost(
    *,
    token_count: int,
    usd_per_million: Decimal,
) -> Decimal:
    """Рассчитывает стоимость части токенов."""

    return (
        Decimal(token_count)
        / TOKENS_PER_MILLION
        * usd_per_million
    ).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def calculate_openai_cost(
    usage: OpenAITokenUsage,
    pricing: OpenAIModelPricing,
) -> OpenAICostEstimate:
    """Рассчитывает стоимость одного запроса."""

    regular_input_cost = (
        _calculate_component_cost(
            token_count=(
                usage.regular_input_tokens
            ),
            usd_per_million=(
                pricing.input_usd_per_million
            ),
        )
    )

    cached_input_cost = (
        _calculate_component_cost(
            token_count=(
                usage.cached_input_tokens
            ),
            usd_per_million=(
                pricing
                .cached_input_usd_per_million
            ),
        )
    )

    cache_write_rate = (
        pricing.input_usd_per_million
        * pricing.cache_write_multiplier
    )

    cache_write_cost = (
        _calculate_component_cost(
            token_count=(
                usage.cache_write_tokens
            ),
            usd_per_million=(
                cache_write_rate
            ),
        )
    )

    output_cost = _calculate_component_cost(
        token_count=usage.output_tokens,
        usd_per_million=(
            pricing.output_usd_per_million
        ),
    )

    total_cost = (
        regular_input_cost
        + cached_input_cost
        + cache_write_cost
        + output_cost
    ).quantize(
        USD_QUANTUM,
        rounding=ROUND_HALF_UP,
    )

    return OpenAICostEstimate(
        model_name=pricing.model_name,
        pricing_version=(
            pricing.pricing_version
        ),
        regular_input_cost_usd=(
            regular_input_cost
        ),
        cached_input_cost_usd=(
            cached_input_cost
        ),
        cache_write_cost_usd=(
            cache_write_cost
        ),
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
    )