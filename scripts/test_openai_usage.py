from dataclasses import dataclass

from app.ranking.openai_usage import (
    GPT_5_6_TERRA_PRICING,
    calculate_openai_cost,
    extract_response_usage,
    get_model_pricing,
)


@dataclass(frozen=True, slots=True)
class FakeInputTokenDetails:
    """Поддельная детализация входа."""

    cached_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True, slots=True)
class FakeOutputTokenDetails:
    """Поддельная детализация выхода."""

    reasoning_tokens: int


@dataclass(frozen=True, slots=True)
class FakeUsage:
    """Поддельный объект usage."""

    input_tokens: int
    input_tokens_details: (
        FakeInputTokenDetails
    )
    output_tokens: int
    output_tokens_details: (
        FakeOutputTokenDetails
    )
    total_tokens: int


@dataclass(frozen=True, slots=True)
class FakeResponse:
    """Поддельный ответ OpenAI SDK."""

    usage: FakeUsage


def build_response() -> FakeResponse:
    """Создаёт тестовый usage."""

    return FakeResponse(
        usage=FakeUsage(
            input_tokens=1500,
            input_tokens_details=(
                FakeInputTokenDetails(
                    cached_tokens=400,
                    cache_write_tokens=100,
                )
            ),
            output_tokens=300,
            output_tokens_details=(
                FakeOutputTokenDetails(
                    reasoning_tokens=200
                )
            ),
            total_tokens=1800,
        )
    )


def test_usage_extraction() -> None:
    """Проверяет извлечение всех счётчиков."""

    usage = extract_response_usage(
        build_response()
    )

    assert usage.input_tokens == 1500
    assert usage.cached_input_tokens == 400
    assert usage.cache_write_tokens == 100
    assert usage.regular_input_tokens == 1000
    assert usage.output_tokens == 300
    assert usage.reasoning_tokens == 200
    assert usage.total_tokens == 1800

    print("Usage extraction: OK")
    print(
        f"input_tokens={usage.input_tokens}"
    )
    print(
        "regular_input_tokens="
        f"{usage.regular_input_tokens}"
    )
    print(
        "cached_input_tokens="
        f"{usage.cached_input_tokens}"
    )
    print(
        "cache_write_tokens="
        f"{usage.cache_write_tokens}"
    )
    print(
        f"output_tokens={usage.output_tokens}"
    )
    print(
        "reasoning_tokens="
        f"{usage.reasoning_tokens}"
    )
    print(
        f"total_tokens={usage.total_tokens}"
    )


def test_cost_calculation() -> None:
    """Проверяет расчёт стоимости Terra."""

    usage = extract_response_usage(
        build_response()
    )

    pricing = get_model_pricing(
        "gpt-5.6-terra"
    )

    assert pricing == (
        GPT_5_6_TERRA_PRICING
    )

    cost = calculate_openai_cost(
        usage,
        pricing,
    )

    assert str(
        cost.regular_input_cost_usd
    ) == "0.00200000"

    assert str(
        cost.cached_input_cost_usd
    ) == "0.00008000"

    assert str(
        cost.cache_write_cost_usd
    ) == "0.00025000"

    assert str(
        cost.output_cost_usd
    ) == "0.00360000"

    assert str(
        cost.total_cost_usd
    ) == "0.00593000"

    print()
    print("Cost calculation: OK")
    print(
        f"model={cost.model_name}"
    )
    print(
        "pricing_version="
        f"{cost.pricing_version}"
    )
    print(
        "regular_input_cost_usd="
        f"{cost.regular_input_cost_usd}"
    )
    print(
        "cached_input_cost_usd="
        f"{cost.cached_input_cost_usd}"
    )
    print(
        "cache_write_cost_usd="
        f"{cost.cache_write_cost_usd}"
    )
    print(
        "output_cost_usd="
        f"{cost.output_cost_usd}"
    )
    print(
        "total_cost_usd="
        f"{cost.total_cost_usd}"
    )


def test_missing_usage_blocking() -> None:
    """Проверяет ответ без usage."""

    class ResponseWithoutUsage:
        pass

    try:
        extract_response_usage(
            ResponseWithoutUsage()
        )
    except ValueError as error:
        assert "не содержит usage" in str(
            error
        )

        print()
        print(
            "Missing usage blocking: OK"
        )
        return

    raise AssertionError(
        "Ответ без usage не был заблокирован."
    )


def test_unknown_model_blocking() -> None:
    """Проверяет неизвестный тариф."""

    try:
        get_model_pricing(
            "unknown-model"
        )
    except LookupError as error:
        assert "не настроен тариф" in str(
            error
        )

        print()
        print(
            "Unknown model blocking: OK"
        )
        return

    raise AssertionError(
        "Неизвестная модель "
        "не была заблокирована."
    )


def main() -> int:
    """Запускает тест usage и стоимости."""

    test_usage_extraction()
    test_cost_calculation()
    test_missing_usage_blocking()
    test_unknown_model_blocking()

    print()
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("OpenAI usage test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())