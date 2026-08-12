from decimal import Decimal

from app.generation.image_generator import (
    OpenAIImageUsage,
)
from app.generation.image_openai_usage import (
    GPT_IMAGE_2_PRICING,
    GPT_IMAGE_2_PRICING_VERSION,
    IMAGE_PRICING_BASIS,
    OpenAIImageModelPricing,
    build_openai_image_cost_payload,
    calculate_openai_image_cost,
    get_image_model_pricing,
)


def test_pricing_configuration() -> None:
    """Проверяет зафиксированный gpt-image-2 тариф."""

    pricing = get_image_model_pricing(
        "  gpt-image-2  "
    )

    assert pricing is GPT_IMAGE_2_PRICING

    assert pricing.model_name == "gpt-image-2"

    assert (
        pricing.text_input_usd_per_million
        == Decimal("5.00")
    )

    assert (
        pricing
        .text_cached_input_usd_per_million
        == Decimal("1.25")
    )

    assert (
        pricing.image_input_usd_per_million
        == Decimal("8.00")
    )

    assert (
        pricing
        .image_cached_input_usd_per_million
        == Decimal("2.00")
    )

    assert (
        pricing.image_output_usd_per_million
        == Decimal("30.00")
    )

    assert (
        pricing.pricing_version
        == GPT_IMAGE_2_PRICING_VERSION
    )

    print("GPT Image 2 pricing configuration: OK")
    print(
        "pricing_version="
        f"{pricing.pricing_version}"
    )
    print(
        "text_input_usd_per_million="
        f"{pricing.text_input_usd_per_million}"
    )
    print(
        "image_input_usd_per_million="
        f"{pricing.image_input_usd_per_million}"
    )
    print(
        "image_output_usd_per_million="
        f"{pricing.image_output_usd_per_million}"
    )


def test_generation_cost() -> None:
    """Проверяет text input + image output."""

    usage = OpenAIImageUsage(
        input_tokens=120,
        input_text_tokens=120,
        input_image_tokens=0,
        output_tokens=900,
        output_text_tokens=0,
        output_image_tokens=900,
        total_tokens=1020,
    )

    cost = calculate_openai_image_cost(
        usage,
        GPT_IMAGE_2_PRICING,
    )

    assert (
        cost.input_text_cost_usd
        == Decimal("0.00060000")
    )

    assert (
        cost.input_image_cost_usd
        == Decimal("0.00000000")
    )

    assert (
        cost.output_text_cost_usd
        == Decimal("0.00000000")
    )

    assert (
        cost.output_image_cost_usd
        == Decimal("0.02700000")
    )

    assert (
        cost.total_cost_usd
        == Decimal("0.02760000")
    )

    assert (
        cost.pricing_basis
        == IMAGE_PRICING_BASIS
    )

    print()
    print("Image generation cost: OK")
    print(
        "input_text_cost_usd="
        f"{cost.input_text_cost_usd}"
    )
    print(
        "output_image_cost_usd="
        f"{cost.output_image_cost_usd}"
    )
    print(
        "total_cost_usd="
        f"{cost.total_cost_usd}"
    )


def test_multimodal_edit_cost() -> None:
    """Проверяет text+image input и image output."""

    usage = OpenAIImageUsage(
        input_tokens=3000,
        input_text_tokens=1000,
        input_image_tokens=2000,
        output_tokens=3000,
        output_text_tokens=0,
        output_image_tokens=3000,
        total_tokens=6000,
    )

    cost = calculate_openai_image_cost(
        usage,
        GPT_IMAGE_2_PRICING,
    )

    assert (
        cost.input_text_cost_usd
        == Decimal("0.00500000")
    )

    assert (
        cost.input_image_cost_usd
        == Decimal("0.01600000")
    )

    assert (
        cost.output_image_cost_usd
        == Decimal("0.09000000")
    )

    assert (
        cost.total_cost_usd
        == Decimal("0.11100000")
    )

    print()
    print("Multimodal image cost: OK")
    print(
        "input_text_cost_usd="
        f"{cost.input_text_cost_usd}"
    )
    print(
        "input_image_cost_usd="
        f"{cost.input_image_cost_usd}"
    )
    print(
        "output_image_cost_usd="
        f"{cost.output_image_cost_usd}"
    )
    print(
        "total_cost_usd="
        f"{cost.total_cost_usd}"
    )


def test_json_cost_payload() -> None:
    """Проверяет DB-совместимый cost payload."""

    usage = OpenAIImageUsage(
        input_tokens=120,
        input_text_tokens=120,
        input_image_tokens=0,
        output_tokens=900,
        output_text_tokens=0,
        output_image_tokens=900,
        total_tokens=1020,
    )

    payload = (
        build_openai_image_cost_payload(
            "gpt-image-2",
            usage,
        )
    )

    expected_payload = {
        "model_name": "gpt-image-2",
        "pricing_version": (
            GPT_IMAGE_2_PRICING_VERSION
        ),
        "pricing_basis": (
            IMAGE_PRICING_BASIS
        ),
        "input_text_cost_usd": (
            "0.00060000"
        ),
        "input_image_cost_usd": (
            "0.00000000"
        ),
        "output_text_cost_usd": (
            "0.00000000"
        ),
        "output_image_cost_usd": (
            "0.02700000"
        ),
        "total_cost_usd": (
            "0.02760000"
        ),
    }

    assert payload == expected_payload

    assert all(
        isinstance(value, str)
        for value in payload.values()
    )

    print()
    print("Image cost JSON payload: OK")
    print("decimal_serialized_as_string=true")
    print("db_json_compatible=true")


def test_unknown_model_blocking() -> None:
    """Проверяет неизвестный image pricing."""

    usage = OpenAIImageUsage(
        input_tokens=1,
        input_text_tokens=1,
        input_image_tokens=0,
        output_tokens=1,
        output_text_tokens=0,
        output_image_tokens=1,
        total_tokens=2,
    )

    try:
        build_openai_image_cost_payload(
            "unknown-image-model",
            usage,
        )
    except LookupError as error:
        assert (
            "не настроен тариф"
            in str(error)
        )

        print()
        print("Unknown image model blocking: OK")
        return

    raise AssertionError(
        "Неизвестный image model "
        "не был заблокирован."
    )


def test_output_text_blocking() -> None:
    """Запрещает неподдержанный text output."""

    usage = OpenAIImageUsage(
        input_tokens=1,
        input_text_tokens=1,
        input_image_tokens=0,
        output_tokens=2,
        output_text_tokens=1,
        output_image_tokens=1,
        total_tokens=3,
    )

    try:
        calculate_openai_image_cost(
            usage,
            GPT_IMAGE_2_PRICING,
        )
    except ValueError as error:
        assert (
            "output_text_tokens должен быть 0"
            in str(error)
        )

        print()
        print("Image text output blocking: OK")
        return

    raise AssertionError(
        "Неподдержанный text output "
        "не был заблокирован."
    )


def test_invalid_pricing_blocking() -> None:
    """Проверяет защиту от ошибочного тарифа."""

    try:
        OpenAIImageModelPricing(
            model_name="bad-image-model",
            text_input_usd_per_million=(
                Decimal("-1")
            ),
            text_cached_input_usd_per_million=(
                Decimal("0")
            ),
            image_input_usd_per_million=(
                Decimal("0")
            ),
            image_cached_input_usd_per_million=(
                Decimal("0")
            ),
            image_output_usd_per_million=(
                Decimal("0")
            ),
            pricing_version="test",
        )
    except ValueError as error:
        assert "отрицательным" in str(
            error
        )

        print()
        print("Invalid image pricing blocking: OK")
        return

    raise AssertionError(
        "Отрицательный тариф "
        "не был заблокирован."
    )


def main() -> int:
    """Запускает unit test image pricing."""

    test_pricing_configuration()
    test_generation_cost()
    test_multimodal_edit_cost()
    test_json_cost_payload()
    test_unknown_model_blocking()
    test_output_text_blocking()
    test_invalid_pricing_blocking()

    print()
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("PNG files created: 0")
    print("Telegram publication: not performed")
    print("OpenAI image usage/cost test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )