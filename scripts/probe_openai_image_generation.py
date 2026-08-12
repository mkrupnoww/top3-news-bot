import asyncio
from pathlib import Path
import tempfile

from app.config import get_settings
from app.generation.image_generator import (
    ImageGenerationNewsItem,
)
from app.generation.image_openai_usage import (
    build_openai_image_cost_payload,
)
from app.generation.image_storage import (
    store_png_image,
)
from app.generation.openai_image_factory import (
    create_openai_image_generation_runtime,
)


PROBE_IMAGE_GENERATION_ID = 1

EXPECTED_MODEL = "gpt-image-2"
EXPECTED_SIZE = "1024x1536"
EXPECTED_QUALITY = "medium"
EXPECTED_OUTPUT_FORMAT = "png"
EXPECTED_BACKGROUND = "opaque"
EXPECTED_MODERATION = "auto"


def build_probe_items() -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """
    Создаёт три синтетические киноновости.

    Они нужны только для проверки реального
    movie_news_image_v1 prompt и Image API.
    """

    return (
        ImageGenerationNewsItem(
            position=1,
            news_id=900001,
            title=(
                "Завершились съёмки "
                "фантастического триллера «Орбита»"
            ),
            summary=(
                "Съёмочная группа завершила основную "
                "работу над камерным научно-фантастическим "
                "триллером о международной космической "
                "экспедиции. Создатели готовят фильм "
                "к постпродакшену."
            ),
        ),
        ImageGenerationNewsItem(
            position=2,
            news_id=900002,
            title=(
                "Драма «Последний сеанс» "
                "лидирует в прокате"
            ),
            summary=(
                "Новая кинодрама вторую неделю подряд "
                "удерживает первое место в национальном "
                "прокате. Визуально новость должна "
                "считываться как история о кинотеатрах "
                "и зрительском интересе без выдуманных "
                "кассовых цифр."
            ),
        ),
        ImageGenerationNewsItem(
            position=3,
            news_id=900003,
            title=(
                "Актёр Алекс Моррис ведёт переговоры "
                "о роли в новом триллере"
            ),
            summary=(
                "Актёр пока только обсуждает участие "
                "в новом психологическом триллере. "
                "Сделка не подтверждена, поэтому "
                "визуализация должна сохранять "
                "неопределённость и не показывать "
                "участие как свершившийся факт."
            ),
        ),
    )


def validate_probe_configuration() -> None:
    """Блокирует неожиданный платный запрос."""

    settings = get_settings()

    if settings.openai_api_key is None:
        raise ValueError(
            "OPENAI_API_KEY не настроен."
        )

    if settings.openai_image_model != EXPECTED_MODEL:
        raise ValueError(
            "Live probe разрешён только для "
            f"{EXPECTED_MODEL}: actual="
            f"{settings.openai_image_model!r}"
        )

    if settings.openai_image_size != EXPECTED_SIZE:
        raise ValueError(
            "Live probe разрешён только для "
            f"{EXPECTED_SIZE}: actual="
            f"{settings.openai_image_size!r}"
        )

    if settings.openai_max_retries != 0:
        raise ValueError(
            "Для live probe требуется "
            "OPENAI_MAX_RETRIES=0, чтобы один запуск "
            "не породил автоматический повторный "
            "платный запрос."
        )


def validate_model_request(
    *,
    model: str,
    size: str,
    quality: str,
    output_format: str,
    background: str,
    moderation: str,
    n: int,
) -> None:
    """Проверяет точные параметры probe-request."""

    expected = {
        "model": EXPECTED_MODEL,
        "size": EXPECTED_SIZE,
        "quality": EXPECTED_QUALITY,
        "output_format": EXPECTED_OUTPUT_FORMAT,
        "background": EXPECTED_BACKGROUND,
        "moderation": EXPECTED_MODERATION,
        "n": 1,
    }

    actual = {
        "model": model,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "moderation": moderation,
        "n": n,
    }

    if actual != expected:
        raise ValueError(
            "Live probe ImageModelRequest отличается "
            "от ожидаемого: "
            f"actual={actual!r}, expected={expected!r}"
        )


async def close_sdk_client(
    sdk_client: object,
) -> None:
    """Закрывает AsyncOpenAI client."""

    close_method = getattr(
        sdk_client,
        "close",
        None,
    )

    if close_method is None:
        raise RuntimeError(
            "AsyncOpenAI client не содержит close()."
        )

    await close_method()


async def main() -> int:
    """
    Выполняет ровно один платный live Image API probe.

    PostgreSQL и Telegram не используются.
    Итоговый PNG хранится только во временной
    директории и удаляется автоматически.
    """

    validate_probe_configuration()

    settings = get_settings()
    items = build_probe_items()

    runtime = None
    temporary_png_path: Path | None = None

    print(
        "WARNING: this script performs exactly "
        "one paid OpenAI Image API request."
    )
    print(
        f"image_model={settings.openai_image_model}"
    )
    print(
        f"image_size={settings.openai_image_size}"
    )
    print(
        "quality="
        f"{EXPECTED_QUALITY}"
    )
    print(
        "output_format="
        f"{EXPECTED_OUTPUT_FORMAT}"
    )
    print(
        "background="
        f"{EXPECTED_BACKGROUND}"
    )
    print(
        "moderation="
        f"{EXPECTED_MODERATION}"
    )
    print(
        "max_retries="
        f"{settings.openai_max_retries}"
    )
    print(
        "database_enabled=false"
    )
    print(
        "telegram_enabled=false"
    )

    try:
        runtime = (
            create_openai_image_generation_runtime(
                settings
            )
        )

        model_request = (
            runtime.generator.build_request(
                items=items
            )
        )

        validate_model_request(
            model=model_request.model,
            size=model_request.size,
            quality=model_request.quality,
            output_format=(
                model_request.output_format
            ),
            background=model_request.background,
            moderation=model_request.moderation,
            n=model_request.n,
        )

        print(
            "prompt_version="
            f"{runtime.generator.metadata.prompt_version}"
        )
        print(
            "prompt_chars="
            f"{len(model_request.prompt)}"
        )
        print(
            "openai_image_request_started=true"
        )

        generation = (
            await runtime.generator.generate(
                items=items
            )
        )

        if (
            generation.model_request
            != model_request
        ):
            raise RuntimeError(
                "Фактический request генератора "
                "не совпал с предварительно "
                "проверенным request."
            )

        response = generation.model_response

        if (
            response.output_format is not None
            and response.output_format
            != EXPECTED_OUTPUT_FORMAT
        ):
            raise ValueError(
                "Image API вернул неожиданный "
                "output_format: "
                f"{response.output_format!r}"
            )

        if (
            response.size is not None
            and response.size != EXPECTED_SIZE
        ):
            raise ValueError(
                "Image API вернул неожиданный size: "
                f"{response.size!r}"
            )

        if (
            response.quality is not None
            and response.quality
            != EXPECTED_QUALITY
        ):
            raise ValueError(
                "Image API вернул неожиданное "
                "quality: "
                f"{response.quality!r}"
            )

        if (
            response.background is not None
            and response.background
            != EXPECTED_BACKGROUND
        ):
            raise ValueError(
                "Image API вернул неожиданный "
                "background: "
                f"{response.background!r}"
            )

        with tempfile.TemporaryDirectory(
            prefix="top3-openai-image-live-probe-"
        ) as temporary_directory:
            output_dir = (
                Path(temporary_directory)
                / "generated"
            )

            artifact = store_png_image(
                response.image_bytes,
                image_generation_id=(
                    PROBE_IMAGE_GENERATION_ID
                ),
                expected_size=(
                    settings.openai_image_size
                ),
                output_dir=output_dir,
            )

            temporary_png_path = (
                artifact.file_path
            )

            if not temporary_png_path.exists():
                raise AssertionError(
                    "PNG не найден после storage."
                )

            if artifact.width != 1024:
                raise AssertionError(
                    "PNG width != 1024."
                )

            if artifact.height != 1536:
                raise AssertionError(
                    "PNG height != 1536."
                )

            print(
                "openai_image_request_completed=true"
            )
            print(
                "base64_decoded=true"
            )
            print(
                f"response_created={response.created}"
            )
            print(
                "response_output_format="
                f"{response.output_format}"
            )
            print(
                f"response_quality={response.quality}"
            )
            print(
                f"response_size={response.size}"
            )
            print(
                "response_background="
                f"{response.background}"
            )
            print(
                "revised_prompt_present="
                f"{response.revised_prompt is not None}"
            )
            print(
                "image_bytes="
                f"{len(response.image_bytes)}"
            )
            print(
                f"png_width={artifact.width}"
            )
            print(
                f"png_height={artifact.height}"
            )
            print(
                "png_sha256="
                f"{artifact.image_sha256}"
            )
            print(
                "temporary_png_stored=true"
            )

            usage = response.usage

            if usage is None:
                print(
                    "usage_present=false"
                )
                print(
                    "cost_estimate_available=false"
                )
            else:
                cost_payload = (
                    build_openai_image_cost_payload(
                        model_request.model,
                        usage,
                    )
                )

                print(
                    "usage_present=true"
                )
                print(
                    "input_tokens="
                    f"{usage.input_tokens}"
                )
                print(
                    "input_text_tokens="
                    f"{usage.input_text_tokens}"
                )
                print(
                    "input_image_tokens="
                    f"{usage.input_image_tokens}"
                )
                print(
                    "output_tokens="
                    f"{usage.output_tokens}"
                )
                print(
                    "output_text_tokens="
                    f"{usage.output_text_tokens}"
                )
                print(
                    "output_image_tokens="
                    f"{usage.output_image_tokens}"
                )
                print(
                    "total_tokens="
                    f"{usage.total_tokens}"
                )
                print(
                    "pricing_version="
                    f"{cost_payload['pricing_version']}"
                )
                print(
                    "pricing_basis="
                    f"{cost_payload['pricing_basis']}"
                )
                print(
                    "total_cost_usd="
                    f"{cost_payload['total_cost_usd']}"
                )

        if (
            temporary_png_path is not None
            and temporary_png_path.exists()
        ):
            raise AssertionError(
                "TemporaryDirectory не удалил PNG."
            )

    finally:
        if runtime is not None:
            await close_sdk_client(
                runtime.sdk_client
            )

    print()
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print("Permanent PNG files created: 0")
    print("temporary_png_deleted=true")
    print("OpenAI Image live probe: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )