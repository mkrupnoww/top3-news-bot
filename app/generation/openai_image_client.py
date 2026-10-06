import base64
import binascii
from collections.abc import Mapping
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from app.generation.image_generator import (
    ImageModelRequest,
    ImageModelResponse,
    OpenAIImageUsage,
)


@runtime_checkable
class AsyncImagesResourceProtocol(
    Protocol
):
    """Минимальный контракт AsyncOpenAI.images."""

    async def generate(
        self,
        **kwargs: Any,
    ) -> Any:
        """Создаёт изображение."""

        ...


@runtime_checkable
class AsyncOpenAIImageClientProtocol(
    Protocol
):
    """Минимальный контракт AsyncOpenAI для Image API."""

    images: AsyncImagesResourceProtocol


def _get_value(
    value: Any,
    field_name: str,
) -> Any:
    """Читает поле из SDK-объекта или mapping."""

    if isinstance(value, Mapping):
        return value.get(field_name)

    return getattr(
        value,
        field_name,
        None,
    )


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательную строку."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть строкой."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_optional_text(
    value: Any,
    *,
    field_name: str,
) -> str | None:
    """Нормализует необязательное текстовое поле."""

    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть строкой "
            "или None."
        )

    normalized_value = value.strip()

    if not normalized_value:
        return None

    return normalized_value


def _normalize_nonnegative_integer(
    value: Any,
    *,
    field_name: str,
) -> int:
    """Проверяет неотрицательное целое значение."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} не может быть "
            "отрицательным."
        )

    return value


def _split_token_details(
    details: Any,
    *,
    total_tokens: int,
    text_field_name: str,
    image_field_name: str,
    default_text_tokens: int,
    default_image_tokens: int,
) -> tuple[int, int]:
    """Извлекает text/image token details."""

    if details is None:
        return (
            default_text_tokens,
            default_image_tokens,
        )

    raw_text_tokens = _get_value(
        details,
        "text_tokens",
    )

    raw_image_tokens = _get_value(
        details,
        "image_tokens",
    )

    if (
        raw_text_tokens is None
        and raw_image_tokens is None
    ):
        return (
            default_text_tokens,
            default_image_tokens,
        )

    if raw_text_tokens is None:
        image_tokens = (
            _normalize_nonnegative_integer(
                raw_image_tokens,
                field_name=image_field_name,
            )
        )

        text_tokens = (
            total_tokens - image_tokens
        )

        if text_tokens < 0:
            raise ValueError(
                f"{image_field_name} превышает "
                "общий token count."
            )

        return text_tokens, image_tokens

    if raw_image_tokens is None:
        text_tokens = (
            _normalize_nonnegative_integer(
                raw_text_tokens,
                field_name=text_field_name,
            )
        )

        image_tokens = (
            total_tokens - text_tokens
        )

        if image_tokens < 0:
            raise ValueError(
                f"{text_field_name} превышает "
                "общий token count."
            )

        return text_tokens, image_tokens

    text_tokens = (
        _normalize_nonnegative_integer(
            raw_text_tokens,
            field_name=text_field_name,
        )
    )

    image_tokens = (
        _normalize_nonnegative_integer(
            raw_image_tokens,
            field_name=image_field_name,
        )
    )

    if (
        text_tokens + image_tokens
        != total_tokens
    ):
        raise ValueError(
            f"{text_field_name} + "
            f"{image_field_name} не совпадают "
            "с общим token count."
        )

    return text_tokens, image_tokens


def _extract_usage(
    response: Any,
) -> OpenAIImageUsage | None:
    """Извлекает usage Image API при наличии."""

    raw_usage = _get_value(
        response,
        "usage",
    )

    if raw_usage is None:
        return None

    input_tokens = (
        _normalize_nonnegative_integer(
            _get_value(
                raw_usage,
                "input_tokens",
            ),
            field_name=(
                "usage.input_tokens"
            ),
        )
    )

    output_tokens = (
        _normalize_nonnegative_integer(
            _get_value(
                raw_usage,
                "output_tokens",
            ),
            field_name=(
                "usage.output_tokens"
            ),
        )
    )

    total_tokens = (
        _normalize_nonnegative_integer(
            _get_value(
                raw_usage,
                "total_tokens",
            ),
            field_name=(
                "usage.total_tokens"
            ),
        )
    )

    if (
        input_tokens + output_tokens
        != total_tokens
    ):
        raise ValueError(
            "usage.total_tokens не совпадает "
            "с input_tokens + output_tokens."
        )

    (
        input_text_tokens,
        input_image_tokens,
    ) = _split_token_details(
        _get_value(
            raw_usage,
            "input_tokens_details",
        ),
        total_tokens=input_tokens,
        text_field_name=(
            "usage.input_tokens_details."
            "text_tokens"
        ),
        image_field_name=(
            "usage.input_tokens_details."
            "image_tokens"
        ),
        default_text_tokens=input_tokens,
        default_image_tokens=0,
    )

    (
        output_text_tokens,
        output_image_tokens,
    ) = _split_token_details(
        _get_value(
            raw_usage,
            "output_tokens_details",
        ),
        total_tokens=output_tokens,
        text_field_name=(
            "usage.output_tokens_details."
            "text_tokens"
        ),
        image_field_name=(
            "usage.output_tokens_details."
            "image_tokens"
        ),
        default_text_tokens=0,
        default_image_tokens=output_tokens,
    )

    return OpenAIImageUsage(
        input_tokens=input_tokens,
        input_text_tokens=(
            input_text_tokens
        ),
        input_image_tokens=(
            input_image_tokens
        ),
        output_tokens=output_tokens,
        output_text_tokens=(
            output_text_tokens
        ),
        output_image_tokens=(
            output_image_tokens
        ),
        total_tokens=total_tokens,
    )


def _decode_single_image(
    response: Any,
) -> tuple[bytes, str | None]:
    """Извлекает единственное base64-изображение."""

    data = _get_value(
        response,
        "data",
    )

    if not isinstance(data, list):
        raise ValueError(
            "Image API response.data должен "
            "быть списком."
        )

    if len(data) != 1:
        raise ValueError(
            "Image API должен вернуть ровно "
            "одно изображение: "
            f"actual={len(data)}"
        )

    image_record = data[0]

    raw_b64_json = _get_value(
        image_record,
        "b64_json",
    )

    b64_json = _normalize_required_text(
        raw_b64_json,
        field_name="response.data[0].b64_json",
    )

    try:
        image_bytes = base64.b64decode(
            b64_json,
            validate=True,
        )
    except (
        binascii.Error,
        ValueError,
    ) as error:
        raise ValueError(
            "response.data[0].b64_json "
            "содержит некорректный base64."
        ) from error

    if not image_bytes:
        raise ValueError(
            "Image API вернул пустое изображение."
        )

    revised_prompt = (
        _normalize_optional_text(
            _get_value(
                image_record,
                "revised_prompt",
            ),
            field_name=(
                "response.data[0].revised_prompt"
            ),
        )
    )

    return image_bytes, revised_prompt


class OpenAIImagesGenerationClient:
    """Адаптер AsyncOpenAI Images API."""

    def __init__(
        self,
        *,
        client: AsyncOpenAIImageClientProtocol,
    ) -> None:
        if not isinstance(
            client,
            AsyncOpenAIImageClientProtocol,
        ):
            raise TypeError(
                "client не соответствует "
                "Image API интерфейсу AsyncOpenAI."
            )

        self._client = client

    async def create_image(
        self,
        request: ImageModelRequest,
    ) -> ImageModelResponse:
        """
        Выполняет один images.generate().

        Для GPT Image response_format не передаётся:
        модель всегда возвращает base64-изображение.
        """

        if not isinstance(
            request,
            ImageModelRequest,
        ):
            raise TypeError(
                "request должен быть "
                "ImageModelRequest."
            )

        if request.n != 1:
            raise ValueError(
                "Проект поддерживает ровно "
                "одно изображение за запрос."
            )

        response = (
            await self._client.images.generate(
                model=request.model,
                prompt=request.prompt,
                n=request.n,
                size=request.size,
                quality=request.quality,
                output_format=(
                    request.output_format
                ),
                background=request.background,
                moderation=request.moderation,
            )
        )

        image_bytes, revised_prompt = (
            _decode_single_image(response)
        )

        created = (
            _normalize_nonnegative_integer(
                _get_value(
                    response,
                    "created",
                ),
                field_name="response.created",
            )
        )

        usage = _extract_usage(
            response
        )

        return ImageModelResponse(
            image_bytes=image_bytes,
            created=created,
            output_format=(
                _normalize_optional_text(
                    _get_value(
                        response,
                        "output_format",
                    ),
                    field_name=(
                        "response.output_format"
                    ),
                )
            ),
            quality=(
                _normalize_optional_text(
                    _get_value(
                        response,
                        "quality",
                    ),
                    field_name=(
                        "response.quality"
                    ),
                )
            ),
            size=(
                _normalize_optional_text(
                    _get_value(
                        response,
                        "size",
                    ),
                    field_name="response.size",
                )
            ),
            background=(
                _normalize_optional_text(
                    _get_value(
                        response,
                        "background",
                    ),
                    field_name=(
                        "response.background"
                    ),
                )
            ),
            usage=usage,
            revised_prompt=revised_prompt,
        )