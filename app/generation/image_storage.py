from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import tempfile

from PIL import Image, UnidentifiedImageError

from app.generation.image_generator import (
    _normalize_image_size,
)


DEFAULT_IMAGE_OUTPUT_DIR = Path(
    "data/images/generated"
)

_IMAGE_FORMAT = "PNG"
_IMAGE_EXTENSION = ".png"


@dataclass(frozen=True, slots=True)
class StoredImageArtifact:
    """Сохранённый неизменяемый PNG-артефакт."""

    image_generation_id: int
    image_path: str
    file_path: Path
    image_sha256: str
    width: int
    height: int
    byte_count: int
    already_stored: bool


def _normalize_positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительный целочисленный ID."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} должен быть больше нуля."
        )

    return value


def _normalize_image_bytes(
    image_bytes: bytes,
) -> bytes:
    """Проверяет наличие бинарного изображения."""

    if not isinstance(image_bytes, bytes):
        raise TypeError(
            "image_bytes должен быть bytes."
        )

    if not image_bytes:
        raise ValueError(
            "image_bytes не может быть пустым."
        )

    return image_bytes


def _parse_expected_size(
    expected_size: str,
) -> tuple[str, int, int]:
    """
    Проверяет expected_size тем же валидатором,
    который использует image_generator.
    """

    normalized_size = _normalize_image_size(
        expected_size
    )

    width_text, separator, height_text = (
        normalized_size.partition("x")
    )

    if separator != "x":
        raise RuntimeError(
            "Нормализованный размер изображения "
            "не содержит разделитель x."
        )

    return (
        normalized_size,
        int(width_text),
        int(height_text),
    )


def _validate_png_bytes(
    image_bytes: bytes,
    *,
    expected_width: int,
    expected_height: int,
) -> tuple[int, int]:
    """Проверяет PNG и его фактический размер."""

    try:
        with Image.open(
            BytesIO(image_bytes)
        ) as image:
            detected_format = image.format
            width, height = image.size
            frame_count = int(
                getattr(image, "n_frames", 1)
            )

            image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as error:
        raise ValueError(
            "image_bytes не содержит "
            "корректный PNG."
        ) from error

    if detected_format != _IMAGE_FORMAT:
        raise ValueError(
            "Ожидался PNG, получен формат: "
            f"{detected_format!r}."
        )

    if frame_count != 1:
        raise ValueError(
            "Ожидается статический PNG "
            "с одним кадром."
        )

    if (
        width != expected_width
        or height != expected_height
    ):
        raise ValueError(
            "Фактический размер PNG "
            "не совпадает с ожидаемым: "
            f"expected={expected_width}x"
            f"{expected_height}, "
            f"actual={width}x{height}."
        )

    return width, height


def _build_final_path(
    *,
    output_dir: Path,
    image_generation_id: int,
    image_sha256: str,
) -> Path:
    """Строит неизменяемое имя артефакта."""

    filename = (
        "image_generation_"
        f"{image_generation_id}_"
        f"{image_sha256}"
        f"{_IMAGE_EXTENSION}"
    )

    return output_dir / filename


def _assert_existing_artifact_matches(
    final_path: Path,
    *,
    image_bytes: bytes,
    image_sha256: str,
) -> bool:
    """Проверяет уже существующий immutable-файл."""

    if not final_path.exists():
        return False

    if final_path.is_symlink():
        raise RuntimeError(
            "Путь готового изображения "
            "не должен быть символической ссылкой: "
            f"{final_path}"
        )

    if not final_path.is_file():
        raise RuntimeError(
            "Путь готового изображения "
            "существует, но не является файлом: "
            f"{final_path}"
        )

    existing_bytes = final_path.read_bytes()
    existing_sha256 = sha256(
        existing_bytes
    ).hexdigest()

    if existing_sha256 != image_sha256:
        raise RuntimeError(
            "Существующий immutable PNG "
            "не совпадает с ожидаемым SHA-256: "
            f"path={final_path}, "
            f"expected={image_sha256}, "
            f"actual={existing_sha256}"
        )

    if existing_bytes != image_bytes:
        raise RuntimeError(
            "Существующий immutable PNG "
            "имеет ожидаемый SHA-256, "
            "но бинарные данные отличаются."
        )

    return True


def _fsync_directory(
    directory: Path,
) -> None:
    """Синхронизирует каталог после atomic replace."""

    directory_flags = os.O_RDONLY

    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY

    directory_fd = os.open(
        directory,
        directory_flags,
    )

    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def store_png_image(
    image_bytes: bytes,
    *,
    image_generation_id: int,
    expected_size: str,
    output_dir: str | Path = (
        DEFAULT_IMAGE_OUTPUT_DIR
    ),
) -> StoredImageArtifact:
    """
    Проверяет и атомарно сохраняет PNG.

    Имя файла включает image_generation_id и полный
    SHA-256, поэтому разные бинарные результаты не
    перезаписывают один immutable-артефакт.
    """

    normalized_image_generation_id = (
        _normalize_positive_integer(
            image_generation_id,
            field_name="image_generation_id",
        )
    )

    normalized_image_bytes = (
        _normalize_image_bytes(
            image_bytes
        )
    )

    (
        _normalized_size,
        expected_width,
        expected_height,
    ) = _parse_expected_size(
        expected_size
    )

    width, height = _validate_png_bytes(
        normalized_image_bytes,
        expected_width=expected_width,
        expected_height=expected_height,
    )

    image_sha256 = sha256(
        normalized_image_bytes
    ).hexdigest()

    normalized_output_dir = Path(
        output_dir
    )

    normalized_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not normalized_output_dir.is_dir():
        raise RuntimeError(
            "Каталог хранения изображений "
            "не является директорией: "
            f"{normalized_output_dir}"
        )

    final_path = _build_final_path(
        output_dir=normalized_output_dir,
        image_generation_id=(
            normalized_image_generation_id
        ),
        image_sha256=image_sha256,
    )

    already_stored = (
        _assert_existing_artifact_matches(
            final_path,
            image_bytes=normalized_image_bytes,
            image_sha256=image_sha256,
        )
    )

    if not already_stored:
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=normalized_output_dir,
                prefix=(
                    ".image_generation_"
                    f"{normalized_image_generation_id}_"
                ),
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(
                    temporary_file.name
                )

                temporary_file.write(
                    normalized_image_bytes
                )
                temporary_file.flush()
                os.fsync(
                    temporary_file.fileno()
                )

            os.chmod(
                temporary_path,
                0o640,
            )

            if final_path.exists():
                already_stored = (
                    _assert_existing_artifact_matches(
                        final_path,
                        image_bytes=(
                            normalized_image_bytes
                        ),
                        image_sha256=(
                            image_sha256
                        ),
                    )
                )
            else:
                os.replace(
                    temporary_path,
                    final_path,
                )
                temporary_path = None

                _fsync_directory(
                    normalized_output_dir
                )
        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    return StoredImageArtifact(
        image_generation_id=(
            normalized_image_generation_id
        ),
        image_path=str(final_path),
        file_path=final_path,
        image_sha256=image_sha256,
        width=width,
        height=height,
        byte_count=len(
            normalized_image_bytes
        ),
        already_stored=already_stored,
    )