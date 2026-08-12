from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile

from PIL import Image

from app.generation.image_storage import (
    store_png_image,
)


TEST_WIDTH = 64
TEST_HEIGHT = 96
TEST_SIZE = "64x96"


def build_png_bytes(
    *,
    width: int = TEST_WIDTH,
    height: int = TEST_HEIGHT,
    value: int = 32,
) -> bytes:
    """Создаёт синтетический PNG в памяти."""

    buffer = BytesIO()

    image = Image.new(
        "RGB",
        (width, height),
        (value, value, value),
    )

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def build_jpeg_bytes() -> bytes:
    """Создаёт синтетический JPEG в памяти."""

    buffer = BytesIO()

    image = Image.new(
        "RGB",
        (TEST_WIDTH, TEST_HEIGHT),
        (64, 64, 64),
    )

    image.save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


def assert_no_temporary_files(
    output_dir: Path,
) -> None:
    """Проверяет отсутствие оставшихся tmp-файлов."""

    temporary_files = list(
        output_dir.glob(
            ".image_generation_*.tmp"
        )
    )

    assert temporary_files == []


def test_successful_storage(
    output_dir: Path,
) -> Path:
    """Проверяет успешную атомарную запись PNG."""

    image_bytes = build_png_bytes()

    result = store_png_image(
        image_bytes,
        image_generation_id=101,
        expected_size=TEST_SIZE,
        output_dir=output_dir,
    )

    expected_sha256 = sha256(
        image_bytes
    ).hexdigest()

    assert result.image_generation_id == 101
    assert result.image_sha256 == expected_sha256
    assert result.width == TEST_WIDTH
    assert result.height == TEST_HEIGHT
    assert result.byte_count == len(image_bytes)
    assert result.already_stored is False
    assert result.file_path.exists()
    assert result.file_path.is_file()
    assert result.file_path.read_bytes() == image_bytes
    assert result.image_path == str(
        result.file_path
    )
    assert (
        expected_sha256
        in result.file_path.name
    )
    assert (
        "image_generation_101_"
        in result.file_path.name
    )

    with Image.open(
        result.file_path
    ) as stored_image:
        assert stored_image.format == "PNG"
        assert stored_image.size == (
            TEST_WIDTH,
            TEST_HEIGHT,
        )
        stored_image.verify()

    assert_no_temporary_files(
        output_dir
    )

    print("Successful PNG storage: OK")
    print(
        f"image_path={result.image_path}"
    )
    print(
        f"image_sha256={result.image_sha256}"
    )
    print(
        "image_size="
        f"{result.width}x{result.height}"
    )
    print("already_stored=false")

    return result.file_path


def test_idempotent_storage(
    output_dir: Path,
    *,
    expected_path: Path,
) -> None:
    """Проверяет повторное сохранение тех же байтов."""

    image_bytes = build_png_bytes()

    result = store_png_image(
        image_bytes,
        image_generation_id=101,
        expected_size=TEST_SIZE,
        output_dir=output_dir,
    )

    assert result.file_path == expected_path
    assert result.already_stored is True
    assert result.file_path.read_bytes() == (
        image_bytes
    )

    assert_no_temporary_files(
        output_dir
    )

    print()
    print("Repeated PNG storage: OK")
    print("same_path=true")
    print("already_stored=true")


def test_different_artifact_is_immutable(
    output_dir: Path,
    *,
    first_path: Path,
) -> None:
    """
    Проверяет, что другой результат не перезаписывает
    уже сохранённый immutable PNG.
    """

    first_bytes = first_path.read_bytes()
    second_bytes = build_png_bytes(
        value=96
    )

    second_result = store_png_image(
        second_bytes,
        image_generation_id=101,
        expected_size=TEST_SIZE,
        output_dir=output_dir,
    )

    assert second_result.file_path != first_path
    assert first_path.read_bytes() == first_bytes
    assert (
        second_result.file_path.read_bytes()
        == second_bytes
    )

    assert_no_temporary_files(
        output_dir
    )

    print()
    print("Immutable artifact naming: OK")
    print("different_bytes_use_different_path=true")
    print("first_artifact_preserved=true")


def test_wrong_format_blocked(
    output_dir: Path,
) -> None:
    """Проверяет блокировку JPEG."""

    before = set(
        output_dir.iterdir()
    )

    try:
        store_png_image(
            build_jpeg_bytes(),
            image_generation_id=102,
            expected_size=TEST_SIZE,
            output_dir=output_dir,
        )
    except ValueError as error:
        if "Ожидался PNG" not in str(error):
            raise
    else:
        raise AssertionError(
            "JPEG не был заблокирован."
        )

    after = set(
        output_dir.iterdir()
    )

    assert after == before
    assert_no_temporary_files(
        output_dir
    )

    print()
    print("Wrong image format blocking: OK")
    print("jpeg_blocked=true")
    print("artifact_created=false")


def test_corrupted_png_blocked(
    output_dir: Path,
) -> None:
    """Проверяет блокировку повреждённых байтов."""

    before = set(
        output_dir.iterdir()
    )

    try:
        store_png_image(
            b"\x89PNG\r\n\x1a\nbroken",
            image_generation_id=103,
            expected_size=TEST_SIZE,
            output_dir=output_dir,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Повреждённый PNG не был заблокирован."
        )

    after = set(
        output_dir.iterdir()
    )

    assert after == before
    assert_no_temporary_files(
        output_dir
    )

    print()
    print("Corrupted PNG blocking: OK")
    print("corrupted_png_blocked=true")
    print("artifact_created=false")


def test_wrong_dimensions_blocked(
    output_dir: Path,
) -> None:
    """Проверяет точное совпадение размеров."""

    before = set(
        output_dir.iterdir()
    )

    try:
        store_png_image(
            build_png_bytes(
                width=66,
                height=99,
            ),
            image_generation_id=104,
            expected_size=TEST_SIZE,
            output_dir=output_dir,
        )
    except ValueError as error:
        if "Фактический размер PNG" not in str(error):
            raise
    else:
        raise AssertionError(
            "PNG неверного размера "
            "не был заблокирован."
        )

    after = set(
        output_dir.iterdir()
    )

    assert after == before
    assert_no_temporary_files(
        output_dir
    )

    print()
    print("Wrong PNG dimensions blocking: OK")
    print("wrong_dimensions_blocked=true")
    print("artifact_created=false")


def test_invalid_expected_size_blocked(
    output_dir: Path,
) -> None:
    """Проверяет общий проектный валидатор 2:3."""

    before = set(
        output_dir.iterdir()
    )

    try:
        store_png_image(
            build_png_bytes(),
            image_generation_id=105,
            expected_size="64x95",
            output_dir=output_dir,
        )
    except ValueError as error:
        if "соотношение сторон 2:3" not in str(
            error
        ):
            raise
    else:
        raise AssertionError(
            "Размер не 2:3 не был заблокирован."
        )

    after = set(
        output_dir.iterdir()
    )

    assert after == before

    print()
    print("Invalid expected size blocking: OK")
    print("non_2_to_3_size_blocked=true")
    print("artifact_created=false")


def test_invalid_input_blocked(
    output_dir: Path,
) -> None:
    """Проверяет базовую валидацию входов."""

    try:
        store_png_image(
            b"",
            image_generation_id=106,
            expected_size=TEST_SIZE,
            output_dir=output_dir,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Пустые image_bytes "
            "не были заблокированы."
        )

    try:
        store_png_image(
            build_png_bytes(),
            image_generation_id=True,
            expected_size=TEST_SIZE,
            output_dir=output_dir,
        )
    except TypeError:
        pass
    else:
        raise AssertionError(
            "bool image_generation_id "
            "не был заблокирован."
        )

    print()
    print("Invalid input blocking: OK")
    print("empty_bytes_blocked=true")
    print("bool_image_generation_id_blocked=true")


def main() -> int:
    """Запускает локальный storage-тест."""

    with tempfile.TemporaryDirectory(
        prefix="top3-image-storage-test-"
    ) as temporary_directory:
        output_dir = (
            Path(temporary_directory)
            / "generated"
        )

        successful_path = (
            test_successful_storage(
                output_dir
            )
        )

        test_idempotent_storage(
            output_dir,
            expected_path=successful_path,
        )

        test_different_artifact_is_immutable(
            output_dir,
            first_path=successful_path,
        )

        test_wrong_format_blocked(
            output_dir
        )

        test_corrupted_png_blocked(
            output_dir
        )

        test_wrong_dimensions_blocked(
            output_dir
        )

        test_invalid_expected_size_blocked(
            output_dir
        )

        test_invalid_input_blocked(
            output_dir
        )

    print()
    print("OpenAI Image requests: not performed")
    print("Database changes: none")
    print("Permanent PNG files created: 0")
    print("Image storage test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )