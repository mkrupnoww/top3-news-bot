import asyncio
from io import BytesIO
from pathlib import Path
import tempfile

import asyncpg
from PIL import Image

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.image_generator import (
    ImageModelRequest,
    ImageModelResponse,
    OpenAIMovieNewsImageGenerator,
)
from app.generation.openai_image_pipeline import (
    run_reserved_openai_image_generation,
)
from scripts import (
    test_event_ranking_run_completion
    as ranking_fixture,
)
from scripts import (
    test_image_generation_reservation
    as reservation_fixture,
)


TEST_IMAGE_MODEL = (
    "synthetic-image-pipeline-model"
)

TEST_IMAGE_SIZE = "64x96"

SYNTHETIC_API_ERROR = (
    "Synthetic Image API pipeline failure."
)


def build_png_bytes(
    *,
    width: int = 64,
    height: int = 96,
    value: int = 72,
) -> bytes:
    """Создаёт настоящий синтетический PNG."""

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


class SyntheticImageClient:
    """Fake Image API с настоящими PNG bytes."""

    def __init__(
        self,
        *,
        image_bytes: bytes,
        fail: bool = False,
    ) -> None:
        self.image_bytes = image_bytes
        self.fail = fail
        self.requests: list[
            ImageModelRequest
        ] = []

    async def create_image(
        self,
        request: ImageModelRequest,
    ) -> ImageModelResponse:
        """Возвращает synthetic response без сети."""

        self.requests.append(
            request
        )

        if self.fail:
            raise RuntimeError(
                SYNTHETIC_API_ERROR
            )

        return ImageModelResponse(
            image_bytes=self.image_bytes,
            created=1_800_000_100,
            output_format="png",
            quality=request.quality,
            size=request.size,
            background=request.background,
            usage=None,
            revised_prompt=None,
        )


def build_generator(
    client: SyntheticImageClient,
) -> OpenAIMovieNewsImageGenerator:
    """Создаёт generator для pipeline-теста."""

    return OpenAIMovieNewsImageGenerator(
        client=client,
        model_name=TEST_IMAGE_MODEL,
        size=TEST_IMAGE_SIZE,
    )


async def load_image_state(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
    generated_post_id: int,
) -> asyncpg.Record:
    """Читает состояние batch/post/image request."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_status,

                gp.post_status,
                gp.post_text,
                gp.image_path,
                gp.image_sha256,
                gp.image_prompt,
                gp.image_model_name,
                gp.image_prompt_version,

                igr.image_generation_id,
                igr.image_status,
                igr.request_kind,
                igr.response_metadata,
                igr.openai_usage,
                igr.openai_cost,
                igr.error_type,
                igr.error_message,
                igr.completed_at,
                igr.failed_at,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.generated_posts AS gp2
                    WHERE gp2.batch_id = b.batch_id
                ) AS generated_post_count,

                (
                    SELECT COUNT(*)::integer
                    FROM top3_news.image_generation_requests AS igr2
                    WHERE igr2.batch_id = b.batch_id
                ) AS image_request_count

            FROM top3_news.publication_batches AS b
            JOIN top3_news.generated_posts AS gp
              ON gp.generated_post_id = $2
            LEFT JOIN top3_news.image_generation_requests AS igr
              ON igr.batch_id = b.batch_id
             AND igr.generated_post_id =
                 gp.generated_post_id
            WHERE b.batch_id = $1
            ORDER BY igr.image_generation_id DESC
            LIMIT 1
            """,
            batch_id,
            generated_post_id,
        )

    if record is None:
        raise AssertionError(
            "Не найдено pipeline test state."
        )

    return record


async def test_initial_success_and_duplicate(
    pool: asyncpg.Pool,
    *,
    selection,
    telegram_chat_id: int,
    created_batch_ids: set[int],
    output_dir: Path,
) -> None:
    """Проверяет полный initial image pipeline."""

    (
        batch_id,
        generated_post_id,
    ) = await (
        reservation_fixture
        .create_test_batch_and_post(
            pool,
            selection=selection,
            telegram_chat_id=telegram_chat_id,
            existing_image=False,
            test_name=(
                "openai_image_pipeline_initial_success"
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )
    )

    client = SyntheticImageClient(
        image_bytes=build_png_bytes()
    )

    generator = build_generator(
        client
    )

    result = await (
        run_reserved_openai_image_generation(
            pool,
            generator=generator,
            selection=selection,
            batch_id=batch_id,
            generated_post_id=(
                generated_post_id
            ),
            output_dir=output_dir,
        )
    )

    assert result.completed is True
    assert result.model_called is True
    assert result.request_kind == "initial"
    assert result.image_status == "completed"
    assert len(client.requests) == 1

    if result.artifact is None:
        raise AssertionError(
            "Pipeline не вернул StoredImageArtifact."
        )

    artifact = result.artifact

    assert artifact.file_path.exists()
    assert artifact.file_path.is_file()
    assert (
        artifact.file_path.read_bytes()
        == client.image_bytes
    )
    assert artifact.width == 64
    assert artifact.height == 96
    assert artifact.already_stored is False

    state = await load_image_state(
        pool,
        batch_id=batch_id,
        generated_post_id=generated_post_id,
    )

    original_post_text = state["post_text"]

    assert state["batch_status"] == "awaiting_review"
    assert state["post_status"] == "awaiting_review"
    assert state["image_status"] == "completed"
    assert state["request_kind"] == "initial"
    assert state["image_path"] == artifact.image_path
    assert (
        state["image_sha256"]
        == artifact.image_sha256
    )
    assert (
        state["image_prompt"]
        == result.model_request.prompt
    )
    assert (
        state["image_model_name"]
        == TEST_IMAGE_MODEL
    )
    assert (
        state["image_prompt_version"]
        == "movie_news_image_v1"
    )
    assert state["openai_usage"] is None
    assert state["openai_cost"] is None
    assert state["error_type"] is None
    assert state["error_message"] is None
    assert state["completed_at"] is not None
    assert state["failed_at"] is None
    assert state["generated_post_count"] == 1
    assert state["image_request_count"] == 1

    repeated = await (
        run_reserved_openai_image_generation(
            pool,
            generator=generator,
            selection=selection,
            batch_id=batch_id,
            generated_post_id=(
                generated_post_id
            ),
            output_dir=output_dir,
        )
    )

    assert repeated.completed is True
    assert repeated.model_called is False
    assert (
        repeated.duplicate_request_blocked
        is True
    )
    assert len(client.requests) == 1

    repeated_state = await load_image_state(
        pool,
        batch_id=batch_id,
        generated_post_id=generated_post_id,
    )

    assert (
        repeated_state["post_text"]
        == original_post_text
    )
    assert (
        repeated_state["image_path"]
        == artifact.image_path
    )
    assert (
        repeated_state["image_sha256"]
        == artifact.image_sha256
    )
    assert (
        repeated_state["image_request_count"]
        == 1
    )

    print("Initial image pipeline: OK")
    print(
        "image_generation_id="
        f"{result.image_generation_id}"
    )
    print(f"batch_id={batch_id}")
    print(
        "generated_post_id="
        f"{generated_post_id}"
    )
    print("fake_image_api_calls=1")
    print("png_validated_and_stored=true")
    print("database_completion=true")
    print("text_post_preserved=true")
    print("duplicate_api_call_blocked=true")


async def test_initial_api_failure(
    pool: asyncpg.Pool,
    *,
    selection,
    telegram_chat_id: int,
    created_batch_ids: set[int],
    output_dir: Path,
) -> None:
    """Проверяет API failure без повреждения поста."""

    (
        batch_id,
        generated_post_id,
    ) = await (
        reservation_fixture
        .create_test_batch_and_post(
            pool,
            selection=selection,
            telegram_chat_id=telegram_chat_id,
            existing_image=False,
            test_name=(
                "openai_image_pipeline_api_failure"
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )
    )

    client = SyntheticImageClient(
        image_bytes=build_png_bytes(),
        fail=True,
    )

    generator = build_generator(
        client
    )

    try:
        await run_reserved_openai_image_generation(
            pool,
            generator=generator,
            selection=selection,
            batch_id=batch_id,
            generated_post_id=(
                generated_post_id
            ),
            output_dir=output_dir,
        )
    except RuntimeError as error:
        if SYNTHETIC_API_ERROR not in str(
            error
        ):
            raise
    else:
        raise AssertionError(
            "Synthetic Image API failure "
            "не был проброшен."
        )

    assert len(client.requests) == 1

    state = await load_image_state(
        pool,
        batch_id=batch_id,
        generated_post_id=generated_post_id,
    )

    assert state["batch_status"] == "awaiting_review"
    assert state["post_status"] == "awaiting_review"
    assert state["image_status"] == "failed"
    assert state["image_path"] is None
    assert state["image_sha256"] is None
    assert state["image_prompt"] is None
    assert state["image_model_name"] is None
    assert state["image_prompt_version"] is None
    assert state["error_type"] == "RuntimeError"
    assert (
        state["error_message"]
        == SYNTHETIC_API_ERROR
    )
    assert state["completed_at"] is None
    assert state["failed_at"] is not None
    assert state["generated_post_count"] == 1

    print()
    print("Initial Image API failure: OK")
    print("image_status=failed")
    print("batch_status=awaiting_review")
    print("post_status=awaiting_review")
    print("generated_post_image_fields_unchanged=true")


async def test_storage_failure(
    pool: asyncpg.Pool,
    *,
    selection,
    telegram_chat_id: int,
    created_batch_ids: set[int],
    output_dir: Path,
) -> None:
    """Проверяет отказ PNG validation после API."""

    (
        batch_id,
        generated_post_id,
    ) = await (
        reservation_fixture
        .create_test_batch_and_post(
            pool,
            selection=selection,
            telegram_chat_id=telegram_chat_id,
            existing_image=False,
            test_name=(
                "openai_image_pipeline_storage_failure"
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )
    )

    client = SyntheticImageClient(
        image_bytes=build_png_bytes(
            width=66,
            height=99,
        )
    )

    generator = build_generator(
        client
    )

    before_files = set(
        output_dir.glob("*.png")
    )

    try:
        await run_reserved_openai_image_generation(
            pool,
            generator=generator,
            selection=selection,
            batch_id=batch_id,
            generated_post_id=(
                generated_post_id
            ),
            output_dir=output_dir,
        )
    except ValueError as error:
        if "Фактический размер PNG" not in str(
            error
        ):
            raise
    else:
        raise AssertionError(
            "PNG неверного размера "
            "не был заблокирован."
        )

    after_files = set(
        output_dir.glob("*.png")
    )

    assert after_files == before_files
    assert len(client.requests) == 1

    state = await load_image_state(
        pool,
        batch_id=batch_id,
        generated_post_id=generated_post_id,
    )

    assert state["image_status"] == "failed"
    assert state["batch_status"] == "awaiting_review"
    assert state["post_status"] == "awaiting_review"
    assert state["image_path"] is None
    assert state["image_sha256"] is None
    assert state["completed_at"] is None
    assert state["failed_at"] is not None

    print()
    print("Image storage failure: OK")
    print("fake_image_api_calls=1")
    print("wrong_dimensions_blocked=true")
    print("permanent_artifact_created=false")
    print("image_status=failed")
    print("text_post_preserved=true")


async def test_regenerate_success(
    pool: asyncpg.Pool,
    *,
    selection,
    telegram_chat_id: int,
    reviewer_telegram_user_id: int,
    created_batch_ids: set[int],
    output_dir: Path,
) -> None:
    """Проверяет полный regenerate image pipeline."""

    (
        batch_id,
        generated_post_id,
    ) = await (
        reservation_fixture
        .create_test_batch_and_post(
            pool,
            selection=selection,
            telegram_chat_id=telegram_chat_id,
            existing_image=True,
            test_name=(
                "openai_image_pipeline_regenerate"
            ),
            created_batch_ids=(
                created_batch_ids
            ),
        )
    )

    review_action_id = await (
        reservation_fixture
        .create_regenerate_review_action(
            pool,
            generated_post_id=(
                generated_post_id
            ),
            reviewer_telegram_user_id=(
                reviewer_telegram_user_id
            ),
            test_name=(
                "openai_image_pipeline_regenerate"
            ),
        )
    )

    client = SyntheticImageClient(
        image_bytes=build_png_bytes(
            value=144
        )
    )

    generator = build_generator(
        client
    )

    result = await (
        run_reserved_openai_image_generation(
            pool,
            generator=generator,
            selection=selection,
            batch_id=batch_id,
            generated_post_id=(
                generated_post_id
            ),
            request_kind="regenerate",
            review_action_id=(
                review_action_id
            ),
            editorial_comment=(
                reservation_fixture
                .EDITORIAL_COMMENT
            ),
            issues=(
                reservation_fixture
                .IMAGE_ISSUES
            ),
            output_dir=output_dir,
        )
    )

    assert result.completed is True
    assert result.model_called is True
    assert result.request_kind == "regenerate"
    assert len(client.requests) == 1

    if result.artifact is None:
        raise AssertionError(
            "Regenerate pipeline не вернул artifact."
        )

    state = await load_image_state(
        pool,
        batch_id=batch_id,
        generated_post_id=generated_post_id,
    )

    assert state["image_status"] == "completed"
    assert state["request_kind"] == "regenerate"
    assert (
        state["image_path"]
        == result.artifact.image_path
    )
    assert (
        state["image_sha256"]
        == result.artifact.image_sha256
    )
    assert (
        state["image_path"]
        != reservation_fixture
        .EXISTING_IMAGE_PATH
    )
    assert (
        state["image_sha256"]
        != reservation_fixture
        .EXISTING_IMAGE_SHA256
    )
    assert state["generated_post_count"] == 1

    print()
    print("Regenerate image pipeline: OK")
    print(
        "review_action_id="
        f"{review_action_id}"
    )
    print("fake_image_api_calls=1")
    print("same_generated_post_updated=true")
    print("previous_image_replaced=true")
    print("new_png_stored=true")


async def main() -> int:
    """Запускает end-to-end synthetic image pipeline test."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    created_batch_ids: set[int] = set()
    created_run_ids: set[int] = set()
    created_news_ids: tuple[int, ...] = ()
    fixture_ranking_run_id: int | None = None

    with tempfile.TemporaryDirectory(
        prefix="top3-openai-image-pipeline-"
    ) as temporary_directory:
        output_dir = (
            Path(temporary_directory)
            / "generated"
        )

        try:
            await (
                reservation_fixture
                .assert_migration_applied(pool)
            )

            created_news_ids = await (
                ranking_fixture
                .create_test_news_items(pool)
            )

            ranking_fixture.configure_test_news_ids(
                created_news_ids
            )

            selection = await (
                reservation_fixture
                .create_test_ranking_selection(
                    pool,
                    created_run_ids=(
                        created_run_ids
                    ),
                )
            )

            fixture_ranking_run_id = (
                selection.ranking_run_id
            )

            reviewer_telegram_user_id = await (
                reservation_fixture
                .load_test_reviewer(pool)
            )

            await test_initial_success_and_duplicate(
                pool,
                selection=selection,
                telegram_chat_id=(
                    settings.telegram_channel_id
                ),
                created_batch_ids=(
                    created_batch_ids
                ),
                output_dir=output_dir,
            )

            await test_initial_api_failure(
                pool,
                selection=selection,
                telegram_chat_id=(
                    settings.telegram_channel_id
                ),
                created_batch_ids=(
                    created_batch_ids
                ),
                output_dir=output_dir,
            )

            await test_storage_failure(
                pool,
                selection=selection,
                telegram_chat_id=(
                    settings.telegram_channel_id
                ),
                created_batch_ids=(
                    created_batch_ids
                ),
                output_dir=output_dir,
            )

            await test_regenerate_success(
                pool,
                selection=selection,
                telegram_chat_id=(
                    settings.telegram_channel_id
                ),
                reviewer_telegram_user_id=(
                    reviewer_telegram_user_id
                ),
                created_batch_ids=(
                    created_batch_ids
                ),
                output_dir=output_dir,
            )
        finally:
            try:
                if created_batch_ids:
                    if fixture_ranking_run_id is None:
                        raise RuntimeError(
                            "Неизвестен ranking_run_id "
                            "для cleanup временных batches."
                        )

                    await (
                        reservation_fixture
                        .cleanup_test_batches(
                            pool,
                            created_batch_ids=(
                                created_batch_ids
                            ),
                            ranking_run_id=(
                                fixture_ranking_run_id
                            ),
                        )
                    )
            finally:
                try:
                    await (
                        ranking_fixture
                        .cleanup_test_runs(
                            pool,
                            created_run_ids=(
                                created_run_ids
                            ),
                        )
                    )
                finally:
                    try:
                        await (
                            ranking_fixture
                            .cleanup_test_news_items(
                                pool,
                                news_ids=(
                                    created_news_ids
                                ),
                            )
                        )
                    finally:
                        await close_database_pool(
                            pool
                        )

    print()
    print("Real API key required: no")
    print("OpenAI Image requests: not performed")
    print("Synthetic Image API calls: performed")
    print("Permanent PNG files created: 0")
    print(
        "Database changes: temporary ranking, "
        "batch, image request and completion data "
        "inserted and deleted"
    )
    print("Permanent generated_posts created: 0")
    print("Telegram publication: not performed")
    print("OpenAI image pipeline test: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )