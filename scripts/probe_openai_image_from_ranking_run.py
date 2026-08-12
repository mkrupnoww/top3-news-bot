import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import asyncpg

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


EXPECTED_MODEL = "gpt-image-2"
EXPECTED_SIZE = "1024x1536"
EXPECTED_QUALITY = "medium"
EXPECTED_OUTPUT_FORMAT = "png"
EXPECTED_BACKGROUND = "opaque"
EXPECTED_MODERATION = "auto"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "generated"
    / "live_probe"
)


@dataclass(frozen=True, slots=True)
class RankingRunTop3Record:
    """Одна новость из реального TOP-3."""

    ranking_run_id: int
    top3_position: int
    news_id: int
    source_name: str
    title: str
    summary: str
    selection_reason: str | None
    primary_image_url: str | None


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Выполняет один реальный OpenAI Image API "
            "probe по реальному ranking_run TOP-3 "
            "и сохраняет PNG для визуальной проверки."
        )
    )

    parser.add_argument(
        "--ranking-run-id",
        type=int,
        required=True,
        help=(
            "ranking_run_id с уже выбранным TOP-3."
        ),
    )

    return parser.parse_args()


def normalize_required_text(
    value: object,
    *,
    field_name: str,
) -> str:
    """Нормализует обязательное текстовое поле."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть str."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def normalize_optional_text(
    value: object,
    *,
    field_name: str,
) -> str | None:
    """Нормализует необязательное текстовое поле."""

    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть str | None."
        )

    normalized_value = value.strip()

    if not normalized_value:
        return None

    return normalized_value


def validate_probe_configuration() -> None:
    """Блокирует неожиданный платный запрос."""

    settings = get_settings()

    if settings.openai_api_key is None:
        raise ValueError(
            "OPENAI_API_KEY не настроен."
        )

    if settings.openai_image_model != EXPECTED_MODEL:
        raise ValueError(
            "Live visual probe разрешён только для "
            f"{EXPECTED_MODEL}: actual="
            f"{settings.openai_image_model!r}"
        )

    if settings.openai_image_size != EXPECTED_SIZE:
        raise ValueError(
            "Live visual probe разрешён только для "
            f"{EXPECTED_SIZE}: actual="
            f"{settings.openai_image_size!r}"
        )

    if settings.openai_max_retries != 0:
        raise ValueError(
            "Для live visual probe требуется "
            "OPENAI_MAX_RETRIES=0, чтобы один запуск "
            "не породил автоматический повторный "
            "платный запрос."
        )


async def create_database_pool() -> asyncpg.Pool:
    """Создаёт pool без побочных эффектов."""

    settings = get_settings()

    return await asyncpg.create_pool(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=(
            settings.db_password
            .get_secret_value()
        ),
        min_size=1,
        max_size=2,
    )


async def load_ranking_run_top3(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> tuple[
    RankingRunTop3Record,
    RankingRunTop3Record,
    RankingRunTop3Record,
]:
    """Загружает реальный TOP-3 из БД."""

    if isinstance(ranking_run_id, bool):
        raise TypeError(
            "ranking_run_id не может быть bool."
        )

    if not isinstance(ranking_run_id, int):
        raise TypeError(
            "ranking_run_id должен быть int."
        )

    if ranking_run_id <= 0:
        raise ValueError(
            "ranking_run_id должен быть > 0."
        )

    query = """
    SELECT
        rr.ranking_run_id,
        rr.run_status,
        ns.top3_position,
        ni.news_id,
        s.source_name,
        COALESCE(
            NULLIF(BTRIM(ni.normalized_title), ''),
            NULLIF(BTRIM(ni.raw_title), '')
        ) AS title,
        COALESCE(
            NULLIF(BTRIM(ni.normalized_summary), ''),
            NULLIF(BTRIM(ni.raw_summary), ''),
            NULLIF(BTRIM(ni.article_text), '')
        ) AS summary,
        ns.score_explanation AS selection_reason,
        ni.primary_image_url
    FROM top3_news.ranking_runs AS rr
    JOIN top3_news.news_scores AS ns
      ON ns.ranking_run_id = rr.ranking_run_id
    JOIN top3_news.news_items AS ni
      ON ni.news_id = ns.news_id
    JOIN top3_news.sources AS s
      ON s.source_id = ni.source_id
    WHERE rr.ranking_run_id = $1
      AND rr.run_status = 'completed'
      AND ns.selected_for_top3 = TRUE
    ORDER BY ns.top3_position;
    """

    rows = await pool.fetch(
        query,
        ranking_run_id,
    )

    if len(rows) != 3:
        raise LookupError(
            "Для ranking_run_id="
            f"{ranking_run_id} не найден "
            "полный TOP-3."
        )

    records: list[RankingRunTop3Record] = []

    for row in rows:
        top3_position = row["top3_position"]

        if top3_position not in (1, 2, 3):
            raise ValueError(
                "Некорректная позиция TOP-3: "
                f"{top3_position!r}"
            )

        records.append(
            RankingRunTop3Record(
                ranking_run_id=row["ranking_run_id"],
                top3_position=top3_position,
                news_id=row["news_id"],
                source_name=normalize_required_text(
                    row["source_name"],
                    field_name="source_name",
                ),
                title=normalize_required_text(
                    row["title"],
                    field_name="title",
                ),
                summary=normalize_required_text(
                    row["summary"],
                    field_name="summary",
                ),
                selection_reason=(
                    normalize_optional_text(
                        row["selection_reason"],
                        field_name=(
                            "selection_reason"
                        ),
                    )
                ),
                primary_image_url=(
                    normalize_optional_text(
                        row["primary_image_url"],
                        field_name=(
                            "primary_image_url"
                        ),
                    )
                ),
            )
        )

    positions = [
        record.top3_position
        for record in records
    ]

    if positions != [1, 2, 3]:
        raise ValueError(
            "TOP-3 должен содержать позиции "
            "строго 1, 2, 3."
        )

    return (
        records[0],
        records[1],
        records[2],
    )


def build_image_items(
    records: tuple[
        RankingRunTop3Record,
        RankingRunTop3Record,
        RankingRunTop3Record,
    ],
) -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """Преобразует записи БД в input генератора."""

    items: list[ImageGenerationNewsItem] = []

    for record in records:
        items.append(
            ImageGenerationNewsItem(
                position=record.top3_position,
                news_id=record.news_id,
                title=record.title,
                summary=record.summary,
            )
        )

    return (
        items[0],
        items[1],
        items[2],
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
    """Проверяет точные параметры платного запроса."""

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
            "ImageModelRequest отличается от "
            f"ожидаемого: actual={actual!r}, "
            f"expected={expected!r}"
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
    Выполняет один реальный visual probe.

    БД используется только на чтение.
    Telegram не используется.
    PNG сохраняется постоянно для просмотра.
    """

    args = parse_args()

    validate_probe_configuration()

    pool = None
    runtime = None

    try:
        pool = await create_database_pool()

        records = await load_ranking_run_top3(
            pool,
            ranking_run_id=args.ranking_run_id,
        )

        items = build_image_items(records)

        runtime = (
            create_openai_image_generation_runtime(
                get_settings()
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
            "WARNING: this script performs exactly "
            "one paid OpenAI Image API request."
        )
        print(
            "ranking_run_id="
            f"{args.ranking_run_id}"
        )
        print(
            "image_model="
            f"{model_request.model}"
        )
        print(
            "image_size="
            f"{model_request.size}"
        )
        print(
            "quality="
            f"{model_request.quality}"
        )
        print(
            "database_mode=read_only"
        )
        print(
            "telegram_enabled=false"
        )
        print(
            "prompt_version="
            f"{runtime.generator.metadata.prompt_version}"
        )
        print(
            "prompt_chars="
            f"{len(model_request.prompt)}"
        )
        print()

        for record in records:
            print(
                f"top3_position={record.top3_position}"
            )
            print(
                f"news_id={record.news_id}"
            )
            print(
                f"source_name={record.source_name}"
            )
            print(
                f"title={record.title}"
            )
            print()

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

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact = store_png_image(
            response.image_bytes,
            image_generation_id=(
                args.ranking_run_id
            ),
            expected_size=EXPECTED_SIZE,
            output_dir=OUTPUT_DIR,
        )

        print(
            "openai_image_request_completed=true"
        )
        print(
            "base64_decoded=true"
        )
        print(
            "response_created="
            f"{response.created}"
        )
        print(
            "response_output_format="
            f"{response.output_format}"
        )
        print(
            "response_quality="
            f"{response.quality}"
        )
        print(
            "response_size="
            f"{response.size}"
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
            "png_width="
            f"{artifact.width}"
        )
        print(
            "png_height="
            f"{artifact.height}"
        )
        print(
            "png_sha256="
            f"{artifact.image_sha256}"
        )
        print(
            "saved_image_path="
            f"{artifact.file_path}"
        )

        usage = response.usage

        if usage is None:
            print("usage_present=false")
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

            print("usage_present=true")
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

        print()
        print(
            "Database changes: not performed"
        )
        print(
            "Telegram publication: not performed"
        )
        print(
            "Persistent PNG saved: 1"
        )
        print(
            "OpenAI real TOP-3 visual probe: OK"
        )

        return 0

    finally:
        if runtime is not None:
            await close_sdk_client(
                runtime.sdk_client
            )

        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )