import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path

import asyncpg

from app.config import get_settings
from app.db.generation_selection import (
    load_generation_combination,
)
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.generation.image_generator import (
    OPENAI_IMAGE_FALLBACK_PROMPT_VERSION,
    ImageGenerationNewsItem,
)
from app.generation.openai_image_factory import (
    create_openai_image_generation_runtime,
)


RANKING_RUN_ID = 142
COMBINATION_ID = 1844

OUTPUT_DIR = Path(
    "data/images/generated/semantic-fallback-tests"
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class _SingleConnectionAcquire:
    """Context manager одной asyncpg connection."""

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    async def __aenter__(
        self,
    ) -> asyncpg.Connection:
        return self._connection

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        return None


class _SingleConnectionPool:
    """
    Pool-like wrapper одной connection.

    Используется внутри READ ONLY PostgreSQL transaction,
    чтобы даже случайное изменение loader-а не могло
    записать данные в production DB.
    """

    def __init__(
        self,
        connection: asyncpg.Connection,
    ) -> None:
        self._connection = connection

    def acquire(
        self,
    ) -> _SingleConnectionAcquire:
        return _SingleConnectionAcquire(
            self._connection
        )


async def _close_sdk_client(
    sdk_client: object,
) -> None:
    """Best-effort закрывает AsyncOpenAI-compatible client."""

    close_method = getattr(
        sdk_client,
        "close",
        None,
    )

    if close_method is None:
        return

    result = close_method()

    if inspect.isawaitable(result):
        await result


def _build_image_items(
    selection,
) -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """Строит factual image input из сохранённой combination."""

    if len(selection.items) != 3:
        raise ValueError(
            "Для image test требуется "
            "ровно три новости."
        )

    items = tuple(
        ImageGenerationNewsItem(
            position=item.position,
            news_id=item.news_id,
            title=item.title,
            summary=item.summary,
        )
        for item in selection.items
    )

    positions = tuple(
        item.position
        for item in items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Image items должны иметь "
            "позиции 1, 2, 3."
        )

    news_ids = tuple(
        item.news_id
        for item in items
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Image items содержат "
            "дублирующиеся news_id."
        )

    return (
        items[0],
        items[1],
        items[2],
    )


def _build_output_path() -> Path:
    """Строит уникальный путь PNG для ручного просмотра."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return OUTPUT_DIR / (
        "semantic-fallback-v3-"
        f"ranking-{RANKING_RUN_ID}-"
        f"combination-{COMBINATION_ID}-"
        f"{timestamp}.png"
    )


async def main() -> int:
    """
    Выполняет один изолированный реальный semantic fallback v3.

    PostgreSQL:
    - только READ ONLY transaction;
    - никаких image reservations;
    - никаких workflow checkpoints;
    - никаких generated_post updates.

    Telegram:
    - не используется.

    OpenAI:
    - один вызов generator.generate();
    - возможные транспортные retry определяются
      текущим settings.openai_max_retries.
    """

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    runtime = None

    try:
        # --------------------------------------------------------------
        # 1. Загружаем historical TOP-3 строго в READ ONLY transaction.
        # --------------------------------------------------------------

        async with database_pool.acquire() as connection:
            transaction = connection.transaction(
                readonly=True
            )

            await transaction.start()

            try:
                read_only_pool = (
                    _SingleConnectionPool(
                        connection
                    )
                )

                ranked_combination = (
                    await load_generation_combination(
                        read_only_pool,
                        ranking_run_id=(
                            RANKING_RUN_ID
                        ),
                        combination_id=(
                            COMBINATION_ID
                        ),
                    )
                )

            finally:
                await transaction.rollback()

        selection = (
            ranked_combination.selection
        )

        items = _build_image_items(
            selection
        )

        news_ids = tuple(
            item.news_id
            for item in items
        )

        if (
            ranked_combination.combination_id
            != COMBINATION_ID
        ):
            raise AssertionError(
                "Loader вернул другую "
                "combination_id."
            )

        print(
            "Selection loaded in READ ONLY "
            "transaction: OK"
        )
        print(
            f"ranking_run_id={RANKING_RUN_ID}"
        )
        print(
            f"combination_id={COMBINATION_ID}"
        )
        print(
            "news_ids="
            + ",".join(
                str(news_id)
                for news_id in news_ids
            )
        )

        # --------------------------------------------------------------
        # 2. Создаём runtime и явно включаем semantic fallback v3.
        # --------------------------------------------------------------

        runtime = (
            create_openai_image_generation_runtime(
                settings
            )
        )

        runtime.generator.set_moderation_safe_editorial_fallback(
            True
        )

        metadata = (
            runtime.generator.metadata
        )

        if (
            metadata.prompt_version
            != OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
        ):
            raise AssertionError(
                "Generator не переключился "
                "на semantic fallback v3: "
                f"actual="
                f"{metadata.prompt_version}, "
                f"expected="
                f"{OPENAI_IMAGE_FALLBACK_PROMPT_VERSION}"
            )

        request = (
            runtime.generator.build_request(
                items=items
            )
        )

        print(
            "Semantic fallback enabled: OK"
        )
        print(
            "prompt_version="
            f"{metadata.prompt_version}"
        )
        print(
            "model="
            f"{request.model}"
        )
        print(
            "size="
            f"{request.size}"
        )
        print(
            "quality="
            f"{request.quality}"
        )
        print(
            "moderation="
            f"{request.moderation}"
        )
        print(
            "sdk_max_retries="
            f"{settings.openai_max_retries}"
        )

        # --------------------------------------------------------------
        # 3. Ровно один логический generator.generate().
        #    Pipeline/DB reservation намеренно не используются.
        # --------------------------------------------------------------

        print()
        print(
            "Calling OpenAI Image API..."
        )

        generation = (
            await runtime.generator.generate(
                items=items
            )
        )

        if (
            generation.model_request
            != request
        ):
            raise AssertionError(
                "Фактический Image API request "
                "отличается от предварительно "
                "проверенного request."
            )

        image_bytes = (
            generation
            .model_response
            .image_bytes
        )

        if not image_bytes.startswith(
            PNG_SIGNATURE
        ):
            raise ValueError(
                "Image API result не имеет "
                "PNG signature."
            )

        # --------------------------------------------------------------
        # 4. Сохраняем только локальный PNG-файл.
        # --------------------------------------------------------------

        output_path = (
            _build_output_path()
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_bytes(
            image_bytes
        )

        absolute_path = (
            output_path.resolve()
        )

        usage = (
            generation
            .model_response
            .usage
        )

        print()
        print(
            "Semantic fallback v3 "
            "real Image API call: OK"
        )
        print(
            "output_path="
            f"{absolute_path}"
        )
        print(
            "file_size_bytes="
            f"{len(image_bytes)}"
        )

        if usage is None:
            print(
                "usage=null"
            )
        else:
            print(
                "usage="
                + json.dumps(
                    asdict(usage),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

        print()
        print(
            "Database changes=not_performed_read_only"
        )
        print(
            "Telegram requests=not_performed"
        )
        print(
            "OpenAI generator.generate calls=1"
        )
        print(
            "OpenAI transport retries="
            f"up_to_{settings.openai_max_retries}"
        )
        print(
            "Semantic fallback v3 "
            "isolated live test: OK"
        )

        return 0

    finally:
        if runtime is not None:
            await _close_sdk_client(
                runtime.sdk_client
            )

        await close_database_pool(
            database_pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )