import asyncio
from datetime import UTC, date, datetime
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from aiogram.enums import ChatType
from aiogram.types import InputRichMessage
from PIL import Image

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
import app.publication.approved_service as approved_service


FAKE_MESSAGE_ID = 987654321

TEST_POST_TEXT = (
    "**TOP-3 НОВОСТЕЙ КИНО ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n\n"
    "---\n\n"
    "1️⃣ **Первая тестовая новость**\n\n"
    "Описание первой тестовой новости.\n\n"
    "2️⃣ **Вторая тестовая новость**\n\n"
    "Описание второй тестовой новости.\n\n"
    "3️⃣ **Третья тестовая новость**\n\n"
    "Описание третьей тестовой новости."
)


class FakeSession:
    """Фальшивая HTTP-сессия Telegram."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeBot:
    """Фальшивый Telegram Bot без сетевых запросов."""

    instances: list["FakeBot"] = []

    def __init__(
        self,
        *,
        token: str,
        **kwargs: Any,
    ) -> None:
        self.token = token
        self.session = FakeSession()

        self.get_chat_calls = 0
        self.send_rich_message_calls = 0

        self.last_chat_id: int | None = None
        self.last_rich_message: (
            InputRichMessage | None
        ) = None
        self.last_disable_notification: (
            bool | None
        ) = None

        type(self).instances.append(self)

    async def get_chat(
        self,
        chat_id: int,
    ) -> SimpleNamespace:
        self.get_chat_calls += 1
        self.last_chat_id = chat_id

        return SimpleNamespace(
            id=chat_id,
            type=ChatType.CHANNEL,
            title="Synthetic TOP 3 channel",
        )

    async def send_rich_message(
        self,
        *,
        chat_id: int,
        rich_message: InputRichMessage,
        disable_notification: bool | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.send_rich_message_calls += 1
        self.last_chat_id = chat_id
        self.last_rich_message = rich_message
        self.last_disable_notification = (
            disable_notification
        )

        return SimpleNamespace(
            message_id=FAKE_MESSAGE_ID,
            chat=SimpleNamespace(
                id=chat_id,
                type=ChatType.CHANNEL,
                title="Synthetic TOP 3 channel",
            ),
            date=datetime.now(UTC),
        )


def calculate_sha256(
    path: Path,
) -> str:
    """Вычисляет SHA-256 файла."""

    digest = sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


async def create_test_post(
    pool,
    *,
    telegram_chat_id: int,
    image_path: str,
    image_sha256: str,
) -> tuple[int, int]:
    """
    Создаёт временные approved batch/post.

    publication_attempt здесь не создаётся:
    его обязан создать сам production publisher.
    """

    publication_date = date.today()

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    $1::bigint
                )
                """,
                publication_date.toordinal(),
            )

            edition = await connection.fetchval(
                """
                SELECT
                    COALESCE(
                        MAX(edition),
                        0
                    )::integer + 1
                FROM publication_batches
                WHERE publication_date = $1
                """,
                publication_date,
            )

            batch_id = await connection.fetchval(
                """
                INSERT INTO publication_batches (
                    publication_date,
                    edition,
                    batch_status,
                    target_telegram_chat_id,
                    metadata
                )
                VALUES (
                    $1,
                    $2,
                    'approved',
                    $3,
                    $4::jsonb
                )
                RETURNING batch_id
                """,
                publication_date,
                edition,
                telegram_chat_id,
                json.dumps(
                    {
                        "technical_test": True,
                        "test_name": (
                            "approved_rich_publication"
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

            generated_post_id = await connection.fetchval(
                """
                INSERT INTO generated_posts (
                    batch_id,
                    version_number,
                    post_status,
                    post_text,
                    text_format,
                    image_path,
                    image_sha256,
                    generation_metadata
                )
                VALUES (
                    $1,
                    1,
                    'approved',
                    $2,
                    'markdown',
                    $3,
                    $4,
                    $5::jsonb
                )
                RETURNING generated_post_id
                """,
                batch_id,
                TEST_POST_TEXT,
                image_path,
                image_sha256,
                json.dumps(
                    {
                        "technical_test": True,
                        "test_name": (
                            "approved_rich_publication"
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

    return (
        int(batch_id),
        int(generated_post_id),
    )


async def load_publication_state(
    pool,
    *,
    batch_id: int,
    generated_post_id: int,
) -> dict[str, Any]:
    """Читает результат publication lifecycle."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_status,
                b.published_at,
                p.post_status,
                (
                    SELECT count(*)
                    FROM publication_attempts AS pa
                    WHERE
                        pa.generated_post_id
                        = p.generated_post_id
                )::integer
                    AS publication_attempt_count,
                pa.publication_attempt_id,
                pa.attempt_number,
                pa.attempt_status,
                pa.telegram_message_id,
                pa.request_payload::text
                    AS request_payload_text,
                pa.response_payload::text
                    AS response_payload_text
            FROM publication_batches AS b
            JOIN generated_posts AS p
                ON p.batch_id = b.batch_id
            LEFT JOIN LATERAL (
                SELECT
                    publication_attempt_id,
                    attempt_number,
                    attempt_status,
                    telegram_message_id,
                    request_payload,
                    response_payload
                FROM publication_attempts
                WHERE
                    generated_post_id
                    = p.generated_post_id
                ORDER BY attempt_number DESC
                LIMIT 1
            ) AS pa
                ON true
            WHERE
                b.batch_id = $1
                AND p.generated_post_id = $2
            """,
            batch_id,
            generated_post_id,
        )

    if record is None:
        raise RuntimeError(
            "Не удалось прочитать "
            "временный publication context."
        )

    return dict(record)


async def cleanup_test_data(
    pool,
    *,
    batch_id: int | None,
    generated_post_id: int | None,
) -> None:
    """Удаляет только созданные этим тестом данные."""

    if (
        batch_id is None
        or generated_post_id is None
    ):
        return

    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                DELETE FROM publication_attempts
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM generated_posts
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM publication_batches
                WHERE batch_id = $1
                """,
                batch_id,
            )

    async with pool.acquire() as connection:
        remaining = await connection.fetchval(
            """
            SELECT
                (
                    SELECT count(*)
                    FROM generated_posts
                    WHERE generated_post_id = $1
                )
                +
                (
                    SELECT count(*)
                    FROM publication_batches
                    WHERE batch_id = $2
                )
                +
                (
                    SELECT count(*)
                    FROM publication_attempts
                    WHERE generated_post_id = $1
                )
            """,
            generated_post_id,
            batch_id,
        )

    if remaining != 0:
        raise RuntimeError(
            "Временные publication-данные "
            "удалены не полностью."
        )


def validate_fake_telegram_call(
    *,
    expected_chat_id: int,
) -> None:
    """Проверяет фактический fake Telegram request."""

    if len(FakeBot.instances) != 1:
        raise RuntimeError(
            "Ожидался ровно один экземпляр FakeBot: "
            f"{len(FakeBot.instances)}"
        )

    bot = FakeBot.instances[0]

    if bot.get_chat_calls != 1:
        raise RuntimeError(
            "get_chat должен быть вызван ровно один раз."
        )

    if bot.send_rich_message_calls != 1:
        raise RuntimeError(
            "send_rich_message должен быть "
            "вызван ровно один раз."
        )

    if bot.last_chat_id != expected_chat_id:
        raise RuntimeError(
            "Некорректный chat_id fake publication."
        )

    if bot.last_disable_notification is not True:
        raise RuntimeError(
            "disable_notification должен быть true."
        )

    rich_message = bot.last_rich_message

    if rich_message is None:
        raise RuntimeError(
            "FakeBot не получил InputRichMessage."
        )

    if rich_message.html is None:
        raise RuntimeError(
            "Rich Message не содержит HTML."
        )

    if (
        '<img src="tg://photo?id=top3_image"/>'
        not in rich_message.html
    ):
        raise RuntimeError(
            "Rich Message не содержит image media reference."
        )

    if (
        "<p><b>TOP-3 НОВОСТЕЙ КИНО "
        "ЗА ПОСЛЕДНИЕ 24 ЧАСА</b></p>"
        not in rich_message.html
    ):
        raise RuntimeError(
            "Rich Message заголовок "
            "сформирован неверно."
        )

    if "<hr/>" not in rich_message.html:
        raise RuntimeError(
            "Rich Message не содержит разделитель."
        )

    if not rich_message.media:
        raise RuntimeError(
            "Rich Message не содержит media."
        )

    if len(rich_message.media) != 1:
        raise RuntimeError(
            "Rich Message должен содержать "
            "ровно один media-элемент."
        )

    media_item = rich_message.media[0]

    if media_item.id != "top3_image":
        raise RuntimeError(
            "Некорректный Rich Message media ID: "
            f"{media_item.id!r}"
        )


async def main() -> int:
    """Запускает DB + fake Telegram integration test."""

    settings = get_settings()

    pool = await create_database_pool(
        settings
    )

    batch_id: int | None = None
    generated_post_id: int | None = None

    original_bot = approved_service.Bot

    FakeBot.instances.clear()

    try:
        with TemporaryDirectory() as directory:
            image_file = (
                Path(directory)
                / "approved_rich_test.png"
            )

            image = Image.new(
                "RGB",
                (24, 36),
            )
            image.save(
                image_file,
                format="PNG",
            )

            image_sha256 = calculate_sha256(
                image_file
            )

            (
                batch_id,
                generated_post_id,
            ) = await create_test_post(
                pool,
                telegram_chat_id=(
                    settings.telegram_channel_id
                ),
                image_path=str(
                    image_file.resolve()
                ),
                image_sha256=image_sha256,
            )

            approved_service.Bot = FakeBot

            result = (
                await approved_service
                .publish_approved_post(
                    pool,
                    bot_token=(
                        "synthetic-token-no-network"
                    ),
                    generated_post_id=(
                        generated_post_id
                    ),
                    disable_notification=True,
                )
            )

            if result.database_status != "published":
                raise RuntimeError(
                    "Ожидался database_status=published: "
                    f"{result.database_status}"
                )

            if result.requires_review:
                raise RuntimeError(
                    "Успешная fake publication "
                    "не должна требовать review."
                )

            if (
                result.telegram_message_id
                != FAKE_MESSAGE_ID
            ):
                raise RuntimeError(
                    "Некорректный telegram_message_id."
                )

            validate_fake_telegram_call(
                expected_chat_id=(
                    settings.telegram_channel_id
                )
            )

            state = await load_publication_state(
                pool,
                batch_id=batch_id,
                generated_post_id=(
                    generated_post_id
                ),
            )

            if state["batch_status"] != "published":
                raise RuntimeError(
                    "batch_status должен быть published."
                )

            if state["post_status"] != "published":
                raise RuntimeError(
                    "post_status должен быть published."
                )

            if (
                state["attempt_status"]
                != "published"
            ):
                raise RuntimeError(
                    "attempt_status должен быть published."
                )

            if (
                state["telegram_message_id"]
                != FAKE_MESSAGE_ID
            ):
                raise RuntimeError(
                    "Telegram message ID "
                    "не сохранён в PostgreSQL."
                )

            if (
                state["publication_attempt_count"]
                != 1
            ):
                raise RuntimeError(
                    "Ожидался ровно один "
                    "publication_attempt."
                )

            request_payload = json.loads(
                state["request_payload_text"]
            )

            response_payload = json.loads(
                state["response_payload_text"]
            )

            if (
                request_payload.get("transport")
                != "rich_message"
            ):
                raise RuntimeError(
                    "request_payload.transport "
                    "должен быть rich_message."
                )

            request_rich_message = (
                request_payload.get(
                    "rich_message"
                )
            )

            if not isinstance(
                request_rich_message,
                dict,
            ):
                raise RuntimeError(
                    "request_payload.rich_message "
                    "отсутствует."
                )

            if (
                request_rich_message.get(
                    "media_id"
                )
                != "top3_image"
            ):
                raise RuntimeError(
                    "request media_id неверен."
                )

            request_image = request_payload.get(
                "image"
            )

            if not isinstance(
                request_image,
                dict,
            ):
                raise RuntimeError(
                    "request_payload.image отсутствует."
                )

            if (
                request_image.get("sha256")
                != image_sha256
            ):
                raise RuntimeError(
                    "request image SHA-256 неверен."
                )

            if (
                response_payload.get("transport")
                != "rich_message"
            ):
                raise RuntimeError(
                    "response_payload.transport "
                    "должен быть rich_message."
                )

            if (
                response_payload.get(
                    "image_sha256"
                )
                != image_sha256
            ):
                raise RuntimeError(
                    "response image SHA-256 неверен."
                )

            duplicate_blocked = False

            try:
                await approved_service.publish_approved_post(
                    pool,
                    bot_token=(
                        "synthetic-token-no-network"
                    ),
                    generated_post_id=(
                        generated_post_id
                    ),
                    disable_notification=True,
                )
            except ValueError:
                duplicate_blocked = True

            if not duplicate_blocked:
                raise RuntimeError(
                    "Повторная публикация "
                    "не была заблокирована."
                )

            if (
                FakeBot.instances[0]
                .send_rich_message_calls
                != 1
            ):
                raise RuntimeError(
                    "При повторном вызове Telegram "
                    "не должен вызываться снова."
                )

            state_after_duplicate = (
                await load_publication_state(
                    pool,
                    batch_id=batch_id,
                    generated_post_id=(
                        generated_post_id
                    ),
                )
            )

            if (
                state_after_duplicate[
                    "publication_attempt_count"
                ]
                != 1
            ):
                raise RuntimeError(
                    "Повторный вызов создал "
                    "лишний publication_attempt."
                )

            if not FakeBot.instances[0].session.closed:
                raise RuntimeError(
                    "Fake Telegram session "
                    "не была закрыта."
                )

            print(
                "Approved Rich Message publication "
                "integration test: OK"
            )
            print(f"batch_id={batch_id}")
            print(
                "generated_post_id="
                f"{generated_post_id}"
            )
            print(
                "publication_attempt_id="
                f"{result.publication_attempt_id}"
            )
            print(
                "telegram_message_id="
                f"{result.telegram_message_id}"
            )
            print("transport=rich_message")
            print("media_id=top3_image")
            print(
                "publication_attempt_count=1"
            )
            print(
                "duplicate_request_blocked=true"
            )
            print(
                "fake_send_rich_message_calls=1"
            )
            print(
                "database_status=published"
            )

    finally:
        approved_service.Bot = original_bot

        try:
            await cleanup_test_data(
                pool,
                batch_id=batch_id,
                generated_post_id=(
                    generated_post_id
                ),
            )
        finally:
            await close_database_pool(pool)

    print(
        "Database changes: temporary publication "
        "data inserted and deleted"
    )
    print("OpenAI requests: not performed")
    print("Telegram requests: not performed")
    print("Permanent publication data created: 0")
    print(
        "Approved Rich Message publication "
        "test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )