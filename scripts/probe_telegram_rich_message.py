import argparse
import asyncio
from pathlib import Path

import asyncpg
from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.types import (
    FSInputFile,
    InputMediaPhoto,
    InputRichMessage,
    InputRichMessageMedia,
)

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.publication.telegram_text import (
    prepare_telegram_text,
)


RICH_MESSAGE_MEDIA_ID = "top3_image"
RICH_MESSAGE_MAX_CHARACTERS = 32768


def parse_arguments() -> argparse.Namespace:
    """Разбирает параметры live Telegram probe."""

    parser = argparse.ArgumentParser(
        description=(
            "Отправляет один существующий generated_post "
            "вместе с PNG как Telegram Rich Message. "
            "PostgreSQL используется только на чтение."
        )
    )

    parser.add_argument(
        "--confirm-live-request",
        action="store_true",
        help=(
            "Разрешает реальную отправку одного сообщения "
            "в Telegram."
        ),
    )

    parser.add_argument(
        "--generated-post-id",
        type=int,
        required=True,
        help=(
            "generated_post_id существующего поста."
        ),
    )

    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help=(
            "Путь к PNG, который будет встроен "
            "в rich message."
        ),
    )

    return parser.parse_args()


def validate_generated_post_id(
    generated_post_id: int,
) -> int:
    """Проверяет generated_post_id."""

    if isinstance(generated_post_id, bool):
        raise TypeError(
            "generated_post_id не может быть bool."
        )

    if not isinstance(generated_post_id, int):
        raise TypeError(
            "generated_post_id должен быть int."
        )

    if generated_post_id <= 0:
        raise ValueError(
            "generated_post_id должен быть > 0."
        )

    return generated_post_id


def validate_image_path(
    image_path: Path,
) -> Path:
    """Проверяет локальный PNG перед Telegram API."""

    resolved_path = (
        image_path.expanduser().resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "PNG не найден: "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "image_path должен указывать "
            "на обычный файл: "
            f"{resolved_path}"
        )

    if resolved_path.suffix.lower() != ".png":
        raise ValueError(
            "Для этого probe ожидается PNG: "
            f"{resolved_path}"
        )

    file_size = resolved_path.stat().st_size

    if file_size <= 0:
        raise ValueError(
            "PNG-файл пуст."
        )

    return resolved_path


async def load_generated_post(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
) -> asyncpg.Record:
    """Читает существующий generated_post без изменений БД."""

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                gp.generated_post_id,
                gp.batch_id,
                gp.version_number,
                gp.post_status,
                gp.post_text,
                gp.text_format,
                pb.ranking_run_id,
                pb.publication_date,
                pb.batch_status,
                pb.target_telegram_chat_id
            FROM top3_news.generated_posts AS gp
            JOIN top3_news.publication_batches AS pb
              ON pb.batch_id = gp.batch_id
            WHERE gp.generated_post_id = $1
            """,
            generated_post_id,
        )

    if record is None:
        raise LookupError(
            "generated_post не найден: "
            f"generated_post_id={generated_post_id}"
        )

    post_text = record["post_text"]
    text_format = record["text_format"]

    if not isinstance(post_text, str):
        raise ValueError(
            "generated_posts.post_text "
            "должен быть строкой."
        )

    if not post_text.strip():
        raise ValueError(
            "generated_posts.post_text пуст."
        )

    if not isinstance(text_format, str):
        raise ValueError(
            "generated_posts.text_format "
            "должен быть строкой."
        )

    return record


def build_rich_message(
    *,
    post_text: str,
    text_format: str,
    image_path: Path,
) -> tuple[
    InputRichMessage,
    str,
]:
    """
    Создаёт rich message:

    PNG как первый media block,
    затем полный Telegram HTML поста.
    """

    prepared_text = prepare_telegram_text(
        post_text,
        text_format=text_format,
    )

    if prepared_text.text_format != "html":
        raise ValueError(
            "Для Telegram rich-message probe "
            "ожидается HTML после подготовки: "
            f"actual={prepared_text.text_format!r}"
        )

    media_reference = (
        f'<img src="tg://photo?id='
        f'{RICH_MESSAGE_MEDIA_ID}"/>'
    )

    rich_html = (
        media_reference
        + "\n\n"
        + prepared_text.text
    )

    if len(rich_html) > RICH_MESSAGE_MAX_CHARACTERS:
        raise ValueError(
            "Rich message превышает лимит "
            f"{RICH_MESSAGE_MAX_CHARACTERS}: "
            f"characters={len(rich_html)}"
        )

    media = InputRichMessageMedia(
        id=RICH_MESSAGE_MEDIA_ID,
        media=InputMediaPhoto(
            media=FSInputFile(
                image_path
            )
        ),
    )

    rich_message = InputRichMessage(
        html=rich_html,
        media=[media],
    )

    return (
        rich_message,
        rich_html,
    )


async def close_bot_session(
    bot: Bot,
) -> None:
    """Закрывает HTTP-сессию Telegram."""

    await bot.session.close()


async def main(
    arguments: argparse.Namespace,
) -> int:
    """Выполняет один контролируемый Telegram live probe."""

    if not arguments.confirm_live_request:
        raise RuntimeError(
            "Реальная Telegram-отправка заблокирована. "
            "Для запуска укажи "
            "--confirm-live-request."
        )

    generated_post_id = (
        validate_generated_post_id(
            arguments.generated_post_id
        )
    )

    image_path = validate_image_path(
        arguments.image_path
    )

    settings = get_settings()

    database_pool = await create_database_pool(
        settings
    )

    bot: Bot | None = None

    try:
        record = await load_generated_post(
            database_pool,
            generated_post_id=(
                generated_post_id
            ),
        )

        rich_message, rich_html = (
            build_rich_message(
                post_text=record["post_text"],
                text_format=record["text_format"],
                image_path=image_path,
            )
        )

        telegram_chat_id = (
            settings.telegram_channel_id
        )

        if (
            record["target_telegram_chat_id"]
            != telegram_chat_id
        ):
            raise RuntimeError(
                "TELEGRAM_CHANNEL_ID не совпадает "
                "с target_telegram_chat_id поста: "
                "configured="
                f"{telegram_chat_id}, "
                "post_target="
                f"{record['target_telegram_chat_id']}"
            )

        bot = Bot(
            token=(
                settings.telegram_bot_token
                .get_secret_value()
            )
        )

        chat = await bot.get_chat(
            telegram_chat_id
        )

        if chat.type != ChatType.CHANNEL:
            raise RuntimeError(
                "TELEGRAM_CHANNEL_ID указывает "
                "не на канал: "
                f"chat_type={chat.type}"
            )

        print(
            "WARNING: this script sends exactly "
            "one real Telegram rich message."
        )
        print(
            f"generated_post_id="
            f"{record['generated_post_id']}"
        )
        print(
            f"batch_id="
            f"{record['batch_id']}"
        )
        print(
            f"ranking_run_id="
            f"{record['ranking_run_id']}"
        )
        print(
            f"publication_date="
            f"{record['publication_date']}"
        )
        print(
            f"version_number="
            f"{record['version_number']}"
        )
        print(
            f"post_status="
            f"{record['post_status']}"
        )
        print(
            f"batch_status="
            f"{record['batch_status']}"
        )
        print(
            f"telegram_chat_id="
            f"{telegram_chat_id}"
        )
        print(
            f"telegram_chat_title="
            f"{chat.title}"
        )
        print(
            f"image_path="
            f"{image_path}"
        )
        print(
            f"image_bytes="
            f"{image_path.stat().st_size}"
        )
        print(
            f"rich_html_characters="
            f"{len(rich_html)}"
        )
        print(
            "database_mode=read_only"
        )
        print(
            "publication_attempts_enabled=false"
        )
        print(
            "telegram_rich_message_send_started=true"
        )

        message = await bot.send_rich_message(
            chat_id=telegram_chat_id,
            rich_message=rich_message,
            disable_notification=True,
        )

        print(
            "telegram_rich_message_send_completed=true"
        )
        print(
            f"telegram_message_id="
            f"{message.message_id}"
        )
        print(
            f"telegram_response_chat_id="
            f"{message.chat.id}"
        )
        print(
            f"telegram_response_chat_type="
            f"{message.chat.type.value}"
        )
        print(
            "rich_message_present="
            f"{message.rich_message is not None}"
        )
        print(
            "top_level_photo_present="
            f"{bool(message.photo)}"
        )
        print()
        print(
            "Database changes: not performed"
        )
        print(
            "publication_attempts created: 0"
        )
        print(
            "Telegram messages sent: 1"
        )
        print(
            "Telegram Rich Message live probe: OK"
        )

        return 0

    finally:
        if bot is not None:
            await close_bot_session(
                bot
            )

        await close_database_pool(
            database_pool
        )


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main(
                parse_arguments()
            )
        )
    )