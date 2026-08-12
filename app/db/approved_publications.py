from dataclasses import dataclass
import json
from typing import Any, Mapping

import asyncpg

from app.db.publications import PublicationAttempt


@dataclass(frozen=True, slots=True)
class PreparedApprovedPublication:
    """Одобренный пост, подготовленный к отправке в Telegram."""

    publication: PublicationAttempt

    telegram_chat_id: int

    rich_html: str
    rich_media_id: str

    image_path: str
    image_sha256: str

    source_text_format: str


def _encode_json(
    payload: Mapping[str, Any],
) -> str:
    """Преобразует словарь в JSON для передачи в asyncpg."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_prepared_content(
    *,
    rich_html: str,
    rich_media_id: str,
    image_path: str,
    image_sha256: str,
    source_post_text: str,
    source_text_format: str,
) -> None:
    """Проверяет подготовленные данные публикации."""

    if not rich_html.strip():
        raise ValueError(
            "Подготовленный Rich Message HTML "
            "не может быть пустым."
        )

    if not rich_media_id.strip():
        raise ValueError(
            "Rich Message media ID "
            "не может быть пустым."
        )

    if not image_path.strip():
        raise ValueError(
            "Путь к изображению "
            "не может быть пустым."
        )

    if not image_sha256.strip():
        raise ValueError(
            "SHA-256 изображения "
            "не может быть пустым."
        )

    if not source_post_text.strip():
        raise ValueError(
            "Исходный текст публикации "
            "не может быть пустым."
        )

    if not source_text_format.strip():
        raise ValueError(
            "Исходный формат публикации "
            "не может быть пустым."
        )


async def prepare_approved_publication(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
    disable_notification: bool,
    rich_html: str,
    rich_media_id: str,
    image_path: str,
    image_sha256: str,
    source_post_text: str,
    source_text_format: str,
) -> PreparedApprovedPublication:
    """
    Создаёт попытку отправки существующего одобренного поста.

    Исходный текст и изображение сверяются
    с текущим generated_post.

    В request_payload сохраняются данные
    Rich Message, которые будут переданы
    Telegram Bot API.

    Новый publication_batch и generated_post
    не создаются.

    Повторная отправка блокируется при наличии
    попытки со статусом started, unknown
    или published.
    """

    _validate_prepared_content(
        rich_html=rich_html,
        rich_media_id=rich_media_id,
        image_path=image_path,
        image_sha256=image_sha256,
        source_post_text=source_post_text,
        source_text_format=source_text_format,
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            record = await connection.fetchrow(
                """
                SELECT
                    p.generated_post_id,
                    p.batch_id,
                    p.post_status,
                    p.post_text,
                    p.text_format,
                    p.image_path,
                    p.image_sha256,
                    b.publication_date,
                    b.edition,
                    b.batch_status,
                    b.target_telegram_chat_id
                FROM generated_posts AS p
                JOIN publication_batches AS b
                    ON b.batch_id = p.batch_id
                WHERE p.generated_post_id = $1
                FOR UPDATE OF p, b
                """,
                generated_post_id,
            )

            if record is None:
                raise LookupError(
                    "Одобренный пост не найден: "
                    f"generated_post_id="
                    f"{generated_post_id}"
                )

            if (
                record["post_status"] == "published"
                or record["batch_status"] == "published"
            ):
                raise ValueError(
                    "Пост уже опубликован. "
                    "Повторная отправка запрещена: "
                    f"generated_post_id="
                    f"{generated_post_id}"
                )

            if (
                record["post_status"]
                != "approved"
            ):
                raise ValueError(
                    "Пост не имеет статус approved: "
                    f"post_status="
                    f"{record['post_status']}"
                )

            if (
                record["batch_status"]
                != "approved"
            ):
                raise ValueError(
                    "Подборка не имеет статус "
                    "approved: "
                    f"batch_status="
                    f"{record['batch_status']}"
                )

            if (
                record["post_text"]
                != source_post_text
            ):
                raise ValueError(
                    "Текст generated_post изменился "
                    "после подготовки Rich Message. "
                    "Публикация остановлена."
                )

            if (
                record["text_format"]
                != source_text_format
            ):
                raise ValueError(
                    "Формат generated_post изменился "
                    "после подготовки Rich Message. "
                    "Публикация остановлена."
                )

            stored_image_path = record[
                "image_path"
            ]

            stored_image_sha256 = record[
                "image_sha256"
            ]

            if stored_image_path is None:
                raise ValueError(
                    "У generated_post отсутствует "
                    "image_path."
                )

            if stored_image_sha256 is None:
                raise ValueError(
                    "У generated_post отсутствует "
                    "image_sha256."
                )

            if (
                stored_image_path
                != image_path
            ):
                raise ValueError(
                    "image_path generated_post изменился "
                    "после подготовки Rich Message. "
                    "Публикация остановлена."
                )

            if (
                stored_image_sha256
                != image_sha256
            ):
                raise ValueError(
                    "image_sha256 generated_post "
                    "изменился после подготовки "
                    "Rich Message. "
                    "Публикация остановлена."
                )

            telegram_chat_id = record[
                "target_telegram_chat_id"
            ]

            if telegram_chat_id is None:
                raise ValueError(
                    "У подборки не задан "
                    "target_telegram_chat_id."
                )

            blocking_attempt = (
                await connection.fetchrow(
                    """
                    SELECT
                        publication_attempt_id,
                        attempt_number,
                        attempt_status,
                        telegram_message_id
                    FROM publication_attempts
                    WHERE generated_post_id = $1
                      AND attempt_status IN (
                          'started',
                          'unknown',
                          'published'
                      )
                    ORDER BY
                        attempt_number DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    generated_post_id,
                )
            )

            if blocking_attempt is not None:
                raise ValueError(
                    "Для поста уже существует "
                    "попытка, запрещающая "
                    "повторную отправку: "
                    "publication_attempt_id="
                    f"{blocking_attempt[
                        'publication_attempt_id'
                    ]}, "
                    "attempt_status="
                    f"{blocking_attempt[
                        'attempt_status'
                    ]}, "
                    "telegram_message_id="
                    f"{blocking_attempt[
                        'telegram_message_id'
                    ]}"
                )

            attempt_number = (
                await connection.fetchval(
                    """
                    SELECT
                        COALESCE(
                            MAX(attempt_number),
                            0
                        )::integer + 1
                    FROM publication_attempts
                    WHERE generated_post_id = $1
                    """,
                    generated_post_id,
                )
            )

            request_payload = _encode_json(
                {
                    "chat_id": (
                        telegram_chat_id
                    ),
                    "transport": (
                        "rich_message"
                    ),
                    "rich_message": {
                        "html": rich_html,
                        "media_id": (
                            rich_media_id
                        ),
                    },
                    "image": {
                        "path": image_path,
                        "sha256": (
                            image_sha256
                        ),
                    },
                    "source_text_format": (
                        source_text_format
                    ),
                    "disable_notification": (
                        disable_notification
                    ),
                    "existing_approved_post": True,
                }
            )

            publication_attempt_id = (
                await connection.fetchval(
                    """
                    INSERT INTO
                        publication_attempts (
                            generated_post_id,
                            attempt_number,
                            attempt_status,
                            telegram_chat_id,
                            request_payload
                        )
                    VALUES (
                        $1,
                        $2,
                        'started',
                        $3,
                        $4::jsonb
                    )
                    RETURNING
                        publication_attempt_id
                    """,
                    generated_post_id,
                    attempt_number,
                    telegram_chat_id,
                    request_payload,
                )
            )

            await connection.execute(
                """
                UPDATE publication_batches
                SET
                    batch_status = 'publishing',
                    error_message = NULL
                WHERE batch_id = $1
                """,
                record["batch_id"],
            )

    publication = PublicationAttempt(
        batch_id=record["batch_id"],
        generated_post_id=(
            record["generated_post_id"]
        ),
        publication_attempt_id=(
            publication_attempt_id
        ),
        publication_date=(
            record["publication_date"]
        ),
        edition=record["edition"],
    )

    return PreparedApprovedPublication(
        publication=publication,
        telegram_chat_id=telegram_chat_id,
        rich_html=rich_html,
        rich_media_id=rich_media_id,
        image_path=image_path,
        image_sha256=image_sha256,
        source_text_format=source_text_format,
    )