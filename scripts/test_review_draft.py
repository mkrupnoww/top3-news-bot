import asyncio
from datetime import UTC, datetime
import json

from app.config import get_settings
from app.db.pool import (
    close_database_pool,
    create_database_pool,
)
from app.db.review_drafts import create_review_draft
from app.db.review_queue import (
    get_latest_review_draft,
    record_human_review_decision,
)


REVIEW_COMMENT = (
    "Исправить текст тестового черновика "
    "перед публикацией."
)

REVIEW_ISSUES = (
    "Проверить фактические формулировки.",
    "Убрать неподтверждённые детали.",
)


async def _get_test_reviewer_id(
    database_pool,
) -> int:
    """Возвращает активного admin/reviewer."""

    record = await database_pool.fetchrow(
        """
        SELECT
            telegram_user_id
        FROM bot_users
        WHERE is_active = true
          AND user_role IN (
              'admin',
              'reviewer'
          )
        ORDER BY
            CASE
                WHEN user_role = 'admin'
                    THEN 0
                ELSE 1
            END,
            telegram_user_id
        LIMIT 1
        """
    )

    if record is None:
        raise RuntimeError(
            "В bot_users нет активного "
            "admin/reviewer для теста."
        )

    return int(record["telegram_user_id"])


async def _cleanup_test_draft(
    database_pool,
    *,
    batch_id: int,
    generated_post_id: int,
) -> None:
    """Удаляет только записи этого теста."""

    async with database_pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                DELETE FROM review_actions
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM publication_attempts
                WHERE generated_post_id = $1
                """,
                generated_post_id,
            )

            await connection.execute(
                """
                DELETE FROM batch_items
                WHERE batch_id = $1
                """,
                batch_id,
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


async def main() -> None:
    """
    Проверяет ручное решение changes_required.

    Реальные OpenAI- и Telegram-запросы
    не выполняются.
    """

    settings = get_settings()
    now = datetime.now(UTC)

    post_text = (
        "🎬 Предпросмотр TOP 3 Movie News\n\n"
        "Это тестовый черновик будущей "
        "публикации.\n\n"
        "Он сохранён в PostgreSQL со статусом "
        "awaiting_review, но не отправляется "
        "в Telegram-канал.\n\n"
        f"Время создания: "
        f"{now:%Y-%m-%d %H:%M:%S} UTC"
    )

    metadata = {
        "technical_test": True,
        "scenario": (
            "review_changes_required_test"
        ),
        "script": "scripts.test_review_draft",
        "created_at": now.isoformat(),
        "batch_items_expected": False,
        "telegram_publication_expected": False,
    }

    database_pool = await create_database_pool(
        settings
    )

    draft = None
    cleanup_completed = False

    try:
        reviewer_telegram_user_id = (
            await _get_test_reviewer_id(
                database_pool
            )
        )

        draft = await create_review_draft(
            database_pool,
            publication_date=now.date(),
            telegram_chat_id=(
                settings.telegram_channel_id
            ),
            post_text=post_text,
            text_format="plain_text",
            metadata=metadata,
        )

        attempt_count_before = (
            await database_pool.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM publication_attempts
                WHERE generated_post_id = $1
                """,
                draft.generated_post_id,
            )
        )

        if attempt_count_before != 0:
            raise RuntimeError(
                "Для тестового черновика "
                "неожиданно создана попытка "
                "публикации."
            )

        first_result = (
            await record_human_review_decision(
                database_pool,
                generated_post_id=(
                    draft.generated_post_id
                ),
                reviewer_telegram_user_id=(
                    reviewer_telegram_user_id
                ),
                decision="changes_required",
                requested_action="regenerate_text",
                comment_text=REVIEW_COMMENT,
                issues=REVIEW_ISSUES,
            )
        )

        if first_result.already_processed:
            raise RuntimeError(
                "Первый changes_required "
                "не должен считаться повторным."
            )

        if first_result.review_action_id is None:
            raise RuntimeError(
                "Не создан review_action "
                "для changes_required."
            )

        if first_result.post_status != (
            "awaiting_review"
        ):
            raise RuntimeError(
                "После changes_required "
                "post_status должен остаться "
                "awaiting_review."
            )

        if first_result.batch_status != (
            "awaiting_review"
        ):
            raise RuntimeError(
                "После changes_required "
                "batch_status должен остаться "
                "awaiting_review."
            )

        review_record = (
            await database_pool.fetchrow(
                """
                SELECT
                    review_action_id,
                    generated_post_id,
                    reviewer_type,
                    reviewer_telegram_user_id,
                    decision,
                    requested_action,
                    requires_human_review,
                    comment_text,
                    issues::text AS issues_json,
                    review_details::text
                        AS review_details_json
                FROM review_actions
                WHERE review_action_id = $1
                """,
                first_result.review_action_id,
            )
        )

        if review_record is None:
            raise RuntimeError(
                "Созданный review_action "
                "не найден."
            )

        if review_record["decision"] != (
            "changes_required"
        ):
            raise RuntimeError(
                "В review_actions сохранено "
                "неверное decision."
            )

        if review_record["requested_action"] != (
            "regenerate_text"
        ):
            raise RuntimeError(
                "В review_actions сохранено "
                "неверное requested_action."
            )

        if review_record["comment_text"] != (
            REVIEW_COMMENT
        ):
            raise RuntimeError(
                "Комментарий редактора "
                "сохранён неверно."
            )

        stored_issues = tuple(
            json.loads(
                review_record["issues_json"]
            )
        )

        if stored_issues != REVIEW_ISSUES:
            raise RuntimeError(
                "issues сохранены неверно."
            )

        review_details = json.loads(
            review_record[
                "review_details_json"
            ]
        )

        if review_details.get(
            "generated_post_id"
        ) != draft.generated_post_id:
            raise RuntimeError(
                "review_details содержит "
                "неверный generated_post_id."
            )

        if review_details.get(
            "requested_action"
        ) != "regenerate_text":
            raise RuntimeError(
                "review_details содержит "
                "неверный requested_action."
            )

        repeated_result = (
            await record_human_review_decision(
                database_pool,
                generated_post_id=(
                    draft.generated_post_id
                ),
                reviewer_telegram_user_id=(
                    reviewer_telegram_user_id
                ),
                decision="changes_required",
                requested_action="regenerate_text",
                comment_text=REVIEW_COMMENT,
                issues=REVIEW_ISSUES,
            )
        )

        if not repeated_result.already_processed:
            raise RuntimeError(
                "Повторный changes_required "
                "должен быть идемпотентным."
            )

        if (
            repeated_result.review_action_id
            != first_result.review_action_id
        ):
            raise RuntimeError(
                "Повторный changes_required "
                "создал другой review_action."
            )

        review_action_count = (
            await database_pool.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM review_actions
                WHERE generated_post_id = $1
                  AND reviewer_type = 'human'
                  AND decision =
                      'changes_required'
                  AND requested_action =
                      'regenerate_text'
                """,
                draft.generated_post_id,
            )
        )

        if review_action_count != 1:
            raise RuntimeError(
                "Для одного черновика должен "
                "существовать ровно один "
                "changes_required review_action."
            )

        status_record = (
            await database_pool.fetchrow(
                """
                SELECT
                    p.post_status,
                    b.batch_status
                FROM generated_posts AS p
                JOIN publication_batches AS b
                    ON b.batch_id = p.batch_id
                WHERE p.generated_post_id = $1
                """,
                draft.generated_post_id,
            )
        )

        if status_record is None:
            raise RuntimeError(
                "Тестовый generated_post "
                "не найден после review."
            )

        if (
            status_record["post_status"]
            != "awaiting_review"
            or status_record["batch_status"]
            != "awaiting_review"
        ):
            raise RuntimeError(
                "changes_required не должен "
                "переводить пост или batch "
                "в approved/rejected."
            )

        latest_draft = (
            await get_latest_review_draft(
                database_pool
            )
        )

        if (
            latest_draft is not None
            and latest_draft.generated_post_id
            == draft.generated_post_id
        ):
            raise RuntimeError(
                "Черновик с changes_required "
                "не должен повторно попадать "
                "в очередь ручной проверки."
            )

        attempt_count_after = (
            await database_pool.fetchval(
                """
                SELECT COUNT(*)::integer
                FROM publication_attempts
                WHERE generated_post_id = $1
                """,
                draft.generated_post_id,
            )
        )

        if attempt_count_after != 0:
            raise RuntimeError(
                "changes_required неожиданно "
                "создал попытку публикации."
            )

        print(
            "Review changes_required test"
        )
        print(
            "database_connection=performed"
        )
        print(
            "openai_requests=not_performed"
        )
        print(
            "telegram_requests=not_performed"
        )
        print(
            f"batch_id={draft.batch_id}"
        )
        print(
            "generated_post_id="
            f"{draft.generated_post_id}"
        )
        print(
            "review_action_id="
            f"{first_result.review_action_id}"
        )
        print(
            "reviewer_telegram_user_id="
            f"{reviewer_telegram_user_id}"
        )
        print(
            "decision=changes_required"
        )
        print(
            "requested_action=regenerate_text"
        )
        print(
            "post_status=awaiting_review"
        )
        print(
            "batch_status=awaiting_review"
        )
        print(
            "repeated_request_blocked=true"
        )
        print(
            "review_action_count=1"
        )
        print(
            "publication_attempt_count=0"
        )
        print(
            "review_queue_exclusion=true"
        )

    finally:
        if draft is not None:
            await _cleanup_test_draft(
                database_pool,
                batch_id=draft.batch_id,
                generated_post_id=(
                    draft.generated_post_id
                ),
            )
            cleanup_completed = True

        await close_database_pool(
            database_pool
        )

    print(
        "temporary_review_data_deleted="
        f"{str(cleanup_completed).lower()}"
    )
    print(
        "Review changes_required test: OK"
    )


if __name__ == "__main__":
    asyncio.run(main())