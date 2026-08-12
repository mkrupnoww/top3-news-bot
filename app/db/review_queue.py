from dataclasses import dataclass
from datetime import date
import json
from typing import Literal

import asyncpg


ReviewDecision = Literal[
    "approve",
    "reject",
    "changes_required",
]

ReviewRequestedAction = Literal[
    "regenerate_text",
]


@dataclass(frozen=True, slots=True)
class ReviewDraftPreview:
    """Черновик, ожидающий ручной проверки."""

    batch_id: int
    generated_post_id: int
    publication_date: date
    edition: int
    version_number: int
    post_text: str
    text_format: str
    image_path: str | None
    image_sha256: str | None


@dataclass(frozen=True, slots=True)
class ReviewDecisionResult:
    """Результат ручного решения по черновику."""

    batch_id: int
    generated_post_id: int
    review_action_id: int | None
    decision: ReviewDecision
    post_status: str
    batch_status: str
    already_processed: bool


def _normalize_changes_required_input(
    *,
    requested_action: ReviewRequestedAction | None,
    comment_text: str | None,
    issues: tuple[str, ...],
) -> tuple[
    ReviewRequestedAction,
    str,
    tuple[str, ...],
]:
    """Проверяет параметры редакционной доработки."""

    if requested_action != "regenerate_text":
        raise ValueError(
            "Для changes_required требуется "
            "requested_action='regenerate_text'."
        )

    if not isinstance(comment_text, str):
        raise TypeError(
            "comment_text должен быть строкой."
        )

    normalized_comment = comment_text.strip()

    if not normalized_comment:
        raise ValueError(
            "comment_text не может быть пустым "
            "для changes_required."
        )

    if not isinstance(issues, tuple):
        raise TypeError(
            "issues должен быть tuple."
        )

    if not issues:
        raise ValueError(
            "issues не может быть пустым "
            "для changes_required."
        )

    normalized_issues: list[str] = []

    for index, issue in enumerate(
        issues,
        start=1,
    ):
        if not isinstance(issue, str):
            raise TypeError(
                "Каждый элемент issues должен "
                "быть строкой: "
                f"index={index}"
            )

        normalized_issue = issue.strip()

        if not normalized_issue:
            raise ValueError(
                "Элемент issues не может быть "
                "пустым: "
                f"index={index}"
            )

        normalized_issues.append(
            normalized_issue
        )

    return (
        requested_action,
        normalized_comment,
        tuple(normalized_issues),
    )


async def get_latest_review_draft(
    pool: asyncpg.Pool,
) -> ReviewDraftPreview | None:
    """
    Возвращает последний черновик,
    ожидающий ручной проверки.

    Черновик с уже зафиксированным
    changes_required не возвращается повторно:
    он ожидает отдельного revision-потока.
    """

    query = """
        SELECT
            b.batch_id,
            b.publication_date,
            b.edition,
            p.generated_post_id,
            p.version_number,
            p.post_text,
            p.text_format
            p.image_path,
            p.image_sha256
        FROM publication_batches AS b
        JOIN generated_posts AS p
            ON p.batch_id = b.batch_id
        WHERE b.batch_status = 'awaiting_review'
          AND p.post_status = 'awaiting_review'
          AND NOT EXISTS (
              SELECT 1
              FROM review_actions AS ra
              WHERE ra.generated_post_id =
                    p.generated_post_id
                AND ra.reviewer_type = 'human'
                AND ra.decision =
                    'changes_required'
                AND ra.requested_action =
                    'regenerate_text'
          )
        ORDER BY
            b.publication_date DESC,
            b.edition DESC,
            p.version_number DESC
        LIMIT 1
    """

    async with pool.acquire() as connection:
        record = await connection.fetchrow(query)

    if record is None:
        return None

    return ReviewDraftPreview(
        batch_id=record["batch_id"],
        generated_post_id=record[
            "generated_post_id"
        ],
        publication_date=record[
            "publication_date"
        ],
        edition=record["edition"],
        version_number=record[
            "version_number"
        ],
        post_text=record["post_text"],
        text_format=record["text_format"],
        image_path=record["image_path"],
        image_sha256=record["image_sha256"],
    )


async def record_human_review_decision(
    pool: asyncpg.Pool,
    *,
    generated_post_id: int,
    reviewer_telegram_user_id: int,
    decision: ReviewDecision,
    requested_action: (
        ReviewRequestedAction | None
    ) = None,
    comment_text: str | None = None,
    issues: tuple[str, ...] = (),
) -> ReviewDecisionResult:
    """
    Фиксирует решение человека по черновику.

    Разрешены решения:

    awaiting_review -> approved
    awaiting_review -> rejected
    awaiting_review -> changes_required

    Для changes_required статусы поста и подборки
    остаются awaiting_review до успешного
    revision-потока. Сам черновик исключается
    из очереди ручной проверки через review_action.
    """

    if decision not in {
        "approve",
        "reject",
        "changes_required",
    }:
        raise ValueError(
            f"Неподдерживаемое решение: {decision}"
        )

    normalized_requested_action: (
        ReviewRequestedAction | None
    ) = None

    normalized_comment_text: str
    normalized_issues: tuple[str, ...]

    if decision == "changes_required":
        (
            normalized_requested_action,
            normalized_comment_text,
            normalized_issues,
        ) = _normalize_changes_required_input(
            requested_action=requested_action,
            comment_text=comment_text,
            issues=issues,
        )
    else:
        if requested_action is not None:
            raise ValueError(
                "requested_action разрешён только "
                "для changes_required."
            )

        if issues:
            raise ValueError(
                "issues разрешены только для "
                "changes_required."
            )

        if comment_text is None:
            normalized_comment_text = (
                "Решение принято через "
                "inline-кнопку Telegram-бота."
            )
        else:
            if not isinstance(
                comment_text,
                str,
            ):
                raise TypeError(
                    "comment_text должен быть "
                    "строкой."
                )

            normalized_comment_text = (
                comment_text.strip()
            )

            if not normalized_comment_text:
                raise ValueError(
                    "comment_text не может "
                    "быть пустым."
                )

        normalized_issues = ()

    review_details = json.dumps(
        {
            "source": "telegram_inline_keyboard",
            "generated_post_id": (
                generated_post_id
            ),
            "requested_action": (
                normalized_requested_action
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    issues_json = json.dumps(
        list(normalized_issues),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    async with pool.acquire() as connection:
        async with connection.transaction():
            reviewer = await connection.fetchrow(
                """
                SELECT
                    telegram_user_id,
                    user_role
                FROM bot_users
                WHERE telegram_user_id = $1
                  AND is_active = true
                  AND user_role IN (
                      'admin',
                      'reviewer'
                  )
                """,
                reviewer_telegram_user_id,
            )

            if reviewer is None:
                raise PermissionError(
                    "Пользователь не имеет права "
                    "проверять публикации."
                )

            record = await connection.fetchrow(
                """
                SELECT
                    p.generated_post_id,
                    p.batch_id,
                    p.post_status,
                    b.batch_status
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
                    "Черновик не найден: "
                    f"generated_post_id="
                    f"{generated_post_id}"
                )

            if decision in {
                "approve",
                "reject",
            }:
                target_status = (
                    "approved"
                    if decision == "approve"
                    else "rejected"
                )

                if (
                    record["post_status"]
                    == target_status
                    and record["batch_status"]
                    == target_status
                ):
                    return ReviewDecisionResult(
                        batch_id=record["batch_id"],
                        generated_post_id=(
                            generated_post_id
                        ),
                        review_action_id=None,
                        decision=decision,
                        post_status=target_status,
                        batch_status=target_status,
                        already_processed=True,
                    )

            existing_changes_required = (
                await connection.fetchrow(
                    """
                    SELECT
                        review_action_id,
                        reviewer_telegram_user_id,
                        comment_text,
                        issues::text AS issues_json
                    FROM review_actions
                    WHERE generated_post_id = $1
                      AND reviewer_type = 'human'
                      AND decision =
                          'changes_required'
                      AND requested_action =
                          'regenerate_text'
                    ORDER BY review_action_id DESC
                    LIMIT 1
                    """,
                    generated_post_id,
                )
            )

            if (
                existing_changes_required
                is not None
            ):
                if decision == "changes_required":
                    existing_issues = tuple(
                        json.loads(
                            existing_changes_required[
                                "issues_json"
                            ]
                        )
                    )

                    same_request = (
                        existing_changes_required[
                            "reviewer_telegram_user_id"
                        ]
                        == reviewer_telegram_user_id
                        and existing_changes_required[
                            "comment_text"
                        ]
                        == normalized_comment_text
                        and existing_issues
                        == normalized_issues
                    )

                    if not same_request:
                        raise ValueError(
                            "По посту уже зафиксирован "
                            "changes_required с другими "
                            "параметрами."
                        )

                    return ReviewDecisionResult(
                        batch_id=record["batch_id"],
                        generated_post_id=(
                            generated_post_id
                        ),
                        review_action_id=(
                            existing_changes_required[
                                "review_action_id"
                            ]
                        ),
                        decision=decision,
                        post_status=record[
                            "post_status"
                        ],
                        batch_status=record[
                            "batch_status"
                        ],
                        already_processed=True,
                    )

                raise ValueError(
                    "По посту уже запрошена "
                    "редакционная доработка."
                )

            if (
                record["post_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Пост уже не ожидает проверки: "
                    f"post_status="
                    f"{record['post_status']}"
                )

            if (
                record["batch_status"]
                != "awaiting_review"
            ):
                raise ValueError(
                    "Подборка уже не ожидает "
                    "проверки: "
                    f"batch_status="
                    f"{record['batch_status']}"
                )

            review_action_id = (
                await connection.fetchval(
                    """
                    INSERT INTO review_actions (
                        generated_post_id,
                        reviewer_type,
                        reviewer_telegram_user_id,
                        decision,
                        requested_action,
                        requires_human_review,
                        comment_text,
                        issues,
                        review_details
                    )
                    VALUES (
                        $1,
                        'human',
                        $2,
                        $3,
                        $4,
                        false,
                        $5,
                        $6::jsonb,
                        $7::jsonb
                    )
                    RETURNING review_action_id
                    """,
                    generated_post_id,
                    reviewer_telegram_user_id,
                    decision,
                    normalized_requested_action,
                    normalized_comment_text,
                    issues_json,
                    review_details,
                )
            )

            if decision == "changes_required":
                return ReviewDecisionResult(
                    batch_id=record["batch_id"],
                    generated_post_id=(
                        generated_post_id
                    ),
                    review_action_id=(
                        review_action_id
                    ),
                    decision=decision,
                    post_status=record[
                        "post_status"
                    ],
                    batch_status=record[
                        "batch_status"
                    ],
                    already_processed=False,
                )

            target_status = (
                "approved"
                if decision == "approve"
                else "rejected"
            )

            await connection.execute(
                """
                UPDATE generated_posts
                SET post_status = $2
                WHERE generated_post_id = $1
                """,
                generated_post_id,
                target_status,
            )

            if decision == "approve":
                await connection.execute(
                    """
                    UPDATE publication_batches
                    SET
                        batch_status = 'approved',
                        approved_at = now(),
                        approved_by_telegram_user_id =
                            $2,
                        error_message = NULL
                    WHERE batch_id = $1
                    """,
                    record["batch_id"],
                    reviewer_telegram_user_id,
                )
            else:
                await connection.execute(
                    """
                    UPDATE publication_batches
                    SET
                        batch_status = 'rejected',
                        approved_at = NULL,
                        approved_by_telegram_user_id =
                            NULL,
                        error_message = NULL
                    WHERE batch_id = $1
                    """,
                    record["batch_id"],
                )

    return ReviewDecisionResult(
        batch_id=record["batch_id"],
        generated_post_id=generated_post_id,
        review_action_id=review_action_id,
        decision=decision,
        post_status=target_status,
        batch_status=target_status,
        already_processed=False,
    )