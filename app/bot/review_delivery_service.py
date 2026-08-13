import asyncio
from dataclasses import dataclass

import asyncpg
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramNetworkError,
    TelegramServerError,
)
from aiogram.types import InlineKeyboardMarkup

from app.bot.review_preview import (
    ReviewPhotoSender,
    prepare_review_photo,
    send_review_draft_photo,
)
from app.db.review_delivery import (
    ReviewDeliveryReservation,
    get_review_draft_by_id,
    list_active_reviewers,
    mark_review_delivery_failed,
    mark_review_delivery_sent,
    mark_review_delivery_unknown,
    reserve_review_delivery,
)
from app.db.review_queue import ReviewDraftPreview
from app.db.users import BotUser


class ReviewDeliveryStateUncertainError(RuntimeError):
    """
    Telegram-доставка могла состояться,
    но состояние в PostgreSQL нельзя
    надёжно финализировать.
    """


@dataclass(frozen=True, slots=True)
class ReviewDeliveryOutcome:
    """Результат доставки одному reviewer."""

    telegram_user_id: int
    review_delivery_attempt_id: int
    attempt_number: int
    delivery_status: str
    telegram_message_id: int | None
    telegram_called: bool


@dataclass(frozen=True, slots=True)
class ReviewDeliveryResult:
    """Результат доставки generated_post reviewer-ам."""

    generated_post_id: int
    reviewer_count: int
    outcomes: tuple[ReviewDeliveryOutcome, ...]

    @property
    def sent_count(self) -> int:
        return sum(
            item.delivery_status == "sent"
            for item in self.outcomes
        )

    @property
    def failed_count(self) -> int:
        return sum(
            item.delivery_status == "failed"
            for item in self.outcomes
        )

    @property
    def unknown_count(self) -> int:
        return sum(
            item.delivery_status == "unknown"
            for item in self.outcomes
        )

    @property
    def skipped_count(self) -> int:
        return sum(
            item.telegram_called is False
            for item in self.outcomes
        )


def _error_message(
    error: BaseException,
) -> str:
    """Возвращает непустое описание ошибки."""

    message = str(error).strip()

    if message:
        return message

    return type(error).__name__


def _telegram_error_code(
    error: BaseException,
) -> int | None:
    """Извлекает Telegram error code, если он доступен."""

    value = getattr(
        error,
        "error_code",
        None,
    )

    if isinstance(value, int):
        return value

    return None


def _request_payload(
    *,
    draft: ReviewDraftPreview,
    reviewer: BotUser,
) -> dict[str, object]:
    """Формирует аудит запроса review delivery."""

    return {
        "transport": "native_photo",
        "generated_post_id": (
            draft.generated_post_id
        ),
        "batch_id": draft.batch_id,
        "version_number": (
            draft.version_number
        ),
        "telegram_user_id": (
            reviewer.telegram_user_id
        ),
        "image_sha256": draft.image_sha256,
        "post_length": len(
            draft.post_text
        ),
    }


def _outcome_from_reservation(
    reservation: ReviewDeliveryReservation,
    *,
    delivery_status: str | None = None,
    telegram_message_id: int | None = None,
    telegram_called: bool,
) -> ReviewDeliveryOutcome:
    """Собирает унифицированный outcome."""

    return ReviewDeliveryOutcome(
        telegram_user_id=(
            reservation.telegram_user_id
        ),
        review_delivery_attempt_id=(
            reservation
            .review_delivery_attempt_id
        ),
        attempt_number=(
            reservation.attempt_number
        ),
        delivery_status=(
            delivery_status
            if delivery_status is not None
            else reservation.delivery_status
        ),
        telegram_message_id=(
            telegram_message_id
        ),
        telegram_called=telegram_called,
    )


async def _mark_unknown_after_error(
    pool: asyncpg.Pool,
    reservation: ReviewDeliveryReservation,
    *,
    telegram_message_id: int | None,
    response_payload: dict[str, object],
    error: BaseException,
) -> ReviewDeliveryOutcome:
    """Фиксирует неопределённый Telegram outcome."""

    await mark_review_delivery_unknown(
        pool,
        reservation,
        telegram_message_id=(
            telegram_message_id
        ),
        response_payload=response_payload,
        error_type=type(error).__name__,
        error_message=_error_message(
            error
        ),
    )

    return _outcome_from_reservation(
        reservation,
        delivery_status="unknown",
        telegram_message_id=(
            telegram_message_id
        ),
        telegram_called=True,
    )


async def deliver_review_draft_to_reviewer(
    pool: asyncpg.Pool,
    *,
    bot: ReviewPhotoSender,
    draft: ReviewDraftPreview,
    reviewer: BotUser,
    reply_markup: InlineKeyboardMarkup,
) -> ReviewDeliveryOutcome:
    """
    Доставляет один generated_post одному reviewer.

    Семантика:
    - explicit Telegram API rejection -> failed;
    - network/server/timeout -> unknown;
    - успешный message_id -> sent;
    - существующий reserved/sent/unknown -> skip.
    """

    # Локальные данные проверяем до reservation.
    # Если PNG или caption некорректны, Telegram
    # ещё не затронут и lifecycle не создаётся.
    prepare_review_photo(
        draft
    )

    reservation = await reserve_review_delivery(
        pool,
        generated_post_id=(
            draft.generated_post_id
        ),
        telegram_user_id=(
            reviewer.telegram_user_id
        ),
        request_payload=_request_payload(
            draft=draft,
            reviewer=reviewer,
        ),
    )

    if not reservation.should_send:
        return _outcome_from_reservation(
            reservation,
            telegram_called=False,
        )

    try:
        message = await send_review_draft_photo(
            bot,
            chat_id=(
                reviewer.telegram_user_id
            ),
            draft=draft,
            reply_markup=reply_markup,
        )

    except (
        TelegramNetworkError,
        TelegramServerError,
        asyncio.TimeoutError,
        TimeoutError,
        OSError,
    ) as error:
        return await _mark_unknown_after_error(
            pool,
            reservation,
            telegram_message_id=None,
            response_payload={},
            error=error,
        )

    except TelegramAPIError as error:
        await mark_review_delivery_failed(
            pool,
            reservation,
            error_type=type(error).__name__,
            error_message=_error_message(
                error
            ),
            telegram_error_code=(
                _telegram_error_code(
                    error
                )
            ),
        )

        return _outcome_from_reservation(
            reservation,
            delivery_status="failed",
            telegram_called=True,
        )

    except Exception as error:
        # После входа в send_photo любой
        # неожиданный outcome считаем uncertain.
        return await _mark_unknown_after_error(
            pool,
            reservation,
            telegram_message_id=None,
            response_payload={},
            error=error,
        )

    telegram_message_id = getattr(
        message,
        "message_id",
        None,
    )

    if (
        not isinstance(
            telegram_message_id,
            int,
        )
        or telegram_message_id <= 0
    ):
        error = RuntimeError(
            "Telegram send_photo вернул "
            "ответ без корректного message_id."
        )

        return await _mark_unknown_after_error(
            pool,
            reservation,
            telegram_message_id=None,
            response_payload={},
            error=error,
        )

    response_payload: dict[str, object] = {
        "transport": "native_photo",
        "telegram_user_id": (
            reviewer.telegram_user_id
        ),
        "telegram_message_id": (
            telegram_message_id
        ),
    }

    try:
        await mark_review_delivery_sent(
            pool,
            reservation,
            telegram_message_id=(
                telegram_message_id
            ),
            response_payload=(
                response_payload
            ),
        )

    except Exception as finalize_error:
        try:
            return await _mark_unknown_after_error(
                pool,
                reservation,
                telegram_message_id=(
                    telegram_message_id
                ),
                response_payload=(
                    response_payload
                ),
                error=finalize_error,
            )
        except Exception as unknown_error:
            raise ReviewDeliveryStateUncertainError(
                "Telegram review-message уже "
                "получил message_id, но PostgreSQL "
                "не удалось финализировать как "
                "sent или unknown: "
                "generated_post_id="
                f"{draft.generated_post_id}, "
                "telegram_user_id="
                f"{reviewer.telegram_user_id}, "
                "telegram_message_id="
                f"{telegram_message_id}"
            ) from unknown_error

    return _outcome_from_reservation(
        reservation,
        delivery_status="sent",
        telegram_message_id=(
            telegram_message_id
        ),
        telegram_called=True,
    )


async def deliver_generated_post_to_reviewers(
    pool: asyncpg.Pool,
    *,
    bot: ReviewPhotoSender,
    generated_post_id: int,
    reply_markup: InlineKeyboardMarkup,
) -> ReviewDeliveryResult:
    """
    Доставляет конкретный awaiting_review post
    всем активным admin/reviewer.

    Отправка последовательная: review-сообщений
    мало, а так проще контролировать Telegram
    rate limits и lifecycle каждой попытки.
    """

    draft = await get_review_draft_by_id(
        pool,
        generated_post_id=(
            generated_post_id
        ),
    )

    if draft is None:
        raise LookupError(
            "Awaiting-review draft не найден: "
            "generated_post_id="
            f"{generated_post_id}"
        )

    # Проверяем весь локальный артефакт
    # до первой Telegram reservation.
    prepare_review_photo(
        draft
    )

    reviewers = await list_active_reviewers(
        pool
    )

    if not reviewers:
        raise LookupError(
            "Нет активных admin/reviewer "
            "для автоматической review delivery."
        )

    outcomes: list[
        ReviewDeliveryOutcome
    ] = []

    for reviewer in reviewers:
        outcome = (
            await deliver_review_draft_to_reviewer(
                pool,
                bot=bot,
                draft=draft,
                reviewer=reviewer,
                reply_markup=reply_markup,
            )
        )

        outcomes.append(
            outcome
        )

    return ReviewDeliveryResult(
        generated_post_id=(
            draft.generated_post_id
        ),
        reviewer_count=len(
            reviewers
        ),
        outcomes=tuple(
            outcomes
        ),
    )
