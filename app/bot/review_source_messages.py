from datetime import timezone

from app.db.review_sources import ReviewSourceItem


def _truncate(
    value: str,
    *,
    max_length: int,
) -> str:
    """Сокращает длинное поле для сообщения Telegram."""

    normalized_value = value.strip()

    if len(normalized_value) <= max_length:
        return normalized_value

    return (
        normalized_value[: max_length - 1].rstrip()
        + "…"
    )


def _format_published_at(
    item: ReviewSourceItem,
) -> str:
    """Форматирует дату публикации источника в UTC."""

    if item.source_published_at is None:
        return "не указано"

    published_at = (
        item.source_published_at.astimezone(
            timezone.utc
        )
    )

    return published_at.strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def build_review_source_messages(
    items: tuple[ReviewSourceItem, ...],
) -> tuple[str, ...]:
    """Формирует три карточки источников для проверки."""

    if len(items) != 3:
        raise ValueError(
            "Для проверки должно быть ровно "
            f"три источника: news_count={len(items)}"
        )

    ordered_items = sorted(
        items,
        key=lambda item: item.position,
    )

    messages: list[str] = []

    for item in ordered_items:
        title = _truncate(
            item.title,
            max_length=800,
        )

        source_name = _truncate(
            item.source_name,
            max_length=250,
        )

        selection_reason = _truncate(
            item.selection_reason
            or "Причина выбора не указана.",
            max_length=700,
        )

        message_text = (
            f"Источник новости №{item.position}\n\n"
            f"News ID: {item.news_id}\n"
            f"Заголовок: {title}\n"
            f"Источник: {source_name}\n"
            "Опубликовано: "
            f"{_format_published_at(item)}\n\n"
            f"Причина выбора:\n{selection_reason}\n\n"
            f"Исходная публикация:\n{item.source_url}"
        )

        if len(message_text) > 4096:
            raise ValueError(
                "Карточка источника превышает "
                "лимит Telegram: "
                f"position={item.position}, "
                f"length={len(message_text)}"
            )

        messages.append(message_text)

    return tuple(messages)