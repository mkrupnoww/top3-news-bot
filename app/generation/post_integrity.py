from __future__ import annotations

import re

from app.generation.openai_generator import (
    OpenAIGeneratedPostPayload,
    build_top3_post_text,
)
from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
    TARGET_BODY_LENGTH_MAX,
)


_TERMINAL_PUNCTUATION = frozenset(
    {".", "!", "?", "…"}
)

_TRAILING_CLOSERS = frozenset(
    {
        '"',
        "'",
        "»",
        "”",
        "’",
        ")",
        "]",
        "}",
        "›",
    }
)

_MARKDOWN_LINK_AT_END_PATTERN = re.compile(
    r"\[[^\]]+\]\([^)]+\)\s*$"
)

_PLAIN_URL_AT_END_PATTERN = re.compile(
    r"https?://\S+\s*$",
    re.IGNORECASE,
)

_COMPLETE_SENTENCE_END_PATTERN = re.compile(
    r"[.!?…](?:[\"'»”’)\]}›*_]{0,6})(?=\s|$)"
)

_SUSPICIOUS_UNTERMINATED_BODY_MIN_LENGTH = (
    TARGET_BODY_LENGTH_MAX
)


def _strip_trailing_formatting(
    text: str,
) -> str:
    """Убирает только хвостовые closers/Markdown."""

    normalized = text.rstrip()
    changed = True

    while normalized and changed:
        changed = False

        while (
            normalized
            and normalized[-1]
            in _TRAILING_CLOSERS
        ):
            normalized = (
                normalized[:-1].rstrip()
            )
            changed = True

        for marker in (
            "**",
            "__",
            "*",
            "_",
        ):
            if normalized.endswith(marker):
                normalized = (
                    normalized[
                        :-len(marker)
                    ].rstrip()
                )
                changed = True
                break

    return normalized


def body_has_terminal_punctuation(
    body: str,
) -> bool:
    """Проверяет признак законченной фразы."""

    normalized = body.strip()

    if not normalized:
        return False

    if (
        _MARKDOWN_LINK_AT_END_PATTERN.search(
            normalized
        )
        is not None
    ):
        return True

    if (
        _PLAIN_URL_AT_END_PATTERN.search(
            normalized
        )
        is not None
    ):
        return True

    normalized = _strip_trailing_formatting(
        normalized
    )

    if not normalized:
        return False

    return (
        normalized[-1]
        in _TERMINAL_PUNCTUATION
    )


def body_has_suspicious_unterminated_tail(
    body: str,
) -> bool:
    """
    Срабатывает только для длинного body у лимита.

    Короткая нормальная фраза без финальной точки
    сама по себе больше не является integrity failure.
    """

    normalized = body.strip()

    if not normalized:
        return False

    return (
        len(normalized)
        >= _SUSPICIOUS_UNTERMINATED_BODY_MIN_LENGTH
        and not body_has_terminal_punctuation(
            normalized
        )
    )


def _last_complete_sentence_prefix(
    body: str,
) -> str | None:
    """Возвращает префикс до последней полной фразы."""

    normalized = body.strip()

    matches = list(
        _COMPLETE_SENTENCE_END_PATTERN.finditer(
            normalized
        )
    )

    if not matches:
        return None

    candidate = normalized[
        :matches[-1].end()
    ].rstrip()

    if not candidate:
        return None

    if not body_has_terminal_punctuation(
        candidate
    ):
        return None

    return candidate


def _headline_fallback_body(
    headline: str,
) -> str:
    """Строит минимальный factual body из headline."""

    normalized = headline.strip()

    if not normalized:
        raise ValueError(
            "Нельзя построить emergency body "
            "из пустого headline."
        )

    if normalized[-1] in _TERMINAL_PUNCTUATION:
        return normalized

    return normalized + "."


def _fallback_item_by_news_id(
    fallback_payload: (
        OpenAIGeneratedPostPayload
        | None
    ),
    *,
    news_id: int,
):
    """Ищет соответствующую новость в fallback payload."""

    if fallback_payload is None:
        return None

    for item in fallback_payload.items:
        if item.news_id == news_id:
            return item

    return None


def salvage_body_deterministically(
    *,
    body: str,
    headline: str,
    fallback_body: str | None = None,
) -> str:
    """
    Детерминированно восстанавливает body.

    Приоритет:
    1. Текущий body уже завершён.
    2. Полная часть текущего body до оборванного хвоста.
    3. Корректный body из primary generation.
    4. Уже сгенерированный headline как emergency body.
    """

    normalized_body = body.strip()

    if body_has_terminal_punctuation(
        normalized_body
    ):
        return normalized_body

    complete_prefix = (
        _last_complete_sentence_prefix(
            normalized_body
        )
    )

    if complete_prefix is not None:
        return complete_prefix

    if fallback_body is not None:
        normalized_fallback_body = (
            fallback_body.strip()
        )

        if body_has_terminal_punctuation(
            normalized_fallback_body
        ):
            return normalized_fallback_body

    return _headline_fallback_body(
        headline
    )


def validate_generated_post_integrity(
    payload: OpenAIGeneratedPostPayload,
) -> tuple[str, ...]:
    """Проверяет структурную и текстовую целостность."""

    issues: list[str] = []

    if len(payload.items) != 3:
        issues.append(
            "Payload должен содержать ровно "
            "3 новости."
        )

    positions = [
        item.position
        for item in payload.items
    ]

    if positions != [1, 2, 3]:
        issues.append(
            "Позиции новостей должны идти "
            "строго 1, 2, 3."
        )

    news_ids = [
        item.news_id
        for item in payload.items
    ]

    if len(set(news_ids)) != len(news_ids):
        issues.append(
            "В payload обнаружены "
            "повторяющиеся news_id."
        )

    for item in payload.items:
        headline = item.headline.strip()
        body = item.body.strip()

        if not headline:
            issues.append(
                "Новость position="
                f"{item.position} имеет пустой "
                "headline."
            )

        if not body:
            issues.append(
                "Новость position="
                f"{item.position} имеет пустой "
                "body."
            )
            continue

        if body_has_suspicious_unterminated_tail(
            body
        ):
            issues.append(
                "Новость position="
                f"{item.position} имеет длинный "
                "body без признака завершённого "
                "предложения в конце; возможен "
                "обрезанный хвост."
            )

    post_text = payload.post_text.strip()

    if not post_text:
        issues.append(
            "payload.post_text не может быть "
            "пустым."
        )

    try:
        canonical_post_text = (
            build_top3_post_text(
                payload.items
            )
        )
    except Exception as error:
        issues.append(
            "Не удалось канонически собрать "
            "post_text из payload.items: "
            f"{type(error).__name__}: {error}"
        )
    else:
        if (
            len(canonical_post_text)
            > MAXIMUM_POST_LENGTH
        ):
            issues.append(
                "Канонический post_text "
                "превышает допустимую длину: "
                f"actual={len(canonical_post_text)}, "
                f"limit={MAXIMUM_POST_LENGTH}."
            )

        if post_text != canonical_post_text:
            issues.append(
                "payload.post_text не совпадает "
                "с канонической сборкой из "
                "payload.items."
            )

    return tuple(issues)


def build_post_integrity_editorial_comment(
    issues: tuple[str, ...],
) -> str:
    """Формирует замечание для bounded revision."""

    normalized_issues = tuple(
        issue.strip()
        for issue in issues
        if issue.strip()
    )

    if not normalized_issues:
        return (
            "Проверь целостность текста и "
            "сохрани корректный пост без "
            "лишних изменений."
        )

    issues_text = "\n".join(
        f"- {issue}"
        for issue in normalized_issues
    )

    return (
        "Исправь только обнаруженные "
        "технические проблемы целостности "
        "готового Telegram-поста. "
        "Не меняй набор новостей, порядок, "
        "news_id и фактический смысл. "
        "Не сокращай корректные фрагменты "
        "без необходимости. Если body "
        "упирается в лимит, сократи его так, "
        "чтобы мысль была полностью закончена, "
        "вместо попытки использовать все "
        "доступные символы. Каждый исправляемый "
        "body должен заканчиваться точкой, "
        "вопросительным знаком, восклицательным "
        "знаком или многоточием, если только "
        "в самом конце не стоит допустимая "
        "ссылка.\n\n"
        "Обнаруженные проблемы:\n"
        f"{issues_text}"
    )


def build_deterministic_integrity_fallback(
    payload: OpenAIGeneratedPostPayload,
    *,
    fallback_payload: (
        OpenAIGeneratedPostPayload
        | None
    ) = None,
) -> OpenAIGeneratedPostPayload:
    """Последний локальный fail-safe после model repairs."""

    repaired_items = []
    repaired_news_ids: set[int] = set()

    for item in payload.items:
        if not body_has_suspicious_unterminated_tail(
            item.body
        ):
            repaired_items.append(item)
            continue

        fallback_item = (
            _fallback_item_by_news_id(
                fallback_payload,
                news_id=item.news_id,
            )
        )

        repaired_body = (
            salvage_body_deterministically(
                body=item.body,
                headline=item.headline,
                fallback_body=(
                    fallback_item.body
                    if fallback_item is not None
                    else None
                ),
            )
        )

        repaired_items.append(
            item.model_copy(
                update={
                    "body": repaired_body,
                }
            )
        )
        repaired_news_ids.add(item.news_id)

    try:
        canonical_post_text = (
            build_top3_post_text(
                repaired_items
            )
        )
    except ValueError:
        compact_items = []

        for item in repaired_items:
            if item.news_id in repaired_news_ids:
                compact_items.append(
                    item.model_copy(
                        update={
                            "body": (
                                _headline_fallback_body(
                                    item.headline
                                )
                            ),
                        }
                    )
                )
            else:
                compact_items.append(item)

        repaired_items = compact_items
        canonical_post_text = (
            build_top3_post_text(
                repaired_items
            )
        )

    repaired_payload = payload.model_copy(
        update={
            "items": repaired_items,
            "post_text": canonical_post_text,
        }
    )

    remaining_issues = (
        validate_generated_post_integrity(
            repaired_payload
        )
    )

    if remaining_issues:
        raise ValueError(
            "Deterministic integrity fallback "
            "не смог построить валидный post: "
            + "; ".join(remaining_issues)
        )

    return repaired_payload


def assert_generated_post_integrity(
    payload: OpenAIGeneratedPostPayload,
) -> None:
    """Бросает ValueError при failed gate."""

    issues = validate_generated_post_integrity(
        payload
    )

    if issues:
        raise ValueError(
            "Integrity gate failed: "
            + "; ".join(issues)
        )
