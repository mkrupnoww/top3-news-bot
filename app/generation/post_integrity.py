from __future__ import annotations

import re

from app.generation.openai_generator import (
    OpenAIGeneratedPostPayload,
    build_top3_post_text,
)
from app.generation.post_contract import (
    MAXIMUM_POST_LENGTH,
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


def _strip_trailing_formatting(
    text: str,
) -> str:
    """
    Удаляет только хвостовые закрывающие
    кавычки, скобки и Markdown-маркеры.

    Содержимое текста не изменяется.
    """

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

        for marker in ("**", "__", "*", "_"):
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
    """
    Проверяет, что body имеет признак
    законченной фразы.

    Допускаются:
    - ., !, ?, …;
    - закрывающие кавычки/скобки после них;
    - Markdown-маркеры после них;
    - Markdown-ссылка или plain URL в самом конце.

    Последний вариант нужен для корректных
    публикаций, где body завершается ссылкой
    на подтверждённый официальный материал.
    """

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


def validate_generated_post_integrity(
    payload: OpenAIGeneratedPostPayload,
) -> tuple[str, ...]:
    """
    Детерминированно проверяет целостность
    уже сформированного TOP-3 поста.

    Этот gate дополняет Pydantic/schema-
    валидацию. Его задача — поймать
    формально допустимый, но оборванный
    человеческий текст.
    """

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

        if not body_has_terminal_punctuation(
            body
        ):
            issues.append(
                "Новость position="
                f"{item.position} не имеет "
                "признака завершённого "
                "предложения в конце body."
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
                f"actual="
                f"{len(canonical_post_text)}, "
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
    """
    Формирует точное редакционное замечание
    для автоматического revision-прохода.
    """

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
        "оборван, закончи мысль естественной "
        "краткой фразой в пределах текущего "
        "контракта длины.\n\n"
        "Обнаруженные проблемы:\n"
        f"{issues_text}"
    )


def assert_generated_post_integrity(
    payload: OpenAIGeneratedPostPayload,
) -> None:
    """
    Бросает ValueError, если post integrity
    gate не пройден.
    """

    issues = validate_generated_post_integrity(
        payload
    )

    if issues:
        raise ValueError(
            "Integrity gate failed: "
            + "; ".join(issues)
        )
