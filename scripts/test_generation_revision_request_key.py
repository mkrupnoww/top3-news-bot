from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    GenerationNewsItem,
    OPENAI_POST_PROMPT_VERSION,
    OPENAI_POST_REVISION_PROMPT_VERSION,
    OpenAITelegramPostGenerator,
)
from app.generation.revision_request_key import (
    GENERATION_REVISION_REQUEST_KEY_VERSION,
    REVISION_REQUESTED_ACTION,
    create_generation_revision_request_key,
)


SOURCE_POST_TEXT = (
    "**TOP-3 НОВОСТЕЙ КИНО "
    "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n"
    "_______________\n\n"
    "1️⃣ **Старый заголовок первой новости**\n\n"
    "Старый текст первой новости.\n\n"
    "2️⃣ **Старый заголовок второй новости**\n\n"
    "Старый текст второй новости.\n\n"
    "3️⃣ **Старый заголовок третьей новости**\n\n"
    "Старый текст третьей новости.\n\n"
    "……………\n"
    "Подписаться на VIP канал - @kkm_vip_bot"
)

EDITORIAL_COMMENT = (
    "Исправить фактические неточности и "
    "убрать неподтверждённые детали."
)

REVISION_ISSUES = (
    "Имена людей передавать кириллицей.",
    "Использовать факты только из title и summary.",
)


class UnusedStructuredGenerationClient:
    """Клиент, который не должен вызываться."""

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Блокирует неожиданный вызов модели."""

        raise AssertionError(
            "OpenAI-вызов в тесте request key "
            "не должен выполняться."
        )


def build_generator(
    *,
    model_name: str = "gpt-5.6-terra",
) -> OpenAITelegramPostGenerator:
    """Создаёт генератор без сетевых вызовов."""

    return OpenAITelegramPostGenerator(
        client=UnusedStructuredGenerationClient(),
        model_name=model_name,
    )


def build_news_items() -> tuple[
    GenerationNewsItem,
    ...,
]:
    """Создаёт тестовый TOP-3."""

    return (
        GenerationNewsItem(
            position=1,
            news_id=201,
            title=(
                "Major Studio Reports "
                "Quarterly Revenue Decline"
            ),
            summary=(
                "The film division reported "
                "a revenue decline while another "
                "entertainment segment grew."
            ),
            source_name="Test Film Source",
            source_url=(
                "https://example.com/news/201"
            ),
            source_published_at=datetime(
                2026,
                8,
                6,
                10,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "6.700000"
            ),
            selection_reason=(
                "Крупная корпоративная новость "
                "с измеримой динамикой выручки."
            ),
        ),
        GenerationNewsItem(
            position=2,
            news_id=202,
            title=(
                "International Short Film "
                "Festival Adds AI Competition"
            ),
            summary=(
                "The festival expanded its "
                "programme with competitions "
                "for AI films and screen dance."
            ),
            source_name="Test Film Source",
            source_url=(
                "https://example.com/news/202"
            ),
            source_published_at=datetime(
                2026,
                8,
                6,
                9,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "5.900000"
            ),
            selection_reason=(
                "Международный фестивальный "
                "повод с новой AI-категорией."
            ),
        ),
        GenerationNewsItem(
            position=3,
            news_id=203,
            title=(
                "Production Company Appoints "
                "First Communications Head"
            ),
            summary=(
                "The company appointed an "
                "experienced publicist to lead "
                "communications and marketing."
            ),
            source_name="Test Film Source",
            source_url=(
                "https://example.com/news/203"
            ),
            source_published_at=datetime(
                2026,
                8,
                6,
                8,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "5.100000"
            ),
            selection_reason=(
                "Заметное отраслевое назначение "
                "в известной кинокомпании."
            ),
        ),
    )


def build_revision_request(
    *,
    generator: (
        OpenAITelegramPostGenerator
        | None
    ) = None,
    items: tuple[
        GenerationNewsItem,
        ...,
    ]
    | None = None,
    source_post_text: str = SOURCE_POST_TEXT,
    editorial_comment: str = EDITORIAL_COMMENT,
    issues: tuple[str, ...] = REVISION_ISSUES,
) -> GenerationModelRequest:
    """Создаёт production-like revision request."""

    current_generator = (
        generator
        or build_generator()
    )

    return current_generator.build_revision_request(
        items or build_news_items(),
        source_post_text=source_post_text,
        editorial_comment=editorial_comment,
        issues=issues,
    )


def build_request_key(
    *,
    batch_id: int = 17,
    source_generated_post_id: int = 14,
    review_action_id: int = 9,
    target_version_number: int = 2,
    source_post_text: str = SOURCE_POST_TEXT,
    editorial_comment: str = EDITORIAL_COMMENT,
    issues: tuple[str, ...] = REVISION_ISSUES,
    generator: (
        OpenAITelegramPostGenerator
        | None
    ) = None,
    model_request: (
        GenerationModelRequest
        | None
    ) = None,
    items: tuple[
        GenerationNewsItem,
        ...,
    ]
    | None = None,
    revision_prompt_version: str = (
        OPENAI_POST_REVISION_PROMPT_VERSION
    ),
):
    """Создаёт revision key с типовыми параметрами."""

    current_generator = (
        generator
        or build_generator()
    )

    current_items = (
        items
        or build_news_items()
    )

    current_model_request = (
        model_request
        or build_revision_request(
            generator=current_generator,
            items=current_items,
            source_post_text=source_post_text,
            editorial_comment=editorial_comment,
            issues=issues,
        )
    )

    return create_generation_revision_request_key(
        batch_id=batch_id,
        source_generated_post_id=(
            source_generated_post_id
        ),
        review_action_id=review_action_id,
        target_version_number=(
            target_version_number
        ),
        source_post_text=source_post_text,
        editorial_comment=editorial_comment,
        issues=issues,
        metadata=current_generator.metadata,
        model_request=current_model_request,
        items=current_items,
        revision_prompt_version=(
            revision_prompt_version
        ),
    )


def test_deterministic_key() -> None:
    """Проверяет повторяемость SHA-256."""

    first_key = build_request_key()
    second_key = build_request_key()

    assert first_key.value == second_key.value

    assert (
        first_key.canonical_json
        == second_key.canonical_json
    )

    assert len(first_key.value) == 64

    assert all(
        character in "0123456789abcdef"
        for character in first_key.value
    )

    assert (
        first_key.version
        == GENERATION_REVISION_REQUEST_KEY_VERSION
    )

    print(
        "Deterministic generation revision "
        "request key: OK"
    )
    print(
        "generation_revision_request_key="
        f"{first_key.value}"
    )
    print(
        "generation_revision_request_key_version="
        f"{first_key.version}"
    )


def test_canonical_payload() -> None:
    """Проверяет каноническое содержимое."""

    request_key = build_request_key()

    payload = json.loads(
        request_key.canonical_json
    )

    assert payload[
        "generation_revision_request_key_version"
    ] == GENERATION_REVISION_REQUEST_KEY_VERSION

    assert payload["batch_id"] == 17

    assert (
        payload["source_generated_post_id"]
        == 14
    )

    assert payload["review_action_id"] == 9

    assert (
        payload["target_version_number"]
        == 2
    )

    assert payload[
        "requested_action"
    ] == REVISION_REQUESTED_ACTION

    assert payload[
        "revision"
    ][
        "source_post_text"
    ] == SOURCE_POST_TEXT

    assert payload[
        "revision"
    ][
        "editorial_comment"
    ] == EDITORIAL_COMMENT

    assert payload[
        "revision"
    ][
        "issues"
    ] == list(
        REVISION_ISSUES
    )

    assert payload[
        "top3_news_ids"
    ] == [
        201,
        202,
        203,
    ]

    expected_top3_fields = {
        "position",
        "news_id",
        "title",
        "summary",
    }

    assert all(
        set(item) == expected_top3_fields
        for item in payload["top3"]
    )

    assert payload[
        "generator"
    ][
        "base_prompt_version"
    ] == OPENAI_POST_PROMPT_VERSION

    assert payload[
        "generator"
    ][
        "revision_prompt_version"
    ] == OPENAI_POST_REVISION_PROMPT_VERSION

    assert payload[
        "generator"
    ][
        "model_name"
    ] == "gpt-5.6-terra"

    assert payload[
        "generator"
    ][
        "text_format"
    ] == "markdown"

    assert payload[
        "model_request"
    ][
        "model"
    ] == "gpt-5.6-terra"

    print()
    print(
        "Canonical generation revision "
        "payload: OK"
    )
    print(
        "top3_news_ids="
        + ",".join(
            str(news_id)
            for news_id in payload[
                "top3_news_ids"
            ]
        )
    )


def test_changed_revision_parameters_change_key() -> None:
    """Проверяет чувствительность revision key."""

    original_key = build_request_key()

    changed_keys = (
        build_request_key(
            batch_id=18
        ),
        build_request_key(
            source_generated_post_id=15
        ),
        build_request_key(
            review_action_id=10
        ),
        build_request_key(
            target_version_number=3
        ),
        build_request_key(
            source_post_text=(
                SOURCE_POST_TEXT
                + "\nИзменённый исходный текст."
            )
        ),
        build_request_key(
            editorial_comment=(
                EDITORIAL_COMMENT
                + " Дополнительное замечание."
            )
        ),
        build_request_key(
            issues=(
                REVISION_ISSUES[0],
                (
                    REVISION_ISSUES[1]
                    + " Проверить отдельно."
                ),
            )
        ),
        build_request_key(
            revision_prompt_version=(
                "movie_news_telegram_post_"
                "revision_prompt_v3"
            )
        ),
    )

    assert all(
        changed_key.value
        != original_key.value
        for changed_key in changed_keys
    )

    print()
    print(
        "Changed revision parameters "
        "change key: OK"
    )


def test_changed_factual_content_changes_key() -> None:
    """Проверяет изменение title/summary."""

    original_key = build_request_key()

    items = list(
        build_news_items()
    )

    items[0] = replace(
        items[0],
        summary=(
            items[0].summary
            + " Additional confirmed detail."
        ),
    )

    changed_key = build_request_key(
        items=tuple(items)
    )

    assert (
        changed_key.value
        != original_key.value
    )

    print()
    print(
        "Changed revision factual content "
        "changes key: OK"
    )


def test_technical_news_fields_do_not_change_key() -> None:
    """Технические поля не влияют на revision key."""

    original_items = build_news_items()

    original_key = build_request_key(
        items=original_items
    )

    changed_items = (
        replace(
            original_items[0],
            source_name="Another Source",
            source_url=(
                "https://example.org/changed/201"
            ),
            source_published_at=datetime(
                2026,
                8,
                7,
                12,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "9.999999"
            ),
            selection_reason=(
                "Изменённое техническое "
                "объяснение выбора."
            ),
        ),
        replace(
            original_items[1],
            source_name="Another Source 2",
            individual_score=Decimal(
                "1.000000"
            ),
            selection_reason=(
                "Другое техническое объяснение."
            ),
        ),
        replace(
            original_items[2],
            source_url=(
                "https://example.org/changed/203"
            ),
            individual_score=Decimal(
                "2.000000"
            ),
        ),
    )

    changed_key = build_request_key(
        items=changed_items
    )

    assert (
        changed_key.value
        == original_key.value
    )

    assert (
        changed_key.canonical_json
        == original_key.canonical_json
    )

    print()
    print(
        "Technical revision news fields "
        "do not change key: OK"
    )


def test_duplicate_news_ids_blocking() -> None:
    """Проверяет блокировку дубликатов."""

    valid_items = build_news_items()

    duplicate_item = replace(
        valid_items[2],
        news_id=valid_items[1].news_id,
    )

    try:
        build_request_key(
            items=(
                valid_items[0],
                valid_items[1],
                duplicate_item,
            )
        )
    except ValueError as error:
        assert (
            "уникальными"
            in str(error)
        )

        print()
        print(
            "Duplicate revision news IDs "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Дублирующиеся news_id "
        "не были заблокированы."
    )


def test_invalid_position_order_blocking() -> None:
    """Проверяет порядок позиций."""

    valid_items = build_news_items()

    try:
        build_request_key(
            items=(
                valid_items[1],
                valid_items[0],
                valid_items[2],
            )
        )
    except ValueError as error:
        assert (
            "порядке позиций 1, 2 и 3"
            in str(error)
        )

        print()
        print(
            "Invalid revision position "
            "order blocking: OK"
        )
        return

    raise AssertionError(
        "Неверный порядок позиций "
        "не был заблокирован."
    )


def test_empty_issues_blocking() -> None:
    """Проверяет обязательные issues."""

    try:
        build_request_key(
            issues=()
        )
    except ValueError as error:
        assert (
            "issues не может быть пустым"
            in str(error)
        )

        print()
        print(
            "Empty revision issues "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Пустые issues "
        "не были заблокированы."
    )


def test_invalid_target_version_blocking() -> None:
    """Проверяет номер новой версии."""

    try:
        build_request_key(
            target_version_number=1
        )
    except ValueError as error:
        assert (
            "должен быть больше 1"
            in str(error)
        )

        print()
        print(
            "Invalid revision target version "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "target_version_number=1 "
        "не был заблокирован."
    )


def test_model_mismatch_blocking() -> None:
    """Проверяет несовпадение моделей."""

    generator = build_generator(
        model_name="other-model"
    )

    model_request = build_revision_request(
        generator=build_generator(
            model_name="gpt-5.6-terra"
        )
    )

    try:
        create_generation_revision_request_key(
            batch_id=17,
            source_generated_post_id=14,
            review_action_id=9,
            target_version_number=2,
            source_post_text=SOURCE_POST_TEXT,
            editorial_comment=EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
            metadata=generator.metadata,
            model_request=model_request,
            items=build_news_items(),
        )
    except ValueError as error:
        assert (
            "Модель в metadata не совпадает"
            in str(error)
        )

        print()
        print(
            "Revision model mismatch "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Несовпадение моделей "
        "не было заблокировано."
    )


def main() -> int:
    """Запускает автономный тест."""

    test_deterministic_key()
    test_canonical_payload()
    test_changed_revision_parameters_change_key()
    test_changed_factual_content_changes_key()
    test_technical_news_fields_do_not_change_key()
    test_duplicate_news_ids_blocking()
    test_invalid_position_order_blocking()
    test_empty_issues_blocking()
    test_invalid_target_version_blocking()
    test_model_mismatch_blocking()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "Generation revision request key "
        "test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )