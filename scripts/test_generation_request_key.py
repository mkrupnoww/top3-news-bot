from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import json

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationNewsItem,
    OpenAIPostGeneratorMetadata,
)
from app.generation.request_key import (
    GENERATION_REQUEST_KEY_VERSION,
    create_generation_request_key,
)


OFFICIAL_TRAILER_URL = (
    "https://www.youtube.com/watch?v=5fHXyqQOKL8"
)


def build_metadata(
    *,
    model_name: str = "gpt-5.6-terra",
    prompt_version: str = (
        "movie_news_telegram_post_prompt_v1"
    ),
) -> OpenAIPostGeneratorMetadata:
    """Создаёт метаданные тестового генератора."""

    return OpenAIPostGeneratorMetadata(
        generator_name=(
            "OpenAITelegramPostGenerator"
        ),
        generator_version=(
            "openai_telegram_post_generator_v1"
        ),
        prompt_version=prompt_version,
        model_name=model_name,
        text_format="markdown",
    )


def build_news_items() -> tuple[
    GenerationNewsItem,
    ...,
]:
    """Создаёт тестовый сохранённый TOP-3."""

    return (
        GenerationNewsItem(
            position=1,
            news_id=11,
            title=(
                "Sony Pictures Revenue Drops "
                "in June Quarter"
            ),
            summary=(
                "The film division reported "
                "lower quarterly revenue."
            ),
            source_name="Variety Film",
            source_url=(
                "https://example.com/news/11"
            ),
            source_published_at=datetime(
                2026,
                7,
                31,
                10,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "6.713500"
            ),
            selection_reason=(
                "Крупная корпоративная новость "
                "с измеримой динамикой выручки."
            ),
        ),
        GenerationNewsItem(
            position=2,
            news_id=9,
            title=(
                "Short Film Festival Adds "
                "AI Competition"
            ),
            summary=(
                "The festival added AI-film "
                "and screen-dance competitions."
            ),
            source_name="Variety Film",
            source_url=(
                "https://example.com/news/9"
            ),
            source_published_at=datetime(
                2026,
                7,
                31,
                9,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "5.898000"
            ),
            selection_reason=(
                "Международная фестивальная "
                "новость с новой AI-категорией."
            ),
        ),
        GenerationNewsItem(
            position=3,
            news_id=10,
            title=(
                "Element Pictures Appoints "
                "Communications Head"
            ),
            summary=(
                "The company appointed its first "
                "communications and marketing head."
            ),
            source_name="Variety Film",
            source_url=(
                "https://example.com/news/10"
            ),
            source_published_at=datetime(
                2026,
                7,
                31,
                8,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "5.073000"
            ),
            selection_reason=(
                "Заметное отраслевое назначение "
                "в известной кинокомпании."
            ),
        ),
    )


def build_model_request(
    *,
    model: str = "gpt-5.6-terra",
    input_text: str | None = None,
) -> GenerationModelRequest:
    """Создаёт тестовый запрос модели."""

    return GenerationModelRequest(
        model=model,
        instructions=(
            "Сформируй русскоязычный "
            "Telegram-пост с TOP-3."
        ),
        input_text=(
            input_text
            or (
                '{"news":['
                '{"position":1,"news_id":11},'
                '{"position":2,"news_id":9},'
                '{"position":3,"news_id":10}'
                "]}"
            )
        ),
    )


def build_request_key(
    *,
    ranking_run_id: int = 18,
    publication_date: date = date(
        2026,
        8,
        1,
    ),
    telegram_chat_id: int = (
        -1001224825458
    ),
    metadata: (
        OpenAIPostGeneratorMetadata
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
):
    """Создаёт ключ с типовыми параметрами."""

    return create_generation_request_key(
        ranking_run_id=ranking_run_id,
        publication_date=publication_date,
        telegram_chat_id=telegram_chat_id,
        metadata=(
            metadata
            or build_metadata()
        ),
        model_request=(
            model_request
            or build_model_request()
        ),
        items=(
            items
            or build_news_items()
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
        == GENERATION_REQUEST_KEY_VERSION
    )

    assert (
        GENERATION_REQUEST_KEY_VERSION
        == "generation_request_key_v2"
    )

    assert first_key.value == (
        "bee6b1b04b86210390677b600c534dd0"
        "cddf9a5cc31221b5212b092f112d28e4"
    )

    print(
        "Deterministic generation "
        "request key: OK"
    )
    print(
        f"generation_request_key="
        f"{first_key.value}"
    )
    print(
        "generation_request_key_version="
        f"{first_key.version}"
    )


def test_canonical_payload() -> None:
    """Проверяет каноническое содержимое."""

    request_key = build_request_key()

    payload = json.loads(
        request_key.canonical_json
    )

    assert payload[
        "generation_request_key_version"
    ] == GENERATION_REQUEST_KEY_VERSION

    assert payload["ranking_run_id"] == 18

    assert payload[
        "publication_date"
    ] == "2026-08-01"

    assert payload[
        "telegram_chat_id"
    ] == -1001224825458

    assert payload[
        "top3_news_ids"
    ] == [
        11,
        9,
        10,
    ]

    assert [
        item["position"]
        for item in payload["top3"]
    ] == [
        1,
        2,
        3,
    ]

    assert [
        item["news_id"]
        for item in payload["top3"]
    ] == [
        11,
        9,
        10,
    ]

    assert payload[
        "top3"
    ][0][
        "individual_score"
    ] == "6.713500"

    assert payload[
        "top3"
    ][0][
        "source_published_at"
    ] == "2026-07-31T10:00:00+00:00"

    expected_top3_fields = {
        "position",
        "news_id",
        "title",
        "summary",
        "source_name",
        "source_url",
        "source_published_at",
        "individual_score",
        "selection_reason",
    }

    assert all(
        set(item) == expected_top3_fields
        for item in payload["top3"]
    )

    assert all(
        "official_trailer_url" not in item
        for item in payload["top3"]
    )

    assert payload[
        "generator"
    ][
        "generator_name"
    ] == "OpenAITelegramPostGenerator"

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
        "Canonical generation payload: OK"
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


def test_official_trailer_canonical_payload() -> None:
    """Проверяет trailer URL в canonical payload."""

    items = list(
        build_news_items()
    )

    items[1] = replace(
        items[1],
        official_trailer_url=(
            OFFICIAL_TRAILER_URL
        ),
    )

    request_key = build_request_key(
        items=tuple(items)
    )

    payload = json.loads(
        request_key.canonical_json
    )

    assert (
        "official_trailer_url"
        not in payload["top3"][0]
    )

    assert payload[
        "top3"
    ][1][
        "official_trailer_url"
    ] == OFFICIAL_TRAILER_URL

    assert (
        "official_trailer_url"
        not in payload["top3"][2]
    )

    print()
    print(
        "Official trailer canonical "
        "payload: OK"
    )


def test_changed_parameters_change_key() -> None:
    """Проверяет чувствительность ключа."""

    original_key = build_request_key()

    changed_ranking_run_key = (
        build_request_key(
            ranking_run_id=19
        )
    )

    changed_date_key = build_request_key(
        publication_date=date(
            2026,
            8,
            2,
        )
    )

    changed_chat_key = build_request_key(
        telegram_chat_id=(
            -1009999999999
        )
    )

    changed_model_key = build_request_key(
        metadata=build_metadata(
            model_name="other-model"
        ),
        model_request=build_model_request(
            model="other-model"
        ),
    )

    changed_prompt_key = build_request_key(
        metadata=build_metadata(
            prompt_version=(
                "movie_news_telegram_"
                "post_prompt_v2"
            )
        )
    )

    changed_input_key = build_request_key(
        model_request=build_model_request(
            input_text=(
                '{"news":['
                '{"position":1,"news_id":11},'
                '{"position":2,"news_id":9},'
                '{"position":3,"news_id":7}'
                "]}"
            )
        )
    )

    original_items = build_news_items()

    changed_trailer_items = list(
        original_items
    )

    changed_trailer_items[1] = replace(
        changed_trailer_items[1],
        official_trailer_url=(
            OFFICIAL_TRAILER_URL
        ),
    )

    changed_trailer_key = build_request_key(
        items=tuple(
            changed_trailer_items
        )
    )

    changed_items = (
        original_items[0],
        original_items[1],
        GenerationNewsItem(
            position=3,
            news_id=7,
            title=(
                "Black Zombie Documentary Review"
            ),
            summary=(
                "A documentary examines "
                "zombie imagery and history."
            ),
            source_name="Variety Film",
            source_url=(
                "https://example.com/news/7"
            ),
            source_published_at=datetime(
                2026,
                7,
                31,
                7,
                0,
                tzinfo=timezone.utc,
            ),
            individual_score=Decimal(
                "4.703000"
            ),
            selection_reason=(
                "Необычная тема документального "
                "фильма и культурной истории."
            ),
        ),
    )

    changed_top3_key = build_request_key(
        model_request=build_model_request(
            input_text=(
                '{"news":['
                '{"position":1,"news_id":11},'
                '{"position":2,"news_id":9},'
                '{"position":3,"news_id":7}'
                "]}"
            )
        ),
        items=changed_items,
    )

    changed_keys = (
        changed_ranking_run_key,
        changed_date_key,
        changed_chat_key,
        changed_model_key,
        changed_prompt_key,
        changed_input_key,
        changed_trailer_key,
        changed_top3_key,
    )

    assert all(
        changed_key.value
        != original_key.value
        for changed_key in changed_keys
    )

    print()
    print(
        "Changed generation parameters "
        "change key: OK"
    )


def test_duplicate_news_ids_blocking() -> None:
    """Проверяет блокировку дубликатов."""

    valid_items = build_news_items()

    duplicate_item = GenerationNewsItem(
        position=3,
        news_id=valid_items[1].news_id,
        title=valid_items[2].title,
        summary=valid_items[2].summary,
        source_name=(
            valid_items[2].source_name
        ),
        source_url=(
            valid_items[2].source_url
        ),
        source_published_at=(
            valid_items[2]
            .source_published_at
        ),
        individual_score=(
            valid_items[2]
            .individual_score
        ),
        selection_reason=(
            valid_items[2]
            .selection_reason
        ),
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
            "Duplicate generation news IDs "
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
            "Invalid generation position "
            "order blocking: OK"
        )
        return

    raise AssertionError(
        "Неверный порядок позиций "
        "не был заблокирован."
    )


def test_invalid_chat_id_blocking() -> None:
    """Проверяет неполный ID канала."""

    try:
        build_request_key(
            telegram_chat_id=-1224825458
        )
    except ValueError as error:
        assert (
            "должен начинаться с -100"
            in str(error)
        )

        print()
        print(
            "Invalid Telegram chat ID "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Некорректный Telegram chat ID "
        "не был заблокирован."
    )


def test_datetime_publication_date_blocking() -> None:
    """Проверяет datetime вместо date."""

    try:
        build_request_key(
            publication_date=datetime(
                2026,
                8,
                1,
                0,
                0,
                tzinfo=timezone.utc,
            )
        )
    except TypeError as error:
        assert (
            "date, а не datetime"
            in str(error)
        )

        print()
        print(
            "Datetime publication date "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "datetime вместо date "
        "не был заблокирован."
    )


def test_naive_source_datetime_blocking() -> None:
    """Проверяет дату новости без пояса."""

    valid_items = build_news_items()

    invalid_item = GenerationNewsItem(
        position=valid_items[0].position,
        news_id=valid_items[0].news_id,
        title=valid_items[0].title,
        summary=valid_items[0].summary,
        source_name=(
            valid_items[0].source_name
        ),
        source_url=(
            valid_items[0].source_url
        ),
        source_published_at=datetime(
            2026,
            7,
            31,
            10,
            0,
        ),
        individual_score=(
            valid_items[0]
            .individual_score
        ),
        selection_reason=(
            valid_items[0]
            .selection_reason
        ),
    )

    try:
        build_request_key(
            items=(
                invalid_item,
                valid_items[1],
                valid_items[2],
            )
        )
    except ValueError as error:
        assert (
            "должен содержать часовой пояс"
            in str(error)
        )

        print()
        print(
            "Naive source datetime "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Дата новости без часового пояса "
        "не была заблокирована."
    )


def test_empty_official_trailer_url_blocking() -> None:
    """Блокирует пустой trailer URL."""

    valid_items = list(
        build_news_items()
    )

    valid_items[1] = replace(
        valid_items[1],
        official_trailer_url="   ",
    )

    try:
        build_request_key(
            items=tuple(
                valid_items
            )
        )
    except ValueError as error:
        assert (
            "official_trailer_url"
            in str(error)
        )

        print()
        print(
            "Empty official trailer URL "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Пустой official_trailer_url "
        "не был заблокирован."
    )


def test_model_mismatch_blocking() -> None:
    """Проверяет несовпадение моделей."""

    try:
        build_request_key(
            metadata=build_metadata(
                model_name="other-model"
            ),
            model_request=(
                build_model_request(
                    model="gpt-5.6-terra"
                )
            ),
        )
    except ValueError as error:
        assert (
            "Модель в metadata не совпадает"
            in str(error)
        )

        print()
        print(
            "Generation model mismatch "
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
    test_official_trailer_canonical_payload()
    test_changed_parameters_change_key()
    test_duplicate_news_ids_blocking()
    test_invalid_position_order_blocking()
    test_invalid_chat_id_blocking()
    test_datetime_publication_date_blocking()
    test_naive_source_datetime_blocking()
    test_empty_official_trailer_url_blocking()
    test_model_mismatch_blocking()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "Generation request key test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())