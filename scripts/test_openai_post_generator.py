import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import json

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    GenerationNewsItem,
    OpenAITelegramPostGenerator,
)


class FakeStructuredGenerationClient:
    """Тестовый клиент без сетевых запросов."""

    def __init__(
        self,
        response_text: str,
    ) -> None:
        self._response_text = response_text
        self.requests: list[
            GenerationModelRequest
        ] = []

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Возвращает заранее заданный JSON."""

        self.requests.append(request)

        return GenerationModelResponse(
            output_text=self._response_text,
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
                7,
                31,
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
                7,
                31,
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
                7,
                31,
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


def build_valid_post_text() -> str:
    """Создаёт корректный Telegram-пост."""

    return (
        "**TOP-3 киноновости дня**\n\n"
        "**1. Студия отчиталась "
        "о снижении выручки**\n"
        "Киноподразделение сообщило "
        "о снижении квартальной выручки, "
        "тогда как другое направление "
        "развлекательного бизнеса выросло.\n\n"
        "**2. Фестиваль добавил "
        "конкурс AI-фильмов**\n"
        "Международный фестиваль "
        "короткометражного кино расширил "
        "программу конкурсами AI-фильмов "
        "и экранного танца.\n\n"
        "**3. У кинокомпании появился "
        "руководитель коммуникаций**\n"
        "Компания впервые назначила "
        "специалиста, который возглавит "
        "коммуникации и маркетинг.\n\n"
        "__Какую из новостей "
        "обсудим подробнее?__"
    )


def build_valid_response() -> str:
    """Создаёт корректный ответ модели."""

    return json.dumps(
        {
            "post_text": (
                build_valid_post_text()
            ),
            "items": [
                {
                    "position": 1,
                    "news_id": 201,
                    "headline": (
                        "Студия отчиталась "
                        "о снижении выручки"
                    ),
                    "body": (
                        "Киноподразделение "
                        "сообщило о снижении "
                        "квартальной выручки, "
                        "тогда как другое "
                        "направление выросло."
                    ),
                },
                {
                    "position": 2,
                    "news_id": 202,
                    "headline": (
                        "Фестиваль добавил "
                        "конкурс AI-фильмов"
                    ),
                    "body": (
                        "Международный фестиваль "
                        "расширил программу "
                        "конкурсами AI-фильмов "
                        "и экранного танца."
                    ),
                },
                {
                    "position": 3,
                    "news_id": 203,
                    "headline": (
                        "У кинокомпании появился "
                        "руководитель коммуникаций"
                    ),
                    "body": (
                        "Компания впервые "
                        "назначила специалиста, "
                        "который возглавит "
                        "коммуникации и маркетинг."
                    ),
                },
            ],
        },
        ensure_ascii=False,
    )


def build_generator(
    response_text: str | None = None,
) -> tuple[
    OpenAITelegramPostGenerator,
    FakeStructuredGenerationClient,
]:
    """Создаёт генератор и fake-клиент."""

    client = FakeStructuredGenerationClient(
        response_text or build_valid_response()
    )

    generator = OpenAITelegramPostGenerator(
        client=client,
        model_name="test-model-no-network",
    )

    return generator, client


def test_build_request() -> None:
    """Проверяет подготовку запроса без модели."""

    generator, client = build_generator()

    items = build_news_items()

    request = generator.build_request(
        items
    )

    assert len(client.requests) == 0

    assert request.model == (
        "test-model-no-network"
    )

    assert request.instructions.strip()

    input_payload = json.loads(
        request.input_text
    )

    assert input_payload["task"] == (
        "generate_russian_telegram_"
        "movie_news_top3"
    )

    assert (
        input_payload["text_format"]
        == "markdown"
    )

    assert (
        input_payload[
            "maximum_post_length"
        ]
        == 3900
    )

    assert [
        item["position"]
        for item in input_payload["news"]
    ] == [
        1,
        2,
        3,
    ]

    assert [
        item["news_id"]
        for item in input_payload["news"]
    ] == [
        201,
        202,
        203,
    ]

    assert (
        input_payload["news"][0][
            "individual_score"
        ]
        == "6.700000"
    )

    assert (
        input_payload["news"][1][
            "source_name"
        ]
        == "Test Film Source"
    )

    print("Request preparation: OK")
    print("client_call_count=0")
    print(f"model={request.model}")
    print(
        "news_ids="
        + ",".join(
            str(item["news_id"])
            for item in input_payload["news"]
        )
    )


async def test_prepared_request() -> None:
    """Проверяет заранее подготовленный запрос."""

    generator, client = build_generator()

    items = build_news_items()

    request = generator.build_request(
        items
    )

    result = (
        await generator
        .generate_prepared_request(
            items,
            request,
        )
    )

    assert len(client.requests) == 1
    assert client.requests[0] == request

    assert (
        result.payload.post_text
        == build_valid_post_text()
    )

    assert [
        item.news_id
        for item in result.payload.items
    ] == [
        201,
        202,
        203,
    ]

    assert (
        result.model_response.usage
        is None
    )

    assert (
        result
        .model_response
        .cost_estimate
        is None
    )

    print()
    print(
        "Prepared request generation: OK"
    )
    print("client_call_count=1")
    print(
        "response_news_ids="
        + ",".join(
            str(item.news_id)
            for item in result.payload.items
        )
    )


async def test_modified_request_blocking() -> None:
    """Проверяет блокировку изменённого запроса."""

    generator, client = build_generator()

    items = build_news_items()

    valid_request = generator.build_request(
        items
    )

    modified_request = GenerationModelRequest(
        model=valid_request.model,
        instructions=(
            valid_request.instructions
        ),
        input_text=(
            valid_request.input_text
            + " "
        ),
    )

    try:
        await generator.generate_prepared_request(
            items,
            modified_request,
        )
    except ValueError as error:
        assert (
            "Подготовленный запрос "
            "не соответствует"
            in str(error)
        )

        assert len(client.requests) == 0

        print()
        print(
            "Modified request blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Изменённый запрос "
        "не был заблокирован."
    )


async def test_valid_response() -> None:
    """Проверяет корректный структурированный ответ."""

    generator, client = build_generator()

    result = await generator.generate_detailed(
        build_news_items()
    )

    assert len(client.requests) == 1

    assert (
        result.payload.post_text
        == build_valid_post_text()
    )

    assert len(result.payload.items) == 3

    assert [
        item.position
        for item in result.payload.items
    ] == [
        1,
        2,
        3,
    ]

    assert [
        item.news_id
        for item in result.payload.items
    ] == [
        201,
        202,
        203,
    ]

    assert (
        generator.metadata.generator_name
        == "OpenAITelegramPostGenerator"
    )

    assert (
        generator.metadata.model_name
        == "test-model-no-network"
    )

    assert (
        generator.metadata.text_format
        == "markdown"
    )

    assert (
        result.model_response.usage
        is None
    )

    assert (
        result
        .model_response
        .cost_estimate
        is None
    )

    print()
    print(
        "Valid structured response: OK"
    )
    print(
        "client_call_count="
        f"{len(client.requests)}"
    )
    print(
        "generated_item_count="
        f"{len(result.payload.items)}"
    )
    print(
        "post_length="
        f"{len(result.payload.post_text)}"
    )
    print("usage_present=false")
    print("cost_estimate_present=false")


async def test_simple_interface() -> None:
    """Проверяет метод generate()."""

    generator, client = build_generator()

    post_text = await generator.generate(
        build_news_items()
    )

    assert len(client.requests) == 1

    assert post_text == (
        build_valid_post_text()
    )

    print()
    print(
        "Simple generator interface: OK"
    )
    print(
        f"post_length={len(post_text)}"
    )


async def test_changed_news_order() -> None:
    """Проверяет изменённый порядок news_id."""

    response_payload = json.loads(
        build_valid_response()
    )

    response_payload["items"][0][
        "news_id"
    ] = 202

    response_payload["items"][1][
        "news_id"
    ] = 201

    generator, _ = build_generator(
        json.dumps(
            response_payload,
            ensure_ascii=False,
        )
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert (
            "изменила порядок или набор "
            "news_id"
            in str(error)
        )

        print()
        print(
            "Changed news order blocking: OK"
        )
        return

    raise AssertionError(
        "Изменённый порядок news_id "
        "не был заблокирован."
    )


async def test_duplicate_news_id() -> None:
    """Проверяет повторяющийся news_id."""

    response_payload = json.loads(
        build_valid_response()
    )

    response_payload["items"][1][
        "news_id"
    ] = 201

    generator, _ = build_generator(
        json.dumps(
            response_payload,
            ensure_ascii=False,
        )
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Duplicate news_id blocking: OK"
        )
        return

    raise AssertionError(
        "Повторяющийся news_id "
        "не был заблокирован."
    )


async def test_invalid_position_order() -> None:
    """Проверяет неправильный порядок позиций."""

    response_payload = json.loads(
        build_valid_response()
    )

    response_payload["items"][0][
        "position"
    ] = 2

    response_payload["items"][1][
        "position"
    ] = 1

    generator, _ = build_generator(
        json.dumps(
            response_payload,
            ensure_ascii=False,
        )
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Invalid position order "
            "blocking: OK"
        )
        return

    raise AssertionError(
        "Неверный порядок позиций "
        "не был заблокирован."
    )


async def test_empty_headline() -> None:
    """Проверяет пустой заголовок новости."""

    response_payload = json.loads(
        build_valid_response()
    )

    response_payload["items"][0][
        "headline"
    ] = "   "

    generator, _ = build_generator(
        json.dumps(
            response_payload,
            ensure_ascii=False,
        )
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Empty headline blocking: OK"
        )
        return

    raise AssertionError(
        "Пустой заголовок "
        "не был заблокирован."
    )


async def test_oversized_post() -> None:
    """Проверяет превышение лимита текста."""

    response_payload = json.loads(
        build_valid_response()
    )

    response_payload["post_text"] = (
        "X" * 4097
    )

    generator, _ = build_generator(
        json.dumps(
            response_payload,
            ensure_ascii=False,
        )
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Oversized post blocking: OK"
        )
        return

    raise AssertionError(
        "Слишком длинный пост "
        "не был заблокирован."
    )


async def test_invalid_json() -> None:
    """Проверяет синтаксически неверный JSON."""

    generator, _ = build_generator(
        "{invalid-json"
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert (
            "не соответствует схеме"
            in str(error)
        )

        print()
        print(
            "Invalid JSON blocking: OK"
        )
        return

    raise AssertionError(
        "Некорректный JSON "
        "не был заблокирован."
    )


async def test_empty_response() -> None:
    """Проверяет пустой ответ модели."""

    generator, _ = build_generator(
        "   "
    )

    try:
        await generator.generate(
            build_news_items()
        )
    except ValueError as error:
        assert "пустой ответ" in str(error)

        print()
        print(
            "Empty response blocking: OK"
        )
        return

    raise AssertionError(
        "Пустой ответ модели "
        "не был заблокирован."
    )


async def test_wrong_input_count() -> None:
    """Проверяет вход без третьей новости."""

    generator, client = build_generator()

    items = build_news_items()[:2]

    try:
        generator.build_request(items)
    except ValueError as error:
        assert (
            "ровно три новости"
            in str(error)
        )

        assert len(client.requests) == 0

        print()
        print(
            "Wrong input count blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Неполный входной TOP-3 "
        "не был заблокирован."
    )


async def test_invalid_input_positions() -> None:
    """Проверяет неверные входные позиции."""

    valid_items = build_news_items()

    invalid_items = (
        valid_items[1],
        valid_items[0],
        valid_items[2],
    )

    generator, client = build_generator()

    try:
        generator.build_request(
            invalid_items
        )
    except ValueError as error:
        assert (
            "порядке позиций 1, 2 и 3"
            in str(error)
        )

        assert len(client.requests) == 0

        print()
        print(
            "Invalid input positions "
            "blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Неверный порядок входных позиций "
        "не был заблокирован."
    )


async def test_naive_publication_datetime() -> None:
    """Проверяет дату без часового пояса."""

    valid_items = build_news_items()

    invalid_first_item = GenerationNewsItem(
        position=valid_items[0].position,
        news_id=valid_items[0].news_id,
        title=valid_items[0].title,
        summary=valid_items[0].summary,
        source_name=(
            valid_items[0].source_name
        ),
        source_url=valid_items[0].source_url,
        source_published_at=datetime(
            2026,
            7,
            31,
            10,
            0,
        ),
        individual_score=(
            valid_items[0].individual_score
        ),
        selection_reason=(
            valid_items[0].selection_reason
        ),
    )

    invalid_items = (
        invalid_first_item,
        valid_items[1],
        valid_items[2],
    )

    generator, client = build_generator()

    try:
        generator.build_request(
            invalid_items
        )
    except ValueError as error:
        assert (
            "должен содержать часовой пояс"
            in str(error)
        )

        assert len(client.requests) == 0

        print()
        print(
            "Naive datetime blocking: OK"
        )
        print("client_call_count=0")
        return

    raise AssertionError(
        "Дата без часового пояса "
        "не была заблокирована."
    )


async def main() -> int:
    """Запускает автономный тест генератора."""

    test_build_request()
    await test_prepared_request()
    await test_modified_request_blocking()
    await test_valid_response()
    await test_simple_interface()
    await test_changed_news_order()
    await test_duplicate_news_id()
    await test_invalid_position_order()
    await test_empty_headline()
    await test_oversized_post()
    await test_invalid_json()
    await test_empty_response()
    await test_wrong_input_count()
    await test_invalid_input_positions()
    await test_naive_publication_datetime()

    print()
    print("API key required: no")
    print("OpenAI requests: not performed")
    print("Database changes: not performed")
    print("Telegram publication: not performed")
    print(
        "OpenAI post generator "
        "fake-client test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )