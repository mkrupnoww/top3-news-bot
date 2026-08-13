import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json

from app.generation.post_contract import (
    MAXIMUM_BODY_LENGTH,
    MAXIMUM_HEADLINE_LENGTH,
    MAXIMUM_POST_LENGTH,
    TARGET_BODY_LENGTH_MAX,
    TARGET_BODY_LENGTH_MIN,
    TARGET_HEADLINE_LENGTH_MAX,
    TARGET_HEADLINE_LENGTH_MIN,
    TARGET_POST_LENGTH_MAX,
    TARGET_POST_LENGTH_MIN,
)

from app.generation.openai_generator import (
    GenerationModelRequest,
    GenerationModelResponse,
    GenerationNewsItem,
    OPENAI_POST_GENERATOR_VERSION,
    OPENAI_POST_PROMPT_VERSION,
    OPENAI_POST_REVISION_PROMPT_VERSION,
    OPENAI_POST_TEXT_FORMAT,
    OpenAITelegramPostGenerator,
    POST_FOOTER_SEPARATOR,
    POST_HEADER,
    POST_SUBSCRIPTION_LINE,
    POST_TOP_SEPARATOR,
    build_top3_post_text,
)


REVISION_SOURCE_POST_TEXT = (
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

REVISION_EDITORIAL_COMMENT = (
    "Исправить фактические неточности и "
    "убрать неподтверждённые имена."
)

REVISION_ISSUES = (
    "Имена людей должны быть переданы кириллицей.",
    "Не добавлять факты вне title и summary.",
)

OFFICIAL_TRAILER_URL = (
    "https://www.youtube.com/watch?v=5fHXyqQOKL8"
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


def build_response_items() -> list[
    dict[str, object]
]:
    """Создаёт корректные items ответа модели."""

    return [
        {
            "position": 1,
            "news_id": 201,
            "headline": (
                "Студия отчиталась "
                "о снижении выручки"
            ),
            "body": (
                "Киноподразделение сообщило "
                "о снижении квартальной выручки, "
                "тогда как другое направление "
                "развлекательного бизнеса выросло."
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
                "короткометражного кино расширил "
                "программу конкурсами "
                "**AI-фильмов** и экранного танца."
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
                "Компания впервые назначила "
                "специалиста, который возглавит "
                "коммуникации и маркетинг."
            ),
        },
    ]


def build_expected_post_text() -> str:
    """Возвращает канонический готовый пост."""

    return (
        "**TOP-3 НОВОСТЕЙ КИНО "
        "ЗА ПОСЛЕДНИЕ 24 ЧАСА**\n"
        "_______________\n\n"
        "1️⃣ **Студия отчиталась "
        "о снижении выручки**\n\n"
        "Киноподразделение сообщило "
        "о снижении квартальной выручки, "
        "тогда как другое направление "
        "развлекательного бизнеса выросло.\n\n"
        "2️⃣ **Фестиваль добавил "
        "конкурс AI-фильмов**\n\n"
        "Международный фестиваль "
        "короткометражного кино расширил "
        "программу конкурсами "
        "**AI-фильмов** и экранного танца.\n\n"
        "3️⃣ **У кинокомпании появился "
        "руководитель коммуникаций**\n\n"
        "Компания впервые назначила "
        "специалиста, который возглавит "
        "коммуникации и маркетинг.\n\n"
        "……………\n"
        "Подписаться на VIP канал - "
        "@kkm_vip_bot"
    )


def build_valid_response() -> str:
    """Создаёт корректный ответ модели."""

    return json.dumps(
        {
            "post_text": (
                "Черновик модели, который "
                "должен быть заменён Python."
            ),
            "items": build_response_items(),
        },
        ensure_ascii=False,
    )


def build_generator(
    response_text: str | None = None,
) -> tuple[
    OpenAITelegramPostGenerator,
    FakeStructuredGenerationClient,
]:
    """Создаёт генератор с тестовым клиентом."""

    client = FakeStructuredGenerationClient(
        response_text=(
            response_text
            if response_text is not None
            else build_valid_response()
        )
    )

    generator = OpenAITelegramPostGenerator(
        client=client,
        model_name="gpt-5.6-terra",
    )

    return generator, client


def assert_raises(
    expected_exception: type[Exception],
    expected_text: str,
    callback,
) -> None:
    """Проверяет синхронное исключение."""

    try:
        callback()
    except expected_exception as error:
        assert expected_text in str(error)
        return

    raise AssertionError(
        "Ожидаемое исключение не возникло: "
        f"{expected_exception.__name__}"
    )


async def assert_raises_async(
    expected_exception: type[Exception],
    expected_text: str,
    callback,
) -> None:
    """Проверяет асинхронное исключение."""

    try:
        await callback()
    except expected_exception as error:
        assert expected_text in str(error)
        return

    raise AssertionError(
        "Ожидаемое исключение не возникло: "
        f"{expected_exception.__name__}"
    )


def test_metadata_and_request() -> None:
    """Проверяет версии и подготовку запроса."""

    generator, client = build_generator()

    metadata = generator.metadata

    assert (
        metadata.generator_version
        == OPENAI_POST_GENERATOR_VERSION
    )

    assert (
        metadata.prompt_version
        == OPENAI_POST_PROMPT_VERSION
    )

    assert (
        metadata.text_format
        == OPENAI_POST_TEXT_FORMAT
    )

    assert OPENAI_POST_GENERATOR_VERSION == (
        "openai_telegram_post_generator_v6"
    )

    assert OPENAI_POST_PROMPT_VERSION == (
        "movie_news_telegram_post_prompt_v6"
    )

    assert OPENAI_POST_REVISION_PROMPT_VERSION == (
        "movie_news_telegram_post_revision_prompt_v4"
    )

    request = generator.build_request(
        build_news_items()
    )

    normalized_instructions = " ".join(
        request.instructions.split()
    )

    assert request.model == "gpt-5.6-terra"

    assert (
        "TOP-3 НОВОСТЕЙ КИНО"
        in request.instructions
    )

    assert (
        "телевизионные сериалы"
        in request.instructions
    )

    assert (
        "Прогнозы погоды"
        in request.instructions
    )

    assert (
        "Имена людей передавай кириллицей"
        in request.instructions
    )

    assert (
        "Фактическое содержание headline и body"
        in request.instructions
    )

    assert (
        "только из полей title и summary"
        in request.instructions
    )

    assert (
        POST_SUBSCRIPTION_LINE
        in request.instructions
    )

    payload = json.loads(
        request.input_text
    )

    assert payload["task"] == (
        "generate_russian_telegram_"
        "movie_news_top3"
    )

    assert payload["text_format"] == "markdown"

    assert (
        payload["maximum_post_length"]
        == MAXIMUM_POST_LENGTH
    )

    assert (
        (
            f"{TARGET_POST_LENGTH_MIN}–"
            f"{TARGET_POST_LENGTH_MAX} "
            "символов с пробелами"
        )
        in normalized_instructions
    )

    assert (
        (
            f"{TARGET_HEADLINE_LENGTH_MIN}–"
            f"{TARGET_HEADLINE_LENGTH_MAX} "
            "символов"
        )
        in normalized_instructions
    )

    assert (
        (
            "Абсолютный максимум headline — "
            f"{MAXIMUM_HEADLINE_LENGTH} символов"
        )
        in normalized_instructions
    )

    assert (
        (
            f"{TARGET_BODY_LENGTH_MIN}–"
            f"{TARGET_BODY_LENGTH_MAX} "
            "символов"
        )
        in normalized_instructions
    )

    assert (
        (
            "Абсолютный максимум body — "
            f"{MAXIMUM_BODY_LENGTH} символов"
        )
        in normalized_instructions
    )

    assert (
        (
            "Абсолютный максимум итогового "
            "post_text — "
            f"{MAXIMUM_POST_LENGTH} символов"
        )
        in normalized_instructions
    )

    assert [
        item["news_id"]
        for item in payload["news"]
    ] == [
        201,
        202,
        203,
    ]

    expected_news_fields = {
        "position",
        "news_id",
        "title",
        "summary",
    }

    assert all(
        set(item) == expected_news_fields
        for item in payload["news"]
    )

    assert all(
        "official_trailer_url" not in item
        for item in payload["news"]
    )

    assert len(client.requests) == 0

    print("Request preparation: OK")
    print(
        "generator_version="
        f"{metadata.generator_version}"
    )
    print(
        "prompt_version="
        f"{metadata.prompt_version}"
    )
    print("client_call_count=0")


def test_canonical_builder() -> None:
    """Проверяет точный шаблон Python-сборки."""

    response_payload = json.loads(
        build_valid_response()
    )

    from app.generation.openai_generator import (
        OpenAIGeneratedPostPayload,
    )

    payload = (
        OpenAIGeneratedPostPayload
        .model_validate(
            response_payload
        )
    )

    post_text = build_top3_post_text(
        payload.items
    )

    assert post_text == (
        build_expected_post_text()
    )

    assert post_text.startswith(
        POST_HEADER
        + "\n"
        + POST_TOP_SEPARATOR
    )

    assert post_text.endswith(
        POST_FOOTER_SEPARATOR
        + "\n"
        + POST_SUBSCRIPTION_LINE
    )

    assert post_text.count("1️⃣") == 1
    assert post_text.count("2️⃣") == 1
    assert post_text.count("3️⃣") == 1

    assert "#кино" not in post_text

    print("Canonical post builder: OK")
    print(f"post_length={len(post_text)}")


async def test_generation_success() -> None:
    """Проверяет успешную генерацию."""

    generator, client = build_generator()

    result = await generator.generate_detailed(
        build_news_items()
    )

    assert len(client.requests) == 1

    assert result.payload.post_text == (
        build_expected_post_text()
    )

    assert (
        "Черновик модели"
        not in result.payload.post_text
    )

    assert tuple(
        item.news_id
        for item in result.payload.items
    ) == (
        201,
        202,
        203,
    )

    assert (
        result.model_response.output_text
        == build_valid_response()
    )

    print("Detailed generation: OK")
    print("client_call_count=1")
    print(
        "canonical_post_length="
        f"{len(result.payload.post_text)}"
    )


async def test_generate_text_interface() -> None:
    """Проверяет интерфейс возврата одной строки."""

    generator, client = build_generator()

    post_text = await generator.generate(
        build_news_items()
    )

    assert post_text == (
        build_expected_post_text()
    )

    assert len(client.requests) == 1

    print("Text generation interface: OK")


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

    assert (
        result.payload.post_text
        == build_expected_post_text()
    )

    print("Prepared request generation: OK")


async def test_modified_request_blocking() -> None:
    """Не отправляет изменённый запрос модели."""

    generator, client = build_generator()

    items = build_news_items()

    request = generator.build_request(
        items
    )

    modified_request = (
        GenerationModelRequest(
            model=request.model,
            instructions=(
                request.instructions + " "
            ),
            input_text=request.input_text,
        )
    )

    await assert_raises_async(
        ValueError,
        "не соответствует",
        lambda: (
            generator
            .generate_prepared_request(
                items,
                modified_request,
            )
        ),
    )

    assert len(client.requests) == 0

    print("Modified request blocking: OK")
    print("client_call_count=0")


def test_invalid_input_count() -> None:
    """Блокирует TOP-3 неполного размера."""

    generator, client = build_generator()

    items = build_news_items()[:2]

    assert_raises(
        ValueError,
        "ровно три новости",
        lambda: generator.build_request(
            items
        ),
    )

    assert len(client.requests) == 0

    print("Invalid input count blocking: OK")


def test_invalid_input_order() -> None:
    """Блокирует неправильный порядок TOP-3."""

    generator, client = build_generator()

    items = build_news_items()

    invalid_items = (
        items[1],
        items[0],
        items[2],
    )

    assert_raises(
        ValueError,
        "порядке позиций",
        lambda: generator.build_request(
            invalid_items
        ),
    )

    assert len(client.requests) == 0

    print("Invalid input order blocking: OK")


def test_duplicate_input_news_id() -> None:
    """Блокирует повторяющийся входной news_id."""

    generator, client = build_generator()

    items = list(
        build_news_items()
    )

    items[1] = replace(
        items[1],
        news_id=items[0].news_id,
    )

    assert_raises(
        ValueError,
        "уникальными",
        lambda: generator.build_request(
            tuple(items)
        ),
    )

    assert len(client.requests) == 0

    print("Duplicate input news_id blocking: OK")


def test_naive_datetime() -> None:
    """Блокирует дату без часового пояса."""

    generator, client = build_generator()

    items = list(
        build_news_items()
    )

    items[0] = replace(
        items[0],
        source_published_at=datetime(
            2026,
            7,
            31,
            10,
            0,
        ),
    )

    assert_raises(
        ValueError,
        "содержать часовой пояс",
        lambda: generator.build_request(
            tuple(items)
        ),
    )

    assert len(client.requests) == 0

    print("Naive datetime blocking: OK")


def test_invalid_score() -> None:
    """Блокирует отрицательный балл."""

    generator, client = build_generator()

    items = list(
        build_news_items()
    )

    items[0] = replace(
        items[0],
        individual_score=Decimal("-1"),
    )

    assert_raises(
        ValueError,
        "неотрицательным",
        lambda: generator.build_request(
            tuple(items)
        ),
    )

    assert len(client.requests) == 0

    print("Invalid score blocking: OK")


def test_official_trailer_request_serialization() -> None:
    """Передаёт подтверждённый URL в primary JSON."""

    generator, client = build_generator()

    items = list(
        build_news_items()
    )

    items[1] = replace(
        items[1],
        official_trailer_url=(
            OFFICIAL_TRAILER_URL
        ),
    )

    request = generator.build_request(
        tuple(items)
    )

    payload = json.loads(
        request.input_text
    )

    assert (
        "official_trailer_url"
        not in payload["news"][0]
    )

    assert payload[
        "news"
    ][1][
        "official_trailer_url"
    ] == OFFICIAL_TRAILER_URL

    assert (
        "official_trailer_url"
        not in payload["news"][2]
    )

    assert len(client.requests) == 0

    print(
        "Official trailer primary "
        "serialization: OK"
    )


def test_invalid_official_trailer_url() -> None:
    """Блокирует некорректный trailer URL."""

    generator, client = build_generator()

    items = list(
        build_news_items()
    )

    items[1] = replace(
        items[1],
        official_trailer_url=(
            "youtube.com/watch?v=invalid"
        ),
    )

    assert_raises(
        ValueError,
        "абсолютным HTTP или HTTPS URL",
        lambda: generator.build_request(
            tuple(items)
        ),
    )

    assert len(client.requests) == 0

    print(
        "Invalid official trailer URL "
        "blocking: OK"
    )


def test_self_review_request_preparation() -> None:
    """Передаёт trailer URL во второй проход."""

    generator, client = build_generator()

    items = list(
        build_news_items()
    )

    items[1] = replace(
        items[1],
        official_trailer_url=(
            OFFICIAL_TRAILER_URL
        ),
    )

    request = (
        generator.build_self_review_request(
            tuple(items),
            source_post_text=(
                REVISION_SOURCE_POST_TEXT
            ),
        )
    )

    assert request.model == "gpt-5.6-terra"
    assert request.allow_web_search is True

    payload = json.loads(
        request.input_text
    )

    assert payload["task"] == (
        "self_review_russian_telegram_"
        "movie_news_top3"
    )

    assert (
        payload["source_post_text"]
        == REVISION_SOURCE_POST_TEXT
    )

    expected_news_fields = {
        "position",
        "news_id",
        "title",
        "summary",
        "source_name",
        "source_url",
        "source_published_at",
    }

    assert (
        set(payload["news"][0])
        == expected_news_fields
    )

    assert (
        set(payload["news"][1])
        == (
            expected_news_fields
            | {"official_trailer_url"}
        )
    )

    assert (
        set(payload["news"][2])
        == expected_news_fields
    )

    assert (
        "official_trailer_url"
        not in payload["news"][0]
    )

    assert payload[
        "news"
    ][1][
        "official_trailer_url"
    ] == OFFICIAL_TRAILER_URL

    assert (
        "official_trailer_url"
        not in payload["news"][2]
    )

    assert len(client.requests) == 0

    print(
        "Official trailer self-review "
        "serialization: OK"
    )
    print(
        "self_review_allow_web_search=true"
    )


async def test_invalid_json() -> None:
    """Блокирует синтаксически неверный JSON."""

    generator, client = build_generator(
        "{invalid-json"
    )

    await assert_raises_async(
        ValueError,
        "не соответствует схеме",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Invalid JSON blocking: OK")


async def test_wrong_response_news_ids() -> None:
    """Блокирует изменение набора news_id."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][1]["news_id"] = 999

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "изменила порядок или набор",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Wrong response news_id blocking: OK")


async def test_duplicate_response_news_id() -> None:
    """Блокирует повторяющийся news_id ответа."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][1]["news_id"] = (
        payload["items"][0]["news_id"]
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "не соответствует схеме",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Duplicate response news_id blocking: OK")


async def test_empty_headline() -> None:
    """Блокирует пустой headline."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["headline"] = "   "

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "не соответствует схеме",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Empty headline blocking: OK")


async def test_headline_markdown() -> None:
    """Блокирует Markdown-маркеры вокруг headline."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["headline"] = (
        "**Заголовок с маркерами**"
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "без Markdown-маркеров",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Headline Markdown blocking: OK")


async def test_hashtag_blocking() -> None:
    """Блокирует хештеги в body."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["body"] += (
        "\n\n#кино"
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "Хештеги в body запрещены",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Hashtag blocking: OK")


async def test_service_fragment_in_body() -> None:
    """Блокирует служебные элементы в body."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["body"] += (
        "\n\n"
        + POST_SUBSCRIPTION_LINE
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "служебный элемент шаблона",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Service fragment blocking: OK")


async def test_unpaired_body_markdown() -> None:
    """Блокирует непарный Markdown в body."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["body"] += (
        "\n\n**Незакрытый жирный текст"
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "непарный маркер жирного",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Unpaired body Markdown blocking: OK")


async def test_oversized_headline() -> None:
    """Блокирует headline длиннее технического лимита."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["headline"] = (
        "X" * (MAXIMUM_HEADLINE_LENGTH + 1)
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "не соответствует схеме",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Oversized headline blocking: OK")


async def test_oversized_body() -> None:
    """Блокирует body длиннее технического лимита."""

    payload = json.loads(
        build_valid_response()
    )

    payload["items"][0]["body"] = (
        "X" * (MAXIMUM_BODY_LENGTH + 1)
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    await assert_raises_async(
        ValueError,
        "не соответствует схеме",
        lambda: generator.generate(
            build_news_items()
        ),
    )

    assert len(client.requests) == 1

    print("Oversized body blocking: OK")


async def test_model_draft_is_ignored() -> None:
    """Проверяет, что модель не управляет шаблоном."""

    payload = json.loads(
        build_valid_response()
    )

    payload["post_text"] = (
        "Совершенно неправильный формат модели."
    )

    generator, client = build_generator(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    result = await generator.generate_detailed(
        build_news_items()
    )

    assert (
        result.payload.post_text
        == build_expected_post_text()
    )

    assert (
        "Совершенно неправильный формат"
        not in result.payload.post_text
    )

    assert len(client.requests) == 1

    print("Model draft replacement: OK")


def test_revision_request_preparation() -> None:
    """Проверяет подготовку revision-запроса."""

    generator, client = build_generator()

    request = generator.build_revision_request(
        build_news_items(),
        source_post_text=REVISION_SOURCE_POST_TEXT,
        editorial_comment=REVISION_EDITORIAL_COMMENT,
        issues=REVISION_ISSUES,
    )

    assert request.model == "gpt-5.6-terra"

    assert (
        "Соблюдай все основные правила "
        "генерации поста."
        in request.instructions
    )

    assert (
        "source_post_text — текущая редактируемая версия"
        in request.instructions
    )

    assert (
        "но не как источник новых фактов"
        in request.instructions
    )

    assert (
        "только из полей title и summary"
        in request.instructions
    )

    payload = json.loads(request.input_text)

    assert payload["task"] == (
        "revise_russian_telegram_"
        "movie_news_top3"
    )

    assert (
        payload["text_format"]
        == OPENAI_POST_TEXT_FORMAT
    )

    assert (
        payload["maximum_post_length"]
        == MAXIMUM_POST_LENGTH
    )

    normalized_instructions = " ".join(
        request.instructions.split()
    )

    assert (
        (
            f"{TARGET_POST_LENGTH_MIN}–"
            f"{TARGET_POST_LENGTH_MAX} "
            "символов с пробелами"
        )
        in normalized_instructions
    )

    assert (
        (
            f"{TARGET_HEADLINE_LENGTH_MIN}–"
            f"{TARGET_HEADLINE_LENGTH_MAX} "
            "символов"
        )
        in normalized_instructions
    )

    assert (
        (
            f"{TARGET_BODY_LENGTH_MIN}–"
            f"{TARGET_BODY_LENGTH_MAX} "
            "символов"
        )
        in normalized_instructions
    )

    assert (
        (
            "Абсолютный максимум итогового "
            "post_text — "
            f"{MAXIMUM_POST_LENGTH} символов"
        )
        in normalized_instructions
    )

    assert (
        payload["source_post_text"]
        == REVISION_SOURCE_POST_TEXT
    )

    assert (
        payload["editorial_comment"]
        == REVISION_EDITORIAL_COMMENT
    )

    assert payload["issues"] == list(
        REVISION_ISSUES
    )

    expected_news_fields = {
        "position",
        "news_id",
        "title",
        "summary",
    }

    assert all(
        set(item) == expected_news_fields
        for item in payload["news"]
    )

    assert [
        item["news_id"]
        for item in payload["news"]
    ] == [
        201,
        202,
        203,
    ]

    assert len(client.requests) == 0

    print("Revision request preparation: OK")
    print("revision_client_call_count=0")


def test_empty_revision_issues() -> None:
    """Блокирует пустой список замечаний."""

    generator, client = build_generator()

    assert_raises(
        ValueError,
        "issues не может быть пустым",
        lambda: generator.build_revision_request(
            build_news_items(),
            source_post_text=REVISION_SOURCE_POST_TEXT,
            editorial_comment=(
                REVISION_EDITORIAL_COMMENT
            ),
            issues=(),
        ),
    )

    assert len(client.requests) == 0

    print("Empty revision issues blocking: OK")


def test_empty_revision_comment() -> None:
    """Блокирует пустой комментарий редактора."""

    generator, client = build_generator()

    assert_raises(
        ValueError,
        "editorial_comment не может быть пустым",
        lambda: generator.build_revision_request(
            build_news_items(),
            source_post_text=REVISION_SOURCE_POST_TEXT,
            editorial_comment="   ",
            issues=REVISION_ISSUES,
        ),
    )

    assert len(client.requests) == 0

    print("Empty revision comment blocking: OK")


async def test_revision_generation_success() -> None:
    """Проверяет успешную доработку поста."""

    generator, client = build_generator()
    items = build_news_items()

    request = generator.build_revision_request(
        items,
        source_post_text=REVISION_SOURCE_POST_TEXT,
        editorial_comment=REVISION_EDITORIAL_COMMENT,
        issues=REVISION_ISSUES,
    )

    result = (
        await generator
        .generate_prepared_revision_request(
            items,
            request,
            source_post_text=REVISION_SOURCE_POST_TEXT,
            editorial_comment=REVISION_EDITORIAL_COMMENT,
            issues=REVISION_ISSUES,
        )
    )

    assert len(client.requests) == 1
    assert client.requests[0] == request

    assert (
        result.payload.post_text
        == build_expected_post_text()
    )

    assert (
        "Черновик модели"
        not in result.payload.post_text
    )

    assert tuple(
        item.news_id
        for item in result.payload.items
    ) == (
        201,
        202,
        203,
    )

    print("Revision generation: OK")
    print("revision_client_call_count=1")


async def test_modified_revision_source_blocking() -> None:
    """Блокирует подмену исходного поста."""

    generator, client = build_generator()
    items = build_news_items()

    request = generator.build_revision_request(
        items,
        source_post_text=REVISION_SOURCE_POST_TEXT,
        editorial_comment=REVISION_EDITORIAL_COMMENT,
        issues=REVISION_ISSUES,
    )

    await assert_raises_async(
        ValueError,
        "revision-запрос",
        lambda: (
            generator
            .generate_prepared_revision_request(
                items,
                request,
                source_post_text=(
                    REVISION_SOURCE_POST_TEXT
                    + "\nПодмена."
                ),
                editorial_comment=(
                    REVISION_EDITORIAL_COMMENT
                ),
                issues=REVISION_ISSUES,
            )
        ),
    )

    assert len(client.requests) == 0

    print(
        "Modified revision source blocking: OK"
    )
    print("revision_client_call_count=0")


async def test_modified_revision_comment_blocking() -> None:
    """Блокирует подмену комментария редактора."""

    generator, client = build_generator()
    items = build_news_items()

    request = generator.build_revision_request(
        items,
        source_post_text=REVISION_SOURCE_POST_TEXT,
        editorial_comment=REVISION_EDITORIAL_COMMENT,
        issues=REVISION_ISSUES,
    )

    await assert_raises_async(
        ValueError,
        "revision-запрос",
        lambda: (
            generator
            .generate_prepared_revision_request(
                items,
                request,
                source_post_text=REVISION_SOURCE_POST_TEXT,
                editorial_comment=(
                    REVISION_EDITORIAL_COMMENT
                    + " Дополнительное изменение."
                ),
                issues=REVISION_ISSUES,
            )
        ),
    )

    assert len(client.requests) == 0

    print(
        "Modified revision comment blocking: OK"
    )
    print("revision_client_call_count=0")


async def main() -> int:
    """Запускает изолированные тесты генератора."""

    print(
        "OpenAI post generator isolated test"
    )
    print(
        "database_connections=not_performed"
    )
    print("openai_requests=not_performed")
    print("telegram_requests=not_performed")
    print()

    test_metadata_and_request()
    test_canonical_builder()
    test_invalid_input_count()
    test_invalid_input_order()
    test_duplicate_input_news_id()
    test_naive_datetime()
    test_invalid_score()
    test_official_trailer_request_serialization()
    test_invalid_official_trailer_url()
    test_self_review_request_preparation()
    test_revision_request_preparation()
    test_empty_revision_issues()
    test_empty_revision_comment()

    await test_generation_success()
    await test_generate_text_interface()
    await test_prepared_request()
    await test_modified_request_blocking()
    await test_invalid_json()
    await test_wrong_response_news_ids()
    await test_duplicate_response_news_id()
    await test_empty_headline()
    await test_headline_markdown()
    await test_hashtag_blocking()
    await test_service_fragment_in_body()
    await test_unpaired_body_markdown()
    await test_oversized_headline()
    await test_oversized_body()
    await test_model_draft_is_ignored()
    await test_revision_generation_success()
    await test_modified_revision_source_blocking()
    await test_modified_revision_comment_blocking()

    print()
    print(
        "OpenAI post generator test: OK"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(main())
    )