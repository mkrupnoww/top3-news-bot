from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import re
from typing import Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


OPENAI_POST_GENERATOR_VERSION = (
    "openai_telegram_post_generator_v3"
)

OPENAI_POST_PROMPT_VERSION = (
    "movie_news_telegram_post_prompt_v3"
)

OPENAI_POST_REVISION_PROMPT_VERSION = (
    "movie_news_telegram_post_revision_prompt_v1"
)

OPENAI_POST_TEXT_FORMAT = "markdown"

MAXIMUM_POST_LENGTH = 3900

POST_HEADER = (
    "**TOP-3 НОВОСТЕЙ КИНО "
    "ЗА ПОСЛЕДНИЕ 24 ЧАСА**"
)

POST_TOP_SEPARATOR = "_______________"

POST_FOOTER_SEPARATOR = "……………"

POST_SUBSCRIPTION_LINE = (
    "Подписаться на VIP канал - @kkm_vip_bot"
)

POST_POSITION_MARKERS = (
    "1️⃣",
    "2️⃣",
    "3️⃣",
)

_HASHTAG_PATTERN = re.compile(
    r"(?<![\w/])#[\wА-Яа-яЁё]+",
    flags=re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class GenerationNewsItem:
    """Одна новость из сохранённого TOP-3."""

    position: int
    news_id: int
    title: str
    summary: str
    source_name: str
    source_url: str
    source_published_at: datetime
    individual_score: Decimal
    selection_reason: str


@dataclass(frozen=True, slots=True)
class GenerationModelRequest:
    """Точный запрос к модели генерации текста."""

    model: str
    instructions: str
    input_text: str


@dataclass(frozen=True, slots=True)
class GenerationModelResponse:
    """Ответ модели вместе с телеметрией."""

    output_text: str
    usage: OpenAITokenUsage | None = None
    cost_estimate: OpenAICostEstimate | None = None


@dataclass(frozen=True, slots=True)
class OpenAIPostGeneratorMetadata:
    """Версии компонентов генератора."""

    generator_name: str
    generator_version: str
    prompt_version: str
    model_name: str
    text_format: str


@dataclass(frozen=True, slots=True)
class OpenAIPostGenerationResult:
    """Сгенерированный пост и данные API-запроса."""

    payload: "OpenAIGeneratedPostPayload"
    model_response: GenerationModelResponse


@runtime_checkable
class StructuredGenerationClient(
    Protocol
):
    """
    Транспортный интерфейс генератора.

    Реальная реализация использует OpenAI
    Responses API. В тестах применяется
    локальный клиент без сетевых запросов.
    """

    async def create_response(
        self,
        request: GenerationModelRequest,
    ) -> GenerationModelResponse:
        """Возвращает структурированный ответ."""

        ...


class OpenAIGeneratedNewsPayload(
    BaseModel
):
    """Сгенерированный текст одной новости."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    position: int = Field(
        ge=1,
        le=3,
    )

    news_id: int = Field(
        gt=0,
    )

    headline: str = Field(
        min_length=1,
        max_length=300,
    )

    body: str = Field(
        min_length=1,
        max_length=1400,
    )

    @field_validator(
        "headline",
        "body",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: str,
    ) -> str:
        """Удаляет внешние пробелы."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "Текстовое поле не может "
                "быть пустым."
            )

        return normalized_value


class OpenAIGeneratedPostPayload(
    BaseModel
):
    """Полный структурированный ответ генератора."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    post_text: str = Field(
        min_length=1,
        max_length=4096,
    )

    items: list[
        OpenAIGeneratedNewsPayload
    ] = Field(
        min_length=3,
        max_length=3,
    )

    @field_validator("post_text")
    @classmethod
    def normalize_post_text(
        cls,
        value: str,
    ) -> str:
        """Нормализует черновик Telegram-текста."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "post_text не может быть пустым."
            )

        if len(normalized_value) > 4096:
            raise ValueError(
                "post_text превышает ограничение "
                "Telegram в 4096 символов."
            )

        return normalized_value

    @model_validator(mode="after")
    def validate_items(
        self,
    ) -> "OpenAIGeneratedPostPayload":
        """Проверяет позиции и уникальность новостей."""

        positions = [
            item.position
            for item in self.items
        ]

        if positions != [1, 2, 3]:
            raise ValueError(
                "items должны идти строго "
                "в порядке позиций 1, 2 и 3."
            )

        news_ids = [
            item.news_id
            for item in self.items
        ]

        if len(set(news_ids)) != 3:
            raise ValueError(
                "items должны содержать "
                "три уникальных news_id."
            )

        return self


SYSTEM_INSTRUCTIONS = """
Ты готовишь русскоязычный пост для Telegram-канала
с ежедневной подборкой TOP-3 новостей о кино
и сериалах.

Используй только сведения из переданного JSON.
Не добавляй факты, цитаты, числа, имена, оценки,
просмотры, реакции, даты или последствия, которых
нет во входных данных.

Тематическая область включает:

- полнометражные и короткометражные фильмы;
- документальное кино;
- анимационные фильмы и сериалы;
- телевизионные сериалы;
- мини-сериалы и limited series;
- сериалы стриминговых платформ;
- производство, кастинг и съёмки;
- премьеры, трейлеры и даты выпуска;
- кинотеатральную и стриминговую дистрибуцию;
- фестивали и награды;
- бизнес, право и кадровые решения, если они
  непосредственно связаны с производством или
  распространением фильмов и сериалов.

Слово television, TV или название телевизионной
компании само по себе не означает, что публикация
относится к сериалам.

Прогнозы погоды, новостные эфиры, спортивные
трансляции, ток-шоу, реалити-шоу и обычные
телевизионные программы не являются сериалами.

Требования к содержанию:

1. Сохрани порядок новостей 1, 2 и 3.
2. Для каждой новости подготовь короткий,
   выразительный headline и содержательный body.
3. Body может состоять из одного или двух
   коротких абзацев.
4. Передавай смысл своими словами.
5. Не копируй длинные фрагменты публикаций.
6. Пиши естественно и информативно.
7. Не используй кликбейт, канцелярит и рекламу.
8. Имена людей передавай кириллицей:
   используй общепринятую русскую форму или
   нейтральную транслитерацию. Не оставляй имя
   и фамилию целиком латиницей.
9. Имена известных персонажей и франшиз
   передавай в общепринятой русской форме,
   если она однозначна. Например: Cyclops —
   Циклоп, X-Men — «Люди Икс».
10. Названия компаний и брендов передавай
    точно. Допустимо сохранять их латинское
    написание: A24, Paramount, Marvel.
11. Названия фильмов и сериалов сохраняй
    в оригинальном написании, если во входных
    данных нет подтверждённого русского
    названия. Не придумывай локализованные
    названия.
12. Фактическое содержание headline и body
    формируй только из полей title и summary
    соответствующей новости.
13. Не добавляй других актёров, персонажей,
    сделки, даты, цитаты или обстоятельства,
    которых нет в title или summary этой
    новости.
14. Не включай технические показатели, баллы,
    news_id или внутренние объяснения оценщика
    в текст публикации.
15. Не вставляй ссылки на исходные статьи.
16. Не добавляй длинный перечень источников.
17. Если основным событием новости является
    публикация нового официального трейлера
    фильма или сериала, добавь прямую ссылку
    на этот трейлер только при наличии во
    входных данных отдельного подтверждённого
    поля official_trailer_url.
18. official_trailer_url должен вести на
    официальный источник: канал или сайт студии,
    дистрибьютора, стримингового сервиса либо
    другого правообладателя.
19. Не используй перезаливы, агрегаторы,
    публикации СМИ или неофициальные каналы
    вместо подтверждённого официального
    первоисточника.
20. Если official_trailer_url отсутствует,
    не придумывай URL и не пытайся восстановить
    его по названию фильма или сериала.
21. Если трейлер лишь упоминается в новости,
    но его публикация не является основным
    событием новости, ссылка на трейлер
    не обязательна.
22. Не называй событие скандалом, сенсацией или
    рекордом, если это прямо не подтверждено.
23. Хештеги запрещены.
24. Markdown-заголовки с символом # запрещены.
25. Не используй MarkdownV2 или HTML.
26. Для жирного текста внутри body используй
    только две звёздочки: **текст**.
27. Для курсива внутри body используй только
    двойное нижнее подчёркивание: __текст__.
28. headline возвращай без внешних Markdown-
    маркеров.
29. Верни каждую входную новость ровно один раз.
30. items должны идти строго в порядке:
    position=1, position=2, position=3.
31. Общая длина будущего поста должна позволять
    уложиться в 3900 символов.

Финальный post_text будет программно собран
Python-кодом из headline и body.

Не добавляй в headline или body служебные элементы:

- заголовок всего выпуска;
- строку из подчёркиваний;
- номера 1️⃣, 2️⃣ и 3️⃣;
- нижний разделитель;
- строку подписки.

Python самостоятельно добавит эти элементы по
неизменяемому шаблону:

**TOP-3 НОВОСТЕЙ КИНО ЗА ПОСЛЕДНИЕ 24 ЧАСА**
_______________

1️⃣ **Заголовок первой новости**

Текст первой новости.

2️⃣ **Заголовок второй новости**

Текст второй новости.

3️⃣ **Заголовок третьей новости**

Текст третьей новости.

……………
Подписаться на VIP канал - @kkm_vip_bot

Поле post_text также обязательно верни в JSON.
Оно является черновиком модели.

Окончательный канонический post_text будет
построен Python-кодом из массива items, поэтому
главное — вернуть точные headline и body.

Формат JSON-ответа:

{
  "post_text": "Черновик полного Telegram-поста",
  "items": [
    {
      "position": 1,
      "news_id": 1,
      "headline": "Короткий заголовок",
      "body": "Текст первой новости"
    },
    {
      "position": 2,
      "news_id": 2,
      "headline": "Короткий заголовок",
      "body": "Текст второй новости"
    },
    {
      "position": 3,
      "news_id": 3,
      "headline": "Короткий заголовок",
      "body": "Текст третьей новости"
    }
  ]
}

Верни только JSON-объект без Markdown-обёртки.
""".strip()


REVISION_SYSTEM_INSTRUCTIONS = """
Ты дорабатываешь ранее созданный русскоязычный
Telegram-пост с ежедневной подборкой TOP-3
новостей о кино и сериалах.

Соблюдай все основные правила генерации поста.

Дополнительные правила ревизии:

1. Выполни замечания редактора из полей
   editorial_comment и issues.

2. source_post_text — текущая редактируемая версия
   поста. Используй её как основу новой версии,
   но не как источник новых фактов.

3. Факты для каждой новости разрешено брать
   только из полей title и summary этой новости.

4. Изменяй только те фрагменты source_post_text,
   которые необходимо изменить для выполнения
   текущих замечаний редактора или для устранения
   фактического нарушения основных правил.

5. Если замечание редактора не относится к
   конкретному headline или body, сохрани этот
   headline или body максимально близко к
   source_post_text. Не переписывай его только
   ради стилистического улучшения.

6. Сохраняй уже внесённые в source_post_text
   редакционные исправления. Не отменяй и не
   ухудшай предыдущие исправления, если текущее
   замечание этого прямо не требует.

7. Не перефразируй исправленные ранее фрагменты
   без необходимости. При последовательных
   ревизиях каждая новая версия должна быть
   точечной доработкой предыдущей версии,
   а не новой генерацией всего поста.

8. Если замечание невозможно выполнить на
   основании title и summary, не выдумывай
   недостающие имена, должности, события, даты,
   цитаты или другие факты. В таком случае
   сохрани фактически безопасную формулировку.

9. Удали из предыдущего текста любые имена,
   события, даты, цитаты и обстоятельства,
   которые не подтверждены полями title
   и summary соответствующей новости.

10. Сохрани исходный порядок трёх новостей
    и их news_id.

11. В JSON обязательно верни headline и body
    для всех трёх новостей. Для незатронутых
    редакционным замечанием новостей повтори
    содержание текущей версии без ненужного
    переписывания.

12. Не описывай внесённые исправления и не
    обращайся к редактору.

13. Верни только JSON-объект в той же структуре,
    что требуется основными инструкциями.
""".strip()


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательное текстовое поле."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} должен быть строкой."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_revision_issues(
    issues: tuple[str, ...],
) -> tuple[str, ...]:
    """Проверяет редакционные замечания."""

    if not isinstance(issues, tuple):
        raise TypeError(
            "issues должен быть tuple."
        )

    if not issues:
        raise ValueError(
            "issues не может быть пустым."
        )

    normalized_issues: list[str] = []

    for index, issue in enumerate(
        issues,
        start=1,
    ):
        normalized_issue = (
            _normalize_required_text(
                issue,
                field_name=(
                    f"issues[{index}]"
                ),
            )
        )

        normalized_issues.append(
            normalized_issue
        )

    return tuple(normalized_issues)


def _build_revision_input_text(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
    *,
    source_post_text: str,
    editorial_comment: str,
    issues: tuple[str, ...],
) -> str:
    """Формирует JSON запроса на ревизию."""

    _validate_news_items(items)

    normalized_source_post_text = (
        _normalize_required_text(
            source_post_text,
            field_name="source_post_text",
        )
    )

    normalized_editorial_comment = (
        _normalize_required_text(
            editorial_comment,
            field_name="editorial_comment",
        )
    )

    normalized_issues = (
        _normalize_revision_issues(
            issues
        )
    )

    payload = {
        "task": (
            "revise_russian_telegram_"
            "movie_news_top3"
        ),
        "text_format": (
            OPENAI_POST_TEXT_FORMAT
        ),
        "maximum_post_length": (
            MAXIMUM_POST_LENGTH
        ),
        "source_post_text": (
            normalized_source_post_text
        ),
        "editorial_comment": (
            normalized_editorial_comment
        ),
        "issues": list(
            normalized_issues
        ),
        "news": [
            {
                "position": item.position,
                "news_id": item.news_id,
                "title": item.title.strip(),
                "summary": item.summary.strip(),
            }
            for item in items
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _validate_news_items(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> tuple[int, ...]:
    """Проверяет входной TOP-3."""

    if len(items) != 3:
        raise ValueError(
            "Для генерации требуется ровно "
            "три новости."
        )

    positions = tuple(
        item.position
        for item in items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Новости должны идти строго "
            "в порядке позиций 1, 2 и 3."
        )

    news_ids = tuple(
        item.news_id
        for item in items
    )

    for news_id in news_ids:
        if isinstance(news_id, bool):
            raise TypeError(
                "news_id не может быть bool."
            )

        if not isinstance(news_id, int):
            raise TypeError(
                "news_id должен быть int."
            )

        if news_id <= 0:
            raise ValueError(
                "news_id должен быть "
                "больше нуля."
            )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Все три news_id должны "
            "быть уникальными."
        )

    for item in items:
        _normalize_required_text(
            item.title,
            field_name=(
                f"title news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.summary,
            field_name=(
                f"summary news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.source_name,
            field_name=(
                "source_name "
                f"news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.source_url,
            field_name=(
                "source_url "
                f"news_id={item.news_id}"
            ),
        )

        _normalize_required_text(
            item.selection_reason,
            field_name=(
                "selection_reason "
                f"news_id={item.news_id}"
            ),
        )

        published_at = (
            item.source_published_at
        )

        if not isinstance(
            published_at,
            datetime,
        ):
            raise TypeError(
                "source_published_at должен "
                "быть datetime: "
                f"news_id={item.news_id}"
            )

        if (
            published_at.tzinfo is None
            or published_at.utcoffset() is None
        ):
            raise ValueError(
                "source_published_at должен "
                "содержать часовой пояс: "
                f"news_id={item.news_id}"
            )

        if not isinstance(
            item.individual_score,
            Decimal,
        ):
            raise TypeError(
                "individual_score должен "
                "быть Decimal: "
                f"news_id={item.news_id}"
            )

        if (
            not item.individual_score.is_finite()
            or item.individual_score < 0
        ):
            raise ValueError(
                "individual_score должен быть "
                "конечным неотрицательным числом: "
                f"news_id={item.news_id}"
            )

    return news_ids


def _build_input_text(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
) -> str:
    """Формирует JSON с TOP-3 для модели."""

    payload = {
        "task": (
            "generate_russian_telegram_"
            "movie_news_top3"
        ),
        "text_format": (
            OPENAI_POST_TEXT_FORMAT
        ),
        "maximum_post_length": (
            MAXIMUM_POST_LENGTH
        ),
        "news": [
            {
                "position": item.position,
                "news_id": item.news_id,
                "title": item.title.strip(),
                "summary": item.summary.strip(),
            }
            for item in items
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_response(
    response_text: str,
) -> OpenAIGeneratedPostPayload:
    """Проверяет структурированный ответ."""

    normalized_response = (
        response_text.strip()
    )

    if not normalized_response:
        raise ValueError(
            "Модель вернула пустой ответ."
        )

    try:
        return (
            OpenAIGeneratedPostPayload
            .model_validate_json(
                normalized_response
            )
        )
    except ValidationError as error:
        raise ValueError(
            "Ответ модели не соответствует "
            "схеме Telegram-поста."
        ) from error


def _validate_response_items(
    *,
    expected_news_ids: tuple[int, ...],
    payload: OpenAIGeneratedPostPayload,
) -> None:
    """Сверяет позиции и news_id ответа."""

    response_news_ids = tuple(
        item.news_id
        for item in payload.items
    )

    if response_news_ids != expected_news_ids:
        raise ValueError(
            "Модель изменила порядок или набор "
            "news_id: "
            f"expected={expected_news_ids}, "
            f"actual={response_news_ids}"
        )

    response_positions = tuple(
        item.position
        for item in payload.items
    )

    if response_positions != (1, 2, 3):
        raise ValueError(
            "Модель вернула некорректный "
            "порядок позиций: "
            f"{response_positions}"
        )


def _validate_generated_headline(
    headline: str,
    *,
    position: int,
) -> str:
    """Проверяет заголовок одной новости."""

    normalized_headline = (
        _normalize_required_text(
            headline,
            field_name=(
                f"headline position={position}"
            ),
        )
    )

    if (
        "\n" in normalized_headline
        or "\r" in normalized_headline
    ):
        raise ValueError(
            "headline не должен содержать "
            "перевод строки: "
            f"position={position}"
        )

    if (
        "**" in normalized_headline
        or "__" in normalized_headline
    ):
        raise ValueError(
            "headline должен содержать чистый "
            "текст без Markdown-маркеров: "
            f"position={position}"
        )

    if _HASHTAG_PATTERN.search(
        normalized_headline
    ):
        raise ValueError(
            "Хештеги в headline запрещены: "
            f"position={position}"
        )

    forbidden_fragments = (
        POST_HEADER,
        POST_TOP_SEPARATOR,
        POST_FOOTER_SEPARATOR,
        POST_SUBSCRIPTION_LINE,
        *POST_POSITION_MARKERS,
    )

    for fragment in forbidden_fragments:
        if fragment in normalized_headline:
            raise ValueError(
                "headline содержит служебный "
                "элемент шаблона: "
                f"position={position}, "
                f"fragment={fragment!r}"
            )

    return normalized_headline


def _validate_generated_body(
    body: str,
    *,
    position: int,
) -> str:
    """Проверяет текст одной новости."""

    normalized_body = _normalize_required_text(
        body,
        field_name=f"body position={position}",
    )

    if _HASHTAG_PATTERN.search(normalized_body):
        raise ValueError(
            "Хештеги в body запрещены: "
            f"position={position}"
        )

    forbidden_fragments = (
        POST_HEADER,
        POST_TOP_SEPARATOR,
        POST_FOOTER_SEPARATOR,
        POST_SUBSCRIPTION_LINE,
        *POST_POSITION_MARKERS,
    )

    for fragment in forbidden_fragments:
        if fragment in normalized_body:
            raise ValueError(
                "body содержит служебный "
                "элемент шаблона: "
                f"position={position}, "
                f"fragment={fragment!r}"
            )

    if normalized_body.count("**") % 2 != 0:
        raise ValueError(
            "В body обнаружен непарный "
            "маркер жирного текста **: "
            f"position={position}"
        )

    if normalized_body.count("__") % 2 != 0:
        raise ValueError(
            "В body обнаружен непарный "
            "маркер курсива __: "
            f"position={position}"
        )

    return normalized_body


def build_top3_post_text(
    items: list[
        OpenAIGeneratedNewsPayload
    ],
) -> str:
    """
    Собирает финальный Markdown-пост TOP-3.

    Служебные элементы не зависят от формулировки
    модели и всегда добавляются Python-кодом.
    """

    if len(items) != 3:
        raise ValueError(
            "Для построения поста требуется "
            "ровно три элемента."
        )

    positions = tuple(
        item.position
        for item in items
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Элементы должны идти строго "
            "в порядке позиций 1, 2 и 3: "
            f"actual={positions}"
        )

    news_sections: list[str] = []

    for item, marker in zip(
        items,
        POST_POSITION_MARKERS,
        strict=True,
    ):
        headline = (
            _validate_generated_headline(
                item.headline,
                position=item.position,
            )
        )

        body = _validate_generated_body(
            item.body,
            position=item.position,
        )

        news_sections.append(
            f"{marker} **{headline}**"
            f"\n\n{body}"
        )

    post_text = (
        f"{POST_HEADER}\n"
        f"{POST_TOP_SEPARATOR}\n\n"
        + "\n\n".join(news_sections)
        + "\n\n"
        + f"{POST_FOOTER_SEPARATOR}\n"
        + POST_SUBSCRIPTION_LINE
    )

    if len(post_text) > MAXIMUM_POST_LENGTH:
        raise ValueError(
            "Собранный post_text превышает "
            f"{MAXIMUM_POST_LENGTH} символов: "
            f"actual={len(post_text)}"
        )

    return post_text


class OpenAITelegramPostGenerator:
    """Генератор Telegram-поста через OpenAI."""

    def __init__(
        self,
        *,
        client: StructuredGenerationClient,
        model_name: str,
    ) -> None:
        normalized_model_name = (
            model_name.strip()
        )

        if not normalized_model_name:
            raise ValueError(
                "model_name не может быть пустым."
            )

        self._client = client

        self._metadata = (
            OpenAIPostGeneratorMetadata(
                generator_name=(
                    "OpenAITelegramPostGenerator"
                ),
                generator_version=(
                    OPENAI_POST_GENERATOR_VERSION
                ),
                prompt_version=(
                    OPENAI_POST_PROMPT_VERSION
                ),
                model_name=(
                    normalized_model_name
                ),
                text_format=(
                    OPENAI_POST_TEXT_FORMAT
                ),
            )
        )

    @property
    def metadata(
        self,
    ) -> OpenAIPostGeneratorMetadata:
        """Возвращает метаданные генератора."""

        return self._metadata

    def build_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
    ) -> GenerationModelRequest:
        """Формирует запрос без вызова модели."""

        _validate_news_items(items)

        return GenerationModelRequest(
            model=self._metadata.model_name,
            instructions=SYSTEM_INSTRUCTIONS,
            input_text=_build_input_text(
                items
            ),
        )

    def build_revision_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        *,
        source_post_text: str,
        editorial_comment: str,
        issues: tuple[str, ...],
    ) -> GenerationModelRequest:
        """Формирует запрос на доработку поста."""

        _validate_news_items(items)

        instructions = (
            f"{SYSTEM_INSTRUCTIONS}\n\n"
            f"{REVISION_SYSTEM_INSTRUCTIONS}"
        )

        return GenerationModelRequest(
            model=self._metadata.model_name,
            instructions=instructions,
            input_text=_build_revision_input_text(
                items,
                source_post_text=source_post_text,
                editorial_comment=(
                    editorial_comment
                ),
                issues=issues,
            ),
        )

    async def generate_prepared_revision_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        request: GenerationModelRequest,
        *,
        source_post_text: str,
        editorial_comment: str,
        issues: tuple[str, ...],
    ) -> OpenAIPostGenerationResult:
        """
        Выполняет заранее сформированный запрос
        на редакционную доработку поста.
        """

        expected_news_ids = (
            _validate_news_items(items)
        )

        expected_request = (
            self.build_revision_request(
                items,
                source_post_text=(
                    source_post_text
                ),
                editorial_comment=(
                    editorial_comment
                ),
                issues=issues,
            )
        )

        if request != expected_request:
            raise ValueError(
                "Подготовленный revision-запрос "
                "не соответствует текущему TOP-3, "
                "исходному посту, замечаниям, "
                "модели или промпту."
            )

        model_response = (
            await self._client.create_response(
                request
            )
        )

        payload = _parse_response(
            model_response.output_text
        )

        _validate_response_items(
            expected_news_ids=(
                expected_news_ids
            ),
            payload=payload,
        )

        canonical_post_text = (
            build_top3_post_text(
                payload.items
            )
        )

        payload = payload.model_copy(
            update={
                "post_text": (
                    canonical_post_text
                ),
            }
        )

        return OpenAIPostGenerationResult(
            payload=payload,
            model_response=model_response,
        )

    async def generate_prepared_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        request: GenerationModelRequest,
    ) -> OpenAIPostGenerationResult:
        """
        Выполняет заранее сформированный запрос.

        Перед обращением к модели проверяет,
        что запрос соответствует текущему TOP-3,
        модели и версии промпта.
        """

        expected_news_ids = (
            _validate_news_items(items)
        )

        expected_request = self.build_request(
            items
        )

        if request != expected_request:
            raise ValueError(
                "Подготовленный запрос не "
                "соответствует текущему TOP-3, "
                "модели или промпту."
            )

        model_response = (
            await self._client.create_response(
                request
            )
        )

        payload = _parse_response(
            model_response.output_text
        )

        _validate_response_items(
            expected_news_ids=(
                expected_news_ids
            ),
            payload=payload,
        )

        canonical_post_text = (
            build_top3_post_text(
                payload.items
            )
        )

        payload = payload.model_copy(
            update={
                "post_text": (
                    canonical_post_text
                ),
            }
        )

        return OpenAIPostGenerationResult(
            payload=payload,
            model_response=model_response,
        )

    async def generate_detailed(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
    ) -> OpenAIPostGenerationResult:
        """Генерирует пост с телеметрией."""

        request = self.build_request(
            items
        )

        return await self.generate_prepared_request(
            items,
            request,
        )

    async def generate(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
    ) -> str:
        """Возвращает только готовый текст."""

        result = await self.generate_detailed(
            items
        )

        return result.payload.post_text