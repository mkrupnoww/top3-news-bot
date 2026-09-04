from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
import re
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

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

OPENAI_POST_GENERATOR_VERSION = (
    "openai_telegram_post_generator_v7"
)

OPENAI_POST_PROMPT_VERSION = (
    "movie_news_telegram_post_prompt_v7"
)

OPENAI_POST_REVISION_PROMPT_VERSION = (
    "movie_news_telegram_post_revision_prompt_v5"
)

OPENAI_POST_TEXT_FORMAT = "markdown"

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
    official_trailer_url: str | None = None
    official_trailer_channel_name: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationModelRequest:
    """Точный запрос к модели генерации текста."""

    model: str
    instructions: str
    input_text: str
    allow_web_search: bool = False


@dataclass(frozen=True, slots=True)
class GenerationModelResponse:
    """Ответ модели вместе с телеметрией."""

    output_text: str
    usage: OpenAITokenUsage | None = None
    cost_estimate: OpenAICostEstimate | None = None
    web_search_used: bool = False
    web_search_call_count: int = 0
    web_source_urls: tuple[str, ...] = ()


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
        max_length=MAXIMUM_HEADLINE_LENGTH,
    )

    body: str = Field(
        min_length=1,
        max_length=MAXIMUM_BODY_LENGTH,
    )

    # Service metadata is attached by Python after the model response.
    # The model JSON schema itself does not need to return these fields.
    official_trailer_url: str | None = None
    official_trailer_channel_name: str | None = None

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
        max_length=MAXIMUM_POST_LENGTH,
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

        if (
            len(normalized_value)
            > MAXIMUM_POST_LENGTH
        ):
            raise ValueError(
                "post_text превышает допустимую "
                "длину выпуска: "
                f"{MAXIMUM_POST_LENGTH} символов."
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
6. Пиши естественно и информативно. Избегай
   обрубленных фраз без явного субъекта. Например,
   вместо «Также снялись Рассел Кроу и Шейлин Вудли»
   пиши «В фильме также снялись Рассел Кроу и
   Шейлин Вудли», если это подтверждено исходными
   данными.
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
17. Поля official_trailer_url и
    official_trailer_channel_name являются
    служебными подтверждёнными данными. Никогда
    не вставляй URL трейлера, Markdown-ссылку или
    отдельную строку «Официальный трейлер» в
    headline, body или черновой post_text.
18. Python-код сам детерминированно добавит
    подтверждённую ссылку после body нужной новости.
19. Не изменяй, не дополняй и не придумывай
    official_trailer_url или название канала.
20. Если official_trailer_url отсутствует,
    не придумывай URL и не пытайся восстановить
    его по названию фильма или сериала.
21. Наличие служебных trailer-полей не разрешает
    добавлять в body новые факты, которых нет в
    title и summary.
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
31. Целевой объём итогового post_text —
    __TARGET_POST_LENGTH_MIN__–__TARGET_POST_LENGTH_MAX__
    символов с пробелами, включая заголовок выпуска,
    разделители, номера новостей и строку подписки,
    которые добавит Python-код. Если хотя бы у одной
    новости передан official_trailer_url, оставляй
    примерно 120 символов запаса: Python добавит
    отдельную Markdown-строку официального трейлера.
32. Для каждого headline ориентируйся на
    __TARGET_HEADLINE_LENGTH_MIN__–__TARGET_HEADLINE_LENGTH_MAX__
    символов. Абсолютный максимум headline —
    __MAXIMUM_HEADLINE_LENGTH__ символов.
33. Для каждого body ориентируйся на
    __TARGET_BODY_LENGTH_MIN__–__TARGET_BODY_LENGTH_MAX__
    символов. Абсолютный максимум body —
    __MAXIMUM_BODY_LENGTH__ символов.
34. Не сокращай текст сильнее необходимого. Если
    входных фактов достаточно, каждая новость должна
    содержать не только само событие, но и важный
    контекст, объясняющий читателю, что произошло.
35. Не увеличивай объём искусственно: не добавляй
    воду, повторы или неподтверждённые детали ради
    достижения целевой длины. Если исходных фактов
    объективно мало, headline или body могут быть
    короче целевого диапазона.
36. Абсолютный максимум итогового post_text —
    __MAXIMUM_POST_LENGTH__ символов.

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
""".replace(
    "__TARGET_POST_LENGTH_MIN__",
    str(TARGET_POST_LENGTH_MIN),
).replace(
    "__TARGET_POST_LENGTH_MAX__",
    str(TARGET_POST_LENGTH_MAX),
).replace(
    "__TARGET_HEADLINE_LENGTH_MIN__",
    str(TARGET_HEADLINE_LENGTH_MIN),
).replace(
    "__TARGET_HEADLINE_LENGTH_MAX__",
    str(TARGET_HEADLINE_LENGTH_MAX),
).replace(
    "__MAXIMUM_HEADLINE_LENGTH__",
    str(MAXIMUM_HEADLINE_LENGTH),
).replace(
    "__TARGET_BODY_LENGTH_MIN__",
    str(TARGET_BODY_LENGTH_MIN),
).replace(
    "__TARGET_BODY_LENGTH_MAX__",
    str(TARGET_BODY_LENGTH_MAX),
).replace(
    "__MAXIMUM_BODY_LENGTH__",
    str(MAXIMUM_BODY_LENGTH),
).replace(
    "__MAXIMUM_POST_LENGTH__",
    str(MAXIMUM_POST_LENGTH),
).strip()


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

14. Если во входных news переданы
    official_trailer_url / official_trailer_channel_name,
    не вставляй их в headline, body или post_text.
    Python сам восстановит каноническую ссылку после
    редакционной ревизии. Даже если редактор просит
    «добавить ссылку на трейлер», отредактируй только
    содержательный текст новости — ссылку добавит код.

15. Следи за естественной русской связностью. Не
    оставляй контекстно обрубленные конструкции вроде
    «Также снялись…» без понятного субъекта; используй
    «В фильме также снялись…» или другую естественную
    конструкцию, подтверждённую исходными данными.

16. Не сокращай source_post_text только ради
    лаконичности. Если исходный пост находится
    примерно в диапазоне 850–950 символов,
    по возможности сохраняй сопоставимый объём,
    если замечания редактора не требуют иного. Если
    передан official_trailer_url, оставляй примерно
    120 символов запаса для программно добавляемой
    Markdown-строки официального трейлера.
""".strip()



SELF_REVIEW_SYSTEM_INSTRUCTIONS = """
Ты выполняешь второй редакционный проход по уже
созданному русскоязычному Telegram-посту с TOP-3
новостей о кино и сериалах.

Твоя задача — самостоятельно оценить готовый пост
как внимательный редактор и при необходимости
точечно улучшить его. Не жди замечаний человека.

Проверь весь source_post_text и для каждой новости
оцени:

- понятен ли текст обычному читателю без знания
  исходной статьи;
- понятно ли, кто такие упомянутые люди и какова
  их роль, если это существенно для смысла;
- естественно ли звучат формулировки по-русски;
- есть ли у фраз явный контекст и субъект; исправляй
  обрубленные конструкции вроде «Также снялись…» на
  естественные, например «В фильме также снялись…»;
- нет ли двусмысленных, обрубленных или слишком
  буквальных переводов;
- корректно ли переданы имена людей кириллицей;
- понятны ли даты и относительные указания времени;
- не выглядит ли рекорд, сумма, дата, должность,
  статус сделки, цитата или другой существенный факт
  сомнительным либо недостаточно объяснённым;
- нет ли внутренних противоречий между headline,
  body и исходными данными новости.

У тебя доступен web_search. Сам решай, нужен ли он.
Если поиск нужен, старайся ограничиться одним или
двумя целевыми поисковыми вызовами.

Используй web_search, когда имеющихся входных данных
недостаточно для важного уточнения или проверки,
которая заметно улучшает понятность или фактическую
надёжность поста. Например, поиск уместен при
неполном имени или неясной должности человека,
неоднозначной дате, сомнительном рекорде, точной
сумме, статусе сделки или другом существенном факте.

Не используй поиск только ради обогащения поста
новыми интересными подробностями. Найденные в сети
сведения разрешено добавлять лишь тогда, когда они
непосредственно нужны для устранения обнаруженного
недостатка текущего текста.

При проверке предпочитай первоисточники, официальные
сайты и авторитетные профильные СМИ. Учитывай
переданный source_url исходной новости. Если
источники противоречат друг другу или уверенно
установить факт не удалось, не выдумывай ответ:
используй более безопасную формулировку либо убери
сомнительную деталь.

Редактируй консервативно:

1. source_post_text — текущая версия поста и основа
   результата.
2. Если фрагмент уже хороший и понятный, сохрани его
   максимально близко к исходному.
3. Не переписывай весь пост ради стилистического
   разнообразия.
4. Исправляй только реальные недостатки, которые
   обнаружил при втором проходе.
5. Сохраняй порядок трёх новостей и их news_id.
6. Имена людей передавай кириллицей: используй
   общепринятую русскую форму или нейтральную
   транслитерацию.
7. Названия компаний и брендов можно сохранять
   латиницей.
8. Названия фильмов и сериалов сохраняй в исходном
   написании, если нет надёжно подтверждённого
   русского названия.
9. Не добавляй в headline/body ссылки на веб-источники,
   поисковые цитаты, список источников или объяснение
   своей проверки. Если переданы official_trailer_url /
   official_trailer_channel_name, не вставляй trailer URL
   вручную: Python сам добавит каноническую строку после
   body и сохранит её при revisions.
10. Не добавляй хештеги, Markdown-заголовки, HTML
    или служебные комментарии.
11. headline возвращай без внешних Markdown-маркеров.
12. body может содержать только обычный Markdown,
    разрешённый основными правилами поста.
13. В JSON обязательно верни headline и body для
    всех трёх новостей, даже если часть из них не
    потребовала изменений.
14. Поле post_text также обязательно верни, однако
    окончательный post_text будет заново собран
    Python-кодом из массива items.
15. Верни только JSON-объект без Markdown-обёртки.
16. Не сокращай хороший и понятный source_post_text
    только ради лаконичности. Второй проход должен
    исправлять реальные недостатки, а не превращать
    содержательный текст в краткую выжимку.
17. Ориентир для итогового post_text —
    __TARGET_POST_LENGTH_MIN__–__TARGET_POST_LENGTH_MAX__
    символов с пробелами. Если исходный пост находится
    в этом диапазоне, по возможности сохраняй близкий
    объём. Если хотя бы у одной новости передан
    official_trailer_url, оставляй примерно 120
    символов запаса для Markdown-строки, которую
    программно добавит Python.
18. Для каждого headline ориентируйся на
    __TARGET_HEADLINE_LENGTH_MIN__–__TARGET_HEADLINE_LENGTH_MAX__
    символов, для body —
    __TARGET_BODY_LENGTH_MIN__–__TARGET_BODY_LENGTH_MAX__
    символов.
19. Если source_post_text заметно короче целевого
    диапазона, а title и summary содержат полезные
    подтверждённые детали, используй их, чтобы сделать
    новости содержательнее.
20. Не добавляй воду, повторы или новые факты только
    для увеличения длины. При недостатке исходной
    фактуры допустим более короткий текст.
21. Headline не должен превышать
    __MAXIMUM_HEADLINE_LENGTH__ символов, body —
    __MAXIMUM_BODY_LENGTH__ символов, а итоговый
    post_text — __MAXIMUM_POST_LENGTH__ символов.
""".replace(
    "__TARGET_POST_LENGTH_MIN__",
    str(TARGET_POST_LENGTH_MIN),
).replace(
    "__TARGET_POST_LENGTH_MAX__",
    str(TARGET_POST_LENGTH_MAX),
).replace(
    "__TARGET_HEADLINE_LENGTH_MIN__",
    str(TARGET_HEADLINE_LENGTH_MIN),
).replace(
    "__TARGET_HEADLINE_LENGTH_MAX__",
    str(TARGET_HEADLINE_LENGTH_MAX),
).replace(
    "__MAXIMUM_HEADLINE_LENGTH__",
    str(MAXIMUM_HEADLINE_LENGTH),
).replace(
    "__TARGET_BODY_LENGTH_MIN__",
    str(TARGET_BODY_LENGTH_MIN),
).replace(
    "__TARGET_BODY_LENGTH_MAX__",
    str(TARGET_BODY_LENGTH_MAX),
).replace(
    "__MAXIMUM_BODY_LENGTH__",
    str(MAXIMUM_BODY_LENGTH),
).replace(
    "__MAXIMUM_POST_LENGTH__",
    str(MAXIMUM_POST_LENGTH),
).strip()


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
                **(
                    {
                        "official_trailer_url": (
                            item.official_trailer_url.strip()
                        ),
                        **(
                            {
                                "official_trailer_channel_name": (
                                    item.official_trailer_channel_name.strip()
                                )
                            }
                            if item.official_trailer_channel_name is not None
                            else {}
                        ),
                    }
                    if item.official_trailer_url is not None
                    else {}
                ),
            }
            for item in items
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )



def _build_self_review_input_text(
    items: tuple[
        GenerationNewsItem,
        ...,
    ],
    *,
    source_post_text: str,
) -> str:
    """Формирует JSON для второго прохода редактора."""

    _validate_news_items(items)

    normalized_source_post_text = (
        _normalize_required_text(
            source_post_text,
            field_name="source_post_text",
        )
    )

    payload = {
        "task": (
            "self_review_russian_telegram_"
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
        "news": [
            {
                "position": item.position,
                "news_id": item.news_id,
                "title": item.title.strip(),
                "summary": item.summary.strip(),
                "source_name": (
                    item.source_name.strip()
                ),
                "source_url": (
                    item.source_url.strip()
                ),
                "source_published_at": (
                    item.source_published_at
                    .isoformat()
                ),
                **(
                    {
                        "official_trailer_url": (
                            item.official_trailer_url.strip()
                        ),
                        **(
                            {
                                "official_trailer_channel_name": (
                                    item.official_trailer_channel_name.strip()
                                )
                            }
                            if item.official_trailer_channel_name is not None
                            else {}
                        ),
                    }
                    if (
                        item.official_trailer_url
                        is not None
                    )
                    else {}
                ),
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

        if item.official_trailer_url is not None:
            official_trailer_url = (
                _normalize_required_text(
                    item.official_trailer_url,
                    field_name=(
                        "official_trailer_url "
                        f"news_id={item.news_id}"
                    ),
                )
            )

            parsed_trailer_url = urlsplit(
                official_trailer_url
            )

            if (
                parsed_trailer_url.scheme.casefold()
                not in {"http", "https"}
                or not parsed_trailer_url.netloc
            ):
                raise ValueError(
                    "official_trailer_url должен "
                    "быть абсолютным HTTP или HTTPS "
                    "URL: "
                    f"news_id={item.news_id}"
                )

        if item.official_trailer_channel_name is not None:
            if item.official_trailer_url is None:
                raise ValueError(
                    "official_trailer_channel_name нельзя "
                    "задавать без official_trailer_url: "
                    f"news_id={item.news_id}"
                )

            _normalize_required_text(
                item.official_trailer_channel_name,
                field_name=(
                    "official_trailer_channel_name "
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
                **(
                    {
                        "official_trailer_url": (
                            item.official_trailer_url.strip()
                        ),
                        **(
                            {
                                "official_trailer_channel_name": (
                                    item.official_trailer_channel_name.strip()
                                )
                            }
                            if item.official_trailer_channel_name is not None
                            else {}
                        ),
                    }
                    if (
                        item.official_trailer_url
                        is not None
                    )
                    else {}
                ),
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


def _normalize_trailer_channel_label(
    value: str | None,
) -> str | None:
    """Возвращает безопасную подпись официального YouTube-канала."""

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    normalized = " ".join(value.strip().split())

    if not normalized or len(normalized) > 80:
        return None

    if "http://" in normalized.casefold() or "https://" in normalized.casefold():
        return None

    allowed_punctuation = frozenset(".&+'-")
    for character in normalized:
        if (
            character.isalnum()
            or character.isspace()
            or character in allowed_punctuation
        ):
            continue
        return None

    return normalized


def build_official_trailer_markdown(
    url: str,
    channel_name: str | None,
) -> str:
    """Строит детерминированную Markdown-ссылку на официальный трейлер."""

    normalized_url = _normalize_required_text(
        url,
        field_name="official_trailer_url",
    )

    parsed = urlsplit(normalized_url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "official_trailer_url должен быть абсолютным HTTP/HTTPS URL."
        )

    channel_label = _normalize_trailer_channel_label(channel_name)
    if channel_label is None:
        label = "▶️ Официальный трейлер"
    else:
        label = f"▶️ Официальный трейлер {channel_label}"

    return f"[{label}]({normalized_url})"


def _attach_trailer_metadata_to_payload(
    payload: OpenAIGeneratedPostPayload,
    source_items: tuple[GenerationNewsItem, ...],
) -> OpenAIGeneratedPostPayload:
    """Прикрепляет verified trailer metadata после model response."""

    source_by_news_id = {
        item.news_id: item
        for item in source_items
    }

    updated_items: list[OpenAIGeneratedNewsPayload] = []

    for generated_item in payload.items:
        source_item = source_by_news_id[generated_item.news_id]
        updated_items.append(
            generated_item.model_copy(
                update={
                    "official_trailer_url": source_item.official_trailer_url,
                    "official_trailer_channel_name": (
                        source_item.official_trailer_channel_name
                    ),
                }
            )
        )

    return payload.model_copy(update={"items": updated_items})


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

        section = (
            f"{marker} **{headline}**"
            f"\n\n{body}"
        )

        if item.official_trailer_url is not None:
            section += (
                "\n\n"
                + build_official_trailer_markdown(
                    item.official_trailer_url,
                    item.official_trailer_channel_name,
                )
            )

        news_sections.append(section)

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

    def build_self_review_request(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        *,
        source_post_text: str,
    ) -> GenerationModelRequest:
        """Формирует запрос второго редакционного прохода."""

        _validate_news_items(items)

        return GenerationModelRequest(
            model=self._metadata.model_name,
            instructions=(
                SELF_REVIEW_SYSTEM_INSTRUCTIONS
            ),
            input_text=(
                _build_self_review_input_text(
                    items,
                    source_post_text=(
                        source_post_text
                    ),
                )
            ),
            allow_web_search=True,
        )

    async def generate_self_review_detailed(
        self,
        items: tuple[
            GenerationNewsItem,
            ...,
        ],
        *,
        source_post_text: str,
    ) -> OpenAIPostGenerationResult:
        """Выполняет второй проход с optional web search."""

        expected_news_ids = (
            _validate_news_items(items)
        )

        request = self.build_self_review_request(
            items,
            source_post_text=source_post_text,
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

        payload = _attach_trailer_metadata_to_payload(
            payload,
            items,
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

        payload = _attach_trailer_metadata_to_payload(
            payload,
            items,
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

        payload = _attach_trailer_metadata_to_payload(
            payload,
            items,
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