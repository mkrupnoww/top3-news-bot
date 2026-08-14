from dataclasses import dataclass
import json
import re
from typing import Literal, Protocol, runtime_checkable


OPENAI_IMAGE_GENERATOR_VERSION = (
    "openai_movie_news_image_generator_v2"
)

OPENAI_IMAGE_PROMPT_VERSION = (
    "movie_news_image_v2"
)

OPENAI_IMAGE_FALLBACK_PROMPT_VERSION = (
    "movie_news_image_moderation_fallback_v1"
)

DEFAULT_IMAGE_QUALITY = "medium"
DEFAULT_IMAGE_OUTPUT_FORMAT = "png"
DEFAULT_IMAGE_BACKGROUND = "opaque"
DEFAULT_IMAGE_MODERATION = "auto"
DEFAULT_IMAGE_COUNT = 1

_IMAGE_SIZE_PATTERN = re.compile(
    r"^(?P<width>[1-9][0-9]*)x(?P<height>[1-9][0-9]*)$"
)


ImageQuality = Literal[
    "low",
    "medium",
    "high",
]

ImageOutputFormat = Literal[
    "png",
]

ImageBackground = Literal[
    "opaque",
]

ImageModeration = Literal[
    "auto",
]


@dataclass(frozen=True, slots=True)
class ImageGenerationNewsItem:
    """Одна новость для генерации общей иллюстрации."""

    position: int
    news_id: int
    title: str
    summary: str


@dataclass(frozen=True, slots=True)
class ImageModelRequest:
    """Точный запрос к модели генерации изображения."""

    model: str
    prompt: str
    size: str
    quality: ImageQuality
    output_format: ImageOutputFormat
    background: ImageBackground
    moderation: ImageModeration
    n: int


@dataclass(frozen=True, slots=True)
class OpenAIImageUsage:
    """Фактическое потребление токенов Image API."""

    input_tokens: int
    input_text_tokens: int
    input_image_tokens: int
    output_tokens: int
    output_text_tokens: int
    output_image_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        """Проверяет согласованность счётчиков."""

        values = {
            "input_tokens": self.input_tokens,
            "input_text_tokens": (
                self.input_text_tokens
            ),
            "input_image_tokens": (
                self.input_image_tokens
            ),
            "output_tokens": self.output_tokens,
            "output_text_tokens": (
                self.output_text_tokens
            ),
            "output_image_tokens": (
                self.output_image_tokens
            ),
            "total_tokens": self.total_tokens,
        }

        for field_name, value in values.items():
            if isinstance(value, bool):
                raise TypeError(
                    f"{field_name} не может быть bool."
                )

            if not isinstance(value, int):
                raise TypeError(
                    f"{field_name} должен быть int."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} не может "
                    "быть отрицательным."
                )

        if (
            self.input_text_tokens
            + self.input_image_tokens
            != self.input_tokens
        ):
            raise ValueError(
                "input_tokens не совпадает с "
                "input_text_tokens + "
                "input_image_tokens."
            )

        if (
            self.output_text_tokens
            + self.output_image_tokens
            != self.output_tokens
        ):
            raise ValueError(
                "output_tokens не совпадает с "
                "output_text_tokens + "
                "output_image_tokens."
            )

        if (
            self.input_tokens
            + self.output_tokens
            != self.total_tokens
        ):
            raise ValueError(
                "total_tokens не совпадает с "
                "input_tokens + output_tokens."
            )


@dataclass(frozen=True, slots=True)
class ImageModelResponse:
    """Ответ Image API вместе с телеметрией."""

    image_bytes: bytes
    created: int
    output_format: str | None
    quality: str | None
    size: str | None
    background: str | None
    usage: OpenAIImageUsage | None = None
    revised_prompt: str | None = None

    def __post_init__(self) -> None:
        """Проверяет базовую целостность ответа."""

        if not isinstance(
            self.image_bytes,
            bytes,
        ):
            raise TypeError(
                "image_bytes должен быть bytes."
            )

        if not self.image_bytes:
            raise ValueError(
                "image_bytes не может быть пустым."
            )

        if isinstance(self.created, bool):
            raise TypeError(
                "created не может быть bool."
            )

        if not isinstance(self.created, int):
            raise TypeError(
                "created должен быть int."
            )

        if self.created < 0:
            raise ValueError(
                "created не может быть отрицательным."
            )


@dataclass(frozen=True, slots=True)
class OpenAIImageGeneratorMetadata:
    """Версии компонентов генератора изображения."""

    generator_name: str
    generator_version: str
    prompt_version: str
    model_name: str


@dataclass(frozen=True, slots=True)
class OpenAIImageGenerationResult:
    """Сгенерированное изображение и данные API-запроса."""

    model_request: ImageModelRequest
    model_response: ImageModelResponse


@runtime_checkable
class ImageGenerationClient(
    Protocol
):
    """Транспортный интерфейс генерации изображения."""

    async def create_image(
        self,
        request: ImageModelRequest,
    ) -> ImageModelResponse:
        """Выполняет один запрос к Image API."""

        ...


IMAGE_PROMPT_INSTRUCTIONS = """
Создай одну цельную вертикальную редакционную иллюстрацию для публикации
ежедневного TOP-3 киноновостей в Telegram.

Это должна быть не афиша и не рекламный постер, а качественная современная
кинематографичная редакционная иллюстрация, визуально передающая смысл трёх
отдельных новостей.

============================================================
1. ОБЩАЯ КОМПОЗИЦИЯ
============================================================

Изображение должно состоять ровно из ТРЁХ основных горизонтальных зон,
расположенных одна под другой.

Верхняя зона — новость №1.
Средняя зона — новость №2.
Нижняя зона — новость №3.

Все три основные зоны должны быть примерно одинаковыми по высоте.

Между зонами должны быть хорошо различимые аккуратные горизонтальные
разделители.

Не добавляй отдельный заголовок над изображением.
Не добавляй надпись TOP-3.
Не добавляй номера 1, 2, 3.

Каждая основная зона относится только к своей новости.

Не смешивай персонажей, предметы, места действия или смысловые элементы
разных новостей между зонами.

Изображение должно сразу визуально читаться как три разные киноновости,
но при этом выглядеть как одна профессионально скомпонованная работа.

============================================================
2. КОМПОЗИЦИЯ ВНУТРИ КАЖДОЙ НОВОСТИ
============================================================

Внутри отдельной горизонтальной зоны разрешается использовать:

- одну цельную сцену;
- несколько визуально связанных под-сцен;
- редакционную композицию из нескольких связанных элементов,

если это помогает понятнее и выразительнее передать смысл конкретной новости.

Не дроби блок на множество мелких случайных фрагментов.

Внутренняя композиция должна оставаться визуально понятной даже при просмотре
изображения на экране смартфона.

Главные объекты новости должны быть достаточно крупными и хорошо читаемыми.

============================================================
3. ВИЗУАЛЬНЫЙ СТИЛЬ
============================================================

Стиль — современная кинематографичная редакционная иллюстрация.

Предпочтительно:

- реалистичное или близкое к фотореалистичному изображение;
- выразительный кинематографический свет;
- хорошая глубина кадра;
- естественные лица и анатомия людей;
- качественные материалы, одежда, интерьер и окружение;
- визуальная насыщенность без перегруженности;
- аккуратная композиция;
- профессиональный уровень журнальной или медийной иллюстрации.

Все три основные зоны должны иметь согласованную цветокоррекцию,
контраст и общее визуальное качество.

При этом атмосфера каждой зоны может различаться в зависимости от новости:
драма, триллер, фестиваль, производство фильма, кинотеатр, корпоративная
сделка, приключение, историческая тема и т. д.

Не делай все три зоны искусственно одинаковыми.

============================================================
4. ТОЧНОСТЬ ПО ОТНОШЕНИЮ К НОВОСТИ
============================================================

Используй только смысл и факты, которые следуют из переданных заголовка
и описания новости.

Не придумывай:

- события, которых не было;
- подтверждённые сделки, если сообщается только о переговорах;
- подтверждённый кастинг, если актёр лишь рассматривается на роль;
- отношения между людьми, которых нет в новости;
- вымышленные награды;
- вымышленные кассовые результаты;
- вымышленные даты;
- вымышленные цитаты;
- вымышленные названия компаний или фильмов;
- дополнительные сюжетные обстоятельства.

Если формулировка новости содержит неопределённость, например
«ведёт переговоры», «может присоединиться», «рассматривается»,
«circling», «eyed», «reportedly» или «expected», визуализация также
не должна создавать впечатление уже окончательно подтверждённого события.

Если изображаются известные актёры, режиссёры или другие публичные люди,
они должны выглядеть естественно и без карикатурного искажения внешности.

============================================================
5. ДЕЛИКАТНЫЕ И ТРАГИЧЕСКИЕ НОВОСТИ
============================================================

Если новость касается смерти человека, используй уважительную,
сдержанную редакционную подачу.

Можно использовать:

- достойный портрет;
- атмосферу, связанную с творчеством человека;
- кинотеатр;
- съёмочную площадку;
- элементы кинопроизводства;
- символическую мемориальную атмосферу.

Не изображай:

- тело умершего;
- момент смерти;
- страдания;
- болезнь;
- травмы;
- кровь;
- похороны, если они прямо не являются предметом новости;
- чрезмерно мрачные или сенсационные сцены.

============================================================
6. КИНОПРОКАТ, БИЗНЕС И КОРПОРАТИВНЫЕ НОВОСТИ
============================================================

Для новостей о кассовых сборах, студиях, слияниях, покупке компаний,
дистрибуции или других деловых событиях используй понятные киноиндустриальные
визуальные метафоры.

Допустимы, например:

- кинотеатр;
- киноплёнка;
- съёмочное оборудование;
- студийные здания;
- деловая встреча;
- документы;
- рукопожатие;
- кинопроизводство;
- визуальное противопоставление двух фильмов;
- индустриальная или корпоративная среда.

Не используй бессмысленные изображения денег, золотых монет или финансовых
графиков только для обозначения бизнеса.

Не изображай вымышленные цифры, проценты, рейтинги или кассовые суммы.

============================================================
7. ТЕКСТ ВНУТРИ ИЗОБРАЖЕНИЯ
============================================================

По умолчанию НЕ добавляй текст.

Не добавляй:

- подписи под каждым блоком;
- номера новостей;
- заголовки новостей;
- поясняющие предложения;
- длинные надписи;
- водяные знаки;
- технический текст;
- случайные вывески с нечитаемыми буквами.

Однако текст разрешён, если без него сложно или невозможно однозначно
передать важный объект конкретной новости.

Например, для новости о слиянии двух кинокомпаний допускается написать
названия этих компаний.

В таком случае:

- используй только название, реально присутствующее в данных новости;
- используй обычную нейтральную типографику;
- текст должен быть хорошо читаемым;
- не превращай его в рекламный элемент;
- не имитируй официальный фирменный стиль.

============================================================
8. ЛОГОТИПЫ И БРЕНДИНГ
============================================================

Не используй официальные логотипы компаний, киностудий, телеканалов,
стриминговых сервисов или других брендов.

Если название компании необходимо для понимания новости,
покажи его обычным текстом.

Например, PARAMOUNT, WARNER BROS, A24, NETFLIX, SONY или UNIVERSAL
допускаются как нейтральные текстовые названия, если они непосредственно
относятся к конкретной новости.

Не воспроизводи фирменную эмблему, логотип, товарный знак или точное
брендовое графическое оформление.

============================================================
9. ФИЛЬМЫ И ВИЗУАЛЬНАЯ АССОЦИАЦИЯ
============================================================

Если новость относится к конкретному фильму или франшизе, передай её
через узнаваемую кинематографическую атмосферу, персонажей, жанровые мотивы,
место действия или связанные с кинопроизводством элементы.

Не копируй существующий официальный постер буквально.

Не превращай отдельную зону в точную репродукцию рекламного постера фильма.

Если название фильма критически необходимо для понимания новости,
его допускается показать обычным аккуратным текстом по тем же правилам,
что и названия компаний.

============================================================
10. ТЕХНИЧЕСКОЕ КАЧЕСТВО
============================================================

Особенно внимательно следи за:

- правильным количеством рук и пальцев;
- естественными кистями рук;
- правильным расположением конечностей;
- отсутствием лишних рук, ног или лиц;
- отсутствием сросшихся людей;
- естественным положением тела;
- корректной перспективой;
- корректным расположением экранов ноутбуков и смартфонов;
- тем, чтобы экран находился с правильной стороны устройства;
- естественными предметами в руках;
- отсутствием случайных повторов одного и того же человека;
- отсутствием бессмысленного или повреждённого текста;
- правильными пропорциями лица и тела.

Не допускай визуальных артефактов, особенно в сценах с большим количеством
людей.

============================================================
11. СООТВЕТСТВИЕ ПРАВИЛАМ И БЕЗОПАСНОСТЬ
============================================================

Создавай изображение в соответствии с действующими правилами OpenAI
для генерации изображений и применимыми требованиями безопасности.

Не создавай контент, нарушающий права третьих лиц.

Не копируй существующие официальные постеры, рекламные материалы,
кадры, фотографии, иллюстрации или другие защищённые изображения
буквально или один в один.

Не воспроизводи официальные логотипы и фирменное графическое оформление.

Создавай оригинальную редакционную визуальную композицию,
основанную на смысле переданных киноновостей.

Если какая-либо конкретная деталь предполагаемой визуализации может
противоречить правилам OpenAI, требованиям безопасности или ограничениям
на генерацию изображений, не отклоняй всю композицию из-за этой детали.

Вместо этого замени только такую деталь на допустимый визуальный
эквивалент, максимально сохранив смысл новости, композицию,
кинематографическую атмосферу и редакционный замысел.

Не пытайся обходить ограничения или системы безопасности OpenAI.

============================================================
12. ФИНАЛЬНАЯ ПРОВЕРКА КОМПОЗИЦИИ
============================================================

Перед созданием финального изображения убедись, что:

1. В изображении ровно три основные горизонтальные зоны.
2. Они расположены сверху вниз в порядке новостей №1, №2, №3.
3. Каждая зона визуализирует только свою новость.
4. Между зонами есть чёткие аккуратные горизонтальные разделители.
5. Нет общего заголовка TOP-3.
6. Нет нумерации.
7. Нет ненужных подписей.
8. Официальные логотипы не используются.
9. Текст присутствует только там, где действительно нужен для понимания новости.
10. Все видимые названия взяты из исходной информации и написаны корректно.
11. Люди, руки, лица, техника и предметы выглядят естественно.
12. Три зоны имеют согласованную общую цветокоррекцию.
13. Итог выглядит как единая профессиональная киноновостная иллюстрация,
    а не как три случайных изображения, поставленных друг над другом.
""".strip()


MODERATION_SAFE_EDITORIAL_FALLBACK_INSTRUCTIONS = """
============================================================
13. БЕЗОПАСНЫЙ РЕДАКЦИОННЫЙ FALLBACK ПОСЛЕ MODERATION_BLOCKED
============================================================

Этот режим включён только потому, что предыдущая обычная генерация
этого же набора киноновостей была отклонена системой безопасности
на стадии готового изображения.

Не меняй факты новостей и не перестраивай без необходимости те зоны,
которые можно визуализировать обычным допустимым способом.

Определи только ту или те зоны, где буквальное изображение конкретной
франшизы, узнаваемого защищённого персонажа, официального визуального
образа, фирменной символики или узнаваемой внешности публичного человека
может привести к повторному отклонению готового изображения.

Для таких зон используй безопасную оригинальную редакционную замену,
которая передаёт именно НОВОСТНОЙ СМЫСЛ, а не буквальное воспроизведение
персонажа, актёра, постера или брендового оформления.

Например, если новость посвящена кассовому успеху известного
супергеройского фильма, допустимо передать её через:

- большой современный кинотеатр и зрителей;
- атмосферу крупного кинорелиза;
- динамичный городской фон;
- кинопроизводство и экран кинотеатра без копирования кадров;
- визуальное ощущение рекорда, масштаба, скорости и успеха;
- жанровые цветовые и световые акценты без точного костюма,
  эмблемы, маски, логотипа или другого фирменного образа.

В fallback-зоне не изображай узнаваемую внешность конкретного реального
актёра и не воспроизводи буквальный узнаваемый образ защищённого
персонажа или его точный костюм.

Это ограничение относится ТОЛЬКО к зоне, которую необходимо безопасно
переформулировать после moderation_blocked. Оно не является общим
запретом на изображение актёров, режиссёров или персонажей в обычных
генерациях проекта.

Сохрани:

- позицию новости в соответствующей горизонтальной зоне;
- смысл и факты новости;
- кинематографичность;
- визуальную выразительность;
- общий стиль изображения;
- остальные безопасные зоны максимально близкими к исходному замыслу.

Не добавляй текст о модерации, безопасности, авторских правах,
fallback-режиме или внутренних правилах на само изображение.
""".strip()


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный текст."""

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


def _normalize_positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительное целое число."""

    if isinstance(value, bool):
        raise TypeError(
            f"{field_name} не может быть bool."
        )

    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} должен быть int."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} должен быть больше нуля."
        )

    return value


def _normalize_image_size(
    value: str,
) -> str:
    """Проверяет размер и проектное соотношение 2:3."""

    normalized_value = _normalize_required_text(
        value,
        field_name="size",
    )

    match = _IMAGE_SIZE_PATTERN.fullmatch(
        normalized_value
    )

    if match is None:
        raise ValueError(
            "size должен иметь формат WIDTHxHEIGHT."
        )

    width = int(match.group("width"))
    height = int(match.group("height"))

    if width * 3 != height * 2:
        raise ValueError(
            "Для итоговой иллюстрации требуется "
            "соотношение сторон 2:3."
        )

    return normalized_value


def _normalize_news_items(
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
) -> tuple[
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
    ImageGenerationNewsItem,
]:
    """Проверяет состав и порядок TOP-3."""

    if not isinstance(items, tuple):
        raise TypeError(
            "items должен быть tuple."
        )

    if len(items) != 3:
        raise ValueError(
            "Для генерации изображения требуется "
            "ровно три новости."
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
        _normalize_positive_integer(
            item.news_id,
            field_name=(
                f"news_id position={item.position}"
            ),
        )
        for item in items
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Все три news_id должны быть уникальными."
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

    return (
        items[0],
        items[1],
        items[2],
    )


def _normalize_editorial_revision(
    *,
    editorial_comment: str | None,
    issues: tuple[str, ...],
) -> tuple[str | None, tuple[str, ...]]:
    """Проверяет необязательные правки изображения."""

    if not isinstance(issues, tuple):
        raise TypeError(
            "issues должен быть tuple."
        )

    normalized_comment: str | None

    if editorial_comment is None:
        normalized_comment = None
    else:
        normalized_comment = (
            _normalize_required_text(
                editorial_comment,
                field_name="editorial_comment",
            )
        )

    normalized_issues = tuple(
        _normalize_required_text(
            issue,
            field_name=f"issues[{index}]",
        )
        for index, issue in enumerate(
            issues,
            start=1,
        )
    )

    if (
        normalized_comment is None
        and normalized_issues
    ):
        raise ValueError(
            "issues требуют editorial_comment."
        )

    if (
        normalized_comment is not None
        and not normalized_issues
    ):
        raise ValueError(
            "editorial_comment требует непустой issues."
        )

    return (
        normalized_comment,
        normalized_issues,
    )


def _build_news_payload(
    items: tuple[
        ImageGenerationNewsItem,
        ImageGenerationNewsItem,
        ImageGenerationNewsItem,
    ],
) -> list[dict[str, object]]:
    """Формирует фактический TOP-3 для промпта."""

    return [
        {
            "position": item.position,
            "news_id": item.news_id,
            "title": item.title.strip(),
            "summary": item.summary.strip(),
        }
        for item in items
    ]


def build_image_prompt(
    *,
    items: tuple[
        ImageGenerationNewsItem,
        ...,
    ],
    editorial_comment: str | None = None,
    issues: tuple[str, ...] = (),
    moderation_safe_editorial_fallback: bool = False,
) -> str:
    """Формирует единый русский промпт изображения."""

    normalized_items = _normalize_news_items(
        items
    )

    (
        normalized_editorial_comment,
        normalized_issues,
    ) = _normalize_editorial_revision(
        editorial_comment=editorial_comment,
        issues=issues,
    )

    if not isinstance(
        moderation_safe_editorial_fallback,
        bool,
    ):
        raise TypeError(
            "moderation_safe_editorial_fallback "
            "должен быть bool."
        )

    payload: dict[str, object] = {
        "top3": _build_news_payload(
            normalized_items
        ),
    }

    fallback_instructions = ""

    if moderation_safe_editorial_fallback:
        payload["moderation_safe_editorial_fallback"] = {
            "enabled": True,
            "reason": "retry_after_output_moderation_blocked",
        }

        fallback_instructions = (
            "\n\n"
            + MODERATION_SAFE_EDITORIAL_FALLBACK_INSTRUCTIONS
        )

    revision_instructions = ""

    if normalized_editorial_comment is not None:
        payload["editorial_revision"] = {
            "comment": normalized_editorial_comment,
            "issues": list(normalized_issues),
        }

        revision_instructions = """

============================================================
14. РЕДАКЦИОННЫЕ ПРАВКИ
============================================================

Во входных данных присутствует объект editorial_revision.
Исправь перечисленные замечания при новой генерации изображения.

Редакционные правки уточняют визуальную реализацию, но не разрешают
придумывать новые факты, которых нет в заголовке и описании соответствующей
новости.
""".rstrip()

    input_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return (
        IMAGE_PROMPT_INSTRUCTIONS
        + fallback_instructions
        + revision_instructions
        + "\n\n"
        + "============================================================\n"
        + "ВХОДНЫЕ ДАННЫЕ\n"
        + "============================================================\n\n"
        + input_json
    )


class OpenAIMovieNewsImageGenerator:
    """Строит запрос и запускает генерацию общей картинки."""

    def __init__(
        self,
        *,
        client: ImageGenerationClient,
        model_name: str,
        size: str,
        quality: ImageQuality = (
            DEFAULT_IMAGE_QUALITY
        ),
    ) -> None:
        if not isinstance(
            client,
            ImageGenerationClient,
        ):
            raise TypeError(
                "client не соответствует "
                "интерфейсу ImageGenerationClient."
            )

        self._client = client
        self._model_name = _normalize_required_text(
            model_name,
            field_name="model_name",
        )
        self._size = _normalize_image_size(
            size
        )

        if quality not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "quality должен быть low, medium "
                "или high."
            )

        self._quality: ImageQuality = quality
        self._moderation_safe_editorial_fallback = False

    def set_moderation_safe_editorial_fallback(
        self,
        enabled: bool,
    ) -> None:
        """
        Переключает runtime-local fallback для следующего image request.

        Генератор не должен совместно использоваться несколькими
        конкурентными image pipeline вызовами, пока этот флаг включён.
        """

        if not isinstance(enabled, bool):
            raise TypeError(
                "enabled должен быть bool."
            )

        self._moderation_safe_editorial_fallback = enabled

    @property
    def moderation_safe_editorial_fallback(
        self,
    ) -> bool:
        """Показывает, включён ли moderation-safe fallback."""

        return self._moderation_safe_editorial_fallback

    @property
    def metadata(
        self,
    ) -> OpenAIImageGeneratorMetadata:
        """Возвращает версии компонентов генератора."""

        return OpenAIImageGeneratorMetadata(
            generator_name=(
                "openai_movie_news_image_generator"
            ),
            generator_version=(
                OPENAI_IMAGE_GENERATOR_VERSION
            ),
            prompt_version=(
                OPENAI_IMAGE_FALLBACK_PROMPT_VERSION
                if self._moderation_safe_editorial_fallback
                else OPENAI_IMAGE_PROMPT_VERSION
            ),
            model_name=self._model_name,
        )

    def build_request(
        self,
        *,
        items: tuple[
            ImageGenerationNewsItem,
            ...,
        ],
        editorial_comment: str | None = None,
        issues: tuple[str, ...] = (),
    ) -> ImageModelRequest:
        """Строит точный запрос к Image API."""

        prompt = build_image_prompt(
            items=items,
            editorial_comment=editorial_comment,
            issues=issues,
            moderation_safe_editorial_fallback=(
                self._moderation_safe_editorial_fallback
            ),
        )

        return ImageModelRequest(
            model=self._model_name,
            prompt=prompt,
            size=self._size,
            quality=self._quality,
            output_format=(
                DEFAULT_IMAGE_OUTPUT_FORMAT
            ),
            background=(
                DEFAULT_IMAGE_BACKGROUND
            ),
            moderation=(
                DEFAULT_IMAGE_MODERATION
            ),
            n=DEFAULT_IMAGE_COUNT,
        )

    async def generate(
        self,
        *,
        items: tuple[
            ImageGenerationNewsItem,
            ...,
        ],
        editorial_comment: str | None = None,
        issues: tuple[str, ...] = (),
    ) -> OpenAIImageGenerationResult:
        """Генерирует одну общую иллюстрацию TOP-3."""

        model_request = self.build_request(
            items=items,
            editorial_comment=editorial_comment,
            issues=issues,
        )

        model_response = (
            await self._client.create_image(
                model_request
            )
        )

        if not isinstance(
            model_response,
            ImageModelResponse,
        ):
            raise TypeError(
                "ImageGenerationClient вернул "
                "неподдерживаемый тип ответа."
            )

        return OpenAIImageGenerationResult(
            model_request=model_request,
            model_response=model_response,
        )