from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import json
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
    "openai_telegram_post_generator_v1"
)

OPENAI_POST_PROMPT_VERSION = (
    "movie_news_telegram_post_prompt_v1"
)

OPENAI_POST_TEXT_FORMAT = "markdown"


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
        """Нормализует готовый Telegram-текст."""

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
с ежедневной подборкой TOP-3 киноновостей.

Используй только сведения из переданного JSON.
Не добавляй факты, цитаты, числа, имена, оценки,
просмотры, реакции или последствия, которых нет
во входных данных.

Требования к готовому посту:

1. Сохрани порядок новостей 1, 2 и 3.
2. Для каждой новости сделай короткий,
   выразительный заголовок и понятный абзац.
3. Передавай смысл своими словами, не копируй
   длинные фрагменты исходных публикаций.
4. Пиши естественно и информативно, без кликбейта,
   канцелярита и рекламных формулировок.
5. Англоязычные имена людей, компаний, фильмов
   и проектов передавай точно. Не придумывай
   официальные русские названия.
6. Не включай в post_text технические показатели,
   баллы F, M, R, H, Q, individual_score,
   news_id или внутренние объяснения оценщика.
7. Не вставляй source_url и длинные перечни
   источников в post_text.
8. Не утверждай, что событие является скандалом,
   сенсацией или рекордом, если это прямо
   не подтверждено входными данными.
9. Не используй Markdown-заголовки с символом #.
10. Для жирного текста применяй две звёздочки:
    **текст**.
11. Для курсива применяй двойное нижнее
    подчёркивание: __текст__.
12. Не используй MarkdownV2 и HTML.
13. Общая длина post_text должна быть
    не более 3900 символов.
14. Верни каждую входную новость ровно один раз.
15. Верни только JSON-объект без Markdown-обёртки.

Рекомендуемая структура post_text:

**TOP-3 киноновости дня**

**1. Короткий заголовок**
Абзац новости.

**2. Короткий заголовок**
Абзац новости.

**3. Короткий заголовок**
Абзац новости.

__Какую из новостей обсудим подробнее?__

Формат JSON-ответа:

{
  "post_text": "Полный текст Telegram-поста",
  "items": [
    {
      "position": 1,
      "news_id": 1,
      "headline": "Короткий заголовок",
      "body": "Абзац новости"
    },
    {
      "position": 2,
      "news_id": 2,
      "headline": "Короткий заголовок",
      "body": "Абзац новости"
    },
    {
      "position": 3,
      "news_id": 3,
      "headline": "Короткий заголовок",
      "body": "Абзац новости"
    }
  ]
}
""".strip()


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательное текстовое поле."""

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


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
        "maximum_post_length": 3900,
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
                    item
                    .source_published_at
                    .isoformat()
                ),
                "individual_score": str(
                    item.individual_score
                ),
                "selection_reason": (
                    item.selection_reason.strip()
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