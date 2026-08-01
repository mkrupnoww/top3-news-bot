from dataclasses import dataclass
from decimal import Decimal
import json
from typing import Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from app.db.news_candidates import NewsCandidate
from app.db.ranking_scores import ManualNewsAssessment
from app.ranking.evaluator import (
    RankingEvaluatorMetadata,
)
from app.ranking.openai_usage import (
    OpenAICostEstimate,
    OpenAITokenUsage,
)


OPENAI_EVALUATOR_VERSION = (
    "openai_ranking_evaluator_v1"
)

OPENAI_PROMPT_VERSION = (
    "movie_news_ranking_prompt_v1"
)


@dataclass(frozen=True, slots=True)
class RankingModelRequest:
    """Запрос к модели ранжирования."""

    model: str
    instructions: str
    input_text: str


@dataclass(frozen=True, slots=True)
class RankingModelResponse:
    """Ответ модели вместе с телеметрией."""

    output_text: str
    usage: OpenAITokenUsage | None = None
    cost_estimate: OpenAICostEstimate | None = None


@dataclass(frozen=True, slots=True)
class OpenAIRankingEvaluationResult:
    """Оценки новостей и данные API-запроса."""

    assessments: tuple[
        ManualNewsAssessment,
        ...,
    ]
    model_response: RankingModelResponse


@runtime_checkable
class StructuredRankingClient(Protocol):
    """
    Транспортный интерфейс модели.

    Реальная реализация использует OpenAI
    Responses API. В тестах применяется
    локальный клиент без сетевых запросов.
    """

    async def create_response(
        self,
        request: RankingModelRequest,
    ) -> RankingModelResponse:
        """Возвращает ответ модели и телеметрию."""

        ...


class OpenAINewsScorePayload(BaseModel):
    """Структурированная оценка одной новости."""

    model_config = ConfigDict(
        extra="forbid",
    )

    news_id: int = Field(
        gt=0,
    )

    f_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )

    m_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )

    r_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )

    h_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("10"),
    )

    q_score: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
    )

    explanation: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("explanation")
    @classmethod
    def normalize_explanation(
        cls,
        value: str,
    ) -> str:
        """Удаляет лишние пробелы по краям."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "explanation не может быть пустым."
            )

        return normalized_value


class OpenAIRankingPayload(BaseModel):
    """Полный структурированный ответ оценщика."""

    model_config = ConfigDict(
        extra="forbid",
    )

    scores: list[
        OpenAINewsScorePayload
    ] = Field(
        min_length=1,
        max_length=500,
    )


SYSTEM_INSTRUCTIONS = """
Ты оцениваешь кандидатов для ежедневного TOP-3
мировых киноновостей.

Для каждой новости верни пять компонентов:
F — свежесть и актуальность, шкала 0–10.
Учитывай age_hours. Новость уже прошла строгий
фильтр временного окна не более 24 часов.

M — масштаб и потенциальный охват, шкала 0–10.
Учитывай известность участников, компаний,
проектов и возможную глубину последствий.

R — резонанс и потенциальная вовлечённость,
шкала 0–10. Не выдумывай просмотры,
комментарии или другие отсутствующие метрики.

H — цепляющий, необычный или конфликтный
элемент новости, шкала 0–10.
Q — подтверждённость и качество доступных
данных, шкала 0–1.

Итоговый балл самостоятельно не вычисляй.
Он рассчитывается программой по формуле:

B = 0.20F + 0.30M + 0.20R + 0.15(H × Q)

Правила ответа:

1. Верни каждую переданную новость ровно один раз.
2. Сохрани исходный news_id.
3. Не добавляй новости, которых нет во входе.
4. Для каждой оценки дай краткое объяснение.
5. Верни только JSON-объект без Markdown.
6. Формат ответа:
{
  "scores": [
    {
      "news_id": 1,
      "f_score": 8.5,
      "m_score": 7.0,
      "r_score": 6.0,
      "h_score": 5.5,
      "q_score": 0.9,
      "explanation": "Краткое объяснение."
    }
  ]
}
""".strip()


def _validate_candidates(
    candidates: tuple[
        NewsCandidate,
        ...,
    ],
) -> tuple[int, ...]:
    """Проверяет входной набор кандидатов."""

    if not candidates:
        raise ValueError(
            "Список кандидатов "
            "не может быть пустым."
        )

    news_ids = tuple(
        candidate.news_id
        for candidate in candidates
    )

    if any(
        news_id <= 0
        for news_id in news_ids
    ):
        raise ValueError(
            "Все news_id должны быть "
            "больше нуля."
        )

    if len(set(news_ids)) != len(news_ids):
        raise ValueError(
            "Во входном наборе обнаружены "
            "повторяющиеся news_id."
        )

    return news_ids


def _build_input_text(
    candidates: tuple[
        NewsCandidate,
        ...,
    ],
) -> str:
    """Формирует JSON с кандидатами для модели."""

    payload = {
        "task": "score_movie_news_candidates",
        "formula": (
            "0.20F + 0.30M + 0.20R "
            "+ 0.15(H × Q)"
        ),
        "score_scales": {
            "f_score": "0..10",
            "m_score": "0..10",
            "r_score": "0..10",
            "h_score": "0..10",
            "q_score": "0..1",
        },
        "candidates": [
            {
                "news_id": candidate.news_id,
                "source_code": (
                    candidate.source_code
                ),
                "source_name": (
                    candidate.source_name
                ),
                "collection_priority": (
                    candidate.collection_priority
                ),
                "title": candidate.title,
                "summary": candidate.summary,
                "author_name": (
                    candidate.author_name
                ),
                "published_at": (
                    candidate
                    .source_published_at
                    .isoformat()
                ),
                "age_hours": round(
                    candidate.age_hours,
                    4,
                ),
                "source_url": (
                    candidate.source_url
                ),
            }
            for candidate in candidates
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_response(
    response_text: str,
) -> OpenAIRankingPayload:
    """Проверяет JSON-ответ модели."""

    normalized_response = (
        response_text.strip()
    )

    if not normalized_response:
        raise ValueError(
            "Модель вернула пустой ответ."
        )

    try:
        return (
            OpenAIRankingPayload
            .model_validate_json(
                normalized_response
            )
        )
    except ValidationError as error:
        raise ValueError(
            "Ответ модели не соответствует "
            "схеме рейтинга."
        ) from error


def _validate_response_news_ids(
    *,
    expected_news_ids: tuple[int, ...],
    payload: OpenAIRankingPayload,
) -> None:
    """Проверяет полноту набора news_id."""

    response_news_ids = [
        score.news_id
        for score in payload.scores
    ]

    duplicate_news_ids = sorted(
        {
            news_id
            for news_id in response_news_ids
            if response_news_ids.count(
                news_id
            ) > 1
        }
    )

    if duplicate_news_ids:
        raise ValueError(
            "Модель вернула повторяющиеся "
            "news_id: "
            + ",".join(
                str(news_id)
                for news_id
                in duplicate_news_ids
            )
        )

    expected_set = set(
        expected_news_ids
    )

    response_set = set(
        response_news_ids
    )

    missing_news_ids = sorted(
        expected_set - response_set
    )

    unexpected_news_ids = sorted(
        response_set - expected_set
    )

    if (
        missing_news_ids
        or unexpected_news_ids
    ):
        raise ValueError(
            "Модель вернула некорректный "
            "набор news_id: "
            f"missing={missing_news_ids}, "
            f"unexpected={unexpected_news_ids}"
        )


def _build_assessments(
    *,
    candidates: tuple[
        NewsCandidate,
        ...,
    ],
    payload: OpenAIRankingPayload,
) -> tuple[
    ManualNewsAssessment,
    ...,
]:
    """Преобразует ответ модели в оценки."""

    score_by_news_id = {
        score.news_id: score
        for score in payload.scores
    }

    return tuple(
        ManualNewsAssessment(
            news_id=candidate.news_id,
            f_score=(
                score_by_news_id[
                    candidate.news_id
                ].f_score
            ),
            m_score=(
                score_by_news_id[
                    candidate.news_id
                ].m_score
            ),
            r_score=(
                score_by_news_id[
                    candidate.news_id
                ].r_score
            ),
            h_score=(
                score_by_news_id[
                    candidate.news_id
                ].h_score
            ),
            q_score=(
                score_by_news_id[
                    candidate.news_id
                ].q_score
            ),
            explanation=(
                score_by_news_id[
                    candidate.news_id
                ].explanation
            ),
        )
        for candidate in candidates
    )


class OpenAIRankingEvaluator:
    """Оценщик киноновостей через модель OpenAI."""

    def __init__(
        self,
        *,
        client: StructuredRankingClient,
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
            RankingEvaluatorMetadata(
                run_mode="openai_ranking",
                evaluator_name=(
                    "OpenAIRankingEvaluator"
                ),
                evaluator_version=(
                    OPENAI_EVALUATOR_VERSION
                ),
                prompt_version=(
                    OPENAI_PROMPT_VERSION
                ),
                model_name=(
                    normalized_model_name
                ),
            )
        )

    @property
    def metadata(
        self,
    ) -> RankingEvaluatorMetadata:
        """Возвращает метаданные оценщика."""

        return self._metadata

    def build_request(
        self,
        candidates: tuple[
            NewsCandidate,
            ...,
        ],
    ) -> RankingModelRequest:
        """
        Формирует точный запрос без вызова модели.

        Этот объект можно использовать для
        вычисления request_key и резервирования
        ranking_run до платного API-запроса.
        """

        _validate_candidates(candidates)

        model_name = self._metadata.model_name

        if model_name is None:
            raise RuntimeError(
                "В метаданных оценщика "
                "отсутствует model_name."
            )

        return RankingModelRequest(
            model=model_name,
            instructions=SYSTEM_INSTRUCTIONS,
            input_text=_build_input_text(
                candidates
            ),
        )

    async def evaluate_prepared_request(
        self,
        candidates: tuple[
            NewsCandidate,
            ...,
        ],
        request: RankingModelRequest,
    ) -> OpenAIRankingEvaluationResult:
        """
        Выполняет заранее сформированный запрос.

        Перед обращением к модели проверяет,
        что запрос в точности соответствует
        текущим кандидатам, модели и промпту.
        """

        expected_news_ids = (
            _validate_candidates(
                candidates
            )
        )

        expected_request = self.build_request(
            candidates
        )

        if request != expected_request:
            raise ValueError(
                "Подготовленный запрос не "
                "соответствует текущим "
                "кандидатам, модели или промпту."
            )

        model_response = (
            await self._client.create_response(
                request
            )
        )

        payload = _parse_response(
            model_response.output_text
        )

        _validate_response_news_ids(
            expected_news_ids=(
                expected_news_ids
            ),
            payload=payload,
        )

        assessments = _build_assessments(
            candidates=candidates,
            payload=payload,
        )

        return OpenAIRankingEvaluationResult(
            assessments=assessments,
            model_response=model_response,
        )

    async def evaluate_detailed(
        self,
        candidates: tuple[
            NewsCandidate,
            ...,
        ],
    ) -> OpenAIRankingEvaluationResult:
        """Оценивает новости и возвращает телеметрию."""

        request = self.build_request(
            candidates
        )

        return await self.evaluate_prepared_request(
            candidates,
            request,
        )

    async def evaluate(
        self,
        candidates: tuple[
            NewsCandidate,
            ...,
        ],
    ) -> tuple[
        ManualNewsAssessment,
        ...,
    ]:
        """Оценивает новости через общий интерфейс."""

        result = await self.evaluate_detailed(
            candidates
        )

        return result.assessments