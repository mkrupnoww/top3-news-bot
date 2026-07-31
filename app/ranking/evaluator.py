from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.db.news_candidates import NewsCandidate
from app.db.ranking_scores import ManualNewsAssessment


@dataclass(frozen=True, slots=True)
class RankingEvaluatorMetadata:
    """Идентификация оценщика и его конфигурации."""

    run_mode: str
    evaluator_name: str
    evaluator_version: str
    prompt_version: str
    model_name: str | None = None

    def __post_init__(self) -> None:
        """Проверяет обязательные текстовые поля."""

        required_fields = {
            "run_mode": self.run_mode,
            "evaluator_name": self.evaluator_name,
            "evaluator_version": self.evaluator_version,
            "prompt_version": self.prompt_version,
        }

        for field_name, value in required_fields.items():
            if not value.strip():
                raise ValueError(
                    f"{field_name} не может быть пустым."
                )

        if (
            self.model_name is not None
            and not self.model_name.strip()
        ):
            raise ValueError(
                "model_name не может быть пустой строкой."
            )


@runtime_checkable
class RankingEvaluator(Protocol):
    """
    Универсальный интерфейс оценщика новостей.

    Реализацией может быть:
    - локальная тестовая фикстура;
    - OpenAI-оценщик;
    - другой внешний или локальный сервис.
    """

    @property
    def metadata(
        self,
    ) -> RankingEvaluatorMetadata:
        """Возвращает метаданные оценщика."""

        ...

    async def evaluate(
        self,
        candidates: tuple[
            NewsCandidate,
            ...
        ],
    ) -> tuple[
        ManualNewsAssessment,
        ...
    ]:
        """Оценивает переданный набор кандидатов."""

        ...