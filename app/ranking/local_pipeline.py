from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import asyncpg

from app.db.news_candidates import (
    NewsCandidate,
    select_news_candidates,
)
from app.db.ranking_scores import (
    ManualNewsAssessment,
    RankingPersistenceResult,
    RankingRunMetadata,
    persist_manual_ranking_test,
)
from app.ranking.evaluator import (
    RankingEvaluator,
    RankingEvaluatorMetadata,
)
from app.ranking.score_formula import ScoreInput


LOCAL_EVALUATOR_VERSION = (
    "local_fixture_evaluator_v1"
)

LOCAL_PROMPT_VERSION = (
    "local_fixture_no_prompt_v1"
)


@dataclass(frozen=True, slots=True)
class FixtureScore:
    """Тестовые оценки одной новости."""

    f_score: ScoreInput
    m_score: ScoreInput
    r_score: ScoreInput
    h_score: ScoreInput
    q_score: ScoreInput
    explanation: str


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Новость с сохранённым рейтингом."""

    rank_position: int
    score_id: int
    news_id: int
    title: str
    source_code: str
    source_name: str
    source_url: str
    individual_score: str


@dataclass(frozen=True, slots=True)
class LocalRankingPipelineResult:
    """Результат локального тестового конвейера."""

    ranking_run_id: int
    run_status: str
    already_persisted: bool
    window_started_at: datetime
    window_finished_at: datetime
    candidate_count: int
    scored_count: int
    eligible_count: int
    evaluator_metadata: RankingEvaluatorMetadata
    ranked_candidates: tuple[
        RankedCandidate,
        ...
    ]
    top_candidates: tuple[
        RankedCandidate,
        ...
    ]


class LocalFixtureEvaluator:
    """
    Детерминированный тестовый оценщик.

    Он не анализирует содержание новости.
    Значения передаются явно по news_id.
    """

    def __init__(
        self,
        scores_by_news_id: Mapping[
            int,
            FixtureScore,
        ],
    ) -> None:
        if not scores_by_news_id:
            raise ValueError(
                "Набор тестовых оценок "
                "не может быть пустым."
            )

        invalid_news_ids = sorted(
            news_id
            for news_id in scores_by_news_id
            if news_id <= 0
        )

        if invalid_news_ids:
            raise ValueError(
                "Все news_id в тестовом наборе "
                "должны быть больше нуля: "
                + ",".join(
                    str(news_id)
                    for news_id in invalid_news_ids
                )
            )

        self._scores_by_news_id = dict(
            scores_by_news_id
        )

        self._metadata = (
            RankingEvaluatorMetadata(
                run_mode="local_fixture_test",
                evaluator_name=(
                    "LocalFixtureEvaluator"
                ),
                evaluator_version=(
                    LOCAL_EVALUATOR_VERSION
                ),
                prompt_version=(
                    LOCAL_PROMPT_VERSION
                ),
                model_name=None,
            )
        )

    @property
    def metadata(
        self,
    ) -> RankingEvaluatorMetadata:
        """Возвращает метаданные оценщика."""

        return self._metadata

    @property
    def evaluator_version(
        self,
    ) -> str:
        """
        Возвращает версию оценщика.

        Свойство оставлено для совместимости
        с существующим тестовым сценарием.
        """

        return self._metadata.evaluator_version

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
        """
        Возвращает оценки для всех кандидатов.

        Если хотя бы для одного кандидата нет
        фикстуры, запуск блокируется.
        """

        if not candidates:
            raise ValueError(
                "Список кандидатов "
                "не может быть пустым."
            )

        candidate_ids = {
            candidate.news_id
            for candidate in candidates
        }

        fixture_ids = set(
            self._scores_by_news_id
        )

        missing_fixture_ids = sorted(
            candidate_ids - fixture_ids
        )

        if missing_fixture_ids:
            raise LookupError(
                "Нет тестовых оценок для news_id: "
                + ",".join(
                    str(news_id)
                    for news_id
                    in missing_fixture_ids
                )
            )

        unexpected_fixture_ids = sorted(
            fixture_ids - candidate_ids
        )

        if unexpected_fixture_ids:
            raise ValueError(
                "В фикстуре есть новости, "
                "которые не попали в окно: "
                + ",".join(
                    str(news_id)
                    for news_id
                    in unexpected_fixture_ids
                )
            )

        assessments: list[
            ManualNewsAssessment
        ] = []

        for candidate in candidates:
            fixture = (
                self._scores_by_news_id[
                    candidate.news_id
                ]
            )

            assessments.append(
                ManualNewsAssessment(
                    news_id=candidate.news_id,
                    f_score=fixture.f_score,
                    m_score=fixture.m_score,
                    r_score=fixture.r_score,
                    h_score=fixture.h_score,
                    q_score=fixture.q_score,
                    explanation=(
                        fixture.explanation
                    ),
                )
            )

        return tuple(assessments)


def _build_ranked_candidates(
    *,
    candidates: tuple[
        NewsCandidate,
        ...
    ],
    persistence_result: (
        RankingPersistenceResult
    ),
) -> tuple[
    RankedCandidate,
    ...
]:
    """Объединяет кандидатов и сохранённые оценки."""

    candidate_by_news_id = {
        candidate.news_id: candidate
        for candidate in candidates
    }

    ranked_candidates: list[
        RankedCandidate
    ] = []

    for persisted_score in (
        persistence_result.scores
    ):
        candidate = candidate_by_news_id.get(
            persisted_score.news_id
        )

        if candidate is None:
            raise RuntimeError(
                "Для сохранённой оценки "
                "не найден кандидат: "
                f"news_id="
                f"{persisted_score.news_id}"
            )

        if not persisted_score.scores_match:
            raise RuntimeError(
                "Расчёт Python и PostgreSQL "
                "не совпадает: "
                f"news_id="
                f"{persisted_score.news_id}"
            )

        ranked_candidates.append(
            RankedCandidate(
                rank_position=(
                    persisted_score
                    .rank_position
                ),
                score_id=(
                    persisted_score.score_id
                ),
                news_id=(
                    persisted_score.news_id
                ),
                title=candidate.title,
                source_code=(
                    candidate.source_code
                ),
                source_name=(
                    candidate.source_name
                ),
                source_url=(
                    candidate.source_url
                ),
                individual_score=str(
                    persisted_score
                    .postgres_individual_score
                ),
            )
        )

    ranked_candidates.sort(
        key=lambda candidate: (
            candidate.rank_position,
            candidate.news_id,
        )
    )

    return tuple(ranked_candidates)


async def run_local_ranking_pipeline(
    pool: asyncpg.Pool,
    *,
    as_of: datetime,
    window_hours: float,
    source_codes: tuple[str, ...] | None,
    candidate_limit: int,
    top_size: int,
    test_key: str,
    evaluator: RankingEvaluator,
) -> LocalRankingPipelineResult:
    """
    Запускает локальный тестовый конвейер.

    Последовательность:
    1. Выбрать кандидатов.
    2. Получить оценки через единый интерфейс.
    3. Создать или повторно использовать ranking_run.
    4. Сохранить news_scores.
    5. Вернуть TOP-N.
    """

    if top_size <= 0:
        raise ValueError(
            "top_size должен быть больше нуля."
        )

    normalized_test_key = test_key.strip()

    if not normalized_test_key:
        raise ValueError(
            "test_key не может быть пустым."
        )

    evaluator_metadata = evaluator.metadata

    candidate_result = (
        await select_news_candidates(
            pool,
            as_of=as_of,
            window_hours=window_hours,
            limit=candidate_limit,
            source_codes=source_codes,
        )
    )

    if not candidate_result.candidates:
        raise LookupError(
            "В заданном временном окне "
            "нет кандидатов."
        )

    if (
        top_size
        > len(candidate_result.candidates)
    ):
        raise ValueError(
            "top_size не может превышать "
            "количество кандидатов: "
            f"top_size={top_size}, "
            "candidate_count="
            f"{len(candidate_result.candidates)}"
        )

    assessments = await evaluator.evaluate(
        candidate_result.candidates
    )

    assessed_news_ids = {
        assessment.news_id
        for assessment in assessments
    }

    candidate_news_ids = {
        candidate.news_id
        for candidate
        in candidate_result.candidates
    }

    if assessed_news_ids != candidate_news_ids:
        missing_ids = sorted(
            candidate_news_ids
            - assessed_news_ids
        )

        unexpected_ids = sorted(
            assessed_news_ids
            - candidate_news_ids
        )

        raise RuntimeError(
            "Оценщик вернул некорректный "
            "набор новостей: "
            f"missing={missing_ids}, "
            f"unexpected={unexpected_ids}"
        )

    ranking_run_metadata = RankingRunMetadata(
        run_mode=evaluator_metadata.run_mode,
        evaluator_name=(
            evaluator_metadata.evaluator_name
        ),
        evaluator_version=(
            evaluator_metadata.evaluator_version
        ),
        prompt_version=(
            evaluator_metadata.prompt_version
        ),
        model_name=evaluator_metadata.model_name,
    )

    persistence_result = (
        await persist_manual_ranking_test(
            pool,
            test_key=normalized_test_key,
            window_started_at=(
                candidate_result.window_start
            ),
            window_finished_at=(
                candidate_result.window_end
            ),
            assessments=assessments,
            metadata=ranking_run_metadata,
        )
    )

    ranked_candidates = (
        _build_ranked_candidates(
            candidates=(
                candidate_result.candidates
            ),
            persistence_result=(
                persistence_result
            ),
        )
    )

    return LocalRankingPipelineResult(
        ranking_run_id=(
            persistence_result.ranking_run_id
        ),
        run_status=(
            persistence_result.run_status
        ),
        already_persisted=(
            persistence_result
            .already_persisted
        ),
        window_started_at=(
            candidate_result.window_start
        ),
        window_finished_at=(
            candidate_result.window_end
        ),
        candidate_count=(
            persistence_result
            .candidate_count
        ),
        scored_count=(
            persistence_result.scored_count
        ),
        eligible_count=(
            persistence_result.eligible_count
        ),
        evaluator_metadata=(
            evaluator_metadata
        ),
        ranked_candidates=(
            ranked_candidates
        ),
        top_candidates=(
            ranked_candidates[:top_size]
        ),
    )