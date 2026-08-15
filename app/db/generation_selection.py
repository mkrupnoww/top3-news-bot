from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import asyncpg

from app.generation.openai_generator import (
    GenerationNewsItem,
)


@dataclass(frozen=True, slots=True)
class GenerationTop3Selection:
    """Сохранённый TOP-3 завершённого ранжирования."""

    ranking_run_id: int
    run_status: str
    eligible_count: int
    score_ids: tuple[int, int, int]
    items: tuple[
        GenerationNewsItem,
        GenerationNewsItem,
        GenerationNewsItem,
    ]

    @property
    def news_ids(self) -> tuple[int, int, int]:
        """Возвращает news_id в порядке TOP-3."""

        return (
            self.items[0].news_id,
            self.items[1].news_id,
            self.items[2].news_id,
        )


@dataclass(frozen=True, slots=True)
class RankedGenerationCombination:
    """Одна сохранённая ranking combination для генерации."""

    combination_id: int
    combination_rank: int
    final_top_score: Decimal
    is_winner: bool
    selection: GenerationTop3Selection

    @property
    def news_ids(self) -> tuple[int, int, int]:
        """Возвращает news_id комбинации в сохранённом порядке."""

        return self.selection.news_ids


@dataclass(frozen=True, slots=True)
class GenerationCombinationReplacement:
    """Следующая допустимая комбинация для replacement cascade."""

    combination: RankedGenerationCombination
    overlap_count: int
    removed_news_ids: tuple[int, ...]
    added_news_ids: tuple[int, ...]

    @property
    def selection(self) -> GenerationTop3Selection:
        """Возвращает готовую selection для text/image pipeline."""

        return self.combination.selection


def _normalize_positive_integer(
    value: int,
    *,
    field_name: str,
) -> int:
    """Проверяет положительный идентификатор."""

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
            f"{field_name} должен быть "
            "больше нуля."
        )

    return value


def _normalize_required_text(
    value: str,
    *,
    field_name: str,
) -> str:
    """Проверяет обязательный текст."""

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name} не может быть пустым."
        )

    return normalized_value


def _normalize_top3_news_ids(
    news_ids: tuple[int, int, int],
    *,
    field_name: str,
) -> tuple[int, int, int]:
    """Проверяет ровно три уникальных положительных news_id."""

    if not isinstance(news_ids, tuple):
        raise TypeError(
            f"{field_name} должен быть tuple."
        )

    if len(news_ids) != 3:
        raise ValueError(
            f"{field_name} должен содержать "
            "ровно три news_id."
        )

    normalized = tuple(
        _normalize_positive_integer(
            news_id,
            field_name=(
                f"{field_name}[{index}]"
            ),
        )
        for index, news_id in enumerate(
            news_ids,
            start=1,
        )
    )

    if len(set(normalized)) != 3:
        raise ValueError(
            f"{field_name} содержит "
            "повторяющиеся news_id."
        )

    return (
        normalized[0],
        normalized[1],
        normalized[2],
    )


def _normalize_combination_ids(
    combination_ids: tuple[int, ...],
    *,
    field_name: str,
) -> tuple[int, ...]:
    """Проверяет список уже использованных combination_id."""

    if not isinstance(
        combination_ids,
        tuple,
    ):
        raise TypeError(
            f"{field_name} должен быть tuple."
        )

    normalized = tuple(
        _normalize_positive_integer(
            combination_id,
            field_name=(
                f"{field_name}[{index}]"
            ),
        )
        for index, combination_id in enumerate(
            combination_ids,
            start=1,
        )
    )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{field_name} содержит "
            "повторяющиеся combination_id."
        )

    return normalized


def _resolve_saved_top3_mode(
    selected_for_top3_count: int,
    *,
    ranking_run_id: int,
) -> bool:
    """
    Определяет способ чтения сохранённого TOP-3.

    Возвращает True для нового формата
    selected_for_top3/top3_position и False
    для legacy rank_position 1..3.

    Любое частично сохранённое новое состояние
    считается ошибкой и не маскируется fallback.
    """

    if isinstance(
        selected_for_top3_count,
        bool,
    ):
        raise TypeError(
            "selected_for_top3_count "
            "не может быть bool."
        )

    if not isinstance(
        selected_for_top3_count,
        int,
    ):
        raise TypeError(
            "selected_for_top3_count "
            "должен быть int."
        )

    if selected_for_top3_count < 0:
        raise ValueError(
            "selected_for_top3_count "
            "не может быть отрицательным."
        )

    if selected_for_top3_count == 3:
        return True

    if selected_for_top3_count == 0:
        return False

    raise ValueError(
        "Сохранённый финальный TOP-3 имеет "
        "некорректное число строк "
        "selected_for_top3=true: "
        f"ranking_run_id={ranking_run_id}, "
        f"found={selected_for_top3_count}, "
        "expected=0 legacy или 3."
    )


def _required_record_text(
    record: asyncpg.Record,
    field_name: str,
    *,
    news_id: int,
) -> str:
    """Извлекает обязательный текст из строки БД."""

    value = record[field_name]

    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} отсутствует: "
            f"news_id={news_id}"
        )

    return _normalize_required_text(
        value,
        field_name=(
            f"{field_name} news_id={news_id}"
        ),
    )


def _build_selection(
    *,
    ranking_run_id: int,
    run_status: str,
    eligible_count: int,
    records: list[asyncpg.Record],
) -> GenerationTop3Selection:
    """Преобразует строки БД в сохранённый TOP-3."""

    if run_status != "completed":
        raise ValueError(
            "Для генерации требуется ranking_run "
            "со статусом completed: "
            f"ranking_run_id={ranking_run_id}, "
            f"run_status={run_status}"
        )

    if eligible_count < 3:
        raise ValueError(
            "В ranking_run меньше трёх "
            "подходящих новостей: "
            f"ranking_run_id={ranking_run_id}, "
            f"eligible_count={eligible_count}"
        )

    if len(records) != 3:
        raise ValueError(
            "Для генерации должны существовать "
            "ровно три сохранённые позиции TOP-3: "
            f"ranking_run_id={ranking_run_id}, "
            f"found={len(records)}"
        )

    positions = tuple(
        int(record["generation_position"])
        for record in records
    )

    if positions != (1, 2, 3):
        raise ValueError(
            "Сохранённые позиции TOP-3 должны "
            "быть равны 1, 2 и 3: "
            f"actual={positions}"
        )

    score_ids: list[int] = []
    items: list[GenerationNewsItem] = []

    for record in records:
        position = int(
            record["generation_position"]
        )
        news_id = int(record["news_id"])
        score_id = int(record["score_id"])

        _normalize_positive_integer(
            score_id,
            field_name=(
                f"score_id position={position}"
            ),
        )

        _normalize_positive_integer(
            news_id,
            field_name=(
                f"news_id position={position}"
            ),
        )

        processing_status = (
            record["processing_status"]
        )

        if processing_status not in {
            "collected",
            "normalized",
            "candidate",
        }:
            raise ValueError(
                "Новость TOP-3 имеет "
                "неподходящий processing_status: "
                f"news_id={news_id}, "
                f"status={processing_status}"
            )

        source_published_at = (
            record["source_published_at"]
        )

        if not isinstance(
            source_published_at,
            datetime,
        ):
            raise ValueError(
                "source_published_at отсутствует: "
                f"news_id={news_id}"
            )

        if (
            source_published_at.tzinfo is None
            or source_published_at.utcoffset()
            is None
        ):
            raise ValueError(
                "source_published_at должен "
                "содержать часовой пояс: "
                f"news_id={news_id}"
            )

        individual_score = (
            record["individual_score"]
        )

        if not isinstance(
            individual_score,
            Decimal,
        ):
            raise TypeError(
                "individual_score должен быть "
                "Decimal: "
                f"news_id={news_id}"
            )

        if (
            not individual_score.is_finite()
            or individual_score < 0
        ):
            raise ValueError(
                "individual_score должен быть "
                "конечным неотрицательным числом: "
                f"news_id={news_id}"
            )

        items.append(
            GenerationNewsItem(
                position=position,
                news_id=news_id,
                title=_required_record_text(
                    record,
                    "title",
                    news_id=news_id,
                ),
                summary=_required_record_text(
                    record,
                    "summary",
                    news_id=news_id,
                ),
                source_name=(
                    _required_record_text(
                        record,
                        "source_name",
                        news_id=news_id,
                    )
                ),
                source_url=(
                    _required_record_text(
                        record,
                        "source_url",
                        news_id=news_id,
                    )
                ),
                source_published_at=(
                    source_published_at
                ),
                individual_score=(
                    individual_score
                ),
                selection_reason=(
                    _required_record_text(
                        record,
                        "score_explanation",
                        news_id=news_id,
                    )
                ),
            )
        )

        score_ids.append(score_id)

    news_ids = tuple(
        item.news_id
        for item in items
    )

    if len(set(news_ids)) != 3:
        raise ValueError(
            "Сохранённый TOP-3 содержит "
            "дублирующиеся news_id."
        )

    if len(set(score_ids)) != 3:
        raise ValueError(
            "Сохранённый TOP-3 содержит "
            "дублирующиеся score_id."
        )

    return GenerationTop3Selection(
        ranking_run_id=ranking_run_id,
        run_status=run_status,
        eligible_count=eligible_count,
        score_ids=(
            score_ids[0],
            score_ids[1],
            score_ids[2],
        ),
        items=(
            items[0],
            items[1],
            items[2],
        ),
    )


async def _load_run_record(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
) -> asyncpg.Record:
    """Читает обязательный ranking_run."""

    run_record = await connection.fetchrow(
        """
        SELECT
            ranking_run_id,
            run_status,
            eligible_count
        FROM ranking_runs
        WHERE ranking_run_id = $1
        """,
        ranking_run_id,
    )

    if run_record is None:
        raise LookupError(
            "Не найден ranking_run: "
            f"{ranking_run_id}"
        )

    return run_record


async def _load_generation_top3(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
) -> GenerationTop3Selection:
    """Читает завершённый TOP-3 через соединение."""

    run_record = await _load_run_record(
        connection,
        ranking_run_id=ranking_run_id,
    )

    selected_for_top3_count = (
        await connection.fetchval(
            """
            SELECT COUNT(*)::integer
            FROM news_scores
            WHERE ranking_run_id = $1
              AND selected_for_top3 = true
            """,
            ranking_run_id,
        )
    )

    use_saved_top3 = _resolve_saved_top3_mode(
        int(selected_for_top3_count),
        ranking_run_id=ranking_run_id,
    )

    records = await connection.fetch(
        """
        SELECT
            ns.score_id,
            CASE
                WHEN $2::boolean
                    THEN ns.top3_position
                ELSE ns.rank_position
            END AS generation_position,
            ns.rank_position,
            ns.news_id,
            ns.individual_score,
            ns.score_explanation,
            COALESCE(
                NULLIF(
                    BTRIM(ni.normalized_title),
                    ''
                ),
                NULLIF(
                    BTRIM(ni.raw_title),
                    ''
                )
            ) AS title,
            COALESCE(
                NULLIF(
                    BTRIM(ni.normalized_summary),
                    ''
                ),
                NULLIF(
                    BTRIM(ni.raw_summary),
                    ''
                ),
                NULLIF(
                    BTRIM(ni.article_text),
                    ''
                )
            ) AS summary,
            s.source_name,
            ni.source_url,
            ni.source_published_at,
            ni.processing_status
        FROM news_scores AS ns
        JOIN news_items AS ni
            ON ni.news_id = ns.news_id
        JOIN sources AS s
            ON s.source_id = ni.source_id
        WHERE ns.ranking_run_id = $1
          AND ns.is_eligible = true
          AND (
                (
                    $2::boolean
                    AND ns.selected_for_top3 = true
                    AND ns.top3_position
                        BETWEEN 1 AND 3
                )
                OR
                (
                    NOT $2::boolean
                    AND ns.rank_position
                        BETWEEN 1 AND 3
                )
          )
        ORDER BY
            CASE
                WHEN $2::boolean
                    THEN ns.top3_position
                ELSE ns.rank_position
            END
        """,
        ranking_run_id,
        use_saved_top3,
    )

    return _build_selection(
        ranking_run_id=int(
            run_record["ranking_run_id"]
        ),
        run_status=run_record["run_status"],
        eligible_count=int(
            run_record["eligible_count"]
        ),
        records=list(records),
    )


async def _load_generation_combination(
    connection: asyncpg.Connection,
    *,
    ranking_run_id: int,
    combination_id: int,
) -> RankedGenerationCombination:
    """Читает одну конкретную сохранённую ranking combination."""

    run_record = await _load_run_record(
        connection,
        ranking_run_id=ranking_run_id,
    )

    combination_record = (
        await connection.fetchrow(
            """
            SELECT
                combination_id,
                combination_rank,
                final_top_score,
                is_winner
            FROM ranking_combinations
            WHERE ranking_run_id = $1
              AND combination_id = $2
            """,
            ranking_run_id,
            combination_id,
        )
    )

    if combination_record is None:
        raise LookupError(
            "Не найдена ranking combination: "
            f"ranking_run_id={ranking_run_id}, "
            f"combination_id={combination_id}"
        )

    records = await connection.fetch(
        """
        SELECT
            ns.score_id,
            rci.position AS generation_position,
            ns.rank_position,
            ns.news_id,
            ns.individual_score,
            ns.score_explanation,
            COALESCE(
                NULLIF(
                    BTRIM(ni.normalized_title),
                    ''
                ),
                NULLIF(
                    BTRIM(ni.raw_title),
                    ''
                )
            ) AS title,
            COALESCE(
                NULLIF(
                    BTRIM(ni.normalized_summary),
                    ''
                ),
                NULLIF(
                    BTRIM(ni.raw_summary),
                    ''
                ),
                NULLIF(
                    BTRIM(ni.article_text),
                    ''
                )
            ) AS summary,
            s.source_name,
            ni.source_url,
            ni.source_published_at,
            ni.processing_status
        FROM ranking_combination_items AS rci
        JOIN news_scores AS ns
          ON ns.score_id = rci.score_id
         AND ns.ranking_run_id = rci.ranking_run_id
        JOIN news_items AS ni
          ON ni.news_id = ns.news_id
        JOIN sources AS s
          ON s.source_id = ni.source_id
        WHERE rci.ranking_run_id = $1
          AND rci.combination_id = $2
          AND ns.is_eligible = true
        ORDER BY rci.position
        """,
        ranking_run_id,
        combination_id,
    )

    selection = _build_selection(
        ranking_run_id=int(
            run_record["ranking_run_id"]
        ),
        run_status=run_record["run_status"],
        eligible_count=int(
            run_record["eligible_count"]
        ),
        records=list(records),
    )

    final_top_score = (
        combination_record["final_top_score"]
    )

    if not isinstance(
        final_top_score,
        Decimal,
    ):
        raise TypeError(
            "final_top_score должен быть Decimal: "
            f"combination_id={combination_id}"
        )

    if (
        not final_top_score.is_finite()
        or final_top_score < 0
    ):
        raise ValueError(
            "final_top_score должен быть "
            "конечным неотрицательным числом: "
            f"combination_id={combination_id}"
        )

    return RankedGenerationCombination(
        combination_id=int(
            combination_record[
                "combination_id"
            ]
        ),
        combination_rank=int(
            combination_record[
                "combination_rank"
            ]
        ),
        final_top_score=final_top_score,
        is_winner=bool(
            combination_record[
                "is_winner"
            ]
        ),
        selection=selection,
    )


async def load_generation_top3(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> GenerationTop3Selection:
    """Возвращает сохранённый TOP-3 для генерации."""

    normalized_ranking_run_id = (
        _normalize_positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
    )

    async with pool.acquire() as connection:
        return await _load_generation_top3(
            connection,
            ranking_run_id=(
                normalized_ranking_run_id
            ),
        )


async def load_generation_combination(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    combination_id: int,
) -> RankedGenerationCombination:
    """Возвращает конкретную ranking combination для генерации."""

    normalized_ranking_run_id = (
        _normalize_positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
    )

    normalized_combination_id = (
        _normalize_positive_integer(
            combination_id,
            field_name="combination_id",
        )
    )

    async with pool.acquire() as connection:
        return await _load_generation_combination(
            connection,
            ranking_run_id=(
                normalized_ranking_run_id
            ),
            combination_id=(
                normalized_combination_id
            ),
        )


async def choose_next_generation_combination(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
    current_news_ids: tuple[int, int, int],
    excluded_combination_ids: tuple[int, ...] = (),
) -> GenerationCombinationReplacement | None:
    """
    Выбирает следующую комбинацию для replacement cascade.

    Правила:
    1. Текущий набор из тех же трёх news_id не выбирается.
    2. Уже использованные combination_id исключаются.
    3. Максимизируется overlap с текущим TOP-3:
       сначала замена одной новости, затем двух, затем трёх.
    4. При одинаковом overlap выигрывает лучший combination_rank.

    Функция только читает БД и ничего не изменяет.
    """

    normalized_ranking_run_id = (
        _normalize_positive_integer(
            ranking_run_id,
            field_name="ranking_run_id",
        )
    )

    normalized_current_news_ids = (
        _normalize_top3_news_ids(
            current_news_ids,
            field_name="current_news_ids",
        )
    )

    normalized_excluded_ids = (
        _normalize_combination_ids(
            excluded_combination_ids,
            field_name=(
                "excluded_combination_ids"
            ),
        )
    )

    current_news_set = set(
        normalized_current_news_ids
    )

    excluded_id_set = set(
        normalized_excluded_ids
    )

    async with pool.acquire() as connection:
        run_record = await _load_run_record(
            connection,
            ranking_run_id=(
                normalized_ranking_run_id
            ),
        )

        if run_record["run_status"] != "completed":
            raise ValueError(
                "Replacement требует completed "
                "ranking_run: "
                f"ranking_run_id="
                f"{normalized_ranking_run_id}, "
                f"run_status="
                f"{run_record['run_status']}"
            )

        rows = await connection.fetch(
            """
            SELECT
                rc.combination_id,
                rc.combination_rank,
                ARRAY_AGG(
                    ns.news_id
                    ORDER BY rci.position
                ) AS news_ids
            FROM ranking_combinations AS rc
            JOIN ranking_combination_items AS rci
              ON rci.combination_id = rc.combination_id
             AND rci.ranking_run_id = rc.ranking_run_id
            JOIN news_scores AS ns
              ON ns.score_id = rci.score_id
             AND ns.ranking_run_id = rci.ranking_run_id
            WHERE rc.ranking_run_id = $1
            GROUP BY
                rc.combination_id,
                rc.combination_rank
            ORDER BY rc.combination_rank
            """,
            normalized_ranking_run_id,
        )

        candidates: list[
            tuple[
                int,
                int,
                int,
                tuple[int, int, int],
            ]
        ] = []

        for row in rows:
            combination_id = int(
                row["combination_id"]
            )

            if combination_id in excluded_id_set:
                continue

            raw_news_ids = tuple(
                int(news_id)
                for news_id in row["news_ids"]
            )

            if len(raw_news_ids) != 3:
                raise ValueError(
                    "ranking combination должна "
                    "содержать ровно три news_id: "
                    f"combination_id={combination_id}, "
                    f"news_ids={raw_news_ids!r}"
                )

            candidate_news_ids = (
                raw_news_ids[0],
                raw_news_ids[1],
                raw_news_ids[2],
            )

            candidate_news_set = set(
                candidate_news_ids
            )

            if candidate_news_set == current_news_set:
                continue

            overlap_count = len(
                current_news_set
                & candidate_news_set
            )

            candidates.append(
                (
                    -overlap_count,
                    int(
                        row["combination_rank"]
                    ),
                    combination_id,
                    candidate_news_ids,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        (
            negative_overlap_count,
            _selected_rank,
            selected_combination_id,
            selected_news_ids,
        ) = candidates[0]

        ranked_combination = (
            await _load_generation_combination(
                connection,
                ranking_run_id=(
                    normalized_ranking_run_id
                ),
                combination_id=(
                    selected_combination_id
                ),
            )
        )

    selected_news_set = set(
        selected_news_ids
    )

    removed_news_ids = tuple(
        news_id
        for news_id in normalized_current_news_ids
        if news_id not in selected_news_set
    )

    added_news_ids = tuple(
        news_id
        for news_id in selected_news_ids
        if news_id not in current_news_set
    )

    return GenerationCombinationReplacement(
        combination=ranked_combination,
        overlap_count=(
            -negative_overlap_count
        ),
        removed_news_ids=(
            removed_news_ids
        ),
        added_news_ids=(
            added_news_ids
        ),
    )