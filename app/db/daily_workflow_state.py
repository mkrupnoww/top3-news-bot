from dataclasses import dataclass
from datetime import date, datetime

import asyncpg


@dataclass(frozen=True, slots=True)
class RankingWorkflowState:
    """Фактическое состояние ranking-run в PostgreSQL."""

    ranking_run_id: int
    run_status: str
    window_started_at: datetime
    window_finished_at: datetime
    candidate_count: int
    scored_count: int
    eligible_count: int
    top3_news_ids: tuple[int, ...]

    @property
    def ready_for_generation(self) -> bool:
        """Можно ли использовать run для генерации TOP-3."""

        return (
            self.run_status == "completed"
            and len(self.top3_news_ids) == 3
        )

    @property
    def failed(self) -> bool:
        """Завершился ли ranking необрабатываемой ошибкой."""

        return self.run_status in {
            "failed",
            "completed_with_errors",
        }

    @property
    def in_progress(self) -> bool:
        """Остался ли ranking в running."""

        return self.run_status == "running"


@dataclass(frozen=True, slots=True)
class GenerationWorkflowState:
    """Фактическое состояние publication batch и post."""

    batch_id: int
    ranking_run_id: int | None
    publication_date: date
    edition: int
    batch_status: str

    generated_post_id: int | None
    version_number: int | None
    post_status: str | None

    image_path: str | None
    image_sha256: str | None

    @property
    def failed(self) -> bool:
        """Завершилась ли генерация ошибкой."""

        return self.batch_status == "failed"

    @property
    def generation_in_progress(self) -> bool:
        """Зарезервирован ли batch без готового post."""

        return self.batch_status in {
            "ranked",
            "generated",
        }

    @property
    def awaiting_review(self) -> bool:
        """Существует ли готовый post для review."""

        return (
            self.batch_status == "awaiting_review"
            and self.generated_post_id is not None
            and self.post_status == "awaiting_review"
        )

    @property
    def has_image(self) -> bool:
        """Полностью ли привязано изображение."""

        return (
            self.image_path is not None
            and self.image_sha256 is not None
        )

    @property
    def image_state_inconsistent(self) -> bool:
        """Заполнена ли только половина image-state."""

        return (
            (self.image_path is None)
            != (self.image_sha256 is None)
        )

    @property
    def ready_for_image(self) -> bool:
        """Нужно ли запускать image generation."""

        return (
            self.awaiting_review
            and not self.has_image
            and not self.image_state_inconsistent
        )

    @property
    def ready_for_review_delivery(self) -> bool:
        """Готов ли post к Telegram review delivery."""

        return (
            self.awaiting_review
            and self.has_image
        )

    @property
    def human_review_already_progressed(self) -> bool:
        """
        Ушёл ли batch дальше автоматической review delivery.

        Такой выпуск повторно редактору автоматически
        отправлять нельзя.
        """

        return self.batch_status in {
            "approved",
            "rejected",
            "publishing",
            "published",
        }


async def load_ranking_workflow_state(
    pool: asyncpg.Pool,
    *,
    ranking_run_id: int,
) -> RankingWorkflowState:
    """Читает ranking state для restart-safe workflow."""

    if ranking_run_id <= 0:
        raise ValueError(
            "ranking_run_id должен быть больше нуля."
        )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                ranking_run_id,
                run_status,
                window_started_at,
                window_finished_at,
                candidate_count,
                scored_count,
                eligible_count
            FROM ranking_runs
            WHERE ranking_run_id = $1
            """,
            ranking_run_id,
        )

        if record is None:
            raise LookupError(
                "ranking_run не найден: "
                f"ranking_run_id={ranking_run_id}"
            )

        top3_records = await connection.fetch(
            """
            SELECT
                news_id
            FROM news_scores
            WHERE ranking_run_id = $1
              AND selected_for_top3 = true
              AND top3_position IS NOT NULL
            ORDER BY top3_position
            """,
            ranking_run_id,
        )

    return RankingWorkflowState(
        ranking_run_id=int(
            record["ranking_run_id"]
        ),
        run_status=record["run_status"],
        window_started_at=(
            record["window_started_at"]
        ),
        window_finished_at=(
            record["window_finished_at"]
        ),
        candidate_count=int(
            record["candidate_count"]
        ),
        scored_count=int(
            record["scored_count"]
        ),
        eligible_count=int(
            record["eligible_count"]
        ),
        top3_news_ids=tuple(
            int(item["news_id"])
            for item in top3_records
        ),
    )


async def load_generation_workflow_state(
    pool: asyncpg.Pool,
    *,
    batch_id: int,
) -> GenerationWorkflowState:
    """Читает batch/post state для restart-safe workflow."""

    if batch_id <= 0:
        raise ValueError(
            "batch_id должен быть больше нуля."
        )

    async with pool.acquire() as connection:
        record = await connection.fetchrow(
            """
            SELECT
                b.batch_id,
                b.ranking_run_id,
                b.publication_date,
                b.edition,
                b.batch_status,

                p.generated_post_id,
                p.version_number,
                p.post_status,
                p.image_path,
                p.image_sha256

            FROM publication_batches AS b

            LEFT JOIN LATERAL (
                SELECT
                    generated_post_id,
                    version_number,
                    post_status,
                    image_path,
                    image_sha256
                FROM generated_posts
                WHERE batch_id = b.batch_id
                ORDER BY version_number DESC
                LIMIT 1
            ) AS p
              ON true

            WHERE b.batch_id = $1
            """,
            batch_id,
        )

    if record is None:
        raise LookupError(
            "publication_batch не найден: "
            f"batch_id={batch_id}"
        )

    generated_post_id = (
        int(record["generated_post_id"])
        if record["generated_post_id"] is not None
        else None
    )

    version_number = (
        int(record["version_number"])
        if record["version_number"] is not None
        else None
    )

    ranking_run_id = (
        int(record["ranking_run_id"])
        if record["ranking_run_id"] is not None
        else None
    )

    return GenerationWorkflowState(
        batch_id=int(
            record["batch_id"]
        ),
        ranking_run_id=ranking_run_id,
        publication_date=(
            record["publication_date"]
        ),
        edition=int(
            record["edition"]
        ),
        batch_status=(
            record["batch_status"]
        ),
        generated_post_id=(
            generated_post_id
        ),
        version_number=version_number,
        post_status=record["post_status"],
        image_path=record["image_path"],
        image_sha256=(
            str(record["image_sha256"])
            if record["image_sha256"] is not None
            else None
        ),
    )
