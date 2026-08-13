BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';


-- ============================================================================
-- 013_daily_workflow_runs.sql
--
-- Фиксирует один автоматический production workflow на дату публикации.
--
-- Workflow является верхним уровнем над существующими защищёнными этапами:
--
-- ranking
-- -> text generation + self-review
-- -> image generation
-- -> Telegram review delivery
--
-- Таблица не заменяет request-key/idempotency каждого этапа.
-- Она связывает их в один restart-safe ежедневный запуск.
-- ============================================================================


CREATE TABLE top3_news.daily_workflow_runs (
    daily_workflow_run_id bigint
        GENERATED ALWAYS AS IDENTITY
        PRIMARY KEY,

    publication_date date NOT NULL,

    workflow_version text NOT NULL,

    workflow_status text NOT NULL
        DEFAULT 'running',

    current_stage text NOT NULL
        DEFAULT 'reserved',

    as_of timestamptz NOT NULL,

    window_hours smallint NOT NULL
        DEFAULT 24,

    target_telegram_chat_id bigint NOT NULL,

    ranking_run_id bigint
        REFERENCES top3_news.ranking_runs (
            ranking_run_id
        )
        ON DELETE SET NULL,

    batch_id bigint
        REFERENCES top3_news.publication_batches (
            batch_id
        )
        ON DELETE SET NULL,

    generated_post_id bigint
        REFERENCES top3_news.generated_posts (
            generated_post_id
        )
        ON DELETE SET NULL,

    image_generation_id bigint
        REFERENCES top3_news.image_generation_requests (
            image_generation_id
        )
        ON DELETE SET NULL,

    error_type text,
    error_message text,

    started_at timestamptz NOT NULL
        DEFAULT now(),

    finished_at timestamptz,

    created_at timestamptz NOT NULL
        DEFAULT now(),

    updated_at timestamptz NOT NULL
        DEFAULT now(),

    CONSTRAINT daily_workflow_runs_version_chk
        CHECK (
            length(btrim(workflow_version)) > 0
        ),

    CONSTRAINT daily_workflow_runs_status_chk
        CHECK (
            workflow_status IN (
                'running',
                'awaiting_review',
                'failed'
            )
        ),

    CONSTRAINT daily_workflow_runs_stage_chk
        CHECK (
            current_stage IN (
                'reserved',
                'ranking',
                'generation',
                'image',
                'review_delivery',
                'awaiting_review',
                'failed'
            )
        ),

    CONSTRAINT daily_workflow_runs_window_chk
        CHECK (
            window_hours = 24
        ),

    CONSTRAINT daily_workflow_runs_chat_chk
        CHECK (
            target_telegram_chat_id::text
            LIKE '-100%'
        ),

    CONSTRAINT daily_workflow_runs_dates_chk
        CHECK (
            finished_at IS NULL
            OR finished_at >= started_at
        ),

    CONSTRAINT daily_workflow_runs_state_chk
        CHECK (
            (
                workflow_status = 'running'
                AND current_stage IN (
                    'reserved',
                    'ranking',
                    'generation',
                    'image',
                    'review_delivery'
                )
                AND finished_at IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
            )
            OR
            (
                workflow_status = 'awaiting_review'
                AND current_stage = 'awaiting_review'
                AND finished_at IS NOT NULL
                AND error_type IS NULL
                AND error_message IS NULL
            )
            OR
            (
                workflow_status = 'failed'
                AND current_stage = 'failed'
                AND finished_at IS NOT NULL
                AND error_message IS NOT NULL
                AND length(
                    btrim(error_message)
                ) > 0
            )
        )
);


-- Один production daily workflow на дату.
CREATE UNIQUE INDEX daily_workflow_runs_publication_date_uq
    ON top3_news.daily_workflow_runs (
        publication_date
    );


-- Один ranking-run не должен принадлежать двум daily workflows.
CREATE UNIQUE INDEX daily_workflow_runs_ranking_run_uq
    ON top3_news.daily_workflow_runs (
        ranking_run_id
    )
    WHERE ranking_run_id IS NOT NULL;


-- Один batch не должен принадлежать двум daily workflows.
CREATE UNIQUE INDEX daily_workflow_runs_batch_uq
    ON top3_news.daily_workflow_runs (
        batch_id
    )
    WHERE batch_id IS NOT NULL;


-- Одна версия post не должна принадлежать двум daily workflows.
CREATE UNIQUE INDEX daily_workflow_runs_generated_post_uq
    ON top3_news.daily_workflow_runs (
        generated_post_id
    )
    WHERE generated_post_id IS NOT NULL;


-- Один image request не должен принадлежать двум daily workflows.
CREATE UNIQUE INDEX daily_workflow_runs_image_generation_uq
    ON top3_news.daily_workflow_runs (
        image_generation_id
    )
    WHERE image_generation_id IS NOT NULL;


CREATE INDEX daily_workflow_runs_status_date_idx
    ON top3_news.daily_workflow_runs (
        workflow_status,
        publication_date DESC
    );


CREATE INDEX daily_workflow_runs_stage_date_idx
    ON top3_news.daily_workflow_runs (
        current_stage,
        publication_date DESC
    );


CREATE TRIGGER daily_workflow_runs_set_updated_at
BEFORE UPDATE
ON top3_news.daily_workflow_runs
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


COMMENT ON TABLE
    top3_news.daily_workflow_runs
IS
    'Restart-safe верхнеуровневые ежедневные production workflow TOP-3 NEWS';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.publication_date
IS
    'Логическая дата выпуска; для production допускается один daily workflow';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.as_of
IS
    'Зафиксированный UTC cutoff строгого 24-часового ranking window';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.current_stage
IS
    'Последний известный этап автоматического production workflow';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.ranking_run_id
IS
    'Связанный защищённый OpenAI ranking run';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.batch_id
IS
    'Связанный publication batch после text generation';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.generated_post_id
IS
    'Финальный generated post после primary generation и self-review';

COMMENT ON COLUMN
    top3_news.daily_workflow_runs.image_generation_id
IS
    'Связанный защищённый OpenAI image-generation request';


INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '013',
    'Add restart-safe daily workflow runs'
);

COMMIT;
