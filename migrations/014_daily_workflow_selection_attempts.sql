BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 014_daily_workflow_selection_attempts.sql
--
-- Добавляет restart-safe историю TOP-3 combinations, использованных одним
-- daily workflow. История нужна для автоматической замены комбинации после
-- доказанного Image API moderation_blocked.
--
-- Важные свойства:
-- - одна combination используется в workflow не более одного раза;
-- - одновременно может существовать только одна active selection;
-- - replacement образует линейную цепочку от предыдущей selection;
-- - blocked selection сохраняет batch/post/image, на котором произошёл block;
-- - ranking_run_id selection обязан совпадать с ranking combination.
-- ============================================================================


-- ============================================================================
-- 1. История selections одного daily workflow
-- ============================================================================

CREATE TABLE top3_news.daily_workflow_selection_attempts (
    selection_attempt_id          bigint
                                      GENERATED ALWAYS AS IDENTITY
                                      PRIMARY KEY,

    daily_workflow_run_id         bigint NOT NULL
                                      REFERENCES
                                          top3_news.daily_workflow_runs (
                                              daily_workflow_run_id
                                          )
                                      ON DELETE CASCADE,

    ranking_run_id                bigint NOT NULL,

    combination_id                bigint NOT NULL,

    attempt_number                integer NOT NULL,

    selection_kind                text NOT NULL,

    selection_status              text NOT NULL
                                      DEFAULT 'active',

    source_selection_attempt_id   bigint,

    batch_id                      bigint
                                      REFERENCES
                                          top3_news.publication_batches (
                                              batch_id
                                          )
                                      ON DELETE SET NULL,

    generated_post_id             bigint
                                      REFERENCES
                                          top3_news.generated_posts (
                                              generated_post_id
                                          )
                                      ON DELETE SET NULL,

    image_generation_id           bigint
                                      REFERENCES
                                          top3_news.image_generation_requests (
                                              image_generation_id
                                          )
                                      ON DELETE SET NULL,

    ended_at                      timestamptz,

    created_at                    timestamptz NOT NULL
                                      DEFAULT now(),

    updated_at                    timestamptz NOT NULL
                                      DEFAULT now(),

    CONSTRAINT daily_workflow_selection_attempts_combination_run_fk
        FOREIGN KEY (
            combination_id,
            ranking_run_id
        )
        REFERENCES top3_news.ranking_combinations (
            combination_id,
            ranking_run_id
        )
        ON DELETE CASCADE,

    CONSTRAINT daily_workflow_selection_attempts_id_workflow_uq
        UNIQUE (
            selection_attempt_id,
            daily_workflow_run_id
        ),

    CONSTRAINT daily_workflow_selection_attempts_source_workflow_fk
        FOREIGN KEY (
            source_selection_attempt_id,
            daily_workflow_run_id
        )
        REFERENCES top3_news.daily_workflow_selection_attempts (
            selection_attempt_id,
            daily_workflow_run_id
        )
        ON DELETE CASCADE,

    CONSTRAINT daily_workflow_selection_attempts_attempt_chk
        CHECK (
            attempt_number > 0
        ),

    CONSTRAINT daily_workflow_selection_attempts_kind_chk
        CHECK (
            selection_kind IN (
                'winner',
                'replacement'
            )
        ),

    CONSTRAINT daily_workflow_selection_attempts_status_chk
        CHECK (
            selection_status IN (
                'active',
                'moderation_blocked',
                'ready_for_review'
            )
        ),

    CONSTRAINT daily_workflow_selection_attempts_chain_chk
        CHECK (
            (
                selection_kind = 'winner'
                AND attempt_number = 1
                AND source_selection_attempt_id IS NULL
            )
            OR
            (
                selection_kind = 'replacement'
                AND attempt_number > 1
                AND source_selection_attempt_id IS NOT NULL
            )
        ),

    CONSTRAINT daily_workflow_selection_attempts_end_chk
        CHECK (
            (
                selection_status = 'active'
                AND ended_at IS NULL
            )
            OR
            (
                selection_status <> 'active'
                AND ended_at IS NOT NULL
            )
        ),

    CONSTRAINT daily_workflow_selection_attempts_artifact_order_chk
        CHECK (
            (
                generated_post_id IS NULL
                OR batch_id IS NOT NULL
            )
            AND
            (
                image_generation_id IS NULL
                OR (
                    batch_id IS NOT NULL
                    AND generated_post_id IS NOT NULL
                )
            )
        ),

    CONSTRAINT daily_workflow_selection_attempts_workflow_attempt_uq
        UNIQUE (
            daily_workflow_run_id,
            attempt_number
        ),

    CONSTRAINT daily_workflow_selection_attempts_workflow_combination_uq
        UNIQUE (
            daily_workflow_run_id,
            combination_id
        )
);


-- ============================================================================
-- 2. Индексы линейной state machine и artifact provenance
-- ============================================================================

CREATE UNIQUE INDEX daily_workflow_selection_attempts_active_uq
    ON top3_news.daily_workflow_selection_attempts (
        daily_workflow_run_id
    )
    WHERE selection_status = 'active';


CREATE UNIQUE INDEX daily_workflow_selection_attempts_source_uq
    ON top3_news.daily_workflow_selection_attempts (
        source_selection_attempt_id
    )
    WHERE source_selection_attempt_id IS NOT NULL;


CREATE UNIQUE INDEX daily_workflow_selection_attempts_batch_uq
    ON top3_news.daily_workflow_selection_attempts (
        batch_id
    )
    WHERE batch_id IS NOT NULL;


CREATE UNIQUE INDEX daily_workflow_selection_attempts_post_uq
    ON top3_news.daily_workflow_selection_attempts (
        generated_post_id
    )
    WHERE generated_post_id IS NOT NULL;


CREATE UNIQUE INDEX daily_workflow_selection_attempts_image_uq
    ON top3_news.daily_workflow_selection_attempts (
        image_generation_id
    )
    WHERE image_generation_id IS NOT NULL;


CREATE INDEX daily_workflow_selection_attempts_workflow_status_idx
    ON top3_news.daily_workflow_selection_attempts (
        daily_workflow_run_id,
        selection_status,
        attempt_number
    );


CREATE INDEX daily_workflow_selection_attempts_ranking_idx
    ON top3_news.daily_workflow_selection_attempts (
        ranking_run_id,
        combination_id
    );


-- ============================================================================
-- 3. updated_at
-- ============================================================================

CREATE TRIGGER daily_workflow_selection_attempts_set_updated_at
BEFORE UPDATE
ON top3_news.daily_workflow_selection_attempts
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


-- ============================================================================
-- 4. Комментарии
-- ============================================================================

COMMENT ON TABLE top3_news.daily_workflow_selection_attempts IS
    'Restart-safe история ranking combinations, использованных daily workflow';

COMMENT ON COLUMN
    top3_news.daily_workflow_selection_attempts.attempt_number IS
    '1 для исходной winner combination, 2+ для replacement cascade';

COMMENT ON COLUMN
    top3_news.daily_workflow_selection_attempts.source_selection_attempt_id IS
    'Предыдущая selection в линейной replacement chain';

COMMENT ON COLUMN
    top3_news.daily_workflow_selection_attempts.image_generation_id IS
    'Последний значимый image request этой selection: blocking или successful';


-- ============================================================================
-- 5. Права приложения
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE
    ON TABLE
        top3_news.daily_workflow_selection_attempts
    TO top3_news_app;


GRANT USAGE, SELECT, UPDATE
    ON ALL SEQUENCES IN SCHEMA top3_news
    TO top3_news_app;


-- ============================================================================
-- 6. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '014',
    'Add restart-safe daily workflow selection attempts'
);

COMMIT;