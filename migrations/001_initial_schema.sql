BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 001_initial_schema.sql
--
-- Начальная схема проекта TOP 3 NEWS.
--
-- Владелец объектов: michael_psql
-- Пользователь приложения: top3_news_app
-- ============================================================================


-- ============================================================================
-- 1. История миграций
-- ============================================================================

CREATE TABLE top3_news.schema_migrations (
    version         text PRIMARY KEY,
    description     text NOT NULL,
    applied_at      timestamptz NOT NULL DEFAULT now(),
    applied_by      text NOT NULL DEFAULT current_user
);


-- ============================================================================
-- 2. Пользователи Telegram, которым разрешено работать с ботом
-- ============================================================================

CREATE TABLE top3_news.bot_users (
    telegram_user_id    bigint PRIMARY KEY,
    telegram_username   text,
    display_name        text NOT NULL,
    user_role           text NOT NULL DEFAULT 'reviewer',
    is_active           boolean NOT NULL DEFAULT true,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT bot_users_role_chk
        CHECK (user_role IN ('admin', 'reviewer', 'viewer'))
);

CREATE UNIQUE INDEX bot_users_username_uq
    ON top3_news.bot_users (lower(telegram_username))
    WHERE telegram_username IS NOT NULL;


-- ============================================================================
-- 3. Источники киноновостей
-- ============================================================================

CREATE TABLE top3_news.sources (
    source_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_code         text NOT NULL,
    source_name         text NOT NULL,
    source_type         text NOT NULL,
    base_url            text,
    feed_url            text,
    default_language    text NOT NULL DEFAULT 'ru',
    is_active           boolean NOT NULL DEFAULT true,
    collection_priority integer NOT NULL DEFAULT 100,
    settings            jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT sources_code_uq
        UNIQUE (source_code),

    CONSTRAINT sources_type_chk
        CHECK (
            source_type IN (
                'rss',
                'api',
                'web',
                'telegram',
                'manual'
            )
        ),

    CONSTRAINT sources_priority_chk
        CHECK (collection_priority >= 0),

    CONSTRAINT sources_settings_object_chk
        CHECK (jsonb_typeof(settings) = 'object')
);


-- ============================================================================
-- 4. Запуски сборщиков новостей
-- ============================================================================

CREATE TABLE top3_news.collection_runs (
    collection_run_id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id           bigint REFERENCES top3_news.sources (source_id)
                            ON DELETE SET NULL,
    run_status          text NOT NULL DEFAULT 'running',
    collector_name      text NOT NULL,
    collector_version   text,
    started_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    fetched_count       integer NOT NULL DEFAULT 0,
    inserted_count      integer NOT NULL DEFAULT 0,
    duplicate_count     integer NOT NULL DEFAULT 0,
    rejected_count      integer NOT NULL DEFAULT 0,
    error_message       text,
    run_metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT collection_runs_status_chk
        CHECK (
            run_status IN (
                'running',
                'completed',
                'completed_with_errors',
                'failed'
            )
        ),

    CONSTRAINT collection_runs_counts_chk
        CHECK (
            fetched_count >= 0
            AND inserted_count >= 0
            AND duplicate_count >= 0
            AND rejected_count >= 0
        ),

    CONSTRAINT collection_runs_dates_chk
        CHECK (
            finished_at IS NULL
            OR finished_at >= started_at
        ),

    CONSTRAINT collection_runs_metadata_object_chk
        CHECK (jsonb_typeof(run_metadata) = 'object')
);

CREATE INDEX collection_runs_source_started_idx
    ON top3_news.collection_runs (source_id, started_at DESC);

CREATE INDEX collection_runs_status_idx
    ON top3_news.collection_runs (run_status, started_at DESC);


-- ============================================================================
-- 5. Собранные новости
-- ============================================================================

CREATE TABLE top3_news.news_items (
    news_id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id               bigint NOT NULL
                                REFERENCES top3_news.sources (source_id)
                                ON DELETE RESTRICT,
    collection_run_id       bigint
                                REFERENCES top3_news.collection_runs (
                                    collection_run_id
                                )
                                ON DELETE SET NULL,

    external_id             text,
    source_url              text NOT NULL,
    canonical_url           text,
    url_sha256              char(64),
    content_sha256          char(64),

    raw_title               text NOT NULL,
    normalized_title        text,
    raw_summary             text,
    normalized_summary      text,
    article_text            text,
    author_name             text,

    source_published_at     timestamptz,
    collected_at            timestamptz NOT NULL DEFAULT now(),

    language_code           text NOT NULL DEFAULT 'ru',

    primary_image_url       text,
    primary_image_path      text,
    image_credit            text,

    processing_status       text NOT NULL DEFAULT 'collected',
    duplicate_of_news_id    bigint
                                REFERENCES top3_news.news_items (news_id)
                                ON DELETE SET NULL,
    rejection_reason        text,

    raw_payload             jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata                jsonb NOT NULL DEFAULT '{}'::jsonb,

    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT news_items_status_chk
        CHECK (
            processing_status IN (
                'collected',
                'normalized',
                'candidate',
                'duplicate',
                'excluded',
                'archived'
            )
        ),

    CONSTRAINT news_items_url_sha256_chk
        CHECK (
            url_sha256 IS NULL
            OR url_sha256 ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT news_items_content_sha256_chk
        CHECK (
            content_sha256 IS NULL
            OR content_sha256 ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT news_items_not_self_duplicate_chk
        CHECK (
            duplicate_of_news_id IS NULL
            OR duplicate_of_news_id <> news_id
        ),

    CONSTRAINT news_items_raw_payload_object_chk
        CHECK (jsonb_typeof(raw_payload) = 'object'),

    CONSTRAINT news_items_metadata_object_chk
        CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE UNIQUE INDEX news_items_source_external_id_uq
    ON top3_news.news_items (source_id, external_id)
    WHERE external_id IS NOT NULL;

CREATE UNIQUE INDEX news_items_url_sha256_uq
    ON top3_news.news_items (url_sha256)
    WHERE url_sha256 IS NOT NULL;

CREATE INDEX news_items_source_published_idx
    ON top3_news.news_items (
        source_id,
        source_published_at DESC
    );

CREATE INDEX news_items_collection_run_idx
    ON top3_news.news_items (collection_run_id);

CREATE INDEX news_items_status_published_idx
    ON top3_news.news_items (
        processing_status,
        source_published_at DESC
    );

CREATE INDEX news_items_duplicate_of_idx
    ON top3_news.news_items (duplicate_of_news_id)
    WHERE duplicate_of_news_id IS NOT NULL;

CREATE INDEX news_items_content_sha256_idx
    ON top3_news.news_items (content_sha256)
    WHERE content_sha256 IS NOT NULL;


-- ============================================================================
-- 6. Запуски оценки и ранжирования новостей
-- ============================================================================

CREATE TABLE top3_news.ranking_runs (
    ranking_run_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_status          text NOT NULL DEFAULT 'running',
    formula_version     text NOT NULL,
    model_name          text,
    prompt_version      text,

    window_started_at   timestamptz NOT NULL,
    window_finished_at  timestamptz NOT NULL,

    candidate_count     integer NOT NULL DEFAULT 0,
    scored_count        integer NOT NULL DEFAULT 0,
    eligible_count      integer NOT NULL DEFAULT 0,

    parameters          jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_message       text,

    started_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ranking_runs_status_chk
        CHECK (
            run_status IN (
                'running',
                'completed',
                'completed_with_errors',
                'failed'
            )
        ),

    CONSTRAINT ranking_runs_window_chk
        CHECK (window_finished_at > window_started_at),

    CONSTRAINT ranking_runs_counts_chk
        CHECK (
            candidate_count >= 0
            AND scored_count >= 0
            AND eligible_count >= 0
            AND scored_count <= candidate_count
            AND eligible_count <= scored_count
        ),

    CONSTRAINT ranking_runs_dates_chk
        CHECK (
            finished_at IS NULL
            OR finished_at >= started_at
        ),

    CONSTRAINT ranking_runs_parameters_object_chk
        CHECK (jsonb_typeof(parameters) = 'object')
);

CREATE INDEX ranking_runs_status_started_idx
    ON top3_news.ranking_runs (run_status, started_at DESC);

CREATE INDEX ranking_runs_window_idx
    ON top3_news.ranking_runs (
        window_started_at,
        window_finished_at
    );


-- ============================================================================
-- 7. Оценки отдельных новостей
--
-- Зафиксированная формула:
--
-- B_i = 0.20 F_i
--     + 0.30 M_i
--     + 0.20 R_i
--     + 0.15 (H_i × Q_i)
--
-- Значения F, M, R, H, Q сохраняются отдельно.
-- Итоговый individual_score рассчитывается PostgreSQL автоматически.
-- ============================================================================

CREATE TABLE top3_news.news_scores (
    score_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ranking_run_id      bigint NOT NULL
                            REFERENCES top3_news.ranking_runs (
                                ranking_run_id
                            )
                            ON DELETE CASCADE,
    news_id             bigint NOT NULL
                            REFERENCES top3_news.news_items (news_id)
                            ON DELETE CASCADE,

    f_score             numeric(12, 6) NOT NULL,
    m_score             numeric(12, 6) NOT NULL,
    r_score             numeric(12, 6) NOT NULL,
    h_score             numeric(12, 6) NOT NULL,
    q_score             numeric(12, 6) NOT NULL,

    individual_score    numeric(20, 6)
                            GENERATED ALWAYS AS (
                                0.20 * f_score
                                + 0.30 * m_score
                                + 0.20 * r_score
                                + 0.15 * (h_score * q_score)
                            ) STORED,

    is_eligible         boolean NOT NULL DEFAULT true,
    exclusion_reason    text,
    rank_position       integer,

    score_explanation   text,
    score_details       jsonb NOT NULL DEFAULT '{}'::jsonb,

    scored_at           timestamptz NOT NULL DEFAULT now(),
    created_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT news_scores_run_news_uq
        UNIQUE (ranking_run_id, news_id),

    CONSTRAINT news_scores_values_chk
        CHECK (
            f_score >= 0
            AND m_score >= 0
            AND r_score >= 0
            AND h_score >= 0
            AND q_score >= 0
        ),

    CONSTRAINT news_scores_rank_chk
        CHECK (
            rank_position IS NULL
            OR rank_position > 0
        ),

    CONSTRAINT news_scores_exclusion_chk
        CHECK (
            is_eligible = true
            OR exclusion_reason IS NOT NULL
        ),

    CONSTRAINT news_scores_details_object_chk
        CHECK (jsonb_typeof(score_details) = 'object')
);

CREATE INDEX news_scores_news_idx
    ON top3_news.news_scores (news_id);

CREATE INDEX news_scores_ranking_score_idx
    ON top3_news.news_scores (
        ranking_run_id,
        individual_score DESC
    )
    WHERE is_eligible = true;

CREATE INDEX news_scores_ranking_rank_idx
    ON top3_news.news_scores (
        ranking_run_id,
        rank_position
    )
    WHERE rank_position IS NOT NULL;


COMMENT ON COLUMN top3_news.news_scores.f_score
    IS 'Значение F_i в формуле индивидуального рейтинга новости';

COMMENT ON COLUMN top3_news.news_scores.m_score
    IS 'Значение M_i в формуле индивидуального рейтинга новости';

COMMENT ON COLUMN top3_news.news_scores.r_score
    IS 'Значение R_i в формуле индивидуального рейтинга новости';

COMMENT ON COLUMN top3_news.news_scores.h_score
    IS 'Значение H_i в формуле индивидуального рейтинга новости';

COMMENT ON COLUMN top3_news.news_scores.q_score
    IS 'Значение Q_i в формуле индивидуального рейтинга новости';

COMMENT ON COLUMN top3_news.news_scores.individual_score
    IS 'B_i = 0.20F_i + 0.30M_i + 0.20R_i + 0.15(H_i × Q_i)';


-- ============================================================================
-- 8. Ежедневные подборки TOP-3
-- ============================================================================

CREATE TABLE top3_news.publication_batches (
    batch_id                        bigint
                                        GENERATED ALWAYS AS IDENTITY
                                        PRIMARY KEY,
    publication_date               date NOT NULL,
    edition                        smallint NOT NULL DEFAULT 1,
    ranking_run_id                 bigint
                                        REFERENCES top3_news.ranking_runs (
                                            ranking_run_id
                                        )
                                        ON DELETE SET NULL,

    batch_status                    text NOT NULL DEFAULT 'draft',

    target_telegram_chat_id         bigint,
    target_channel_username         text,

    scheduled_at                    timestamptz,
    approved_at                     timestamptz,
    published_at                    timestamptz,

    approved_by_telegram_user_id    bigint
                                        REFERENCES top3_news.bot_users (
                                            telegram_user_id
                                        )
                                        ON DELETE SET NULL,

    error_message                   text,
    metadata                        jsonb NOT NULL DEFAULT '{}'::jsonb,

    created_at                      timestamptz NOT NULL DEFAULT now(),
    updated_at                      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT publication_batches_date_edition_uq
        UNIQUE (publication_date, edition),

    CONSTRAINT publication_batches_edition_chk
        CHECK (edition > 0),

    CONSTRAINT publication_batches_status_chk
        CHECK (
            batch_status IN (
                'draft',
                'ranked',
                'generated',
                'awaiting_review',
                'approved',
                'rejected',
                'publishing',
                'published',
                'failed'
            )
        ),

    CONSTRAINT publication_batches_metadata_object_chk
        CHECK (jsonb_typeof(metadata) = 'object')
);

CREATE INDEX publication_batches_ranking_run_idx
    ON top3_news.publication_batches (ranking_run_id);

CREATE INDEX publication_batches_status_date_idx
    ON top3_news.publication_batches (
        batch_status,
        publication_date DESC
    );

CREATE INDEX publication_batches_approver_idx
    ON top3_news.publication_batches (
        approved_by_telegram_user_id
    )
    WHERE approved_by_telegram_user_id IS NOT NULL;


-- ============================================================================
-- 9. Новости, выбранные в конкретную подборку
-- ============================================================================

CREATE TABLE top3_news.batch_items (
    batch_item_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id            bigint NOT NULL
                            REFERENCES top3_news.publication_batches (
                                batch_id
                            )
                            ON DELETE CASCADE,
    news_id             bigint NOT NULL
                            REFERENCES top3_news.news_items (news_id)
                            ON DELETE RESTRICT,
    score_id            bigint
                            REFERENCES top3_news.news_scores (score_id)
                            ON DELETE SET NULL,

    position            smallint NOT NULL,
    selection_reason    text,
    created_at          timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT batch_items_batch_position_uq
        UNIQUE (batch_id, position),

    CONSTRAINT batch_items_batch_news_uq
        UNIQUE (batch_id, news_id),

    CONSTRAINT batch_items_position_chk
        CHECK (position BETWEEN 1 AND 3)
);

CREATE INDEX batch_items_news_idx
    ON top3_news.batch_items (news_id);

CREATE INDEX batch_items_score_idx
    ON top3_news.batch_items (score_id)
    WHERE score_id IS NOT NULL;


-- ============================================================================
-- 10. Сгенерированные версии Telegram-поста
-- ============================================================================

CREATE TABLE top3_news.generated_posts (
    generated_post_id       bigint
                                GENERATED ALWAYS AS IDENTITY
                                PRIMARY KEY,
    batch_id                bigint NOT NULL
                                REFERENCES top3_news.publication_batches (
                                    batch_id
                                )
                                ON DELETE CASCADE,

    version_number          integer NOT NULL,
    post_status             text NOT NULL DEFAULT 'draft',

    post_text               text NOT NULL,
    text_format             text NOT NULL DEFAULT 'markdown',

    image_path              text,
    image_sha256            char(64),
    image_prompt            text,

    text_model_name         text,
    image_model_name        text,
    text_prompt_version     text,
    image_prompt_version    text,

    generation_metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,

    generated_at            timestamptz NOT NULL DEFAULT now(),
    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT generated_posts_batch_version_uq
        UNIQUE (batch_id, version_number),

    CONSTRAINT generated_posts_version_chk
        CHECK (version_number > 0),

    CONSTRAINT generated_posts_status_chk
        CHECK (
            post_status IN (
                'draft',
                'awaiting_review',
                'approved',
                'rejected',
                'superseded',
                'published',
                'failed'
            )
        ),

    CONSTRAINT generated_posts_format_chk
        CHECK (
            text_format IN (
                'markdown',
                'markdown_v2',
                'html',
                'plain_text'
            )
        ),

    CONSTRAINT generated_posts_image_sha256_chk
        CHECK (
            image_sha256 IS NULL
            OR image_sha256 ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT generated_posts_metadata_object_chk
        CHECK (jsonb_typeof(generation_metadata) = 'object')
);

CREATE INDEX generated_posts_batch_status_idx
    ON top3_news.generated_posts (
        batch_id,
        post_status,
        version_number DESC
    );


-- ============================================================================
-- 11. Ручные и AI-проверки постов
-- ============================================================================

CREATE TABLE top3_news.review_actions (
    review_action_id               bigint
                                        GENERATED ALWAYS AS IDENTITY
                                        PRIMARY KEY,
    generated_post_id              bigint NOT NULL
                                        REFERENCES top3_news.generated_posts (
                                            generated_post_id
                                        )
                                        ON DELETE CASCADE,

    reviewer_type                  text NOT NULL,
    reviewer_telegram_user_id      bigint
                                        REFERENCES top3_news.bot_users (
                                            telegram_user_id
                                        )
                                        ON DELETE SET NULL,
    reviewer_model_name            text,

    decision                       text NOT NULL,
    requested_action               text,

    factual_consistency            numeric(5, 4),
    source_support                 numeric(5, 4),
    duplicate_risk                 numeric(5, 4),
    style_score                    numeric(5, 4),
    confidence_score               numeric(5, 4),

    requires_human_review          boolean NOT NULL DEFAULT true,

    comment_text                   text,
    issues                         jsonb NOT NULL DEFAULT '[]'::jsonb,
    review_details                 jsonb NOT NULL DEFAULT '{}'::jsonb,

    reviewed_at                    timestamptz NOT NULL DEFAULT now(),
    created_at                     timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT review_actions_reviewer_type_chk
        CHECK (
            reviewer_type IN (
                'human',
                'ai'
            )
        ),

    CONSTRAINT review_actions_decision_chk
        CHECK (
            decision IN (
                'approve',
                'reject',
                'changes_required',
                'comment'
            )
        ),

    CONSTRAINT review_actions_requested_action_chk
        CHECK (
            requested_action IS NULL
            OR requested_action IN (
                'regenerate_text',
                'regenerate_image',
                'replace_news'
            )
        ),

    CONSTRAINT review_actions_reviewer_identity_chk
        CHECK (
            (
                reviewer_type = 'human'
                AND reviewer_telegram_user_id IS NOT NULL
            )
            OR
            (
                reviewer_type = 'ai'
                AND reviewer_model_name IS NOT NULL
            )
        ),

    CONSTRAINT review_actions_scores_chk
        CHECK (
            (factual_consistency IS NULL
                OR factual_consistency BETWEEN 0 AND 1)
            AND
            (source_support IS NULL
                OR source_support BETWEEN 0 AND 1)
            AND
            (duplicate_risk IS NULL
                OR duplicate_risk BETWEEN 0 AND 1)
            AND
            (style_score IS NULL
                OR style_score BETWEEN 0 AND 1)
            AND
            (confidence_score IS NULL
                OR confidence_score BETWEEN 0 AND 1)
        ),

    CONSTRAINT review_actions_issues_array_chk
        CHECK (jsonb_typeof(issues) = 'array'),

    CONSTRAINT review_actions_details_object_chk
        CHECK (jsonb_typeof(review_details) = 'object')
);

CREATE INDEX review_actions_post_reviewed_idx
    ON top3_news.review_actions (
        generated_post_id,
        reviewed_at DESC
    );

CREATE INDEX review_actions_human_reviewer_idx
    ON top3_news.review_actions (
        reviewer_telegram_user_id,
        reviewed_at DESC
    )
    WHERE reviewer_telegram_user_id IS NOT NULL;


-- ============================================================================
-- 12. Попытки публикации в Telegram
-- ============================================================================

CREATE TABLE top3_news.publication_attempts (
    publication_attempt_id bigint
                                GENERATED ALWAYS AS IDENTITY
                                PRIMARY KEY,
    generated_post_id      bigint NOT NULL
                                REFERENCES top3_news.generated_posts (
                                    generated_post_id
                                )
                                ON DELETE CASCADE,

    attempt_number         integer NOT NULL,
    attempt_status         text NOT NULL DEFAULT 'started',

    telegram_chat_id       bigint NOT NULL,
    telegram_message_id    bigint,
    telegram_media_group_id text,

    request_payload        jsonb NOT NULL DEFAULT '{}'::jsonb,
    response_payload       jsonb NOT NULL DEFAULT '{}'::jsonb,

    telegram_error_code    integer,
    error_message          text,

    started_at             timestamptz NOT NULL DEFAULT now(),
    finished_at            timestamptz,
    created_at             timestamptz NOT NULL DEFAULT now(),
    updated_at             timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT publication_attempts_post_number_uq
        UNIQUE (generated_post_id, attempt_number),

    CONSTRAINT publication_attempts_number_chk
        CHECK (attempt_number > 0),

    CONSTRAINT publication_attempts_status_chk
        CHECK (
            attempt_status IN (
                'started',
                'published',
                'failed',
                'unknown'
            )
        ),

    CONSTRAINT publication_attempts_dates_chk
        CHECK (
            finished_at IS NULL
            OR finished_at >= started_at
        ),

    CONSTRAINT publication_attempts_request_object_chk
        CHECK (jsonb_typeof(request_payload) = 'object'),

    CONSTRAINT publication_attempts_response_object_chk
        CHECK (jsonb_typeof(response_payload) = 'object')
);

CREATE INDEX publication_attempts_post_started_idx
    ON top3_news.publication_attempts (
        generated_post_id,
        started_at DESC
    );

CREATE INDEX publication_attempts_status_started_idx
    ON top3_news.publication_attempts (
        attempt_status,
        started_at DESC
    );


-- ============================================================================
-- 13. Автоматическое обновление updated_at
-- ============================================================================

CREATE FUNCTION top3_news.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;


CREATE TRIGGER bot_users_set_updated_at
BEFORE UPDATE ON top3_news.bot_users
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER sources_set_updated_at
BEFORE UPDATE ON top3_news.sources
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER collection_runs_set_updated_at
BEFORE UPDATE ON top3_news.collection_runs
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER news_items_set_updated_at
BEFORE UPDATE ON top3_news.news_items
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER ranking_runs_set_updated_at
BEFORE UPDATE ON top3_news.ranking_runs
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER publication_batches_set_updated_at
BEFORE UPDATE ON top3_news.publication_batches
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER generated_posts_set_updated_at
BEFORE UPDATE ON top3_news.generated_posts
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


CREATE TRIGGER publication_attempts_set_updated_at
BEFORE UPDATE ON top3_news.publication_attempts
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


-- ============================================================================
-- 14. Права пользователя приложения
-- ============================================================================

GRANT CONNECT
    ON DATABASE top3_news_db
    TO top3_news_app;

GRANT USAGE
    ON SCHEMA top3_news
    TO top3_news_app;

GRANT SELECT
    ON TABLE top3_news.schema_migrations
    TO top3_news_app;

GRANT SELECT, INSERT, UPDATE, DELETE
    ON TABLE
        top3_news.bot_users,
        top3_news.sources,
        top3_news.collection_runs,
        top3_news.news_items,
        top3_news.ranking_runs,
        top3_news.news_scores,
        top3_news.publication_batches,
        top3_news.batch_items,
        top3_news.generated_posts,
        top3_news.review_actions,
        top3_news.publication_attempts
    TO top3_news_app;

GRANT USAGE, SELECT, UPDATE
    ON ALL SEQUENCES IN SCHEMA top3_news
    TO top3_news_app;

GRANT EXECUTE
    ON FUNCTION top3_news.set_updated_at()
    TO top3_news_app;


-- ============================================================================
-- 15. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '001',
    'Initial schema for TOP 3 NEWS Telegram bot'
);


COMMIT;