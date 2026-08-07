BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 008_generation_revision_requests.sql
--
-- Добавляет защищённый поток повторной генерации текста уже созданного
-- Telegram-поста после редакционного решения changes_required.
--
-- Отдельная запись резервируется до обращения к OpenAI. Уникальный
-- revision_request_key блокирует повторный платный запрос с теми же
-- параметрами доработки.
--
-- Результатом успешной ревизии становится новая версия generated_posts
-- внутри существующего publication_batch:
--
-- предыдущая версия -> superseded
-- новая версия       -> awaiting_review
-- publication_batch  -> awaiting_review
-- ============================================================================


-- ============================================================================
-- 1. Запросы на повторную генерацию текста
-- ============================================================================

CREATE TABLE top3_news.generation_revision_requests (
    generation_revision_id         bigint
                                        GENERATED ALWAYS AS IDENTITY
                                        PRIMARY KEY,

    batch_id                        bigint NOT NULL
                                        REFERENCES top3_news.publication_batches (
                                            batch_id
                                        )
                                        ON DELETE CASCADE,

    source_generated_post_id        bigint NOT NULL
                                        REFERENCES top3_news.generated_posts (
                                            generated_post_id
                                        )
                                        ON DELETE CASCADE,

    review_action_id                bigint NOT NULL
                                        REFERENCES top3_news.review_actions (
                                            review_action_id
                                        )
                                        ON DELETE CASCADE,

    target_version_number           integer NOT NULL,

    revision_request_key            text NOT NULL,
    request_key_version             text NOT NULL,

    revision_status                 text NOT NULL DEFAULT 'reserved',
    requested_action                text NOT NULL DEFAULT 'regenerate_text',

    editorial_comment               text NOT NULL,
    issues                          jsonb NOT NULL DEFAULT '[]'::jsonb,

    model_name                      text NOT NULL,
    generator_version               text NOT NULL,
    prompt_version                  text NOT NULL,
    text_format                     text NOT NULL,

    request_payload                 jsonb NOT NULL DEFAULT '{}'::jsonb,

    openai_usage                    jsonb,
    openai_cost                     jsonb,

    generated_post_id               bigint
                                        REFERENCES top3_news.generated_posts (
                                            generated_post_id
                                        )
                                        ON DELETE SET NULL,

    error_type                      text,
    error_message                   text,

    reserved_at                     timestamptz NOT NULL DEFAULT now(),
    completed_at                    timestamptz,
    failed_at                       timestamptz,

    created_at                      timestamptz NOT NULL DEFAULT now(),
    updated_at                      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT generation_revision_requests_target_version_chk
        CHECK (target_version_number > 1),

    CONSTRAINT generation_revision_requests_key_chk
        CHECK (
            revision_request_key
            ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT generation_revision_requests_key_version_chk
        CHECK (
            btrim(request_key_version) <> ''
        ),

    CONSTRAINT generation_revision_requests_status_chk
        CHECK (
            revision_status IN (
                'reserved',
                'completed',
                'failed'
            )
        ),

    CONSTRAINT generation_revision_requests_action_chk
        CHECK (
            requested_action = 'regenerate_text'
        ),

    CONSTRAINT generation_revision_requests_comment_chk
        CHECK (
            btrim(editorial_comment) <> ''
        ),

    CONSTRAINT generation_revision_requests_issues_array_chk
        CHECK (
            jsonb_typeof(issues) = 'array'
            AND jsonb_array_length(issues) > 0
        ),

    CONSTRAINT generation_revision_requests_payload_object_chk
        CHECK (
            jsonb_typeof(request_payload) = 'object'
        ),

    CONSTRAINT generation_revision_requests_usage_object_chk
        CHECK (
            openai_usage IS NULL
            OR jsonb_typeof(openai_usage) = 'object'
        ),

    CONSTRAINT generation_revision_requests_cost_object_chk
        CHECK (
            openai_cost IS NULL
            OR jsonb_typeof(openai_cost) = 'object'
        ),

    CONSTRAINT generation_revision_requests_state_chk
        CHECK (
            (
                revision_status = 'reserved'
                AND generated_post_id IS NULL
                AND openai_usage IS NULL
                AND openai_cost IS NULL
                AND completed_at IS NULL
                AND failed_at IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
            )
            OR
            (
                revision_status = 'completed'
                AND generated_post_id IS NOT NULL
                AND openai_usage IS NOT NULL
                AND openai_cost IS NOT NULL
                AND completed_at IS NOT NULL
                AND failed_at IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
            )
            OR
            (
                revision_status = 'failed'
                AND generated_post_id IS NULL
                AND completed_at IS NULL
                AND failed_at IS NOT NULL
                AND error_type IS NOT NULL
                AND btrim(error_type) <> ''
                AND error_message IS NOT NULL
                AND btrim(error_message) <> ''
            )
        )
);


-- ============================================================================
-- 2. Ограничения идемпотентности и конкурентного выполнения
-- ============================================================================

CREATE UNIQUE INDEX generation_revision_requests_key_uq
    ON top3_news.generation_revision_requests (
        revision_request_key
    );

CREATE UNIQUE INDEX generation_revision_requests_review_action_active_uq
    ON top3_news.generation_revision_requests (
        review_action_id
    )
    WHERE revision_status IN (
        'reserved',
        'completed'
    );

CREATE UNIQUE INDEX generation_revision_requests_batch_version_active_uq
    ON top3_news.generation_revision_requests (
        batch_id,
        target_version_number
    )
    WHERE revision_status IN (
        'reserved',
        'completed'
    );

CREATE UNIQUE INDEX generation_revision_requests_generated_post_uq
    ON top3_news.generation_revision_requests (
        generated_post_id
    )
    WHERE generated_post_id IS NOT NULL;


-- ============================================================================
-- 3. Индексы операционного поиска
-- ============================================================================

CREATE INDEX generation_revision_requests_status_reserved_idx
    ON top3_news.generation_revision_requests (
        revision_status,
        reserved_at
    );

CREATE INDEX generation_revision_requests_source_post_idx
    ON top3_news.generation_revision_requests (
        source_generated_post_id
    );

CREATE INDEX generation_revision_requests_batch_idx
    ON top3_news.generation_revision_requests (
        batch_id,
        target_version_number DESC
    );


-- ============================================================================
-- 4. Комментарии
-- ============================================================================

COMMENT ON TABLE
    top3_news.generation_revision_requests
IS
    'Защищённые платные запросы повторной генерации текста после changes_required';

COMMENT ON COLUMN
    top3_news.generation_revision_requests.revision_request_key
IS
    'SHA-256 ключ конкретной редакционной ревизии; блокирует повторный платный запрос';

COMMENT ON COLUMN
    top3_news.generation_revision_requests.source_generated_post_id
IS
    'Версия generated_posts, отправленная редактором на доработку';

COMMENT ON COLUMN
    top3_news.generation_revision_requests.target_version_number
IS
    'Номер новой версии generated_posts, создаваемой после успешной ревизии';

COMMENT ON COLUMN
    top3_news.generation_revision_requests.generated_post_id
IS
    'Новая версия generated_posts, созданная после успешного завершения ревизии';


-- ============================================================================
-- 5. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '008',
    'Add protected Telegram post generation revision requests'
);

COMMIT;