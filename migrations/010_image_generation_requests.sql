BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 010_image_generation_requests.sql
--
-- Добавляет защищённый поток платной генерации единой PNG-иллюстрации
-- для TOP-3 киноновостей.
--
-- Отдельная запись резервируется до обращения к OpenAI Image API.
-- Детерминированный image_request_key блокирует случайный повтор
-- платного запроса с теми же параметрами.
--
-- Поддерживаются два типа запроса:
--
-- initial      -> первая автоматическая генерация изображения;
-- regenerate   -> повторная генерация после редакционного
--                 requested_action='regenerate_image'.
--
-- Успешный результат не создаёт новую версию generated_posts.
-- Он записывается в image-поля существующей версии generated_posts.
--
-- Идемпотентность:
--
-- reserved  -> второй платный вызов с тем же ключом запрещён;
-- completed -> второй платный вызов с тем же ключом запрещён;
-- failed    -> новая попытка с тем же ключом разрешена.
-- ============================================================================


-- ============================================================================
-- 1. Запросы на генерацию изображения
-- ============================================================================

CREATE TABLE top3_news.image_generation_requests (
    image_generation_id             bigint
                                        GENERATED ALWAYS AS IDENTITY
                                        PRIMARY KEY,

    batch_id                        bigint NOT NULL
                                        REFERENCES top3_news.publication_batches (
                                            batch_id
                                        )
                                        ON DELETE CASCADE,

    generated_post_id               bigint NOT NULL
                                        REFERENCES top3_news.generated_posts (
                                            generated_post_id
                                        )
                                        ON DELETE CASCADE,

    review_action_id                bigint
                                        REFERENCES top3_news.review_actions (
                                            review_action_id
                                        )
                                        ON DELETE CASCADE,

    image_request_key               text NOT NULL,
    request_key_version             text NOT NULL,

    image_status                    text NOT NULL DEFAULT 'reserved',
    request_kind                    text NOT NULL,

    editorial_comment               text,
    issues                          jsonb NOT NULL DEFAULT '[]'::jsonb,

    model_name                      text NOT NULL,
    generator_version               text NOT NULL,
    prompt_version                  text NOT NULL,

    image_size                      text NOT NULL,
    image_quality                   text NOT NULL,
    output_format                   text NOT NULL,
    background                      text NOT NULL,
    moderation                      text NOT NULL,
    image_count                     integer NOT NULL,

    request_payload                 jsonb NOT NULL DEFAULT '{}'::jsonb,

    response_metadata               jsonb,

    openai_usage                    jsonb,
    openai_cost                     jsonb,

    image_path                      text,
    image_sha256                    char(64),

    error_type                      text,
    error_message                   text,

    reserved_at                     timestamptz NOT NULL DEFAULT now(),
    completed_at                    timestamptz,
    failed_at                       timestamptz,

    created_at                      timestamptz NOT NULL DEFAULT now(),
    updated_at                      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT image_generation_requests_key_chk
        CHECK (
            image_request_key
            ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT image_generation_requests_key_version_chk
        CHECK (
            btrim(request_key_version) <> ''
        ),

    CONSTRAINT image_generation_requests_status_chk
        CHECK (
            image_status IN (
                'reserved',
                'completed',
                'failed'
            )
        ),

    CONSTRAINT image_generation_requests_kind_chk
        CHECK (
            request_kind IN (
                'initial',
                'regenerate'
            )
        ),

    CONSTRAINT image_generation_requests_context_chk
        CHECK (
            (
                request_kind = 'initial'
                AND review_action_id IS NULL
                AND editorial_comment IS NULL
                AND jsonb_typeof(issues) = 'array'
                AND jsonb_array_length(issues) = 0
            )
            OR
            (
                request_kind = 'regenerate'
                AND review_action_id IS NOT NULL
                AND editorial_comment IS NOT NULL
                AND btrim(editorial_comment) <> ''
                AND jsonb_typeof(issues) = 'array'
                AND jsonb_array_length(issues) > 0
            )
        ),

    CONSTRAINT image_generation_requests_model_chk
        CHECK (
            btrim(model_name) <> ''
        ),

    CONSTRAINT image_generation_requests_generator_version_chk
        CHECK (
            btrim(generator_version) <> ''
        ),

    CONSTRAINT image_generation_requests_prompt_version_chk
        CHECK (
            btrim(prompt_version) <> ''
        ),

    CONSTRAINT image_generation_requests_size_chk
        CHECK (
            image_size
            ~ '^[1-9][0-9]*x[1-9][0-9]*$'
        ),

    CONSTRAINT image_generation_requests_quality_chk
        CHECK (
            image_quality IN (
                'low',
                'medium',
                'high'
            )
        ),

    CONSTRAINT image_generation_requests_output_format_chk
        CHECK (
            output_format = 'png'
        ),

    CONSTRAINT image_generation_requests_background_chk
        CHECK (
            background = 'opaque'
        ),

    CONSTRAINT image_generation_requests_moderation_chk
        CHECK (
            moderation = 'auto'
        ),

    CONSTRAINT image_generation_requests_count_chk
        CHECK (
            image_count = 1
        ),

    CONSTRAINT image_generation_requests_payload_object_chk
        CHECK (
            jsonb_typeof(request_payload) = 'object'
        ),

    CONSTRAINT image_generation_requests_response_object_chk
        CHECK (
            response_metadata IS NULL
            OR jsonb_typeof(response_metadata) = 'object'
        ),

    CONSTRAINT image_generation_requests_usage_object_chk
        CHECK (
            openai_usage IS NULL
            OR jsonb_typeof(openai_usage) = 'object'
        ),

    CONSTRAINT image_generation_requests_cost_object_chk
        CHECK (
            openai_cost IS NULL
            OR jsonb_typeof(openai_cost) = 'object'
        ),

    CONSTRAINT image_generation_requests_usage_cost_pair_chk
        CHECK (
            (
                openai_usage IS NULL
                AND openai_cost IS NULL
            )
            OR
            (
                openai_usage IS NOT NULL
                AND openai_cost IS NOT NULL
            )
        ),

    CONSTRAINT image_generation_requests_path_chk
        CHECK (
            image_path IS NULL
            OR btrim(image_path) <> ''
        ),

    CONSTRAINT image_generation_requests_sha256_chk
        CHECK (
            image_sha256 IS NULL
            OR image_sha256 ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT image_generation_requests_state_chk
        CHECK (
            (
                image_status = 'reserved'
                AND response_metadata IS NULL
                AND openai_usage IS NULL
                AND openai_cost IS NULL
                AND image_path IS NULL
                AND image_sha256 IS NULL
                AND completed_at IS NULL
                AND failed_at IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
            )
            OR
            (
                image_status = 'completed'
                AND response_metadata IS NOT NULL
                AND image_path IS NOT NULL
                AND btrim(image_path) <> ''
                AND image_sha256 IS NOT NULL
                AND completed_at IS NOT NULL
                AND failed_at IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
            )
            OR
            (
                image_status = 'failed'
                AND image_path IS NULL
                AND image_sha256 IS NULL
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

CREATE UNIQUE INDEX image_generation_requests_key_active_uq
    ON top3_news.image_generation_requests (
        image_request_key
    )
    WHERE image_status IN (
        'reserved',
        'completed'
    );

CREATE UNIQUE INDEX image_generation_requests_initial_batch_active_uq
    ON top3_news.image_generation_requests (
        batch_id
    )
    WHERE request_kind = 'initial'
      AND image_status IN (
          'reserved',
          'completed'
      );

CREATE UNIQUE INDEX image_generation_requests_review_action_active_uq
    ON top3_news.image_generation_requests (
        review_action_id
    )
    WHERE request_kind = 'regenerate'
      AND image_status IN (
          'reserved',
          'completed'
      );

CREATE UNIQUE INDEX image_generation_requests_post_reserved_uq
    ON top3_news.image_generation_requests (
        generated_post_id
    )
    WHERE image_status = 'reserved';


-- ============================================================================
-- 3. Индексы операционного поиска
-- ============================================================================

CREATE INDEX image_generation_requests_status_reserved_idx
    ON top3_news.image_generation_requests (
        image_status,
        reserved_at
    );

CREATE INDEX image_generation_requests_batch_idx
    ON top3_news.image_generation_requests (
        batch_id,
        image_generation_id DESC
    );

CREATE INDEX image_generation_requests_generated_post_idx
    ON top3_news.image_generation_requests (
        generated_post_id,
        image_generation_id DESC
    );

CREATE INDEX image_generation_requests_review_action_idx
    ON top3_news.image_generation_requests (
        review_action_id
    )
    WHERE review_action_id IS NOT NULL;


-- ============================================================================
-- 4. Комментарии
-- ============================================================================

COMMENT ON TABLE
    top3_news.image_generation_requests
IS
    'Защищённые платные запросы генерации единой PNG-иллюстрации TOP-3';

COMMENT ON COLUMN
    top3_news.image_generation_requests.image_request_key
IS
    'Детерминированный SHA-256 ключ image-generation; '
    'уникален для reserved/completed, повтор после failed разрешён';

COMMENT ON COLUMN
    top3_news.image_generation_requests.generated_post_id
IS
    'Версия generated_posts, к которой относится результат image-generation';

COMMENT ON COLUMN
    top3_news.image_generation_requests.request_kind
IS
    'Тип запроса: initial для первой генерации, regenerate для редакционной перегенерации';

COMMENT ON COLUMN
    top3_news.image_generation_requests.review_action_id
IS
    'Review action requested_action=regenerate_image; NULL для initial';

COMMENT ON COLUMN
    top3_news.image_generation_requests.request_payload
IS
    'Канонический JSON payload, использованный при построении image_request_key';

COMMENT ON COLUMN
    top3_news.image_generation_requests.response_metadata
IS
    'Метаданные фактического ответа OpenAI Image API без бинарного содержимого изображения';

COMMENT ON COLUMN
    top3_news.image_generation_requests.image_path
IS
    'Путь к успешно сохранённому неизменяемому PNG-файлу';

COMMENT ON COLUMN
    top3_news.image_generation_requests.image_sha256
IS
    'SHA-256 успешно сохранённого PNG-файла';


-- ============================================================================
-- 5. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '010',
    'Add protected OpenAI image generation requests'
);

COMMIT;