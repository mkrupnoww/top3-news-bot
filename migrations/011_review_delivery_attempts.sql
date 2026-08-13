BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 011_review_delivery_attempts.sql
--
-- Добавляет защищённый lifecycle автоматической доставки готового
-- generated_post редактору в Telegram для ручного review.
--
-- Review delivery не является публикацией в Telegram-канал и поэтому
-- хранится отдельно от publication_attempts.
--
-- Для каждого generated_post сообщение может быть доставлено каждому
-- активному admin/reviewer независимо.
--
-- Состояния:
--
-- reserved -> отправка зарезервирована до Telegram Bot API;
-- sent     -> Telegram подтвердил отправку и message_id сохранён;
-- failed   -> отправка не подтверждена, повтор разрешён;
-- unknown  -> Telegram-отправка могла состояться или состоялась,
--             но состояние требует ручной сверки; автоматический
--             повтор запрещён.
--
-- Идемпотентность:
--
-- reserved/sent/unknown блокируют ещё одну автоматическую доставку
-- того же generated_post тому же пользователю;
-- failed разрешает новую попытку с новым attempt_number.
--
-- Ошибка review delivery НЕ меняет generated_posts.post_status
-- и publication_batches.batch_status.
-- ============================================================================


-- ============================================================================
-- 1. Попытки доставки review
-- ============================================================================

CREATE TABLE top3_news.review_delivery_attempts (
    review_delivery_attempt_id      bigint
                                        GENERATED ALWAYS AS IDENTITY
                                        PRIMARY KEY,

    generated_post_id               bigint NOT NULL
                                        REFERENCES top3_news.generated_posts (
                                            generated_post_id
                                        )
                                        ON DELETE CASCADE,

    telegram_user_id                bigint NOT NULL
                                        REFERENCES top3_news.bot_users (
                                            telegram_user_id
                                        ),

    attempt_number                  integer NOT NULL,

    delivery_status                 text NOT NULL DEFAULT 'reserved',

    telegram_chat_id                bigint NOT NULL,
    telegram_message_id             bigint,

    request_payload                 jsonb NOT NULL DEFAULT '{}'::jsonb,
    response_payload                jsonb NOT NULL DEFAULT '{}'::jsonb,

    telegram_error_code             integer,
    error_type                      text,
    error_message                   text,

    reserved_at                     timestamptz NOT NULL DEFAULT now(),
    sent_at                         timestamptz,
    failed_at                       timestamptz,

    created_at                      timestamptz NOT NULL DEFAULT now(),
    updated_at                      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT review_delivery_attempts_number_chk
        CHECK (
            attempt_number > 0
        ),

    CONSTRAINT review_delivery_attempts_status_chk
        CHECK (
            delivery_status IN (
                'reserved',
                'sent',
                'failed',
                'unknown'
            )
        ),

    CONSTRAINT review_delivery_attempts_request_object_chk
        CHECK (
            jsonb_typeof(request_payload) = 'object'
        ),

    CONSTRAINT review_delivery_attempts_response_object_chk
        CHECK (
            jsonb_typeof(response_payload) = 'object'
        ),

    CONSTRAINT review_delivery_attempts_state_chk
        CHECK (
            (
                delivery_status = 'reserved'
                AND telegram_message_id IS NULL
                AND response_payload = '{}'::jsonb
                AND telegram_error_code IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
                AND sent_at IS NULL
                AND failed_at IS NULL
            )
            OR
            (
                delivery_status = 'sent'
                AND telegram_message_id IS NOT NULL
                AND telegram_error_code IS NULL
                AND error_type IS NULL
                AND error_message IS NULL
                AND sent_at IS NOT NULL
                AND failed_at IS NULL
            )
            OR
            (
                delivery_status = 'failed'
                AND telegram_message_id IS NULL
                AND error_type IS NOT NULL
                AND btrim(error_type) <> ''
                AND error_message IS NOT NULL
                AND btrim(error_message) <> ''
                AND sent_at IS NULL
                AND failed_at IS NOT NULL
            )
            OR
            (
                delivery_status = 'unknown'
                AND telegram_message_id IS NOT NULL
                AND error_type IS NOT NULL
                AND btrim(error_type) <> ''
                AND error_message IS NOT NULL
                AND btrim(error_message) <> ''
                AND sent_at IS NOT NULL
                AND failed_at IS NULL
            )
        )
);


-- ============================================================================
-- 2. Идемпотентность и конкурентное выполнение
-- ============================================================================

CREATE UNIQUE INDEX review_delivery_attempts_post_user_number_uq
    ON top3_news.review_delivery_attempts (
        generated_post_id,
        telegram_user_id,
        attempt_number
    );

CREATE UNIQUE INDEX review_delivery_attempts_post_user_active_uq
    ON top3_news.review_delivery_attempts (
        generated_post_id,
        telegram_user_id
    )
    WHERE delivery_status IN (
        'reserved',
        'sent',
        'unknown'
    );


-- ============================================================================
-- 3. Операционные индексы
-- ============================================================================

CREATE INDEX review_delivery_attempts_post_idx
    ON top3_news.review_delivery_attempts (
        generated_post_id,
        review_delivery_attempt_id DESC
    );

CREATE INDEX review_delivery_attempts_user_idx
    ON top3_news.review_delivery_attempts (
        telegram_user_id,
        review_delivery_attempt_id DESC
    );

CREATE INDEX review_delivery_attempts_status_reserved_idx
    ON top3_news.review_delivery_attempts (
        delivery_status,
        reserved_at
    );


-- ============================================================================
-- 4. updated_at trigger
-- ============================================================================

CREATE TRIGGER review_delivery_attempts_set_updated_at
BEFORE UPDATE
ON top3_news.review_delivery_attempts
FOR EACH ROW
EXECUTE FUNCTION top3_news.set_updated_at();


-- ============================================================================
-- 5. Комментарии
-- ============================================================================

COMMENT ON TABLE
    top3_news.review_delivery_attempts
IS
    'Защищённые попытки автоматической доставки generated_post '
    'редактору в Telegram для ручного review';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.generated_post_id
IS
    'Версия generated_posts, отправляемая редактору';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.telegram_user_id
IS
    'Активный admin/reviewer из bot_users, которому адресован review';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.attempt_number
IS
    'Последовательный номер попытки для пары generated_post/user';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.delivery_status
IS
    'reserved, sent, failed или unknown';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.telegram_chat_id
IS
    'Telegram chat_id получателя; для личной доставки равен telegram_user_id';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.telegram_message_id
IS
    'Telegram message_id успешно отправленного native photo review';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.request_payload
IS
    'Аудит параметров native Telegram photo review до отправки';

COMMENT ON COLUMN
    top3_news.review_delivery_attempts.response_payload
IS
    'Метаданные подтверждённого ответа Telegram Bot API без бинарного PNG';


-- ============================================================================
-- 6. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '011',
    'Add protected Telegram review delivery attempts'
);

COMMIT;
