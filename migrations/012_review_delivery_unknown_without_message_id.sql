BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';


-- ============================================================================
-- 012_review_delivery_unknown_without_message_id.sql
--
-- Разрешает review delivery переходить в unknown,
-- даже если Telegram message_id получить не удалось.
--
-- Это необходимо для сетевых ошибок с неопределённым исходом:
-- запрос sendPhoto мог быть принят Telegram, но ответ с message_id
-- мог не вернуться клиенту.
--
-- unknown блокирует автоматическую повторную отправку и тем самым
-- защищает редактора от возможного дубликата.
-- ============================================================================


ALTER TABLE top3_news.review_delivery_attempts
DROP CONSTRAINT review_delivery_attempts_state_chk;


ALTER TABLE top3_news.review_delivery_attempts
ADD CONSTRAINT review_delivery_attempts_state_chk
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
        AND error_type IS NOT NULL
        AND btrim(error_type) <> ''
        AND error_message IS NOT NULL
        AND btrim(error_message) <> ''
        AND failed_at IS NULL
    )
);


INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '012',
    'Allow uncertain review delivery without Telegram message ID'
);

COMMIT;
