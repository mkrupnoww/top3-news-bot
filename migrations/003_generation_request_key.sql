BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 003_generation_request_key.sql
--
-- Добавляет уникальный ключ логического запуска генерации Telegram-поста.
--
-- Ключ создаётся приложением как SHA-256 от:
-- ranking_run_id, выбранного TOP-3, модели, промпта, формата текста,
-- целевого Telegram-канала и других параметров генерации.
--
-- publication_batches используется как запись резервирования до обращения
-- к OpenAI. Это не допускает повторный платный запрос для одного и того же
-- результата ранжирования и одинаковых параметров генерации.
-- ============================================================================

ALTER TABLE top3_news.publication_batches
ADD COLUMN generation_request_key text;

COMMENT ON COLUMN
    top3_news.publication_batches.generation_request_key
IS
    'SHA-256 ключ логического запуска генерации; защищает от повторного платного запроса';

ALTER TABLE top3_news.publication_batches
ADD CONSTRAINT publication_batches_generation_request_key_chk
CHECK (
    generation_request_key IS NULL
    OR generation_request_key ~ '^[0-9a-f]{64}$'
);

CREATE UNIQUE INDEX publication_batches_generation_request_key_uq
ON top3_news.publication_batches (generation_request_key)
WHERE generation_request_key IS NOT NULL;

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '003',
    'Add idempotency request key to publication batches'
);

COMMIT;