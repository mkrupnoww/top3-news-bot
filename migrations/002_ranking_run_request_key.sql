BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 002_ranking_run_request_key.sql
--
-- Добавляет уникальный ключ логического запуска ранжирования.
--
-- Ключ создаётся приложением как SHA-256 от параметров запуска:
-- временного окна, кандидатов, модели, промпта и версии формулы.
--
-- Он позволяет зарезервировать ranking_run до обращения к OpenAI
-- и не допустить повторный платный запрос для того же набора данных.
-- ============================================================================

ALTER TABLE top3_news.ranking_runs
ADD COLUMN request_key text;

COMMENT ON COLUMN top3_news.ranking_runs.request_key IS
    'SHA-256 ключ логического запуска; защищает от повторного платного запроса';

ALTER TABLE top3_news.ranking_runs
ADD CONSTRAINT ranking_runs_request_key_chk
CHECK (
    request_key IS NULL
    OR request_key ~ '^[0-9a-f]{64}$'
);

CREATE UNIQUE INDEX ranking_runs_request_key_uq
ON top3_news.ranking_runs (request_key)
WHERE request_key IS NOT NULL;

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '002',
    'Add idempotency request key to ranking runs'
);

COMMIT;