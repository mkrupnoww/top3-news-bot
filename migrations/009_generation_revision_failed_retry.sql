BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 009_generation_revision_failed_retry.sql
--
-- Разрешает повторную попытку редакционной ревизии после failed
-- с тем же детерминированным revision_request_key.
--
-- Активная или успешно завершённая ревизия по-прежнему остаётся
-- идемпотентной:
--
-- reserved  -> второй платный вызов с тем же ключом запрещён
-- completed -> второй платный вызов с тем же ключом запрещён
-- failed    -> новая попытка с тем же ключом разрешена
-- ============================================================================


-- ============================================================================
-- 1. Меняем глобальную уникальность ключа на активную
-- ============================================================================

DROP INDEX IF EXISTS
    top3_news.generation_revision_requests_key_uq;

CREATE UNIQUE INDEX
    generation_revision_requests_key_active_uq
    ON top3_news.generation_revision_requests (
        revision_request_key
    )
    WHERE revision_status IN (
        'reserved',
        'completed'
    );


-- ============================================================================
-- 2. Уточняем смысл ключа
-- ============================================================================

COMMENT ON COLUMN
    top3_news.generation_revision_requests.revision_request_key
IS
    'Детерминированный SHA-256 ключ редакционной ревизии; '
    'уникален для reserved/completed, повтор после failed разрешён';


-- ============================================================================
-- 3. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '009',
    'Allow retry after failed generation revision'
);

COMMIT;