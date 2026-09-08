BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 018_individual_score_v3_postgres_guard.sql
--
-- Возвращает независимую PostgreSQL-проверку individual_score после migration
-- 017. Исторические значения news_scores не меняются.
--
-- Новые ranking runs версии individual_score_v3 / top3_cinema_v6 используют:
--   B = 0.20F + 0.35M + 0.20R + 0.10(H × Q)
--
-- Приложение перед INSERT передаёт Python-результат как контрольное значение,
-- но сохраняет результат этой PostgreSQL-функции. После INSERT приложение
-- сравнивает PostgreSQL и Python; несовпадение откатывает транзакцию.
-- ============================================================================

DO $$
DECLARE
    migration_017_applied boolean;
    current_is_generated text;
    current_is_nullable text;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM top3_news.schema_migrations
        WHERE version = '017'
    )
    INTO migration_017_applied;

    IF NOT migration_017_applied THEN
        RAISE EXCEPTION
            'Migration 017 must be applied before migration 018';
    END IF;

    SELECT
        c.is_generated,
        c.is_nullable
    INTO
        current_is_generated,
        current_is_nullable
    FROM information_schema.columns AS c
    WHERE c.table_schema = 'top3_news'
      AND c.table_name = 'news_scores'
      AND c.column_name = 'individual_score';

    IF current_is_generated <> 'NEVER' THEN
        RAISE EXCEPTION
            'Expected persisted individual_score before migration 018, got is_generated=%',
            current_is_generated;
    END IF;

    IF current_is_nullable <> 'NO' THEN
        RAISE EXCEPTION
            'Expected NOT NULL individual_score before migration 018, got is_nullable=%',
            current_is_nullable;
    END IF;
END
$$;

CREATE FUNCTION top3_news.calculate_individual_score_v3(
    p_f_score numeric,
    p_m_score numeric,
    p_r_score numeric,
    p_h_score numeric,
    p_q_score numeric
)
RETURNS numeric
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
AS $$
    SELECT CAST(
        0.20 * p_f_score
        + 0.35 * p_m_score
        + 0.20 * p_r_score
        + 0.10 * (p_h_score * p_q_score)
        AS numeric(20, 6)
    )
$$;

COMMENT ON FUNCTION top3_news.calculate_individual_score_v3(
    numeric,
    numeric,
    numeric,
    numeric,
    numeric
) IS
    'Independent PostgreSQL calculation for individual_score_v3: 0.20F + 0.35M + 0.20R + 0.10(H × Q)';

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '018',
    'Add PostgreSQL guard for individual_score_v3'
);

COMMIT;
