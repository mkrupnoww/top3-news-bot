BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 017_versioned_individual_score.sql
--
-- Переводит news_scores.individual_score из GENERATED STORED в обычное
-- сохранённое поле. Это позволяет менять формулу между ranking_runs,
-- не пересчитывая исторические баллы старых запусков.
--
-- До этой миграции:
--   individual_score_v2 / top3_cinema_v5
--   B = 0.20F + 0.30M + 0.20R + 0.15(H × Q)
--
-- После миграции новые ranking runs сохраняют рассчитанный приложением B
-- явно. Текущая версия приложения:
--   individual_score_v3 / top3_cinema_v6
--   B = 0.20F + 0.35M + 0.20R + 0.10(H × Q)
--
-- Исторические значения individual_score должны остаться неизменными.
-- ============================================================================

DO $$
DECLARE
    current_is_generated text;
BEGIN
    SELECT c.is_generated
    INTO current_is_generated
    FROM information_schema.columns AS c
    WHERE c.table_schema = 'top3_news'
      AND c.table_name = 'news_scores'
      AND c.column_name = 'individual_score';

    IF current_is_generated IS NULL THEN
        RAISE EXCEPTION
            'Column top3_news.news_scores.individual_score not found';
    END IF;

    IF current_is_generated <> 'ALWAYS' THEN
        RAISE EXCEPTION
            'Expected generated individual_score before migration 017, got %',
            current_is_generated;
    END IF;
END
$$;

CREATE TEMP TABLE news_scores_individual_score_before
ON COMMIT DROP
AS
SELECT
    score_id,
    individual_score
FROM top3_news.news_scores;

ALTER TABLE top3_news.news_scores
    ALTER COLUMN individual_score DROP EXPRESSION;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM (
            (
                SELECT
                    score_id,
                    individual_score
                FROM news_scores_individual_score_before

                EXCEPT

                SELECT
                    score_id,
                    individual_score
                FROM top3_news.news_scores
            )
            UNION ALL
            (
                SELECT
                    score_id,
                    individual_score
                FROM top3_news.news_scores

                EXCEPT

                SELECT
                    score_id,
                    individual_score
                FROM news_scores_individual_score_before
            )
        ) AS changed_scores
    ) THEN
        RAISE EXCEPTION
            'Historical individual_score values changed while dropping expression';
    END IF;
END
$$;

ALTER TABLE top3_news.news_scores
    ALTER COLUMN individual_score SET NOT NULL;

COMMENT ON COLUMN top3_news.news_scores.individual_score IS
    'Сохранённый B_i, рассчитанный приложением согласно formula_version ranking run; исторические значения не пересчитываются при изменении формулы';

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '017',
    'Persist versioned individual scores without rewriting history'
);

COMMIT;
