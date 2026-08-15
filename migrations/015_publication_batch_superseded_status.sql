BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 015_publication_batch_superseded_status.sql
--
-- Разрешает publication_batch завершаться состоянием superseded.
--
-- Этот статус используется replacement cascade после доказанного
-- Image API moderation_blocked:
--
-- old generated_post  -> superseded
-- old publication_batch -> superseded
-- daily workflow возвращается в generation для новой ranking combination.
-- ============================================================================


-- ============================================================================
-- 1. Расширяем допустимые статусы publication_batches
-- ============================================================================

ALTER TABLE top3_news.publication_batches
    DROP CONSTRAINT publication_batches_status_chk;


ALTER TABLE top3_news.publication_batches
    ADD CONSTRAINT publication_batches_status_chk
        CHECK (
            batch_status IN (
                'draft',
                'ranked',
                'generated',
                'awaiting_review',
                'approved',
                'rejected',
                'superseded',
                'publishing',
                'published',
                'failed'
            )
        );


COMMENT ON COLUMN top3_news.publication_batches.batch_status IS
    'Состояние выпуска; superseded означает замену выпуска новым '
    'TOP-3 внутри того же daily workflow';


-- ============================================================================
-- 2. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '015',
    'Allow superseded publication batches for TOP-3 replacement cascade'
);

COMMIT;