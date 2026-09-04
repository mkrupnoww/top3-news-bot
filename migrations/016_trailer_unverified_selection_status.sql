BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 016_trailer_unverified_selection_status.sql
--
-- Добавляет явную причину replacement TOP-3, когда trailer/teaser-news
-- не имеет подтверждённого официального trailer URL.
--
-- Daily workflow не падает: текущая selection закрывается как
-- trailer_unverified, после чего orchestrator пробует следующую ranking
-- combination. Если replacement отсутствует, production может продолжить
-- текущую selection в degraded режиме, чтобы выпуск не пропал.
-- ============================================================================

ALTER TABLE top3_news.daily_workflow_selection_attempts
    DROP CONSTRAINT daily_workflow_selection_attempts_status_chk;

ALTER TABLE top3_news.daily_workflow_selection_attempts
    ADD CONSTRAINT daily_workflow_selection_attempts_status_chk
        CHECK (
            selection_status IN (
                'active',
                'moderation_blocked',
                'trailer_unverified',
                'ready_for_review'
            )
        );

ALTER TABLE top3_news.daily_workflow_selection_attempts
    ADD COLUMN rejection_reason text,
    ADD COLUMN rejected_news_ids bigint[];

ALTER TABLE top3_news.daily_workflow_selection_attempts
    ADD CONSTRAINT daily_workflow_selection_attempts_rejection_chk
        CHECK (
            (
                selection_status = 'trailer_unverified'
                AND rejection_reason IS NOT NULL
                AND BTRIM(rejection_reason) <> ''
                AND rejected_news_ids IS NOT NULL
                AND cardinality(rejected_news_ids) > 0
            )
            OR
            (
                selection_status <> 'trailer_unverified'
                AND rejection_reason IS NULL
                AND rejected_news_ids IS NULL
            )
        );

COMMENT ON COLUMN
    top3_news.daily_workflow_selection_attempts.rejection_reason IS
    'Причина content-level replacement; currently official_trailer_not_verified';

COMMENT ON COLUMN
    top3_news.daily_workflow_selection_attempts.rejected_news_ids IS
    'news_id текущей combination, из-за которых content-level selection была заменена';

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '016',
    'Track trailer-unverified TOP-3 selection replacements'
);

COMMIT;
