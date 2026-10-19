BEGIN;
SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- Один автоматический выпуск на UTC-день. Получатели фиксируются до отправки.
-- Результаты каждого канала хранятся в существующих publication_attempts.
CREATE TABLE daily_auto_publications (
    generated_post_id bigint PRIMARY KEY REFERENCES generated_posts ON DELETE CASCADE,
    publication_date date NOT NULL UNIQUE,
    target_chat_ids bigint[] NOT NULL,
    deadline_at timestamptz NOT NULL,
    claimed_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz,
    CONSTRAINT daily_auto_publications_targets_chk CHECK (
        cardinality(target_chat_ids) = 2
        AND target_chat_ids[1] IS NOT NULL AND target_chat_ids[2] IS NOT NULL
        AND target_chat_ids[1] <> target_chat_ids[2]
    )
);
COMMENT ON TABLE daily_auto_publications IS
    'Автопубликация ежедневного поста без реакции редактора к 12:00 UTC; не является human approve';
GRANT SELECT, INSERT, UPDATE, DELETE ON daily_auto_publications TO top3_news_app;
INSERT INTO schema_migrations (version, description)
VALUES ('020', 'Add daily no-response dual-channel auto publication');
COMMIT;
