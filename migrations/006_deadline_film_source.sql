BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 006_deadline_film_source.sql
--
-- Добавляет RSS-источник Deadline Film и фиксирует его вес для event-level
-- ранжирования TOP-3.
--
-- Вес хранится в:
-- sources.settings.ranking.source_weight
--
-- Значение 3 соответствует ведущему международному профильному отраслевому
-- изданию.
-- ============================================================================


-- ============================================================================
-- 1. Deadline Film
-- ============================================================================

INSERT INTO top3_news.sources AS target (
    source_code,
    source_name,
    source_type,
    base_url,
    feed_url,
    default_language,
    is_active,
    collection_priority,
    settings
)
VALUES (
    'deadline_film',
    'Deadline Film',
    'rss',
    'https://deadline.com/v/film/',
    'https://deadline.com/v/film/feed/',
    'en',
    true,
    100,
    jsonb_build_object(
        'collector',
        'rss_atom_http',
        'managed_by',
        'feed_collection',
        'ranking',
        jsonb_build_object(
            'source_weight',
            3
        )
    )
)
ON CONFLICT (source_code)
DO UPDATE SET
    source_name = EXCLUDED.source_name,
    source_type = EXCLUDED.source_type,
    base_url = EXCLUDED.base_url,
    feed_url = EXCLUDED.feed_url,
    default_language = EXCLUDED.default_language,
    is_active = EXCLUDED.is_active,
    collection_priority = EXCLUDED.collection_priority,
    settings = (
        target.settings
        || jsonb_build_object(
            'collector',
            'rss_atom_http',
            'managed_by',
            'feed_collection',
            'ranking',
            COALESCE(
                target.settings->'ranking',
                '{}'::jsonb
            )
            || jsonb_build_object(
                'source_weight',
                3
            )
        )
    ),
    updated_at = now();


-- ============================================================================
-- 2. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '006',
    'Add Deadline Film RSS source and ranking weight'
);

COMMIT;
