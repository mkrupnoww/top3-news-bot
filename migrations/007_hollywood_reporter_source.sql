BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 007_hollywood_reporter_source.sql
--
-- Добавляет общую RSS-ленту The Hollywood Reporter.
--
-- Источник содержит не только кино, поэтому перед платным event-level
-- ранжированием для него включается детерминированный фильтр связи с
-- кинематографом.
--
-- Параметры хранятся в:
-- sources.settings.ranking.source_weight
-- sources.settings.ranking.requires_cinema_relevance_filter
--
-- Значение веса 3 соответствует ведущему международному профильному
-- отраслевому изданию.
-- ============================================================================


-- ============================================================================
-- 1. The Hollywood Reporter
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
    'hollywood_reporter',
    'The Hollywood Reporter',
    'rss',
    'https://www.hollywoodreporter.com/',
    'https://www.hollywoodreporter.com/feed/',
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
            3,
            'requires_cinema_relevance_filter',
            true
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
                3,
                'requires_cinema_relevance_filter',
                true
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
    '007',
    'Add The Hollywood Reporter RSS source with cinema relevance filter'
);

COMMIT;