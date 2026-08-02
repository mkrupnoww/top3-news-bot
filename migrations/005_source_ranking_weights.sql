BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 005_source_ranking_weights.sql
--
-- Фиксирует воспроизводимую конфигурацию веса источника для event-level
-- ранжирования TOP-3.
--
-- Вес хранится в:
-- sources.settings.ranking.source_weight
--
-- На чистой базе источник Variety создаётся заранее.
-- На существующей базе сохраняются все текущие настройки источника и
-- обновляется только вложенный ключ ranking.source_weight.
-- ============================================================================


-- ============================================================================
-- 1. Variety Film: ведущий международный отраслевой источник
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
    'variety_film',
    'Variety Film',
    'rss',
    'https://variety.com/v/film/',
    'https://variety.com/v/film/feed/',
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
    settings = (
        target.settings
        || jsonb_build_object(
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
    '005',
    'Add configured source weights for event ranking'
);

COMMIT;
