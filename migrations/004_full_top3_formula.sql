BEGIN;

SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- ============================================================================
-- 004_full_top3_formula.sql
--
-- Расширяет модель ранжирования до полной методики TOP-3:
--
-- F = 10 × sqrt(1 - h / 24)
-- M = 0.60U + 0.40I
-- R = нормализованная комбинация V, C и S
-- H = 0.30K + 0.25N + 0.25E + 0.20X
--
-- B = 0.20F
--   + 0.30M
--   + 0.20R
--   + 0.15(H × Q)
--
-- TOP(S) = average(B) + 0.15D(S)
--
-- Миграция:
-- - вводит сущность инфоповода внутри ranking_run;
-- - хранит публикации, объединённые в один инфоповод;
-- - хранит сырые audience-метрики;
-- - добавляет промежуточные компоненты формулы;
-- - хранит все рассчитанные комбинации из трёх новостей;
-- - отдельно фиксирует победивший TOP-3.
--
-- Старые ranking_runs и news_scores не пересчитываются.
-- ============================================================================


-- ============================================================================
-- 1. Инфоповоды внутри конкретного ranking_run
-- ============================================================================

CREATE TABLE top3_news.ranking_events (
    ranking_event_id          bigint
                                  GENERATED ALWAYS AS IDENTITY
                                  PRIMARY KEY,

    ranking_run_id            bigint NOT NULL
                                  REFERENCES top3_news.ranking_runs (
                                      ranking_run_id
                                  )
                                  ON DELETE CASCADE,

    event_key                 text NOT NULL,

    representative_news_id    bigint NOT NULL
                                  REFERENCES top3_news.news_items (
                                      news_id
                                  )
                                  ON DELETE RESTRICT,

    event_title               text NOT NULL,
    event_time_utc            timestamptz NOT NULL,

    macro_topic               text NOT NULL,

    impact_reason             text NOT NULL,
    hook_reason               text NOT NULL,
    q_reason                  text NOT NULL,

    source_weight_sum         numeric(20, 6)
                                  NOT NULL
                                  DEFAULT 0,

    event_details             jsonb
                                  NOT NULL
                                  DEFAULT '{}'::jsonb,

    created_at                timestamptz
                                  NOT NULL
                                  DEFAULT now(),

    CONSTRAINT ranking_events_run_key_uq
        UNIQUE (
            ranking_run_id,
            event_key
        ),

    CONSTRAINT ranking_events_run_representative_uq
        UNIQUE (
            ranking_run_id,
            representative_news_id
        ),

    CONSTRAINT ranking_events_id_run_uq
        UNIQUE (
            ranking_event_id,
            ranking_run_id
        ),

    CONSTRAINT ranking_events_key_chk
        CHECK (
            event_key ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT ranking_events_title_chk
        CHECK (
            length(btrim(event_title)) > 0
        ),

    CONSTRAINT ranking_events_macro_topic_chk
        CHECK (
            macro_topic IN (
                'business_economy_law',
                'people_conflicts_legal',
                'creative_cast_production',
                'trailers_premieres_releases',
                'festivals_awards_criticism',
                'box_office_audience_distribution',
                'other'
            )
        ),

    CONSTRAINT ranking_events_reasons_chk
        CHECK (
            length(btrim(impact_reason)) > 0
            AND length(btrim(hook_reason)) > 0
            AND length(btrim(q_reason)) > 0
        ),

    CONSTRAINT ranking_events_source_weight_chk
        CHECK (
            source_weight_sum >= 0
        ),

    CONSTRAINT ranking_events_details_object_chk
        CHECK (
            jsonb_typeof(event_details) = 'object'
        )
);

CREATE INDEX ranking_events_run_time_idx
    ON top3_news.ranking_events (
        ranking_run_id,
        event_time_utc DESC
    );

CREATE INDEX ranking_events_run_topic_idx
    ON top3_news.ranking_events (
        ranking_run_id,
        macro_topic
    );

CREATE INDEX ranking_events_representative_idx
    ON top3_news.ranking_events (
        representative_news_id
    );


COMMENT ON TABLE top3_news.ranking_events IS
    'Инфоповоды, сформированные внутри одного запуска ранжирования';

COMMENT ON COLUMN top3_news.ranking_events.event_key IS
    'SHA-256 ключ состава и содержания инфоповода';

COMMENT ON COLUMN top3_news.ranking_events.event_time_utc IS
    'Время самого события или последнего существенного подтверждённого развития';

COMMENT ON COLUMN top3_news.ranking_events.macro_topic IS
    'Макротема инфоповода для расчёта разнообразия D(S)';

COMMENT ON COLUMN top3_news.ranking_events.source_weight_sum IS
    'A_i: сумма весов независимых источников инфоповода';


-- ============================================================================
-- 2. Публикации, объединённые в один инфоповод
-- ============================================================================

CREATE TABLE top3_news.ranking_event_members (
    ranking_event_member_id   bigint
                                  GENERATED ALWAYS AS IDENTITY
                                  PRIMARY KEY,

    ranking_event_id          bigint NOT NULL,
    ranking_run_id            bigint NOT NULL,

    news_id                   bigint NOT NULL
                                  REFERENCES top3_news.news_items (
                                      news_id
                                  )
                                  ON DELETE RESTRICT,

    is_representative         boolean
                                  NOT NULL
                                  DEFAULT false,

    is_independent_source     boolean
                                  NOT NULL
                                  DEFAULT false,

    counts_toward_reach       boolean
                                  NOT NULL
                                  DEFAULT false,

    source_weight             smallint
                                  NOT NULL
                                  DEFAULT 0,

    source_relation           text NOT NULL,

    membership_reason         text,

    created_at                timestamptz
                                  NOT NULL
                                  DEFAULT now(),

    CONSTRAINT ranking_event_members_event_run_fk
        FOREIGN KEY (
            ranking_event_id,
            ranking_run_id
        )
        REFERENCES top3_news.ranking_events (
            ranking_event_id,
            ranking_run_id
        )
        ON DELETE CASCADE,

    CONSTRAINT ranking_event_members_event_news_uq
        UNIQUE (
            ranking_event_id,
            news_id
        ),

    CONSTRAINT ranking_event_members_source_weight_chk
        CHECK (
            source_weight BETWEEN 0 AND 3
        ),

    CONSTRAINT ranking_event_members_relation_chk
        CHECK (
            source_relation IN (
                'primary',
                'independent',
                'syndicated',
                'duplicate'
            )
        ),

    CONSTRAINT ranking_event_members_reach_chk
        CHECK (
            counts_toward_reach = false
            OR (
                is_independent_source = true
                AND source_weight > 0
                AND source_relation IN (
                    'primary',
                    'independent'
                )
            )
        ),

    CONSTRAINT ranking_event_members_reason_chk
        CHECK (
            membership_reason IS NULL
            OR length(
                btrim(membership_reason)
            ) > 0
        )
);

CREATE UNIQUE INDEX
    ranking_event_members_representative_uq
    ON top3_news.ranking_event_members (
        ranking_event_id
    )
    WHERE is_representative = true;

CREATE INDEX ranking_event_members_run_idx
    ON top3_news.ranking_event_members (
        ranking_run_id,
        ranking_event_id
    );

CREATE INDEX ranking_event_members_news_idx
    ON top3_news.ranking_event_members (
        news_id
    );


COMMENT ON TABLE top3_news.ranking_event_members IS
    'Публикации и источники, сгруппированные в один инфоповод';

COMMENT ON COLUMN
    top3_news.ranking_event_members.source_weight IS
    'Вес источника: 3, 2, 1 или 0 согласно методике';

COMMENT ON COLUMN
    top3_news.ranking_event_members.counts_toward_reach IS
    'Признак участия публикации в расчёте A_i и U_i';


-- ============================================================================
-- 3. Сырые фактические метрики реакции аудитории
-- ============================================================================

CREATE TABLE top3_news.ranking_audience_metrics (
    audience_metric_id        bigint
                                  GENERATED ALWAYS AS IDENTITY
                                  PRIMARY KEY,

    ranking_event_id          bigint NOT NULL,
    ranking_run_id            bigint NOT NULL,

    platform_code             text NOT NULL,
    metric_source_url         text,

    measured_at               timestamptz NOT NULL,

    metric_window_hours       numeric(12, 6),

    view_count                bigint,
    comment_count             bigint,
    share_count               bigint,

    is_trusted                boolean
                                  NOT NULL
                                  DEFAULT true,

    raw_payload               jsonb
                                  NOT NULL
                                  DEFAULT '{}'::jsonb,

    created_at                timestamptz
                                  NOT NULL
                                  DEFAULT now(),

    CONSTRAINT ranking_audience_metrics_event_run_fk
        FOREIGN KEY (
            ranking_event_id,
            ranking_run_id
        )
        REFERENCES top3_news.ranking_events (
            ranking_event_id,
            ranking_run_id
        )
        ON DELETE CASCADE,

    CONSTRAINT ranking_audience_metrics_platform_chk
        CHECK (
            length(
                btrim(platform_code)
            ) > 0
        ),

    CONSTRAINT ranking_audience_metrics_window_chk
        CHECK (
            metric_window_hours IS NULL
            OR metric_window_hours >= 0
        ),

    CONSTRAINT ranking_audience_metrics_counts_chk
        CHECK (
            (
                view_count IS NOT NULL
                OR comment_count IS NOT NULL
                OR share_count IS NOT NULL
            )
            AND (
                view_count IS NULL
                OR view_count >= 0
            )
            AND (
                comment_count IS NULL
                OR comment_count >= 0
            )
            AND (
                share_count IS NULL
                OR share_count >= 0
            )
        ),

    CONSTRAINT ranking_audience_metrics_payload_object_chk
        CHECK (
            jsonb_typeof(raw_payload) = 'object'
        )
);

CREATE INDEX ranking_audience_metrics_event_idx
    ON top3_news.ranking_audience_metrics (
        ranking_event_id,
        measured_at DESC
    );

CREATE INDEX ranking_audience_metrics_run_platform_idx
    ON top3_news.ranking_audience_metrics (
        ranking_run_id,
        platform_code,
        measured_at DESC
    );


COMMENT ON TABLE top3_news.ranking_audience_metrics IS
    'Сырые просмотры, комментарии и распространение для расчёта V, C, S и R';


-- ============================================================================
-- 4. Промежуточные компоненты полной формулы в news_scores
-- ============================================================================

ALTER TABLE top3_news.news_scores
ADD COLUMN ranking_event_id bigint;

ALTER TABLE top3_news.news_scores
ADD COLUMN age_hours numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN u_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN i_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN v_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN c_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN s_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN k_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN n_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN e_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN x_score numeric(12, 6);

ALTER TABLE top3_news.news_scores
ADD COLUMN resonance_confidence text;

ALTER TABLE top3_news.news_scores
ADD COLUMN selected_for_top3 boolean
    NOT NULL
    DEFAULT false;

ALTER TABLE top3_news.news_scores
ADD COLUMN top3_position smallint;


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_score_run_uq
UNIQUE (
    score_id,
    ranking_run_id
);


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_event_run_fk
FOREIGN KEY (
    ranking_event_id,
    ranking_run_id
)
REFERENCES top3_news.ranking_events (
    ranking_event_id,
    ranking_run_id
)
ON DELETE CASCADE;


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_values_upper_chk
CHECK (
    f_score <= 10
    AND m_score <= 10
    AND r_score <= 10
    AND h_score <= 10
    AND q_score <= 1
);


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_age_hours_chk
CHECK (
    age_hours IS NULL
    OR age_hours BETWEEN 0 AND 24
);


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_full_components_chk
CHECK (
    (
        u_score IS NULL
        OR u_score BETWEEN 0 AND 10
    )
    AND (
        i_score IS NULL
        OR i_score BETWEEN 0 AND 10
    )
    AND (
        v_score IS NULL
        OR v_score BETWEEN 0 AND 10
    )
    AND (
        c_score IS NULL
        OR c_score BETWEEN 0 AND 10
    )
    AND (
        s_score IS NULL
        OR s_score BETWEEN 0 AND 10
    )
    AND (
        k_score IS NULL
        OR k_score BETWEEN 0 AND 10
    )
    AND (
        n_score IS NULL
        OR n_score BETWEEN 0 AND 10
    )
    AND (
        e_score IS NULL
        OR e_score BETWEEN 0 AND 10
    )
    AND (
        x_score IS NULL
        OR x_score BETWEEN 0 AND 10
    )
);


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_resonance_confidence_chk
CHECK (
    resonance_confidence IS NULL
    OR resonance_confidence IN (
        'full',
        'partial',
        'unavailable'
    )
);


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_top3_position_chk
CHECK (
    top3_position IS NULL
    OR top3_position BETWEEN 1 AND 3
);


ALTER TABLE top3_news.news_scores
ADD CONSTRAINT news_scores_top3_selection_chk
CHECK (
    (
        selected_for_top3 = true
        AND top3_position IS NOT NULL
        AND is_eligible = true
    )
    OR (
        selected_for_top3 = false
        AND top3_position IS NULL
    )
);


CREATE UNIQUE INDEX news_scores_event_uq
    ON top3_news.news_scores (
        ranking_event_id
    )
    WHERE ranking_event_id IS NOT NULL;


CREATE UNIQUE INDEX news_scores_run_top3_position_uq
    ON top3_news.news_scores (
        ranking_run_id,
        top3_position
    )
    WHERE top3_position IS NOT NULL;


CREATE INDEX news_scores_run_selected_idx
    ON top3_news.news_scores (
        ranking_run_id,
        top3_position
    )
    WHERE selected_for_top3 = true;


COMMENT ON COLUMN top3_news.news_scores.ranking_event_id IS
    'Инфоповод, для которого рассчитана эта оценка';

COMMENT ON COLUMN top3_news.news_scores.age_hours IS
    'h_i: возраст события или существенного развития в часах';

COMMENT ON COLUMN top3_news.news_scores.u_score IS
    'U_i: нормализованный медийный охват';

COMMENT ON COLUMN top3_news.news_scores.i_score IS
    'I_i: экспертная глубина последствий';

COMMENT ON COLUMN top3_news.news_scores.v_score IS
    'V_i: нормализованные просмотры';

COMMENT ON COLUMN top3_news.news_scores.c_score IS
    'C_i: нормализованные комментарии и дискуссии';

COMMENT ON COLUMN top3_news.news_scores.s_score IS
    'S_i: нормализованное распространение и репосты';

COMMENT ON COLUMN top3_news.news_scores.k_score IS
    'K_i: конфликтность или скандальность';

COMMENT ON COLUMN top3_news.news_scores.n_score IS
    'N_i: неожиданность';

COMMENT ON COLUMN top3_news.news_scores.e_score IS
    'E_i: эмоциональная сила';

COMMENT ON COLUMN top3_news.news_scores.x_score IS
    'X_i: уникальность';

COMMENT ON COLUMN
    top3_news.news_scores.resonance_confidence IS
    'Полнота фактических данных для расчёта R_i';

COMMENT ON COLUMN
    top3_news.news_scores.selected_for_top3 IS
    'Признак участия новости в победившей комбинации TOP-3';

COMMENT ON COLUMN top3_news.news_scores.top3_position IS
    'Позиция 1–3 внутри победившей комбинации; порядок по B_i';


-- ============================================================================
-- 5. Все рассчитанные комбинации по три допустимые новости
-- ============================================================================

CREATE TABLE top3_news.ranking_combinations (
    combination_id             bigint
                                   GENERATED ALWAYS AS IDENTITY
                                   PRIMARY KEY,

    ranking_run_id             bigint NOT NULL
                                   REFERENCES top3_news.ranking_runs (
                                       ranking_run_id
                                   )
                                   ON DELETE CASCADE,

    combination_key            text NOT NULL,

    combination_rank           integer NOT NULL,

    mean_individual_score      numeric(20, 6)
                                   NOT NULL,

    diversity_score            numeric(12, 6)
                                   NOT NULL,

    final_top_score            numeric(20, 6)
                                   GENERATED ALWAYS AS (
                                       mean_individual_score
                                       + (
                                           0.15
                                           * diversity_score
                                       )
                                   ) STORED,

    mean_m_score               numeric(20, 6)
                                   NOT NULL,

    mean_q_score               numeric(20, 6)
                                   NOT NULL,

    mean_f_score               numeric(20, 6)
                                   NOT NULL,

    distinct_macro_topic_count smallint NOT NULL,

    is_winner                  boolean
                                   NOT NULL
                                   DEFAULT false,

    selection_reason           text NOT NULL,

    combination_details        jsonb
                                   NOT NULL
                                   DEFAULT '{}'::jsonb,

    created_at                 timestamptz
                                   NOT NULL
                                   DEFAULT now(),

    CONSTRAINT ranking_combinations_run_key_uq
        UNIQUE (
            ranking_run_id,
            combination_key
        ),

    CONSTRAINT ranking_combinations_run_rank_uq
        UNIQUE (
            ranking_run_id,
            combination_rank
        ),

    CONSTRAINT ranking_combinations_id_run_uq
        UNIQUE (
            combination_id,
            ranking_run_id
        ),

    CONSTRAINT ranking_combinations_key_chk
        CHECK (
            combination_key ~ '^[0-9a-f]{64}$'
        ),

    CONSTRAINT ranking_combinations_rank_chk
        CHECK (
            combination_rank > 0
        ),

    CONSTRAINT ranking_combinations_mean_b_chk
        CHECK (
            mean_individual_score
            BETWEEN 0 AND 8.5
        ),

    CONSTRAINT ranking_combinations_diversity_chk
        CHECK (
            diversity_score IN (
                0,
                6,
                10
            )
        ),

    CONSTRAINT ranking_combinations_final_score_chk
        CHECK (
            final_top_score
            BETWEEN 0 AND 10
        ),

    CONSTRAINT ranking_combinations_tie_breakers_chk
        CHECK (
            mean_m_score BETWEEN 0 AND 10
            AND mean_q_score BETWEEN 0 AND 1
            AND mean_f_score BETWEEN 0 AND 10
        ),

    CONSTRAINT ranking_combinations_topic_count_chk
        CHECK (
            distinct_macro_topic_count
            BETWEEN 1 AND 3
        ),

    CONSTRAINT ranking_combinations_topic_diversity_match_chk
        CHECK (
            (
                distinct_macro_topic_count = 3
                AND diversity_score = 10
            )
            OR (
                distinct_macro_topic_count = 2
                AND diversity_score = 6
            )
            OR (
                distinct_macro_topic_count = 1
                AND diversity_score = 0
            )
        ),

    CONSTRAINT ranking_combinations_reason_chk
        CHECK (
            length(
                btrim(selection_reason)
            ) > 0
        ),

    CONSTRAINT ranking_combinations_details_object_chk
        CHECK (
            jsonb_typeof(
                combination_details
            ) = 'object'
        )
);


CREATE UNIQUE INDEX ranking_combinations_run_winner_uq
    ON top3_news.ranking_combinations (
        ranking_run_id
    )
    WHERE is_winner = true;


CREATE INDEX ranking_combinations_run_score_idx
    ON top3_news.ranking_combinations (
        ranking_run_id,
        final_top_score DESC,
        mean_m_score DESC,
        mean_q_score DESC,
        mean_f_score DESC,
        combination_rank
    );


COMMENT ON TABLE top3_news.ranking_combinations IS
    'Все допустимые комбинации из трёх новостей и расчёт TOP(S)';

COMMENT ON COLUMN
    top3_news.ranking_combinations.diversity_score IS
    'D(S): 10 для трёх макротем, 6 для двух, 0 для одной';

COMMENT ON COLUMN
    top3_news.ranking_combinations.final_top_score IS
    'TOP(S) = average(B) + 0.15D(S)';

COMMENT ON COLUMN
    top3_news.ranking_combinations.mean_m_score IS
    'Первый дополнительный критерий разрешения ничьей';

COMMENT ON COLUMN
    top3_news.ranking_combinations.mean_q_score IS
    'Второй дополнительный критерий разрешения ничьей';

COMMENT ON COLUMN
    top3_news.ranking_combinations.mean_f_score IS
    'Третий дополнительный критерий разрешения ничьей';


-- ============================================================================
-- 6. Участники каждой комбинации
-- ============================================================================

CREATE TABLE top3_news.ranking_combination_items (
    combination_item_id    bigint
                               GENERATED ALWAYS AS IDENTITY
                               PRIMARY KEY,

    combination_id         bigint NOT NULL,
    ranking_run_id         bigint NOT NULL,
    score_id               bigint NOT NULL,

    position               smallint NOT NULL,

    created_at             timestamptz
                               NOT NULL
                               DEFAULT now(),

    CONSTRAINT ranking_combination_items_combination_run_fk
        FOREIGN KEY (
            combination_id,
            ranking_run_id
        )
        REFERENCES top3_news.ranking_combinations (
            combination_id,
            ranking_run_id
        )
        ON DELETE CASCADE,

    CONSTRAINT ranking_combination_items_score_run_fk
        FOREIGN KEY (
            score_id,
            ranking_run_id
        )
        REFERENCES top3_news.news_scores (
            score_id,
            ranking_run_id
        )
        ON DELETE CASCADE,

    CONSTRAINT ranking_combination_items_position_chk
        CHECK (
            position BETWEEN 1 AND 3
        ),

    CONSTRAINT ranking_combination_items_position_uq
        UNIQUE (
            combination_id,
            position
        ),

    CONSTRAINT ranking_combination_items_score_uq
        UNIQUE (
            combination_id,
            score_id
        )
);


CREATE INDEX ranking_combination_items_run_idx
    ON top3_news.ranking_combination_items (
        ranking_run_id,
        combination_id
    );


CREATE INDEX ranking_combination_items_score_idx
    ON top3_news.ranking_combination_items (
        score_id
    );


COMMENT ON TABLE top3_news.ranking_combination_items IS
    'Три news_scores, входящие в конкретную комбинацию TOP(S)';

COMMENT ON COLUMN
    top3_news.ranking_combination_items.position IS
    'Порядок новости внутри комбинации по убыванию B_i';


-- ============================================================================
-- 7. Права пользователя приложения
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE
    ON TABLE
        top3_news.ranking_events,
        top3_news.ranking_event_members,
        top3_news.ranking_audience_metrics,
        top3_news.ranking_combinations,
        top3_news.ranking_combination_items
    TO top3_news_app;


GRANT USAGE, SELECT, UPDATE
    ON ALL SEQUENCES IN SCHEMA top3_news
    TO top3_news_app;


-- ============================================================================
-- 8. Регистрация миграции
-- ============================================================================

INSERT INTO top3_news.schema_migrations (
    version,
    description
)
VALUES (
    '004',
    'Add full event and combination model for TOP-3 formula'
);

COMMIT;