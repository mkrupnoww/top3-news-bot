BEGIN;
SET LOCAL search_path = top3_news, public;
SET LOCAL TIME ZONE 'UTC';

-- Планирование — отдельная платная стадия. Reservation фиксируется до
-- Responses API, а completed exact prompt — до резервирования Image API.
CREATE TABLE top3_news.image_prompt_plans (
    image_prompt_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    batch_id bigint NOT NULL REFERENCES publication_batches ON DELETE CASCADE,
    generated_post_id bigint NOT NULL REFERENCES generated_posts ON DELETE CASCADE,
    daily_workflow_run_id bigint REFERENCES daily_workflow_runs ON DELETE SET NULL,
    review_action_id bigint REFERENCES review_actions ON DELETE CASCADE,
    source_prompt_id bigint REFERENCES image_prompt_plans ON DELETE SET NULL,
    source_image_generation_id bigint REFERENCES image_generation_requests ON DELETE SET NULL,
    prompt_request_key text NOT NULL CHECK (prompt_request_key ~ '^[0-9a-f]{64}$'),
    prompt_status text NOT NULL DEFAULT 'reserved'
        CHECK (prompt_status IN ('reserved', 'completed', 'failed')),
    request_kind text NOT NULL CHECK (request_kind IN ('initial', 'regenerate')),
    attempt_kind text NOT NULL CHECK (attempt_kind IN (
        'initial', 'moderation_recovery', 'safe_editorial_fallback', 'editorial_revision'
    )),
    attempt_number integer NOT NULL CHECK (attempt_number > 0),
    prompt_mode text NOT NULL CHECK (prompt_mode IN (
        'normal_creative', 'moderation_recovery', 'safe_editorial_fallback'
    )),
    agent_model text NOT NULL CHECK (btrim(agent_model) <> ''),
    agent_version text NOT NULL,
    agent_prompt_version text NOT NULL,
    image_prompt_version text NOT NULL,
    agent_instructions text NOT NULL,
    input_payload jsonb NOT NULL CHECK (jsonb_typeof(input_payload) = 'object'),
    final_image_prompt text,
    structured_plan jsonb,
    response_metadata jsonb,
    openai_usage jsonb,
    openai_cost jsonb,
    error_type text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    failed_at timestamptz,
    CONSTRAINT image_prompt_context_chk CHECK (
        (request_kind = 'initial' AND review_action_id IS NULL)
        OR (request_kind = 'regenerate' AND review_action_id IS NOT NULL)
    ),
    CONSTRAINT image_prompt_usage_cost_chk CHECK (
        (openai_usage IS NULL AND openai_cost IS NULL)
        OR (openai_usage IS NOT NULL AND openai_cost IS NOT NULL
            AND jsonb_typeof(openai_usage) = 'object' AND jsonb_typeof(openai_cost) = 'object')
    ),
    CONSTRAINT image_prompt_completed_chk CHECK (
        (prompt_status = 'reserved' AND final_image_prompt IS NULL
            AND structured_plan IS NULL AND completed_at IS NULL AND failed_at IS NULL)
        OR (prompt_status = 'completed' AND final_image_prompt IS NOT NULL
            AND structured_plan IS NOT NULL AND btrim(final_image_prompt) <> ''
            AND octet_length(final_image_prompt) <= 4000
            AND jsonb_typeof(structured_plan) = 'object'
            AND structured_plan ->> 'final_image_prompt' = final_image_prompt
            AND completed_at IS NOT NULL AND failed_at IS NULL)
        OR (prompt_status = 'failed' AND final_image_prompt IS NULL
            AND completed_at IS NULL AND failed_at IS NOT NULL
            AND error_type IS NOT NULL AND error_message IS NOT NULL)
    )
);

CREATE UNIQUE INDEX image_prompt_plans_active_key_uq
    ON image_prompt_plans (prompt_request_key)
    WHERE prompt_status IN ('reserved', 'completed');
CREATE UNIQUE INDEX image_prompt_plans_post_reserved_uq
    ON image_prompt_plans (generated_post_id) WHERE prompt_status = 'reserved';
CREATE UNIQUE INDEX image_prompt_plans_attempt_uq
    ON image_prompt_plans (generated_post_id, request_kind, COALESCE(review_action_id, 0), attempt_number);
CREATE INDEX image_prompt_plans_post_idx ON image_prompt_plans (generated_post_id, image_prompt_id DESC);

ALTER TABLE image_generation_requests ADD COLUMN image_prompt_id bigint
    REFERENCES image_prompt_plans ON DELETE SET NULL;
CREATE INDEX image_generation_requests_prompt_idx
    ON image_generation_requests (image_prompt_id) WHERE image_prompt_id IS NOT NULL;

GRANT SELECT, INSERT, UPDATE, DELETE ON image_prompt_plans TO top3_news_app;
GRANT USAGE, SELECT ON SEQUENCE image_prompt_plans_image_prompt_id_seq TO top3_news_app;
COMMENT ON TABLE image_prompt_plans IS
    'Сохранённые попытки Image Prompt Agent: точный prompt, TOP-3, режим, parent и стоимость';
COMMENT ON COLUMN image_generation_requests.image_prompt_id IS
    'Отдельный completed ImagePromptPlan; NULL для исторических и локальных image requests';

INSERT INTO schema_migrations (version, description)
VALUES ('019', 'Add saved Image Prompt Agent plans and image request linkage');
COMMIT;
