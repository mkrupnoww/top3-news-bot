BEGIN;
SET LOCAL search_path=top3_news,public;
CREATE TABLE editorial_replacement_requests (
 review_action_id bigint PRIMARY KEY REFERENCES review_actions ON DELETE CASCADE,
 source_generated_post_id bigint NOT NULL REFERENCES generated_posts,
 combination_id bigint NOT NULL REFERENCES ranking_combinations,
 source_news_ids bigint[] NOT NULL CHECK(cardinality(source_news_ids)=3),
 excluded_news_ids bigint[] NOT NULL,
 replacement_positions smallint[] NOT NULL,
 new_batch_id bigint REFERENCES publication_batches,
 generated_post_id bigint REFERENCES generated_posts,
 replacement_status text NOT NULL DEFAULT 'planned' CHECK(replacement_status IN ('planned','text_ready','completed')),
 error_message text,
 created_at timestamptz NOT NULL DEFAULT now(),
 completed_at timestamptz
);
COMMENT ON TABLE editorial_replacement_requests IS 'Restart-safe замена выбранных новостей по замечанию редактора; новая подборка и изображение';
GRANT SELECT,INSERT,UPDATE,DELETE ON editorial_replacement_requests TO top3_news_app;
INSERT INTO schema_migrations(version,description) VALUES('021','Add explicit editorial news replacement requests');
COMMIT;
