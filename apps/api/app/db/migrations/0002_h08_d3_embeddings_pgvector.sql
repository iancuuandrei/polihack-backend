CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS legal_embeddings (
    id bigserial PRIMARY KEY,
    record_id text NOT NULL,
    legal_unit_id text NULL,
    chunk_id text NULL,
    model_name text NOT NULL,
    embedding_dim integer NOT NULL,
    text_hash text NOT NULL,
    embedding vector(2560) NOT NULL,
    metadata jsonb NULL,
    source_path text NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT legal_embeddings_identity_unique UNIQUE (record_id, model_name, text_hash)
);

ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS record_id text NOT NULL DEFAULT '';
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS legal_unit_id text NULL;
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS chunk_id text NULL;
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS model_name text NOT NULL DEFAULT '';
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS embedding_dim integer NOT NULL DEFAULT 2560;
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS text_hash text NOT NULL DEFAULT '';
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS embedding vector(2560);
ALTER TABLE legal_embeddings ALTER COLUMN embedding SET NOT NULL;
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS metadata jsonb NULL;
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS source_path text NULL;
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE legal_embeddings ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

CREATE UNIQUE INDEX IF NOT EXISTS legal_embeddings_identity_unique
    ON legal_embeddings (record_id, model_name, text_hash);
CREATE INDEX IF NOT EXISTS idx_embeddings_record_id
    ON legal_embeddings (record_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_legal_unit_id
    ON legal_embeddings (legal_unit_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name
    ON legal_embeddings (model_name);
