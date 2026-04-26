CREATE TABLE IF NOT EXISTS legal_units (
    id text PRIMARY KEY,
    canonical_id text NULL,
    source_id text NULL,
    law_id text NOT NULL,
    law_title text NOT NULL,
    act_type text NULL,
    act_number text NULL,
    publication_date date NULL,
    effective_date date NULL,
    version_start date NULL,
    version_end date NULL,
    status text NOT NULL DEFAULT 'unknown',
    hierarchy_path jsonb NULL,
    article_number text NULL,
    paragraph_number text NULL,
    letter_number text NULL,
    point_number text NULL,
    raw_text text NOT NULL,
    normalized_text text NULL,
    legal_domain text NULL,
    legal_concepts jsonb NULL,
    source_url text NULL,
    parent_id text NULL,
    parser_warnings jsonb NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS canonical_id text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS source_id text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS law_id text NOT NULL DEFAULT 'unknown';
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS law_title text NOT NULL DEFAULT 'unknown';
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS act_type text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS act_number text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS publication_date date NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS effective_date date NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS version_start date NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS version_end date NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'unknown';
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS hierarchy_path jsonb NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS article_number text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS paragraph_number text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS letter_number text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS point_number text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS raw_text text NOT NULL DEFAULT '';
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS normalized_text text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS legal_domain text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS legal_concepts jsonb NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS source_url text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS parent_id text NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS parser_warnings jsonb NULL;
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE legal_units ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS legal_edges (
    id text PRIMARY KEY,
    source_id text NOT NULL,
    target_id text NOT NULL,
    type text NOT NULL,
    weight double precision NULL,
    confidence double precision NULL,
    metadata jsonb NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS source_id text NOT NULL DEFAULT '';
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS target_id text NOT NULL DEFAULT '';
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS type text NOT NULL DEFAULT '';
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS weight double precision NULL;
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS confidence double precision NULL;
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS metadata jsonb NULL;
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE legal_edges ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS import_runs (
    id text PRIMARY KEY,
    source_dir text NULL,
    mode text NULL,
    status text NULL,
    counts jsonb NULL,
    warnings jsonb NULL,
    errors jsonb NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz NULL
);

ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS source_dir text NULL;
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS mode text NULL;
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS status text NULL;
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS counts jsonb NULL;
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS warnings jsonb NULL;
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS errors jsonb NULL;
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE import_runs ADD COLUMN IF NOT EXISTS finished_at timestamptz NULL;
