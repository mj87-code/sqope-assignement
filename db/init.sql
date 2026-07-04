CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE text_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384),
    page_number INT,
    chunk_index INT,
    metadata JSONB DEFAULT '{}'
);
-- No ANN index by design: at this scale (a few PDFs → up to a few thousand
-- chunks) exact KNN via sequential scan is sub-millisecond and gives 100% recall.
-- An ivfflat index over few vectors leaves most lists empty, so the default
-- probes=1 returns 0 rows for novel query vectors. For a large corpus, switch to
-- HNSW: CREATE INDEX ON text_chunks USING hnsw (embedding vector_cosine_ops);

-- Schema-agnostic: column names live inside row_data JSONB.
-- Schema is discovered at query time via get_schema_catalog().
CREATE TABLE table_rows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    table_name TEXT NOT NULL,
    row_data JSONB NOT NULL,
    row_index INT,
    metadata JSONB DEFAULT '{}'
);
CREATE INDEX ON table_rows (table_name);
CREATE INDEX ON table_rows USING gin (row_data);
