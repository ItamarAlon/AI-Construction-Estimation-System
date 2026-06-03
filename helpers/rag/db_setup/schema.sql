CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS categories (
    id              BIGSERIAL PRIMARY KEY,
    code            TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS merchants (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    normalized_name     TEXT,
    tax_id              TEXT,
    country_code        CHAR(2),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_merchants_name ON merchants(name);
CREATE INDEX IF NOT EXISTS idx_merchants_normalized_name ON merchants(normalized_name);

CREATE TABLE IF NOT EXISTS receipts (
    id                  UUID PRIMARY KEY,
    merchant_id         UUID REFERENCES merchants(id) ON DELETE SET NULL,
    category_id         BIGINT REFERENCES categories(id) ON DELETE SET NULL,
    receipt_date        DATE NOT NULL,
    currency            CHAR(3) NOT NULL DEFAULT 'USD',
    subtotal_amount     NUMERIC(12, 2),
    tax_amount          NUMERIC(12, 2),
    tip_amount          NUMERIC(12, 2),
    total_amount        NUMERIC(12, 2) NOT NULL,
    payment_method      TEXT,
    source              TEXT,
    notes               TEXT,
    raw_payload         JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (total_amount >= 0)
);

CREATE INDEX IF NOT EXISTS idx_receipts_receipt_date ON receipts(receipt_date);
CREATE INDEX IF NOT EXISTS idx_receipts_merchant_id ON receipts(merchant_id);
CREATE INDEX IF NOT EXISTS idx_receipts_category_id ON receipts(category_id);
CREATE INDEX IF NOT EXISTS idx_receipts_currency ON receipts(currency);
CREATE INDEX IF NOT EXISTS idx_receipts_total_amount ON receipts(total_amount);
CREATE INDEX IF NOT EXISTS idx_receipts_raw_payload_gin ON receipts USING GIN (raw_payload);

CREATE TABLE IF NOT EXISTS receipt_items (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_id          UUID NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    line_no             INTEGER NOT NULL,
    description         TEXT NOT NULL,
    quantity            NUMERIC(12, 3) NOT NULL DEFAULT 1,
    unit_price          NUMERIC(12, 2),
    tax_rate_percent    NUMERIC(5, 2),
    line_total          NUMERIC(12, 2) NOT NULL,
    metadata            JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (receipt_id, line_no)
);

CREATE INDEX IF NOT EXISTS idx_receipt_items_receipt_id ON receipt_items(receipt_id);
CREATE INDEX IF NOT EXISTS idx_receipt_items_description ON receipt_items(description);
CREATE INDEX IF NOT EXISTS idx_receipt_items_metadata_gin ON receipt_items USING GIN (metadata);

CREATE TABLE IF NOT EXISTS tags (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS receipt_tags (
    receipt_id      UUID NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    tag_id          BIGINT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (receipt_id, tag_id)
);

CREATE TABLE IF NOT EXISTS receipt_chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_id          UUID NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    content             TEXT NOT NULL,
    metadata            JSONB,
    embedding           VECTOR(1536) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (receipt_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_receipt_chunks_receipt_id ON receipt_chunks(receipt_id);
CREATE INDEX IF NOT EXISTS idx_receipt_chunks_metadata_gin ON receipt_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_receipt_chunks_embedding_ivfflat
    ON receipt_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS schema_embeddings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_name         TEXT NOT NULL DEFAULT 'public',
    object_type         TEXT NOT NULL,
    object_name         TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    content             TEXT NOT NULL,
    metadata            JSONB,
    embedding           VECTOR(1536) NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (schema_name, object_type, object_name, content_hash),
    CHECK (object_type IN ('table', 'column', 'index', 'view'))
);

CREATE INDEX IF NOT EXISTS idx_schema_embeddings_type_name
    ON schema_embeddings(schema_name, object_type, object_name);
CREATE INDEX IF NOT EXISTS idx_schema_embeddings_embedding_ivfflat
    ON schema_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

INSERT INTO categories (code, name, description)
VALUES
    ('food', 'Food', 'Meals, groceries, cafes and restaurants'),
    ('travel', 'Travel', 'Flights, trains, taxis, fuel, lodging'),
    ('office', 'Office', 'Office supplies, equipment and subscriptions'),
    ('utilities', 'Utilities', 'Electricity, water, internet, phone'),
    ('health', 'Health', 'Medical and pharmacy expenses'),
    ('other', 'Other', 'Unclassified expenses')
ON CONFLICT (code) DO NOTHING;

COMMENT ON TABLE receipts IS 'Main receipt header data and extracted raw payload.';
COMMENT ON TABLE receipt_items IS 'Itemized lines parsed from each receipt.';
COMMENT ON TABLE receipt_chunks IS 'Chunked receipt text with embeddings for semantic retrieval.';
COMMENT ON TABLE schema_embeddings IS 'Embedded DB schema descriptions for agent SQL planning.';
