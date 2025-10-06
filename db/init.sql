-- MTG Vector Database Initialization Script
-- Creates tables for Cards, Rules, and Glossary with separate embedding tables
-- Uses HNSW indexing for fast vector similarity search

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- CARDS TABLES
-- ============================================================================

-- Main cards document table
CREATE TABLE IF NOT EXISTS mtg_cards (
    id SERIAL PRIMARY KEY,
    card_name TEXT UNIQUE NOT NULL,
    card_data JSONB NOT NULL,
    text_content TEXT,
    card_type TEXT,
    colors TEXT[],
    mana_value NUMERIC,
    keywords TEXT[],
    legalities JSONB,
    related_faces TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on frequently queried fields
CREATE INDEX IF NOT EXISTS idx_cards_name ON mtg_cards(card_name);
CREATE INDEX IF NOT EXISTS idx_cards_type ON mtg_cards(card_type);
CREATE INDEX IF NOT EXISTS idx_cards_colors ON mtg_cards USING GIN(colors);
CREATE INDEX IF NOT EXISTS idx_cards_mana_value ON mtg_cards(mana_value);
CREATE INDEX IF NOT EXISTS idx_cards_keywords ON mtg_cards USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_cards_data ON mtg_cards USING GIN(card_data);
CREATE INDEX IF NOT EXISTS idx_cards_related_faces ON mtg_cards(related_faces);

-- Card embeddings table
CREATE TABLE IF NOT EXISTS mtg_card_embeddings (
    id SERIAL PRIMARY KEY,
    card_id INTEGER NOT NULL REFERENCES mtg_cards(id) ON DELETE CASCADE,
    embedding vector(768),
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast similarity search on card embeddings
CREATE INDEX IF NOT EXISTS idx_card_embeddings_hnsw
ON mtg_card_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for foreign key lookups
CREATE INDEX IF NOT EXISTS idx_card_embeddings_card_id ON mtg_card_embeddings(card_id);

-- ============================================================================
-- RULES TABLES
-- ============================================================================

-- Main rules document table
CREATE TABLE IF NOT EXISTS mtg_rules (
    id SERIAL PRIMARY KEY,
    rule_number TEXT UNIQUE NOT NULL,
    text TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN ('main_rule', 'subrule')),
    section_parent TEXT NOT NULL,
    section_number TEXT NOT NULL,
    section_name TEXT NOT NULL,
    parent_rule TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on frequently queried fields
CREATE INDEX IF NOT EXISTS idx_rules_number ON mtg_rules(rule_number);
CREATE INDEX IF NOT EXISTS idx_rules_type ON mtg_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_rules_section_parent ON mtg_rules(section_parent);
CREATE INDEX IF NOT EXISTS idx_rules_section_name ON mtg_rules(section_name);
CREATE INDEX IF NOT EXISTS idx_rules_parent_rule ON mtg_rules(parent_rule);

-- Rule embeddings table
CREATE TABLE IF NOT EXISTS mtg_rule_embeddings (
    id SERIAL PRIMARY KEY,
    rule_id INTEGER NOT NULL REFERENCES mtg_rules(id) ON DELETE CASCADE,
    embedding vector(768),
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast similarity search on rule embeddings
CREATE INDEX IF NOT EXISTS idx_rule_embeddings_hnsw
ON mtg_rule_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for foreign key lookups
CREATE INDEX IF NOT EXISTS idx_rule_embeddings_rule_id ON mtg_rule_embeddings(rule_id);

-- ============================================================================
-- GLOSSARY TABLES
-- ============================================================================

-- Main glossary document table
CREATE TABLE IF NOT EXISTS mtg_glossary (
    id SERIAL PRIMARY KEY,
    term TEXT UNIQUE NOT NULL,
    definition TEXT NOT NULL,
    related_rules TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on frequently queried fields
CREATE INDEX IF NOT EXISTS idx_glossary_term ON mtg_glossary(term);
CREATE INDEX IF NOT EXISTS idx_glossary_related_rules ON mtg_glossary USING GIN(related_rules);

-- Glossary embeddings table
CREATE TABLE IF NOT EXISTS mtg_glossary_embeddings (
    id SERIAL PRIMARY KEY,
    glossary_id INTEGER NOT NULL REFERENCES mtg_glossary(id) ON DELETE CASCADE,
    embedding vector(768),
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast similarity search on glossary embeddings
CREATE INDEX IF NOT EXISTS idx_glossary_embeddings_hnsw
ON mtg_glossary_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for foreign key lookups
CREATE INDEX IF NOT EXISTS idx_glossary_embeddings_glossary_id ON mtg_glossary_embeddings(glossary_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to search cards by vector similarity
CREATE OR REPLACE FUNCTION search_similar_cards(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    card_name TEXT,
    card_data JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.card_name,
        c.card_data,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM mtg_card_embeddings e
    JOIN mtg_cards c ON e.card_id = c.id
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search rules by vector similarity
CREATE OR REPLACE FUNCTION search_similar_rules(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    rule_number TEXT,
    text TEXT,
    rule_type TEXT,
    section_name TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.rule_number,
        r.text,
        r.rule_type,
        r.section_name,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM mtg_rule_embeddings e
    JOIN mtg_rules r ON e.rule_id = r.id
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search glossary by vector similarity
CREATE OR REPLACE FUNCTION search_similar_glossary(
    query_embedding vector(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    term TEXT,
    definition TEXT,
    related_rules TEXT[],
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        g.term,
        g.definition,
        g.related_rules,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM mtg_glossary_embeddings e
    JOIN mtg_glossary g ON e.glossary_id = g.id
    WHERE 1 - (e.embedding <=> query_embedding) > match_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS (optional, adjust based on your security needs)
-- ============================================================================

-- Grant permissions to postgres user (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;
