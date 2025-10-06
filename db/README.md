# MTG Vector Database - Ingestion Scripts

This directory contains scripts for ingesting MTG data (cards, rules, and glossary) into a PostgreSQL database with pgvector embeddings.

## Database Schema

The database consists of 6 tables organized in 3 pairs:

### Cards
- `mtg_cards` - Card documents with metadata
- `mtg_card_embeddings` - Vector embeddings for semantic search (HNSW indexed)

### Rules
- `mtg_rules` - Individual rule entries from MTG comprehensive rules
- `mtg_rule_embeddings` - Vector embeddings for semantic search (HNSW indexed)

### Glossary
- `mtg_glossary` - Glossary terms and definitions
- `mtg_glossary_embeddings` - Vector embeddings for semantic search (HNSW indexed)

## Prerequisites

1. **Docker containers running:**
   ```bash
   docker-compose up -d
   ```

2. **Python dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ollama model pulled:**
   ```bash
   docker exec -it mtg-ollama ollama pull nomic-embed-text
   ```

## Data Files Required

- `../cardsCleaning/ModernAtomic_cleaned.json` - Cleaned card data
- `../rulesCleaning/rules_individual.json` - Individual rules
- `../rulesCleaning/MagicRulesGlossary.txt` - Glossary text file

## Usage

### Option 1: Run All Ingestion Scripts

```bash
python ingest_all.py
```

This will sequentially ingest:
1. Glossary entries
2. Rules
3. Cards

### Option 2: Run Individual Ingestion Scripts

```bash
# Ingest glossary only
python ingest_glossary.py

# Ingest rules only
python ingest_rules.py

# Ingest cards only (this will take the longest)
python ingest_cards.py
```

## Scripts Overview

### Core Utilities

- **`db_utils.py`** - Database connection and Ollama embedding utilities
- **`glossary_parser.py`** - Parses glossary text file into JSON

### Ingestion Scripts

- **`ingest_cards.py`** - Loads cards and generates embeddings
- **`ingest_rules.py`** - Loads rules and generates embeddings
- **`ingest_glossary.py`** - Parses and loads glossary with embeddings
- **`ingest_all.py`** - Master script that runs all ingestion in sequence

### Database Schema

- **`init.sql`** - Creates all tables, indexes, and helper functions

## Database Schema Features

### HNSW Indexing
All embedding tables use HNSW (Hierarchical Navigable Small World) indexes for fast approximate nearest neighbor search:
- `m = 16` - Number of connections per layer
- `ef_construction = 64` - Size of dynamic candidate list during construction

### Helper Functions

The schema includes SQL functions for similarity search:

```sql
-- Search similar cards
SELECT * FROM search_similar_cards(query_embedding, threshold, limit);

-- Search similar rules
SELECT * FROM search_similar_rules(query_embedding, threshold, limit);

-- Search similar glossary entries
SELECT * FROM search_similar_glossary(query_embedding, threshold, limit);
```

## Configuration

Default configuration (can be modified in each script):

```python
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "mtg_vectors",
    "user": "postgres",
    "password": "postgres"
}

ollama_config = {
    "base_url": "http://localhost:11434",
    "model": "nomic-embed-text:latest"
}
```

## Performance Notes

- **Cards**: ~30,000+ cards, expect ~30-60 minutes for full ingestion
- **Rules**: ~2,000+ rules, expect ~5-10 minutes
- **Glossary**: ~700+ entries, expect ~2-5 minutes

Total time: ~45-75 minutes (depends on your GPU/CPU and Ollama model)

## Troubleshooting

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker ps | grep mtg-pgvector

# Check PostgreSQL logs
docker logs mtg-pgvector
```

### Ollama Connection Issues
```bash
# Check if Ollama is running
docker ps | grep mtg-ollama

# Check Ollama logs
docker logs mtg-ollama

# Test Ollama endpoint
curl http://localhost:11434/api/tags
```

### Re-running Ingestion
Each ingestion script clears existing data before inserting new data. To append instead, comment out the DELETE statements in each script.

## Example Queries

### Find similar cards by embedding
```python
from db_utils import DatabaseConnection, OllamaEmbedder

embedder = OllamaEmbedder()
query_embedding = embedder.generate_embedding("flying creature with lifelink")

with DatabaseConnection() as db:
    results = db.execute(
        "SELECT * FROM search_similar_cards(%s, 0.7, 10)",
        (format_vector_for_postgres(query_embedding),)
    ).fetchall()
```

### Search by card name
```sql
SELECT card_name, card_type, colors, text_content
FROM mtg_cards
WHERE card_name ILIKE '%dragon%'
LIMIT 10;
```

### Find rules by section
```sql
SELECT rule_number, text
FROM mtg_rules
WHERE section_name = 'Turn Structure'
ORDER BY rule_number;
```
