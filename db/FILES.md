# Database Files Reference

This document describes all files in the `db/` directory.

## Core Files

### `init.sql`
**Purpose:** Database schema initialization script
**Used by:** Docker PostgreSQL container (auto-executed on startup)

**Contents:**
- Creates pgvector extension
- Defines 6 tables (cards, rules, glossary + their embedding tables)
- Creates HNSW indexes on all embedding columns
- Defines helper functions for similarity search:
  - `search_similar_cards()`
  - `search_similar_rules()`
  - `search_similar_glossary()`

**Key Parameters:**
- Vector dimension: `vector(1024)` (adjust based on your embedding model)
- HNSW index: `m=16, ef_construction=64`

---

### `db_utils.py`
**Purpose:** Shared utilities for database and embedding operations

**Classes:**
- `DatabaseConnection`: PostgreSQL connection manager with context manager support
- `OllamaEmbedder`: Generates embeddings using Ollama API

**Functions:**
- `wait_for_postgres()`: Health check for PostgreSQL
- `wait_for_ollama()`: Health check for Ollama
- `format_vector_for_postgres()`: Formats embedding arrays for pgvector

**Usage:**
```python
from db_utils import DatabaseConnection, OllamaEmbedder

embedder = OllamaEmbedder()
embedding = embedder.generate_embedding("sample text")

with DatabaseConnection() as db:
    db.execute("SELECT 1")
```

---

### `glossary_parser.py`
**Purpose:** Parses `MagicRulesGlossary.txt` into structured JSON

**Class:**
- `GlossaryParser`: Extracts terms, definitions, and related rule references

**Features:**
- Splits glossary by double newlines
- Extracts rule references (e.g., "rule 123.4")
- Extracts section references (e.g., "section 5")

**Usage:**
```python
from glossary_parser import GlossaryParser

parser = GlossaryParser('../rulesCleaning/MagicRulesGlossary.txt')
entries = parser.parse()
parser.save('glossary.json')
```

---

## Ingestion Scripts

### `ingest_cards.py`
**Purpose:** Load cards from `ModernAtomic_cleaned.json` and generate embeddings

**Key Functions:**
- `create_card_embedding_text()`: Formats card data for embedding
- `extract_card_fields()`: Extracts searchable fields
- `ingest_cards()`: Main ingestion logic

**Process:**
1. Load card JSON
2. Insert into `mtg_cards` table
3. Generate embeddings via Ollama
4. Insert into `mtg_card_embeddings` table
5. Process in batches (default: 100 cards)

**Estimated Time:** 30-60 minutes (~30k cards)

---

### `ingest_rules.py`
**Purpose:** Load rules from `rules_individual.json` and generate embeddings

**Key Functions:**
- `create_rule_embedding_text()`: Formats rule data for embedding
- `ingest_rules()`: Main ingestion logic

**Process:**
1. Load rules JSON
2. Insert into `mtg_rules` table
3. Generate embeddings via Ollama
4. Insert into `mtg_rule_embeddings` table
5. Process in batches (default: 100 rules)

**Estimated Time:** 5-10 minutes (~2k rules)

---

### `ingest_glossary.py`
**Purpose:** Parse glossary text file and load with embeddings

**Key Functions:**
- `create_glossary_embedding_text()`: Formats glossary data for embedding
- `ingest_glossary()`: Main ingestion logic

**Process:**
1. Parse `MagicRulesGlossary.txt` using `GlossaryParser`
2. Insert into `mtg_glossary` table
3. Generate embeddings via Ollama
4. Insert into `mtg_glossary_embeddings` table
5. Process in batches (default: 50 entries)

**Estimated Time:** 2-5 minutes (~700 entries)

---

### `ingest_all.py`
**Purpose:** Master script that runs all ingestion processes sequentially

**Process:**
1. Ingest glossary
2. Ingest rules
3. Ingest cards

**Features:**
- Shows progress for each step
- Reports timing for each ingestion
- Final summary with success/failure status

**Usage:**
```bash
python ingest_all.py
```

**Estimated Time:** 45-75 minutes total

---

## Example & Utility Scripts

### `query_example.py`
**Purpose:** Demonstrates how to query the database

**Functions:**
- `search_similar_cards()`: Vector similarity search for cards
- `search_similar_rules()`: Vector similarity search for rules
- `search_similar_glossary()`: Vector similarity search for glossary
- `database_stats()`: Shows counts and statistics
- `run_example_queries()`: Runs several example searches

**Usage:**
```bash
python query_example.py
```

---

## Documentation

### `README.md`
Detailed documentation for the database scripts, including:
- Database schema overview
- Installation instructions
- Usage examples
- Configuration options
- Troubleshooting guide

### `FILES.md` (this file)
Reference documentation for all files in the `db/` directory

---

## Configuration

All scripts use similar configuration:

```python
# Database connection
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "mtg_vectors",
    "user": "postgres",
    "password": "postgres"
}

# Ollama embeddings
ollama_config = {
    "base_url": "http://localhost:11434",
    "model": "nomic-embed-text:latest"
}
```

---

## Dependencies

See `../requirements.txt`:
- `psycopg2-binary` - PostgreSQL adapter
- `pgvector` - pgvector support
- `requests` - Ollama API calls
- `tqdm` - Progress bars

---

## File Dependencies Graph

```
init.sql
    └── (used by docker-compose.yaml)

db_utils.py
    ├── ingest_cards.py
    ├── ingest_rules.py
    ├── ingest_glossary.py
    └── query_example.py

glossary_parser.py
    └── ingest_glossary.py

ingest_cards.py
    └── ingest_all.py

ingest_rules.py
    └── ingest_all.py

ingest_glossary.py
    └── ingest_all.py
```

---

## Data Flow

```
1. Data Preparation:
   cardsCleaning/ModernAtomic_cleaned.json → ingest_cards.py
   rulesCleaning/rules_individual.json → ingest_rules.py
   rulesCleaning/MagicRulesGlossary.txt → glossary_parser.py → ingest_glossary.py

2. Database Insertion:
   ingest_*.py → PostgreSQL (mtg_cards, mtg_rules, mtg_glossary)

3. Embedding Generation:
   ingest_*.py → Ollama API → PostgreSQL (mtg_*_embeddings)

4. Querying:
   query_example.py → PostgreSQL → Results
```
