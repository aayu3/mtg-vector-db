# MTG Vector Database

PGVector database for hosting MTGJSON data with semantic search capabilities using Ollama embeddings.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

```bash
# 1. Start services
docker-compose up -d

# 2. Pull embedding model
docker exec -it mtg-ollama ollama pull embeddinggemma:300m

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Ingest all data
cd db && python ingest_all.py
```
`You are an expert game designer for Magic: The Gathering. Please identify any possible Magic: The Gathering cards and card descriptions and indentifiers and return them as a JSON object with two lists {"cardNames": [], "cardDescriptions" : []} in the following text "I cast Humility and my opponent casts an enchantment that makes all enchantments into 1/1 creatures"`

You are an expert Magic: The Gathering game designer, please reword the following descriptions of Magic: The Gatheri
... ng cards to use language like existing cards. Don't add any information, just reword the description given in a way that doesn't add any information but uses the terminology of existing cards.
## Database Schema

The database contains 6 tables organized in 3 document/embedding pairs:

### Cards
- **`mtg_cards`** - ~30,000 card documents with metadata (name, type, colors, text, etc.)
- **`mtg_card_embeddings`** - Vector embeddings for semantic card search (HNSW indexed)

### Rules
- **`mtg_rules`** - ~2,000 individual rule entries from MTG comprehensive rules
- **`mtg_rule_embeddings`** - Vector embeddings for semantic rule search (HNSW indexed)

### Glossary
- **`mtg_glossary`** - ~700 glossary terms and definitions
- **`mtg_glossary_embeddings`** - Vector embeddings for semantic glossary search (HNSW indexed)

All embedding tables use **HNSW indexes** for fast approximate nearest neighbor search.

## Project Structure

```
mtg-vector-db/
├── db/                          # Database scripts and utilities
│   ├── init.sql                 # Database schema with HNSW indexes
│   ├── db_utils.py              # Database & Ollama utilities
│   ├── ingest_*.py              # Data ingestion scripts
│   └── query_example.py         # Example queries
├── cardsCleaning/               # Card data processing
├── rulesCleaning/               # Rules data processing
├── docker-compose.yaml          # PostgreSQL + Ollama services
└── requirements.txt             # Python dependencies
```

## Features

- ✅ Semantic search across cards, rules, and glossary
- ✅ HNSW indexing for fast vector similarity search
- ✅ Separate document and embedding tables for clean architecture
- ✅ PostgreSQL with pgvector extension
- ✅ Local embeddings using Ollama (no API costs)
- ✅ GPU acceleration support
- ✅ Batch processing with progress tracking

## Usage Examples

### Search Similar Cards
```python
from db.db_utils import DatabaseConnection, OllamaEmbedder, format_vector_for_postgres

embedder = OllamaEmbedder()
query = embedder.generate_embedding("flying creature with deathtouch")

with DatabaseConnection() as db:
    results = db.execute(
        "SELECT * FROM search_similar_cards(%s, 0.7, 10)",
        (format_vector_for_postgres(query),)
    ).fetchall()
```

### Direct SQL Queries
```sql
-- Search cards by name
SELECT card_name, card_type FROM mtg_cards WHERE card_name ILIKE '%dragon%';

-- Find rules by section
SELECT rule_number, text FROM mtg_rules WHERE section_name = 'Combat';

-- Glossary lookup
SELECT term, definition FROM mtg_glossary WHERE term ILIKE '%flash%';
```













Logic Flow

First Step
* Take in user prompt
* Use a lighter weight model to extract the following 
  * Card Names
  * Card Descriptions
    * Reference Glossary to indentify key terms extract
  * Maybe identify rules
  