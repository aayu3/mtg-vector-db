# MTG Vector Database - Quick Start Guide

This guide will help you get the MTG vector database up and running with pgvector and Ollama.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MTG Vector Database                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Cards     │  │    Rules     │  │   Glossary   │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ mtg_cards    │  │ mtg_rules    │  │mtg_glossary  │      │
│  │   (~30k)     │  │   (~2k)      │  │   (~700)     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │   Embeddings │  │   Embeddings │  │   Embeddings │      │
│  │  (HNSW idx)  │  │  (HNSW idx)  │  │  (HNSW idx)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
                   ┌────────┴────────┐
                   │  Ollama (nomic) │
                   │  Embeddings API │
                   └─────────────────┘
```

## Prerequisites

1. **Docker & Docker Compose** installed
2. **Python 3.8+** installed
3. **NVIDIA GPU** (optional, for faster embeddings)

## Step-by-Step Setup

### 1. Start Docker Services

```bash
# Start PostgreSQL with pgvector and Ollama
docker-compose up -d

# Verify containers are running
docker ps
```

You should see:
- `mtg-pgvector` - PostgreSQL with pgvector extension
- `mtg-ollama` - Ollama for generating embeddings

### 2. Pull Ollama Embedding Model

```bash
# Pull the embeddinggemma:300m model (768-dimensional embeddings)
docker exec -it mtg-ollama ollama pull embeddinggemma:300m
```

**Note:** The init.sql schema is configured for 768-dimensional vectors. You may need to adjust this based on your model:
- `embeddinggemma:300m` = 768 dimensions
- Update `vector(768)` to `vector(1025)` in [init.sql](db/init.sql) if needed

### 3. Install Python Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 4. Verify Database Schema

The database schema is automatically created on first startup via `docker-entrypoint-initdb.d/init.sql`.

To verify:
```bash
docker exec -it mtg-pgvector psql -U postgres -d mtg_vectors -c "\dt"
```

You should see 6 tables:
- `mtg_cards`, `mtg_card_embeddings`
- `mtg_rules`, `mtg_rule_embeddings`
- `mtg_glossary`, `mtg_glossary_embeddings`

### 5. Run Data Ingestion

**Option A: Ingest Everything (Recommended)**
```bash
cd db
python ingest_all.py
```

**Option B: Ingest Individually**
```bash
cd db

# Ingest glossary (~2-5 minutes)
python ingest_glossary.py

# Ingest rules (~5-10 minutes)
python ingest_rules.py

# Ingest cards (~30-60 minutes)
python ingest_cards.py
```

⏱️ **Expected time:** 45-75 minutes total for full ingestion

### 6. Test the Database

```bash
cd db
python query_example.py
```

This will:
- Show database statistics
- Run example similarity searches on cards, rules, and glossary
- Demonstrate the vector search capabilities

## Usage Examples

### Python API

```python
from db.db_utils import DatabaseConnection, OllamaEmbedder, format_vector_for_postgres

# Generate embedding for a query
embedder = OllamaEmbedder()
query_embedding = embedder.generate_embedding("flying creature with deathtouch")

# Search for similar cards
with DatabaseConnection() as db:
    results = db.execute(
        "SELECT * FROM search_similar_cards(%s, 0.7, 10)",
        (format_vector_for_postgres(query_embedding),)
    ).fetchall()

    for card_name, card_data, similarity in results:
        print(f"{card_name}: {similarity:.3f}")
```

### Direct SQL

```sql
-- Find cards by name
SELECT card_name, card_type, colors
FROM mtg_cards
WHERE card_name ILIKE '%dragon%'
LIMIT 10;

-- Find rules by section
SELECT rule_number, text
FROM mtg_rules
WHERE section_name = 'Combat'
ORDER BY rule_number;

-- Search glossary
SELECT term, definition
FROM mtg_glossary
WHERE term ILIKE '%flash%';
```

## Configuration

### Database Connection
Edit connection parameters in the ingestion scripts:
```python
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "mtg_vectors",
    "user": "postgres",
    "password": "postgres"
}
```

### Ollama Configuration
```python
ollama_config = {
    "base_url": "http://localhost:11434",
    "model": "embeddinggemma:300m"
}
```

## Troubleshooting

### PostgreSQL Won't Start
```bash
# Check logs
docker logs mtg-pgvector

# Restart container
docker-compose restart postgres
```

### Ollama Won't Start
```bash
# Check logs
docker logs mtg-ollama

# Test Ollama API
curl http://localhost:11434/api/tags
```

### Embedding Generation Fails
```bash
# Pull the model again
docker exec -it mtg-ollama ollama pull embeddinggemma:300m

# Check if model is available
docker exec -it mtg-ollama ollama list
```

### Schema Already Exists Error
If you need to recreate the schema:
```bash
# Drop and recreate database
docker exec -it mtg-pgvector psql -U postgres -c "DROP DATABASE mtg_vectors;"
docker exec -it mtg-pgvector psql -U postgres -c "CREATE DATABASE mtg_vectors;"
docker exec -it mtg-pgvector psql -U postgres -d mtg_vectors -f /docker-entrypoint-initdb.d/init.sql
```

## Performance Tips

1. **Use GPU for Ollama** - Uncomment the GPU section in `docker-compose.yaml`
2. **Adjust batch sizes** - Increase batch sizes in ingestion scripts if you have more RAM
3. **HNSW index tuning** - Adjust `m` and `ef_construction` in `init.sql` for your use case

## Next Steps

- Build a RAG (Retrieval-Augmented Generation) system using the embeddings
- Create a web interface for searching cards and rules
- Experiment with different embedding models
- Fine-tune similarity thresholds for your specific queries

## Directory Structure

```
mtg-vector-db/
├── db/
│   ├── init.sql                 # Database schema
│   ├── db_utils.py              # Shared utilities
│   ├── glossary_parser.py       # Glossary text parser
│   ├── ingest_all.py            # Master ingestion script
│   ├── ingest_cards.py          # Card ingestion
│   ├── ingest_rules.py          # Rules ingestion
│   ├── ingest_glossary.py       # Glossary ingestion
│   ├── query_example.py         # Example queries
│   └── README.md                # Detailed documentation
├── cardsCleaning/
│   ├── ModernAtomic_cleaned.json
│   └── cleanCardJson.py
├── rulesCleaning/
│   ├── rules_individual.json
│   ├── MagicRulesGlossary.txt
│   └── parse_rules_both_versions.py
├── docker-compose.yaml
├── requirements.txt
└── QUICKSTART.md               # This file
```

## Resources

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Ollama Documentation](https://ollama.ai/)
- [MTGJSON Documentation](https://mtgjson.com/)
