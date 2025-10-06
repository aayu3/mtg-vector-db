"""
Ingest MTG glossary into PostgreSQL with vector embeddings.
First parses MagicRulesGlossary.txt, then loads into database with embeddings.
"""

import json
import sys
import os
from typing import Dict, List, Any
from db_utils import DatabaseConnection, OllamaEmbedder, wait_for_postgres, wait_for_ollama, format_vector_for_postgres

# Add parent directory to path to import glossary_parser from rulesCleaning
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rulesCleaning.glossary_parser import GlossaryParser


def create_glossary_embedding_text(entry: Dict[str, Any]) -> str:
    """
    Create text representation of a glossary entry for embedding.

    Combines term and definition.
    """
    parts = [f"Term: {entry['term']}"]
    parts.append(f"Definition: {entry['definition']}")

    if entry.get('related_rules'):
        parts.append(f"Related Rules: {', '.join(entry['related_rules'])}")

    return '\n'.join(parts)


def ingest_glossary(
    glossary_file: str,
    db_config: Dict[str, Any] = None,
    ollama_config: Dict[str, Any] = None,
    batch_size: int = 50
):
    """
    Parse glossary file and ingest into database.

    Args:
        glossary_file: Path to MagicRulesGlossary.txt
        db_config: Database connection configuration
        ollama_config: Ollama embedder configuration
        batch_size: Number of entries to process before committing
    """
    # Default configurations
    if db_config is None:
        db_config = {}
    if ollama_config is None:
        ollama_config = {}

    print("=" * 80)
    print("MTG Glossary Ingestion")
    print("=" * 80)

    # Wait for services
    if not wait_for_postgres(**db_config):
        print("Error: PostgreSQL not available")
        return False

    if not wait_for_ollama(**ollama_config):
        print("Error: Ollama not available")
        return False

    # Parse glossary file
    print(f"\nParsing glossary from {glossary_file}...")
    try:
        parser = GlossaryParser(glossary_file)
        glossary_data = parser.parse()
    except Exception as e:
        print(f"Error parsing glossary file: {e}")
        return False

    print(f"Parsed {len(glossary_data)} glossary entries")

    # Initialize database and embedder
    embedder = OllamaEmbedder(**ollama_config)

    with DatabaseConnection(**db_config) as db:
        # Clear existing data (optional - remove if you want to append)
        print("\nClearing existing glossary data...")
        db.execute("DELETE FROM mtg_glossary_embeddings")
        db.execute("DELETE FROM mtg_glossary")
        db.commit()

        # Process glossary in batches
        total_entries = len(glossary_data)
        entries_inserted = 0
        embeddings_inserted = 0

        print(f"\nProcessing {total_entries} glossary entries in batches of {batch_size}...")

        for i in range(0, total_entries, batch_size):
            batch = glossary_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_entries + batch_size - 1) // batch_size

            print(f"\n--- Batch {batch_num}/{total_batches} ---")

            # Prepare glossary records
            glossary_records = []
            embedding_texts = []

            for entry in batch:
                glossary_records.append((
                    entry['term'],
                    entry['definition'],
                    entry['related_rules'] if entry['related_rules'] else []
                ))

                # Create embedding text
                embedding_text = create_glossary_embedding_text(entry)
                embedding_texts.append(embedding_text)

            # Insert glossary entries
            print(f"  Inserting {len(glossary_records)} glossary entries into database...")
            insert_glossary_query = """
                INSERT INTO mtg_glossary (term, definition, related_rules)
                VALUES (%s, %s, %s)
                RETURNING id
            """

            glossary_ids = []
            for record in glossary_records:
                db.execute(insert_glossary_query, record)
                glossary_id = db.cursor.fetchone()[0]
                glossary_ids.append(glossary_id)

            entries_inserted += len(glossary_ids)

            # Generate embeddings
            print(f"  Generating embeddings for {len(embedding_texts)} glossary entries...")
            embeddings = embedder.generate_embeddings_batch(
                embedding_texts,
                batch_size=10,
                show_progress=True
            )

            # Insert embeddings
            print(f"  Inserting embeddings into database...")
            insert_embedding_query = """
                INSERT INTO mtg_glossary_embeddings (glossary_id, embedding, embedding_model)
                VALUES (%s, %s, %s)
            """

            embedding_records = []
            for glossary_id, embedding in zip(glossary_ids, embeddings):
                if embedding is not None:
                    embedding_records.append((
                        glossary_id,
                        format_vector_for_postgres(embedding),
                        embedder.model
                    ))

            if embedding_records:
                db.executemany(insert_embedding_query, embedding_records)
                embeddings_inserted += len(embedding_records)

            # Commit batch
            db.commit()
            print(f"  ✓ Batch committed ({entries_inserted}/{total_entries} entries processed)")

    # Final summary
    print("\n" + "=" * 80)
    print("Ingestion Complete!")
    print("=" * 80)
    print(f"Glossary entries inserted: {entries_inserted}")
    print(f"Embeddings inserted: {embeddings_inserted}")
    print(f"Success rate: {embeddings_inserted/entries_inserted*100:.1f}%")

    # Show statistics
    with DatabaseConnection(**db_config) as db:
        db.execute("""
            SELECT COUNT(*) FROM mtg_glossary WHERE array_length(related_rules, 1) > 0
        """)
        entries_with_rules = db.cursor.fetchone()[0]
        print(f"\nEntries with related rules: {entries_with_rules}")

        db.execute("""
            SELECT term, array_length(related_rules, 1) as rule_count
            FROM mtg_glossary
            WHERE array_length(related_rules, 1) > 0
            ORDER BY rule_count DESC
            LIMIT 10
        """)
        print("\nTop entries by related rule count:")
        for row in db.cursor.fetchall():
            print(f"  {row[0]}: {row[1]} rules")

    return True


def main():
    """Main execution."""
    # Configuration
    glossary_file = "../rulesCleaning/MagicRulesGlossary.txt"

    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "mtg_vectors",
        "user": "postgres",
        "password": "postgres"
    }

    ollama_config = {
        "base_url": "http://localhost:11434",
        "model": "embeddinggemma:latest"
    }

    # Run ingestion
    success = ingest_glossary(
        glossary_file=glossary_file,
        db_config=db_config,
        ollama_config=ollama_config,
        batch_size=50
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
