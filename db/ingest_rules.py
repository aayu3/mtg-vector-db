"""
Ingest MTG rules into PostgreSQL with vector embeddings.
Loads rules from rules_individual.json and generates embeddings.
"""

import json
import sys
from typing import Dict, List, Any
from db_utils import DatabaseConnection, OllamaEmbedder, wait_for_postgres, wait_for_ollama, format_vector_for_postgres


def create_rule_embedding_text(rule: Dict[str, Any]) -> str:
    """
    Create text representation of a rule for embedding.

    Combines rule number, section context, and text.
    """
    parts = [f"Rule {rule['rule_number']}"]

    # Add section context
    parts.append(f"Section: {rule['section_name']}")

    # Add rule type
    parts.append(f"Type: {rule['rule_type']}")

    # Add parent rule if it's a subrule
    if rule.get('parent_rule'):
        parts.append(f"Parent Rule: {rule['parent_rule']}")

    # Add the actual rule text (most important)
    parts.append(f"Text: {rule['text']}")

    return '\n'.join(parts)


def ingest_rules(
    json_file: str,
    db_config: Dict[str, Any] = None,
    ollama_config: Dict[str, Any] = None,
    batch_size: int = 100
):
    """
    Ingest rules from JSON file into database.

    Args:
        json_file: Path to rules_individual.json
        db_config: Database connection configuration
        ollama_config: Ollama embedder configuration
        batch_size: Number of rules to process before committing
    """
    # Default configurations
    if db_config is None:
        db_config = {}
    if ollama_config is None:
        ollama_config = {}

    print("=" * 80)
    print("MTG Rules Ingestion")
    print("=" * 80)

    # Wait for services
    if not wait_for_postgres(**db_config):
        print("Error: PostgreSQL not available")
        return False

    if not wait_for_ollama(**ollama_config):
        print("Error: Ollama not available")
        return False

    # Load rule data
    print(f"\nLoading rules from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False

    print(f"Loaded {len(rules_data)} rules")

    # Initialize database and embedder
    embedder = OllamaEmbedder(**ollama_config)

    with DatabaseConnection(**db_config) as db:
        # Clear existing data (optional - remove if you want to append)
        print("\nClearing existing rule data...")
        db.execute("DELETE FROM mtg_rule_embeddings")
        db.execute("DELETE FROM mtg_rules")
        db.commit()

        # Process rules in batches
        total_rules = len(rules_data)
        rules_inserted = 0
        embeddings_inserted = 0

        print(f"\nProcessing {total_rules} rules in batches of {batch_size}...")

        for i in range(0, total_rules, batch_size):
            batch = rules_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_rules + batch_size - 1) // batch_size

            print(f"\n--- Batch {batch_num}/{total_batches} ---")

            # Prepare rule records
            rule_records = []
            embedding_texts = []

            for rule in batch:
                rule_records.append((
                    rule['rule_number'],
                    rule['text'],
                    rule['rule_type'],
                    rule['section_parent'],
                    rule['section_number'],
                    rule['section_name'],
                    rule.get('parent_rule')  # NULL for main rules
                ))

                # Create embedding text
                embedding_text = create_rule_embedding_text(rule)
                embedding_texts.append(embedding_text)

            # Insert rules
            print(f"  Inserting {len(rule_records)} rules into database...")
            insert_rule_query = """
                INSERT INTO mtg_rules (
                    rule_number, text, rule_type, section_parent,
                    section_number, section_name, parent_rule
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            rule_ids = []
            for record in rule_records:
                db.execute(insert_rule_query, record)
                rule_id = db.cursor.fetchone()[0]
                rule_ids.append(rule_id)

            rules_inserted += len(rule_ids)

            # Generate embeddings
            print(f"  Generating embeddings for {len(embedding_texts)} rules...")
            embeddings = embedder.generate_embeddings_batch(
                embedding_texts,
                batch_size=10,
                show_progress=True
            )

            # Insert embeddings
            print(f"  Inserting embeddings into database...")
            insert_embedding_query = """
                INSERT INTO mtg_rule_embeddings (rule_id, embedding, embedding_model)
                VALUES (%s, %s, %s)
            """

            embedding_records = []
            for rule_id, embedding in zip(rule_ids, embeddings):
                if embedding is not None:
                    embedding_records.append((
                        rule_id,
                        format_vector_for_postgres(embedding),
                        embedder.model
                    ))

            if embedding_records:
                db.executemany(insert_embedding_query, embedding_records)
                embeddings_inserted += len(embedding_records)

            # Commit batch
            db.commit()
            print(f"  ✓ Batch committed ({rules_inserted}/{total_rules} rules processed)")

    # Final summary
    print("\n" + "=" * 80)
    print("Ingestion Complete!")
    print("=" * 80)
    print(f"Rules inserted: {rules_inserted}")
    print(f"Embeddings inserted: {embeddings_inserted}")
    print(f"Success rate: {embeddings_inserted/rules_inserted*100:.1f}%")

    # Show statistics by rule type
    with DatabaseConnection(**db_config) as db:
        db.execute("""
            SELECT rule_type, COUNT(*) as count
            FROM mtg_rules
            GROUP BY rule_type
            ORDER BY rule_type
        """)
        print("\nRules by type:")
        for row in db.cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")

        db.execute("""
            SELECT section_name, COUNT(*) as count
            FROM mtg_rules
            GROUP BY section_name
            ORDER BY count DESC
            LIMIT 10
        """)
        print("\nTop sections by rule count:")
        for row in db.cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")

    return True


def main():
    """Main execution."""
    # Configuration
    json_file = "../rulesCleaning/rules_individual.json"

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
    success = ingest_rules(
        json_file=json_file,
        db_config=db_config,
        ollama_config=ollama_config,
        batch_size=100
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
