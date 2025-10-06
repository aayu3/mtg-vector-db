"""
Ingest MTG cards into PostgreSQL with vector embeddings.
Loads cards from ModernAtomic_cleaned.json and generates embeddings.
"""

import json
import sys
from typing import Dict, List, Any
from db_utils import DatabaseConnection, OllamaEmbedder, wait_for_postgres, wait_for_ollama, format_vector_for_postgres
from psycopg2.extras import Json


def create_card_embedding_text(card: Dict[str, Any]) -> str:
    """
    Create text representation of a card for embedding.

    Combines card name, type, text, and other relevant info.
    Uses asciiName when available to avoid unicode issues.
    """
    # Use asciiName if available, otherwise fall back to name
    card_name = card.get('asciiName') or card.get('name', 'Unknown')
    parts = [f"Card: {card_name}"]

    # Add type
    if card.get('type'):
        parts.append(f"Type: {card['type']}")

    # Add mana cost
    if card.get('manaCost'):
        parts.append(f"Mana Cost: {card['manaCost']}")

    # Add colors
    if card.get('colors'):
        parts.append(f"Colors: {', '.join(card['colors'])}")

    # Add card text (most important for semantic search)
    if card.get('text'):
        parts.append(f"Text: {card['text']}")

    # Add power/toughness for creatures
    if card.get('power') and card.get('toughness'):
        parts.append(f"P/T: {card['power']}/{card['toughness']}")

    # Add keywords
    if card.get('keywords'):
        parts.append(f"Keywords: {', '.join(card['keywords'])}")

    return '\n'.join(parts)


def extract_card_fields(card: Dict[str, Any]) -> Dict[str, Any]:
    """Extract commonly queried fields from card data.

    Uses asciiName when available to avoid unicode issues in database.
    """
    # Use asciiName if available, otherwise fall back to name
    card_name = card.get('asciiName') or card.get('name', 'Unknown')

    return {
        'card_name': card_name,
        'text_content': card.get('text', ''),
        'card_type': card.get('type', ''),
        'colors': card.get('colors', []),
        'mana_value': card.get('manaValue', 0),
        'keywords': card.get('keywords', []),
        'legalities': card.get('legalities', {})
    }


def ingest_cards(
    json_file: str,
    db_config: Dict[str, Any] = None,
    ollama_config: Dict[str, Any] = None,
    batch_size: int = 100
):
    """
    Ingest cards from JSON file into database.

    Args:
        json_file: Path to ModernAtomic_cleaned.json
        db_config: Database connection configuration
        ollama_config: Ollama embedder configuration
        batch_size: Number of cards to process before committing
    """
    # Default configurations
    if db_config is None:
        db_config = {}
    if ollama_config is None:
        ollama_config = {}

    print("=" * 80)
    print("MTG Cards Ingestion")
    print("=" * 80)

    # Wait for services
    if not wait_for_postgres(**db_config):
        print("Error: PostgreSQL not available")
        return False

    if not wait_for_ollama(**ollama_config):
        print("Error: Ollama not available")
        return False

    # Load card data
    print(f"\nLoading cards from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            cards_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False

    # Flatten card data (each card name maps to array of cards)
    all_cards = []
    for card_name, card_array in cards_data.items():
        for card in card_array:
            all_cards.append(card)

    print(f"Loaded {len(all_cards)} cards")

    # Initialize database and embedder
    embedder = OllamaEmbedder(**ollama_config)

    with DatabaseConnection(**db_config) as db:
        # Clear existing data (optional - remove if you want to append)
        print("\nClearing existing card data...")
        db.execute("DELETE FROM mtg_card_embeddings")
        db.execute("DELETE FROM mtg_cards")
        db.commit()

        # Process cards in batches
        total_cards = len(all_cards)
        cards_inserted = 0
        embeddings_inserted = 0

        print(f"\nProcessing {total_cards} cards in batches of {batch_size}...")

        for i in range(0, total_cards, batch_size):
            batch = all_cards[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_cards + batch_size - 1) // batch_size

            print(f"\n--- Batch {batch_num}/{total_batches} ---")

            # Prepare card records
            card_records = []
            embedding_texts = []

            for card in batch:
                fields = extract_card_fields(card)
                card_records.append((
                    fields['card_name'],
                    Json(card),  # Store full card as JSONB
                    fields['text_content'],
                    fields['card_type'],
                    fields['colors'],
                    fields['mana_value'],
                    fields['keywords'],
                    Json(fields['legalities'])
                ))

                # Create embedding text
                embedding_text = create_card_embedding_text(card)
                embedding_texts.append(embedding_text)

            # Insert cards
            print(f"  Inserting {len(card_records)} cards into database...")
            insert_card_query = """
                INSERT INTO mtg_cards (
                    card_name, card_data, text_content, card_type,
                    colors, mana_value, keywords, legalities
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            card_ids = []
            for record in card_records:
                db.execute(insert_card_query, record)
                card_id = db.cursor.fetchone()[0]
                card_ids.append(card_id)

            cards_inserted += len(card_ids)

            # Generate embeddings
            print(f"  Generating embeddings for {len(embedding_texts)} cards...")
            embeddings = embedder.generate_embeddings_batch(
                embedding_texts,
                batch_size=10,
                show_progress=True
            )

            # Insert embeddings
            print(f"  Inserting embeddings into database...")
            insert_embedding_query = """
                INSERT INTO mtg_card_embeddings (card_id, embedding, embedding_model)
                VALUES (%s, %s, %s)
            """

            embedding_records = []
            for card_id, embedding in zip(card_ids, embeddings):
                if embedding is not None:
                    embedding_records.append((
                        card_id,
                        format_vector_for_postgres(embedding),
                        embedder.model
                    ))

            if embedding_records:
                db.executemany(insert_embedding_query, embedding_records)
                embeddings_inserted += len(embedding_records)

            # Commit batch
            db.commit()
            print(f"  ✓ Batch committed ({cards_inserted}/{total_cards} cards processed)")

    # Final summary
    print("\n" + "=" * 80)
    print("Ingestion Complete!")
    print("=" * 80)
    print(f"Cards inserted: {cards_inserted}")
    print(f"Embeddings inserted: {embeddings_inserted}")
    print(f"Success rate: {embeddings_inserted/cards_inserted*100:.1f}%")

    return True


def main():
    """Main execution."""
    # Configuration
    json_file = "../cardsCleaning/ModernAtomic_cleaned.json"

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
    success = ingest_cards(
        json_file=json_file,
        db_config=db_config,
        ollama_config=ollama_config,
        batch_size=100
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
