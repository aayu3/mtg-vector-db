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

    Concatenates raw values without labels for better semantic matching.
    Example: "4B Bingus the Wizard Legendary Cat Wizard When Bingus enters the battlefield you win the game 5/5"

    Order: ManaCost Name Type CardText Power/Toughness
    """
    parts = []

    # Mana cost first (helps identify mana value)
    # Remove braces from mana cost: {1}{B}{B} -> 1BB
    if card.get('manaCost'):
        mana_cost = card['manaCost'].replace('{', '').replace('}', '')
        parts.append(mana_cost)

    # Card name
    card_name = card.get('asciiName') or card.get('name', '')
    if card_name:
        parts.append(card_name)

    # Type line (remove em dash separator)
    # "Legendary Creature — Vampire" -> "Legendary Creature Vampire"
    if card.get('type'):
        type_line = card['type'].replace(' — ', ' ').replace('—', ' ')
        parts.append(type_line)

    # Card text (most important for semantic search)
    if card.get('text'):
        parts.append(card['text'])

    # Power/toughness for creatures
    if card.get('power') and card.get('toughness'):
        parts.append(f"{card['power']}/{card['toughness']}")

    return ' '.join(parts)


def create_card_nl_text(card: Dict[str, Any]) -> str:
    """
    Create natural language description of a card for embedding.

    Uses just the card text plus power/toughness without labels or structure.
    Example: "Flying Deathtouch Lifelink 2/3"
    """
    parts = []

    # Card text
    card_text = card.get('text', '')
    if card_text:
        parts.append(card_text)

    # Power/toughness for creatures
    if card.get('power') and card.get('toughness'):
        parts.append(f"{card['power']}/{card['toughness']}")

    return ' '.join(parts).strip()


def extract_card_fields(card: Dict[str, Any]) -> Dict[str, Any]:
    """Extract commonly queried fields from card data.

    Handles double-sided cards by using faceName or extracting from full name.
    Uses asciiName when available to avoid unicode issues in database.
    """
    full_name = card.get('name', '')

    # Determine unique card_name for this face
    if card.get('faceName'):
        # Front face (side a) has faceName - use it
        card_name = card.get('asciiName') or card['faceName']
    elif card.get('side') == 'b' and '//' in full_name:
        # Back face (side b) - extract second part of double-sided name
        back_name = full_name.split('//')[1].strip()
        card_name = card.get('asciiName') or back_name
    else:
        # Single-faced card
        card_name = card.get('asciiName') or full_name

    # Store related_faces for double-sided cards to preserve relationship
    related_faces = full_name if '//' in full_name else None

    # Get Oracle ID for duplicate detection
    oracle_id = card.get('identifiers', {}).get('scryfallOracleId', '')

    return {
        'card_name': card_name,
        'text_content': card.get('text', ''),
        'card_type': card.get('type', ''),
        'colors': card.get('colors', []),
        'mana_value': card.get('manaValue', 0),
        'keywords': card.get('keywords', []),
        'legalities': card.get('legalities', {}),
        'related_faces': related_faces,
        'oracle_id': oracle_id
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
        # Check if we already have cards
        db.execute("SELECT COUNT(*) FROM mtg_cards")
        existing_cards = db.cursor.fetchone()[0]

        if existing_cards > 0:
            print(f"\nFound {existing_cards} existing cards in database")
            print("Skipping card and structured embedding insertion (already exist)")
            print("Only generating natural language embeddings for existing cards...")
            skip_structured = True
        else:
            print("\nNo existing cards found - will insert all data")
            skip_structured = False

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
            nl_texts = []
            card_fields_list = []

            for card in batch:
                fields = extract_card_fields(card)
                card_fields_list.append(fields)
                card_records.append((
                    fields['card_name'],
                    Json(card),  # Store full card as JSONB
                    fields['text_content'],
                    fields['card_type'],
                    fields['colors'],
                    fields['mana_value'],
                    fields['keywords'],
                    Json(fields['legalities']),
                    fields['related_faces']
                ))

                # Create structured embedding text
                embedding_text = create_card_embedding_text(card)
                embedding_texts.append(embedding_text)

                # Create natural language embedding text (just card text)
                nl_text = create_card_nl_text(card)
                nl_texts.append(nl_text)

            # Insert cards with duplicate detection
            print(f"  Inserting {len(card_records)} cards into database...")
            insert_card_query = """
                INSERT INTO mtg_cards (
                    card_name, card_data, text_content, card_type,
                    colors, mana_value, keywords, legalities, related_faces
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            card_ids = []
            duplicates_skipped = 0

            for i, record in enumerate(card_records):
                fields = card_fields_list[i]

                # Check if card already exists first (pre-check to avoid rollbacks)
                try:
                    db.execute("SELECT id FROM mtg_cards WHERE card_name = %s LIMIT 1", (fields['card_name'],))
                    existing = db.cursor.fetchone()

                    if existing:
                        # Duplicate exists - skip and log
                        duplicates_skipped += 1

                        with open('card_duplicates.log', 'a', encoding='utf-8') as f:
                            f.write(f"Duplicate: {fields['card_name']}\n")
                            f.write(f"  Type: {fields['card_type']}\n")
                            f.write(f"  Mana Value: {fields['mana_value']}\n")
                            f.write(f"  Related Faces: {fields['related_faces']}\n")
                            f.write(f"  Oracle ID: {fields['oracle_id']}\n")
                            f.write("-" * 80 + "\n")

                        card_ids.append(None)
                        continue
                except:
                    pass

                # Try to insert new card
                try:
                    db.execute(insert_card_query, record)
                    card_id = db.cursor.fetchone()[0]
                    card_ids.append(card_id)
                except Exception as e:
                    error_str = str(e)

                    # Log error and skip
                    print(f"    Error inserting {fields['card_name']}: {e}")
                    with open('card_errors.log', 'a', encoding='utf-8') as f:
                        f.write(f"Error: {fields['card_name']}\n")
                        f.write(f"  {error_str}\n")
                        f.write("-" * 80 + "\n")

                    card_ids.append(None)

            cards_inserted += (len(card_ids) - duplicates_skipped)

            if duplicates_skipped > 0:
                print(f"  Skipped {duplicates_skipped} duplicate cards (logged to card_duplicates.log)")

            # Generate structured embeddings
            print(f"  Generating structured embeddings for {len(embedding_texts)} cards...")
            embeddings = embedder.generate_embeddings_batch(
                embedding_texts,
                batch_size=10,
                show_progress=True
            )

            # Insert structured embeddings
            print(f"  Inserting structured embeddings into database...")
            insert_embedding_query = """
                INSERT INTO mtg_card_embeddings (card_id, embedding, embedding_model)
                VALUES (%s, %s, %s)
            """

            embedding_records = []
            for card_id, embedding in zip(card_ids, embeddings):
                if embedding is not None and card_id is not None:
                    embedding_records.append((
                        card_id,
                        format_vector_for_postgres(embedding),
                        embedder.model
                    ))

            if embedding_records:
                db.executemany(insert_embedding_query, embedding_records)
                embeddings_inserted += len(embedding_records)

            # Generate natural language embeddings (only for cards with text)
            nl_texts_to_embed = []
            nl_card_ids = []
            for card_id, nl_text in zip(card_ids, nl_texts):
                if card_id is not None and nl_text:  # Only embed if card exists and has text
                    nl_texts_to_embed.append(nl_text)
                    nl_card_ids.append(card_id)

            if nl_texts_to_embed:
                print(f"  Generating natural language embeddings for {len(nl_texts_to_embed)} cards...")
                nl_embeddings = embedder.generate_embeddings_batch(
                    nl_texts_to_embed,
                    batch_size=10,
                    show_progress=True
                )

                # Insert natural language embeddings
                print(f"  Inserting natural language embeddings into database...")
                insert_nl_embedding_query = """
                    INSERT INTO mtg_card_nl_embeddings (card_id, embedding, embedding_model, nl_text)
                    VALUES (%s, %s, %s, %s)
                """

                nl_embedding_records = []
                for card_id, nl_text, embedding in zip(nl_card_ids, nl_texts_to_embed, nl_embeddings):
                    if embedding is not None:
                        nl_embedding_records.append((
                            card_id,
                            format_vector_for_postgres(embedding),
                            embedder.model,
                            nl_text
                        ))

                if nl_embedding_records:
                    db.executemany(insert_nl_embedding_query, nl_embedding_records)
                    print(f"  Inserted {len(nl_embedding_records)} natural language embeddings")

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
    json_file = "../cardsCleaning/AtomicCards_cleaned.json"

    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "mtg_vectors",
        "user": "postgres",
        "password": "postgres"
    }

    ollama_config = {
        "base_url": "http://localhost:11434",
        "model": "embeddinggemma:300m"
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
