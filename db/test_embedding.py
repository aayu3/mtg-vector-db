"""
Test script for MTG Vector Database embedding pipeline.
Validates database connection, Ollama embeddings, and vector similarity search.
"""

import sys
from db_utils import DatabaseConnection, OllamaEmbedder, wait_for_postgres, wait_for_ollama, format_vector_for_postgres


def test_database_connection():
    """Test PostgreSQL connection."""
    print("\n" + "=" * 80)
    print("TEST 1: PostgreSQL Connection")
    print("=" * 80)

    try:
        with DatabaseConnection() as db:
            db.execute("SELECT version()")
            version = db.cursor.fetchone()[0]
            print(f"✓ Connected to PostgreSQL")
            print(f"  Version: {version[:50]}...")

            # Check if pgvector extension is installed
            db.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            if db.cursor.fetchone():
                print("✓ pgvector extension is installed")
            else:
                print("✗ pgvector extension NOT installed")
                return False

        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_tables_exist():
    """Test that all required tables exist."""
    print("\n" + "=" * 80)
    print("TEST 2: Database Schema")
    print("=" * 80)

    expected_tables = [
        'mtg_cards', 'mtg_card_embeddings',
        'mtg_rules', 'mtg_rule_embeddings',
        'mtg_glossary', 'mtg_glossary_embeddings'
    ]

    try:
        with DatabaseConnection() as db:
            db.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            tables = [row[0] for row in db.cursor.fetchall()]

            print(f"Found {len(tables)} tables:")
            for table in tables:
                status = "✓" if table in expected_tables else " "
                print(f"  {status} {table}")

            missing = set(expected_tables) - set(tables)
            if missing:
                print(f"\n✗ Missing tables: {', '.join(missing)}")
                return False

            print("\n✓ All required tables exist")
            return True

    except Exception as e:
        print(f"✗ Failed to check tables: {e}")
        return False


def test_ollama_connection():
    """Test Ollama connection and embedding generation."""
    print("\n" + "=" * 80)
    print("TEST 3: Ollama Embedding Generation")
    print("=" * 80)

    try:
        embedder = OllamaEmbedder()
        print(f"Model: {embedder.model}")
        print(f"API URL: {embedder.embed_url}")

        # Generate test embedding
        test_text = "This is a test embedding for Magic: The Gathering cards"
        print(f"\nGenerating embedding for: '{test_text}'")

        embedding = embedder.generate_embedding(test_text)

        if not embedding:
            print("✗ Failed to generate embedding")
            return False

        print(f"✓ Generated embedding successfully")
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")

        # Verify dimension is 768
        if len(embedding) != 768:
            print(f"✗ Unexpected embedding dimension: {len(embedding)} (expected 768)")
            return False

        print("✓ Embedding dimension is correct (768)")
        return True, embedding

    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        return False, None


def test_insert_and_query():
    """Test inserting a glossary entry with embedding and querying it back."""
    print("\n" + "=" * 80)
    print("TEST 4: Insert & Query Test Document")
    print("=" * 80)

    try:
        embedder = OllamaEmbedder()

        # Test data
        test_term = "TEST_FLYING"
        test_definition = "A keyword ability that allows a creature to fly over blockers. A creature with flying can only be blocked by creatures with flying or reach."
        test_text = f"Term: {test_term}\nDefinition: {test_definition}"

        print(f"Test term: {test_term}")
        print(f"Generating embedding...")

        # Generate embedding
        embedding = embedder.generate_embedding(test_text)
        if not embedding:
            print("✗ Failed to generate test embedding")
            return False

        print(f"✓ Generated embedding ({len(embedding)} dimensions)")

        with DatabaseConnection() as db:
            # Clean up any existing test data
            db.execute("DELETE FROM mtg_glossary WHERE term = %s", (test_term,))
            db.commit()

            # Insert test glossary entry
            print("\nInserting test glossary entry...")
            db.execute("""
                INSERT INTO mtg_glossary (term, definition, related_rules)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (test_term, test_definition, ['701.19']))

            glossary_id = db.cursor.fetchone()[0]
            print(f"✓ Inserted glossary entry (ID: {glossary_id})")

            # Insert embedding
            print("Inserting embedding...")
            db.execute("""
                INSERT INTO mtg_glossary_embeddings (glossary_id, embedding, embedding_model)
                VALUES (%s, %s, %s)
            """, (glossary_id, format_vector_for_postgres(embedding), embedder.model))

            db.commit()
            print("✓ Inserted embedding")

            # Test vector similarity search
            print("\nTesting vector similarity search...")
            query_text = "flying creatures and how they work"
            query_embedding = embedder.generate_embedding(query_text)

            if not query_embedding:
                print("✗ Failed to generate query embedding")
                return False

            # Use the helper function
            db.execute("""
                SELECT * FROM search_similar_glossary(%s, 0.1, 5)
            """, (format_vector_for_postgres(query_embedding),))

            results = db.cursor.fetchall()

            print(f"\nQuery: '{query_text}'")
            print(f"Found {len(results)} similar entries:")

            for term, definition, related_rules, similarity in results:
                print(f"\n  Term: {term}")
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Definition: {definition[:100]}...")

            # Clean up test data
            print("\nCleaning up test data...")
            db.execute("DELETE FROM mtg_glossary WHERE term = %s", (test_term,))
            db.commit()
            print("✓ Test data cleaned up")

            return True

    except Exception as e:
        print(f"✗ Insert/Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("MTG Vector Database - Embedding Pipeline Test")
    print("=" * 80)

    results = {}

    # Test 1: Database connection
    results['database'] = test_database_connection()

    if not results['database']:
        print("\n✗ Database connection failed. Cannot continue.")
        sys.exit(1)

    # Test 2: Tables exist
    results['schema'] = test_tables_exist()

    if not results['schema']:
        print("\n✗ Database schema incomplete. Cannot continue.")
        sys.exit(1)

    # Test 3: Ollama connection
    ollama_result = test_ollama_connection()
    if isinstance(ollama_result, tuple):
        results['ollama'], _ = ollama_result
    else:
        results['ollama'] = ollama_result

    if not results['ollama']:
        print("\n✗ Ollama connection failed. Cannot continue.")
        sys.exit(1)

    # Test 4: Insert and query
    results['insert_query'] = test_insert_and_query()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 All tests passed! The system is ready for data ingestion.")
        print("\nNext steps:")
        print("  1. Run: python ingest_glossary.py  (~2-5 minutes)")
        print("  2. Run: python ingest_rules.py     (~5-10 minutes)")
        print("  3. Run: python ingest_cards.py     (~30-60 minutes)")
        print("\nOr run all at once:")
        print("  python ingest_all.py               (~45-75 minutes)")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
