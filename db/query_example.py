"""
Example queries demonstrating how to use the MTG vector database.
Shows vector similarity search for cards, rules, and glossary.
"""

from db_utils import DatabaseConnection, OllamaEmbedder, format_vector_for_postgres
import json


def search_similar_cards(query_text: str, limit: int = 5):
    """Search for cards similar to the query text."""
    print(f"\n{'='*80}")
    print(f"SEARCHING CARDS: '{query_text}'")
    print(f"{'='*80}")

    # Generate embedding for query
    embedder = OllamaEmbedder()
    query_embedding = embedder.generate_embedding(query_text)

    if not query_embedding:
        print("Failed to generate embedding")
        return

    # Search database
    with DatabaseConnection() as db:
        results = db.execute(
            "SELECT * FROM search_similar_cards(%s, 0.5, %s)",
            (format_vector_for_postgres(query_embedding), limit)
        ).fetchall()

        if not results:
            print("No results found")
            return

        print(f"\nFound {len(results)} similar cards:\n")
        for i, (card_name, card_data, similarity) in enumerate(results, 1):
            card = card_data  # card_data is already a dict from JSONB
            print(f"{i}. {card_name} (similarity: {similarity:.3f})")
            print(f"   Type: {card.get('type', 'N/A')}")
            print(f"   Cost: {card.get('manaCost', 'N/A')}")
            if card.get('text'):
                text = card['text'][:150] + "..." if len(card['text']) > 150 else card['text']
                print(f"   Text: {text}")
            print()


def search_similar_rules(query_text: str, limit: int = 5):
    """Search for rules similar to the query text."""
    print(f"\n{'='*80}")
    print(f"SEARCHING RULES: '{query_text}'")
    print(f"{'='*80}")

    # Generate embedding for query
    embedder = OllamaEmbedder()
    query_embedding = embedder.generate_embedding(query_text)

    if not query_embedding:
        print("Failed to generate embedding")
        return

    # Search database
    with DatabaseConnection() as db:
        results = db.execute(
            "SELECT * FROM search_similar_rules(%s, 0.5, %s)",
            (format_vector_for_postgres(query_embedding), limit)
        ).fetchall()

        if not results:
            print("No results found")
            return

        print(f"\nFound {len(results)} similar rules:\n")
        for i, (rule_number, text, rule_type, section_name, similarity) in enumerate(results, 1):
            print(f"{i}. Rule {rule_number} (similarity: {similarity:.3f})")
            print(f"   Section: {section_name}")
            print(f"   Type: {rule_type}")
            rule_text = text[:200] + "..." if len(text) > 200 else text
            print(f"   Text: {rule_text}")
            print()


def search_similar_glossary(query_text: str, limit: int = 5):
    """Search for glossary entries similar to the query text."""
    print(f"\n{'='*80}")
    print(f"SEARCHING GLOSSARY: '{query_text}'")
    print(f"{'='*80}")

    # Generate embedding for query
    embedder = OllamaEmbedder()
    query_embedding = embedder.generate_embedding(query_text)

    if not query_embedding:
        print("Failed to generate embedding")
        return

    # Search database
    with DatabaseConnection() as db:
        results = db.execute(
            "SELECT * FROM search_similar_glossary(%s, 0.5, %s)",
            (format_vector_for_postgres(query_embedding), limit)
        ).fetchall()

        if not results:
            print("No results found")
            return

        print(f"\nFound {len(results)} similar glossary entries:\n")
        for i, (term, definition, related_rules, similarity) in enumerate(results, 1):
            print(f"{i}. {term} (similarity: {similarity:.3f})")
            def_text = definition[:200] + "..." if len(definition) > 200 else definition
            print(f"   Definition: {def_text}")
            if related_rules:
                print(f"   Related Rules: {', '.join(related_rules[:5])}")
            print()


def run_example_queries():
    """Run a series of example queries."""
    print("=" * 80)
    print("MTG Vector Database - Example Queries")
    print("=" * 80)

    # Example 1: Search for flying creatures
    search_similar_cards("flying creature with lifelink", limit=3)

    # Example 2: Search for combo cards
    search_similar_cards("draw cards when creatures die", limit=3)

    # Example 3: Search for rules about combat
    search_similar_rules("how does combat damage work", limit=3)

    # Example 4: Search for rules about the stack
    search_similar_rules("what happens when you cast a spell", limit=3)

    # Example 5: Search glossary for keywords
    search_similar_glossary("ability that prevents damage", limit=3)

    # Example 6: Search glossary for game concepts
    search_similar_glossary("turn structure and phases", limit=3)


def database_stats():
    """Show statistics about the database contents."""
    print(f"\n{'='*80}")
    print("DATABASE STATISTICS")
    print(f"{'='*80}\n")

    with DatabaseConnection() as db:
        # Card stats
        db.execute("SELECT COUNT(*) FROM mtg_cards")
        card_count = db.cursor.fetchone()[0]
        db.execute("SELECT COUNT(*) FROM mtg_card_embeddings")
        card_embedding_count = db.cursor.fetchone()[0]

        # Rule stats
        db.execute("SELECT COUNT(*) FROM mtg_rules")
        rule_count = db.cursor.fetchone()[0]
        db.execute("SELECT COUNT(*) FROM mtg_rule_embeddings")
        rule_embedding_count = db.cursor.fetchone()[0]

        # Glossary stats
        db.execute("SELECT COUNT(*) FROM mtg_glossary")
        glossary_count = db.cursor.fetchone()[0]
        db.execute("SELECT COUNT(*) FROM mtg_glossary_embeddings")
        glossary_embedding_count = db.cursor.fetchone()[0]

        print(f"Cards:    {card_count:,} documents, {card_embedding_count:,} embeddings")
        print(f"Rules:    {rule_count:,} documents, {rule_embedding_count:,} embeddings")
        print(f"Glossary: {glossary_count:,} documents, {glossary_embedding_count:,} embeddings")
        print(f"\nTotal:    {card_count + rule_count + glossary_count:,} documents")
        print(f"Total:    {card_embedding_count + rule_embedding_count + glossary_embedding_count:,} embeddings")


def main():
    """Main execution."""
    # Show database stats
    database_stats()

    # Run example queries
    run_example_queries()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
