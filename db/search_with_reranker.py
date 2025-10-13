"""
Multi-source search with reranking across cards, rules, and glossary.

This demonstrates:
1. Parallel vector search across all 3 tables
2. Reranking results from each source
3. Optional cross-source fusion of results
"""

from db_utils import DatabaseConnection, OllamaEmbedder, OllamaReranker, format_vector_for_postgres
import sys
from typing import List, Dict, Any


def search_cards_with_reranking(
    query: str,
    query_embedding: List[float],
    reranker: OllamaReranker,
    initial_k: int = 30,
    final_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search cards with reranking.

    Args:
        query: Search query text
        query_embedding: Query embedding vector
        reranker: Reranker instance
        initial_k: Number of initial candidates
        final_k: Number of final results

    Returns:
        List of reranked card results
    """
    print(f"\n[Cards] Retrieving top {initial_k} candidates...")

    with DatabaseConnection() as db:
        db.execute("""
            SELECT
                c.card_name,
                c.card_type,
                c.text_content,
                c.related_faces,
                1 - (e.embedding <=> %s) as similarity
            FROM mtg_card_embeddings e
            JOIN mtg_cards c ON e.card_id = c.id
            ORDER BY e.embedding <=> %s
            LIMIT %s
        """, (format_vector_for_postgres(query_embedding),
              format_vector_for_postgres(query_embedding),
              initial_k))

        candidates = db.cursor.fetchall()

    if not candidates:
        return []

    # Prepare documents for reranking
    documents = []
    metadata = []

    for card_name, card_type, text_content, related_faces, similarity in candidates:
        doc_text = f"Card: {card_name}\nType: {card_type}\nText: {text_content or 'No text'}"
        documents.append(doc_text)
        metadata.append({
            'name': card_name,
            'type': card_type,
            'text': text_content,
            'related_faces': related_faces,
            'vector_similarity': similarity
        })

    # Rerank
    print(f"[Cards] Reranking {len(documents)} candidates...")
    reranked = reranker.rerank(query, documents, top_k=final_k)

    # Combine with metadata
    results = []
    for item in reranked:
        meta = metadata[item['index']]
        results.append({
            'source': 'card',
            'name': meta['name'],
            'type': meta['type'],
            'text': meta['text'],
            'related_faces': meta['related_faces'],
            'vector_similarity': meta['vector_similarity'],
            'rerank_score': item['score']
        })

    return results


def search_rules_with_reranking(
    query: str,
    query_embedding: List[float],
    reranker: OllamaReranker,
    initial_k: int = 30,
    final_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search rules with reranking.

    Args:
        query: Search query text
        query_embedding: Query embedding vector
        reranker: Reranker instance
        initial_k: Number of initial candidates
        final_k: Number of final results

    Returns:
        List of reranked rule results
    """
    print(f"\n[Rules] Retrieving top {initial_k} candidates...")

    with DatabaseConnection() as db:
        db.execute("""
            SELECT
                r.rule_number,
                r.text,
                r.rule_type,
                r.section_name,
                1 - (e.embedding <=> %s) as similarity
            FROM mtg_rule_embeddings e
            JOIN mtg_rules r ON e.rule_id = r.id
            ORDER BY e.embedding <=> %s
            LIMIT %s
        """, (format_vector_for_postgres(query_embedding),
              format_vector_for_postgres(query_embedding),
              initial_k))

        candidates = db.cursor.fetchall()

    if not candidates:
        return []

    # Prepare documents for reranking
    documents = []
    metadata = []

    for rule_number, text, rule_type, section_name, similarity in candidates:
        doc_text = f"Rule {rule_number} ({section_name}): {text}"
        documents.append(doc_text)
        metadata.append({
            'rule_number': rule_number,
            'text': text,
            'rule_type': rule_type,
            'section_name': section_name,
            'vector_similarity': similarity
        })

    # Rerank
    print(f"[Rules] Reranking {len(documents)} candidates...")
    reranked = reranker.rerank(query, documents, top_k=final_k)

    # Combine with metadata
    results = []
    for item in reranked:
        meta = metadata[item['index']]
        results.append({
            'source': 'rule',
            'rule_number': meta['rule_number'],
            'text': meta['text'],
            'rule_type': meta['rule_type'],
            'section_name': meta['section_name'],
            'vector_similarity': meta['vector_similarity'],
            'rerank_score': item['score']
        })

    return results


def search_glossary_with_reranking(
    query: str,
    query_embedding: List[float],
    reranker: OllamaReranker,
    initial_k: int = 30,
    final_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search glossary with reranking.

    Args:
        query: Search query text
        query_embedding: Query embedding vector
        reranker: Reranker instance
        initial_k: Number of initial candidates
        final_k: Number of final results

    Returns:
        List of reranked glossary results
    """
    print(f"\n[Glossary] Retrieving top {initial_k} candidates...")

    with DatabaseConnection() as db:
        db.execute("""
            SELECT
                g.term,
                g.definition,
                g.related_rules,
                1 - (e.embedding <=> %s) as similarity
            FROM mtg_glossary_embeddings e
            JOIN mtg_glossary g ON e.glossary_id = g.id
            ORDER BY e.embedding <=> %s
            LIMIT %s
        """, (format_vector_for_postgres(query_embedding),
              format_vector_for_postgres(query_embedding),
              initial_k))

        candidates = db.cursor.fetchall()

    if not candidates:
        return []

    # Prepare documents for reranking
    documents = []
    metadata = []

    for term, definition, related_rules, similarity in candidates:
        doc_text = f"Term: {term}\nDefinition: {definition}"
        documents.append(doc_text)
        metadata.append({
            'term': term,
            'definition': definition,
            'related_rules': related_rules,
            'vector_similarity': similarity
        })

    # Rerank
    print(f"[Glossary] Reranking {len(documents)} candidates...")
    reranked = reranker.rerank(query, documents, top_k=final_k)

    # Combine with metadata
    results = []
    for item in reranked:
        meta = metadata[item['index']]
        results.append({
            'source': 'glossary',
            'term': meta['term'],
            'definition': meta['definition'],
            'related_rules': meta['related_rules'],
            'vector_similarity': meta['vector_similarity'],
            'rerank_score': item['score']
        })

    return results


def search_all_sources(
    query: str,
    cards_k: int = 10,
    rules_k: int = 10,
    glossary_k: int = 5
):
    """
    Search across all sources (cards, rules, glossary) with reranking.

    Args:
        query: Search query
        cards_k: Number of card results
        rules_k: Number of rule results
        glossary_k: Number of glossary results
    """
    print("=" * 80)
    print(f"MULTI-SOURCE SEARCH: '{query}'")
    print("=" * 80)

    # Initialize
    embedder = OllamaEmbedder()
    reranker = OllamaReranker()

    # Generate query embedding
    print("\n[1/4] Generating query embedding...")
    query_embedding = embedder.generate_embedding(query)

    if not query_embedding:
        print("✗ Failed to generate query embedding")
        return

    print(f"✓ Generated {len(query_embedding)}-dimensional embedding")

    # Search each source
    print("\n[2/4] Searching cards...")
    card_results = search_cards_with_reranking(query, query_embedding, reranker, initial_k=30, final_k=cards_k)

    print("\n[3/4] Searching rules...")
    rule_results = search_rules_with_reranking(query, query_embedding, reranker, initial_k=30, final_k=rules_k)

    print("\n[4/4] Searching glossary...")
    glossary_results = search_glossary_with_reranking(query, query_embedding, reranker, initial_k=20, final_k=glossary_k)

    # Display results
    print("\n" + "=" * 80)
    print(f"RESULTS FOR: '{query}'")
    print("=" * 80)

    # Cards
    if card_results:
        print(f"\n📇 TOP {len(card_results)} CARDS")
        print("-" * 80)
        for i, result in enumerate(card_results, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Type: {result['type']}")
            print(f"   Rerank Score: {result['rerank_score']:.4f} | Vector: {result['vector_similarity']:.4f}")
            if result['text']:
                text = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
                print(f"   Text: {text}")

    # Rules
    if rule_results:
        print(f"\n📖 TOP {len(rule_results)} RULES")
        print("-" * 80)
        for i, result in enumerate(rule_results, 1):
            print(f"\n{i}. Rule {result['rule_number']} - {result['section_name']}")
            print(f"   Rerank Score: {result['rerank_score']:.4f} | Vector: {result['vector_similarity']:.4f}")
            text = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            print(f"   {text}")

    # Glossary
    if glossary_results:
        print(f"\n📚 TOP {len(glossary_results)} GLOSSARY TERMS")
        print("-" * 80)
        for i, result in enumerate(glossary_results, 1):
            print(f"\n{i}. {result['term']}")
            print(f"   Rerank Score: {result['rerank_score']:.4f} | Vector: {result['vector_similarity']:.4f}")
            definition = result['definition'][:200] + "..." if len(result['definition']) > 200 else result['definition']
            print(f"   {definition}")


def main():
    """Main execution."""
    print("=" * 80)
    print("MTG Multi-Source Search with Reranking")
    print("=" * 80)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search_all_sources(query)
    else:
        print("\nUsage: python search_with_reranker.py <your search query>")
        print("\nExample queries:")
        print("  python search_with_reranker.py flying creatures with deathtouch")
        print("  python search_with_reranker.py how does double strike work")
        print("  python search_with_reranker.py cards that destroy artifacts")
        print("\nRunning default example...")
        search_all_sources("flying creatures with deathtouch", cards_k=5, rules_k=3, glossary_k=2)


if __name__ == "__main__":
    main()
