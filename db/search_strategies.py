"""
Multi-strategy search for MTG cards, rules, and glossary.

Search strategies:
1. Card Name Search: Fuzzy matching on card names
2. Card Description Search: Vector similarity + reranking on card text
3. Ambiguous Card Search: Pure vector similarity
4. Rules Text Search: Vector similarity + reranking on rules
5. Glossary/Terms Search: Vector similarity + reranking on glossary
"""

import sys
from typing import List, Dict, Any, Optional
from db_utils import DatabaseConnection, OllamaEmbedder, OllamaReranker


class CardSearchStrategies:
    """Search strategies for MTG cards."""

    def __init__(self, db: DatabaseConnection, embedder: OllamaEmbedder, reranker: Optional[OllamaReranker] = None):
        self.db = db
        self.embedder = embedder
        self.reranker = reranker  # Only used for semantic search, not name search

    def search_by_card_name(self, query: str, top_k: int = 10, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for cards by name using PostgreSQL pg_trgm fuzzy matching.

        This is a pure character-based search - no semantic understanding or reranking needed.
        pg_trgm already provides the distance metric for ranking results.

        Strategy (3 tiers):
        1. Exact match (fastest - indexed lookup)
        2. Prefix match (very fast - btree index scan + trigram similarity)
        3. Fuzzy trigram similarity (fast - GIN trigram index for typos/variations)

        Args:
            query: Card name to search for
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1) for pg_trgm matches (default 0.3)

        Returns:
            List of cards with similarity scores from pg_trgm

        Examples:
            "lightning bolt" → exact match
            "light" → prefix match (finds "Lightning Bolt")
            "litening bolt" → fuzzy match (typo correction)
        """
        # Tier 1: Exact match (case-insensitive)
        self.db.execute(
            """
            SELECT card_name, card_data, text_content, 1.0 as similarity
            FROM mtg_cards
            WHERE LOWER(card_name) = LOWER(%s)
            """,
            (query,)
        )
        exact_match = self.db.cursor.fetchone()

        if exact_match:
            return [{
                'card_name': exact_match[0],
                'card_data': exact_match[1],
                'text_content': exact_match[2],
                'similarity': float(exact_match[3]),
                'match_type': 'exact'
            }]

        # Tier 2: Prefix match (handles partial names like "light" -> "Lightning Bolt")
        # Uses btree index for fast prefix scan, then computes similarity for ranking
        self.db.execute(
            """
            SELECT card_name, card_data, text_content,
                   similarity(card_name, %s) as sim
            FROM mtg_cards
            WHERE card_name ILIKE %s || '%%'
            ORDER BY sim DESC
            LIMIT %s
            """,
            (query, query, top_k)
        )
        prefix_matches = self.db.cursor.fetchall()

        # If we have good prefix matches (similarity > 0.5), return them
        if prefix_matches and prefix_matches[0][3] > 0.5:
            return [{
                'card_name': row[0],
                'card_data': row[1],
                'text_content': row[2],
                'similarity': float(row[3]),
                'match_type': 'prefix'
            } for row in prefix_matches]

        # Tier 3: Fuzzy trigram similarity (handles typos, character swaps, etc.)
        # Uses GIN trigram index for fast similarity search
        self.db.execute(
            """
            SELECT card_name, card_data, text_content,
                   similarity(card_name, %s) as sim
            FROM mtg_cards
            WHERE similarity(card_name, %s) > %s
            ORDER BY card_name <-> %s
            LIMIT %s
            """,
            (query, query, similarity_threshold, query, top_k)
        )
        fuzzy_matches = self.db.cursor.fetchall()

        return [{
            'card_name': row[0],
            'card_data': row[1],
            'text_content': row[2],
            'similarity': float(row[3]),
            'match_type': 'fuzzy_trigram'
        } for row in fuzzy_matches]

    def search_by_card_description(
        self,
        query: str,
        initial_k: int = 30,
        final_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for cards by description using vector similarity + optional reranking.

        This is a semantic search - understands concepts, not just character matching.
        Vector embeddings provide the initial semantic similarity.
        Reranker (LLM) refines results by understanding context (slow but accurate).

        Args:
            query: Description or text to search for (e.g., "flying creature with deathtouch")
            initial_k: Number of candidates to retrieve via vector search (fast)
            final_k: Number of results to return after reranking (if reranker available)

        Returns:
            List of cards with relevance scores (vector distance or reranker score)

        Examples:
            "draw cards" → finds cards with card draw effects
            "flying creature with deathtouch" → finds creatures with both abilities
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Vector similarity search using natural language embeddings
        self.db.execute(
            """
            SELECT
                c.card_name,
                c.card_data,
                c.text_content,
                (e.embedding <=> %s::vector) as distance
            FROM mtg_cards c
            JOIN mtg_card_nl_embeddings e ON c.id = e.card_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, initial_k)
        )

        candidates = self.db.cursor.fetchall()

        if not candidates:
            return []

        # If no reranker, return vector search results
        if not self.reranker:
            return [{
                'card_name': card[0],
                'card_data': card[1],
                'text_content': card[2],
                'distance': float(card[3]),
                'match_type': 'vector_only'
            } for card in candidates]

        # Prepare documents for reranking
        documents = []
        for card in candidates:
            card_name = card[0]
            card_data = card[1]
            text_content = card[2] or ""

            # Build comprehensive document text
            doc_text = f"Card: {card_name}\n"
            if text_content:
                doc_text += f"Text: {text_content}\n"

            # Add relevant card data
            if card_data:
                if 'type' in card_data:
                    doc_text += f"Type: {card_data['type']}\n"
                if 'manaCost' in card_data:
                    doc_text += f"Mana Cost: {card_data['manaCost']}\n"
                if 'power' in card_data and 'toughness' in card_data:
                    doc_text += f"P/T: {card_data['power']}/{card_data['toughness']}\n"

            documents.append(doc_text.strip())

        # Rerank documents
        reranked = self.reranker.rerank(query, documents, top_k=final_k)

        # Map reranked results back to cards
        results = []
        for item in reranked:
            original_card = candidates[item['index']]
            results.append({
                'card_name': original_card[0],
                'card_data': original_card[1],
                'text_content': original_card[2],
                'score': item['score'],
                'match_type': 'vector_reranked'
            })

        return results

    def search_ambiguous(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Pure vector similarity search for ambiguous queries.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of cards with distance scores
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Vector similarity search only (using natural language embeddings)
        self.db.execute(
            """
            SELECT
                c.card_name,
                c.card_data,
                c.text_content,
                (e.embedding <=> %s::vector) as distance
            FROM mtg_cards c
            JOIN mtg_card_nl_embeddings e ON c.id = e.card_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k)
        )

        results = self.db.cursor.fetchall()

        return [{
            'card_name': card[0],
            'card_data': card[1],
            'text_content': card[2],
            'distance': float(card[3]),
            'match_type': 'vector_ambiguous'
        } for card in results]


class RulesSearchStrategies:
    """Search strategies for MTG rules."""

    def __init__(self, db: DatabaseConnection, embedder: OllamaEmbedder, reranker: Optional[OllamaReranker] = None):
        self.db = db
        self.embedder = embedder
        self.reranker = reranker

    def search_rules_text(
        self,
        query: str,
        initial_k: int = 30,
        final_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search rules using vector similarity + reranking.

        Args:
            query: Query about rules
            initial_k: Number of candidates to retrieve via vector search
            final_k: Number of results to return after reranking

        Returns:
            List of rules with relevance scores
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Vector similarity search
        self.db.execute(
            """
            SELECT
                r.rule_number,
                r.rule_text,
                r.section_title,
                r.rule_data,
                (e.embedding <=> %s::vector) as distance
            FROM mtg_rules r
            JOIN mtg_rule_embeddings e ON r.id = e.rule_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, initial_k)
        )

        candidates = self.db.cursor.fetchall()

        if not candidates:
            return []

        # If no reranker, return vector search results
        if not self.reranker:
            return [{
                'rule_number': rule[0],
                'rule_text': rule[1],
                'section_title': rule[2],
                'rule_data': rule[3],
                'distance': float(rule[4]),
                'match_type': 'vector_only'
            } for rule in candidates]

        # Prepare documents for reranking
        documents = []
        for rule in candidates:
            rule_number = rule[0] or ""
            rule_text = rule[1] or ""
            section_title = rule[2] or ""

            doc_text = f"Rule {rule_number}"
            if section_title:
                doc_text += f" ({section_title})"
            doc_text += f": {rule_text}"

            documents.append(doc_text.strip())

        # Rerank documents
        reranked = self.reranker.rerank(query, documents, top_k=final_k)

        # Map reranked results back to rules
        results = []
        for item in reranked:
            original_rule = candidates[item['index']]
            results.append({
                'rule_number': original_rule[0],
                'rule_text': original_rule[1],
                'section_title': original_rule[2],
                'rule_data': original_rule[3],
                'score': item['score'],
                'match_type': 'vector_reranked'
            })

        return results


class GlossarySearchStrategies:
    """Search strategies for MTG glossary terms."""

    def __init__(self, db: DatabaseConnection, embedder: OllamaEmbedder, reranker: Optional[OllamaReranker] = None):
        self.db = db
        self.embedder = embedder
        self.reranker = reranker

    def search_glossary_terms(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search glossary terms using vector similarity + reranking.

        Args:
            query: Query about a term or concept
            initial_k: Number of candidates to retrieve via vector search
            final_k: Number of results to return after reranking

        Returns:
            List of glossary entries with relevance scores
        """
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Vector similarity search
        self.db.execute(
            """
            SELECT
                g.term,
                g.definition,
                g.glossary_data,
                (e.embedding <=> %s::vector) as distance
            FROM mtg_glossary g
            JOIN mtg_glossary_embeddings e ON g.id = e.glossary_id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, initial_k)
        )

        candidates = self.db.cursor.fetchall()

        if not candidates:
            return []

        # If no reranker, return vector search results
        if not self.reranker:
            return [{
                'term': entry[0],
                'definition': entry[1],
                'glossary_data': entry[2],
                'distance': float(entry[3]),
                'match_type': 'vector_only'
            } for entry in candidates]

        # Prepare documents for reranking
        documents = []
        for entry in candidates:
            term = entry[0] or ""
            definition = entry[1] or ""

            doc_text = f"Term: {term}\nDefinition: {definition}"
            documents.append(doc_text.strip())

        # Rerank documents
        reranked = self.reranker.rerank(query, documents, top_k=final_k)

        # Map reranked results back to glossary entries
        results = []
        for item in reranked:
            original_entry = candidates[item['index']]
            results.append({
                'term': original_entry[0],
                'definition': original_entry[1],
                'glossary_data': original_entry[2],
                'score': item['score'],
                'match_type': 'vector_reranked'
            })

        return results


def format_card_result(card: Dict[str, Any], index: int) -> str:
    """Format a card search result for display."""
    output = f"\n[{index + 1}] {card['card_name']}"

    if 'similarity' in card:
        output += f" (Similarity: {card['similarity']:.2%})"
    elif 'score' in card:
        output += f" (Score: {card['score']:.3f})"
    elif 'distance' in card:
        output += f" (Distance: {card['distance']:.3f})"

    output += f" [{card['match_type']}]"

    if card.get('text_content'):
        output += f"\n  Text: {card['text_content']}"

    if card.get('card_data'):
        data = card['card_data']
        if 'type' in data:
            output += f"\n  Type: {data['type']}"
        if 'manaCost' in data:
            output += f"\n  Mana: {data['manaCost']}"

    return output


def format_rule_result(rule: Dict[str, Any], index: int) -> str:
    """Format a rules search result for display."""
    output = f"\n[{index + 1}] Rule {rule.get('rule_number', 'N/A')}"

    if rule.get('section_title'):
        output += f" - {rule['section_title']}"

    if 'score' in rule:
        output += f" (Score: {rule['score']:.3f})"
    elif 'distance' in rule:
        output += f" (Distance: {rule['distance']:.3f})"

    output += f" [{rule['match_type']}]"
    output += f"\n  {rule['rule_text']}"

    return output


def format_glossary_result(entry: Dict[str, Any], index: int) -> str:
    """Format a glossary search result for display."""
    output = f"\n[{index + 1}] {entry['term']}"

    if 'score' in entry:
        output += f" (Score: {entry['score']:.3f})"
    elif 'distance' in entry:
        output += f" (Distance: {entry['distance']:.3f})"

    output += f" [{entry['match_type']}]"
    output += f"\n  {entry['definition']}"

    return output


def main():
    """CLI for testing search strategies."""
    if len(sys.argv) < 3:
        print("Usage: python search_strategies.py <search_type> <query>")
        print("\nSearch types:")
        print("  card-name       - Fuzzy search by card name")
        print("  card-desc       - Vector + rerank search by card description")
        print("  card-ambiguous  - Pure vector search for ambiguous card queries")
        print("  rules           - Vector + rerank search for rules")
        print("  glossary        - Vector + rerank search for glossary terms")
        print("\nExamples:")
        print('  python search_strategies.py card-name "lightning bolt"')
        print('  python search_strategies.py card-desc "flying creature with deathtouch"')
        print('  python search_strategies.py card-ambiguous "red removal"')
        print('  python search_strategies.py rules "how does combat work"')
        print('  python search_strategies.py glossary "what is vigilance"')
        sys.exit(1)

    search_type = sys.argv[1].lower()
    query = sys.argv[2]

    # Initialize connections
    print(f"Connecting to database...")
    db = DatabaseConnection()
    db.connect()

    print(f"Initializing embedder...")
    embedder = OllamaEmbedder()

    # Initialize reranker (optional, will be None if model not available)
    reranker = None
    try:
        print(f"Initializing reranker...")
        reranker = OllamaReranker()
    except Exception as e:
        print(f"Warning: Reranker not available: {e}")
        print("Continuing with vector search only...")

    print(f"\nSearching for: '{query}'")
    print(f"Search type: {search_type}\n")
    print("=" * 80)

    try:
        if search_type == 'card-name':
            searcher = CardSearchStrategies(db, embedder, reranker)
            results = searcher.search_by_card_name(query, top_k=10)

            if not results:
                print("No matching cards found.")
            else:
                print(f"\nFound {len(results)} cards:")
                for i, card in enumerate(results):
                    print(format_card_result(card, i))

        elif search_type == 'card-desc':
            searcher = CardSearchStrategies(db, embedder, reranker)
            results = searcher.search_by_card_description(query, initial_k=30, final_k=10)

            if not results:
                print("No matching cards found.")
            else:
                print(f"\nFound {len(results)} cards:")
                for i, card in enumerate(results):
                    print(format_card_result(card, i))

        elif search_type == 'card-ambiguous':
            searcher = CardSearchStrategies(db, embedder, reranker)
            results = searcher.search_ambiguous(query, top_k=10)

            if not results:
                print("No matching cards found.")
            else:
                print(f"\nFound {len(results)} cards:")
                for i, card in enumerate(results):
                    print(format_card_result(card, i))

        elif search_type == 'rules':
            searcher = RulesSearchStrategies(db, embedder, reranker)
            results = searcher.search_rules_text(query, initial_k=30, final_k=10)

            if not results:
                print("No matching rules found.")
            else:
                print(f"\nFound {len(results)} rules:")
                for i, rule in enumerate(results):
                    print(format_rule_result(rule, i))

        elif search_type == 'glossary':
            searcher = GlossarySearchStrategies(db, embedder, reranker)
            results = searcher.search_glossary_terms(query, initial_k=20, final_k=5)

            if not results:
                print("No matching glossary entries found.")
            else:
                print(f"\nFound {len(results)} glossary entries:")
                for i, entry in enumerate(results):
                    print(format_glossary_result(entry, i))

        else:
            print(f"Unknown search type: {search_type}")
            print("Valid types: card-name, card-desc, card-ambiguous, rules, glossary")

    finally:
        db.close()


if __name__ == "__main__":
    main()
