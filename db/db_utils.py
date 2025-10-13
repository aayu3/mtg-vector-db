"""
Database utility functions for MTG Vector Database.
Handles PostgreSQL connections and Ollama embedding generation.
"""

import psycopg2
from psycopg2.extras import execute_batch, Json
import requests
import time
from typing import List, Dict, Any, Optional
import os


class DatabaseConnection:
    """Manages PostgreSQL database connections."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "mtg_vectors",
        user: str = "postgres",
        password: str = "postgres"
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.conn.cursor()
            print(f"[OK] Connected to database: {self.database}")
            return self
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("[OK] Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.close()

    def execute(self, query: str, params: tuple = None):
        """Execute a query."""
        self.cursor.execute(query, params)
        return self.cursor

    def executemany(self, query: str, params: List[tuple]):
        """Execute a query with multiple parameter sets."""
        execute_batch(self.cursor, query, params, page_size=100)

    def commit(self):
        """Commit the current transaction."""
        self.conn.commit()

    def rollback(self):
        """Rollback the current transaction."""
        self.conn.rollback()


class OllamaEmbedder:
    """Generates embeddings using Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "embeddinggemma:300m"
    ):
        self.base_url = base_url
        self.model = model
        self.embed_url = f"{base_url}/api/embeddings"

    def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            max_retries: Number of retry attempts on failure

        Returns:
            List of floats representing the embedding vector
        """
        payload = {
            "model": self.model,
            "prompt": text
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.embed_url,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('embedding')
                else:
                    print(f"Warning: Ollama API returned status {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Warning: Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        print(f"Error: Failed to generate embedding after {max_retries} attempts")
        return None

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            batch_size: Number of embeddings to generate between progress updates
            show_progress: Whether to show progress information

        Returns:
            List of embedding vectors (same length as input texts)
        """
        embeddings = []
        total = len(texts)

        for i, text in enumerate(texts):
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)

            if show_progress and (i + 1) % batch_size == 0:
                print(f"  Generated {i + 1}/{total} embeddings...")

        if show_progress:
            successful = sum(1 for e in embeddings if e is not None)
            print(f"  Completed: {successful}/{total} embeddings generated successfully")

        return embeddings

    def test_connection(self) -> bool:
        """Test if Ollama is accessible and model is available."""
        try:
            # Try to generate a simple embedding
            test_embedding = self.generate_embedding("test", max_retries=1)
            if test_embedding:
                print(f"[OK] Ollama connection successful (model: {self.model})")
                print(f"  Embedding dimension: {len(test_embedding)}")
                return True
            else:
                print(f"[ERROR] Ollama connection failed")
                return False
        except Exception as e:
            print(f"[ERROR] Ollama connection error: {e}")
            return False


def wait_for_postgres(
    max_attempts: int = 30,
    delay: int = 2,
    **db_kwargs
) -> bool:
    """
    Wait for PostgreSQL to be ready.

    Args:
        max_attempts: Maximum number of connection attempts
        delay: Delay between attempts in seconds
        **db_kwargs: Database connection parameters

    Returns:
        True if connection successful, False otherwise
    """
    print("Waiting for PostgreSQL to be ready...")

    for attempt in range(max_attempts):
        try:
            with DatabaseConnection(**db_kwargs) as db:
                db.execute("SELECT 1")
                print("[OK] PostgreSQL is ready")
                return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"  Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"[ERROR] PostgreSQL not ready after {max_attempts} attempts")
                return False

    return False


def wait_for_ollama(
    max_attempts: int = 30,
    delay: int = 2,
    **ollama_kwargs
) -> bool:
    """
    Wait for Ollama to be ready.

    Args:
        max_attempts: Maximum number of connection attempts
        delay: Delay between attempts in seconds
        **ollama_kwargs: Ollama connection parameters

    Returns:
        True if connection successful, False otherwise
    """
    print("Waiting for Ollama to be ready...")

    embedder = OllamaEmbedder(**ollama_kwargs)

    for attempt in range(max_attempts):
        try:
            if embedder.test_connection():
                return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"  Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"[ERROR] Ollama not ready after {max_attempts} attempts")
                return False

    return False


def format_vector_for_postgres(embedding: List[float]) -> str:
    """
    Format embedding vector for PostgreSQL insertion.

    Args:
        embedding: List of floats

    Returns:
        String formatted for pgvector (e.g., '[0.1, 0.2, 0.3]')
    """
    return '[' + ','.join(str(x) for x in embedding) + ']'


class OllamaReranker:
    """Reranks search results using Ollama reranker models."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "dengcao/Qwen3-Reranker-8B:F16"
    ):
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank Magic: The Gathering cards based on relevance to query.

        Uses batch prompt to compare all cards at once for better context.

        Args:
            query: The search query (e.g., "flying creature with deathtouch")
            documents: List of card descriptions to rerank
            top_k: Number of top results to return (None = return all)
            max_retries: Number of retry attempts on failure

        Returns:
            List of dicts with 'index', 'text', and 'score' sorted by relevance
        """
        if not documents:
            return []

        # Build batch prompt with all cards
        prompt = f"""You are ranking Magic: The Gathering cards by relevance.

User is searching for: "{query}"

Here are the candidate MTG cards:

"""
        for idx, doc in enumerate(documents, 1):
            prompt += f"{idx}. {doc}\n\n"

        prompt += f"""Rank these {len(documents)} cards from most to least relevant to the search query "{query}".
Output ONLY the numbers in order, separated by spaces (e.g., "3 1 5 2 4").
Most relevant first."""

        # Get ranking from model
        ranking = self._get_batch_ranking(prompt, len(documents), max_retries)

        if not ranking:
            # If batch ranking fails, return original order with equal scores
            return [{'index': i, 'text': doc, 'score': 0.5} for i, doc in enumerate(documents)][:top_k]

        # Convert ranking to scores (inverse rank normalized to 0-1)
        results = []
        for rank_position, doc_index in enumerate(ranking):
            # Higher rank = higher score
            score = 1.0 - (rank_position / len(documents))
            results.append({
                'index': doc_index,
                'text': documents[doc_index],
                'score': score
            })

        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def _get_batch_ranking(self, prompt: str, num_docs: int, max_retries: int) -> Optional[List[int]]:
        """
        Get ranking from model for batch of documents.

        Args:
            prompt: The formatted batch ranking prompt
            num_docs: Expected number of documents
            max_retries: Number of retry attempts

        Returns:
            List of document indices in ranked order (0-indexed), or None on failure
        """
        import re

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 100
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()

                    # Parse ranking from response
                    # Look for numbers separated by spaces, commas, or newlines
                    numbers = re.findall(r'\d+', response_text)

                    if numbers:
                        # Convert to 0-indexed (prompt uses 1-indexed)
                        ranking = [int(n) - 1 for n in numbers]

                        # Validate: should have valid indices
                        ranking = [idx for idx in ranking if 0 <= idx < num_docs]

                        # Add any missing indices at the end
                        missing = set(range(num_docs)) - set(ranking)
                        ranking.extend(sorted(missing))

                        return ranking[:num_docs]

                else:
                    print(f"Warning: Reranker API returned status {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Warning: Reranker request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _get_relevance_score(self, prompt: str, max_retries: int) -> Optional[float]:
        """
        Get relevance score from reranker model.

        Args:
            prompt: The formatted prompt
            max_retries: Number of retry attempts

        Returns:
            Relevance score (0-1) or None on failure
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 10
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '').strip()

                    # Try to parse score from response
                    score = self._parse_score(response_text)
                    return score
                else:
                    print(f"Warning: Reranker API returned status {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"Warning: Reranker request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _parse_score(self, response_text: str) -> float:
        """
        Parse relevance score from model response.

        Args:
            response_text: Raw text from model

        Returns:
            Score between 0 and 1
        """
        import re

        # Try to extract a number from the response
        # Look for patterns like "0.85", "85%", "8.5/10", etc.
        patterns = [
            r'(\d+\.?\d*)\s*%',  # "85%"
            r'(\d+\.?\d*)\s*/\s*10',  # "8.5/10"
            r'(\d+\.?\d*)\s*/\s*100',  # "85/100"
            r'^(\d+\.?\d*)',  # Just a number at start
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                score = float(match.group(1))

                # Normalize to 0-1 range
                if '%' in pattern or '/100' in pattern:
                    score = score / 100.0
                elif '/10' in pattern:
                    score = score / 10.0

                return max(0.0, min(1.0, score))

        # If no pattern matched, try to convert entire response to float
        try:
            score = float(response_text.strip())
            return max(0.0, min(1.0, score))
        except:
            # Default low score if parsing failed
            return 0.0

    def test_connection(self) -> bool:
        """Test if reranker model is accessible."""
        try:
            test_score = self._get_relevance_score(
                "Query: test\n\nDocument: test document\n\nRelevance score:",
                max_retries=1
            )
            if test_score is not None:
                print(f"[OK] Reranker connection successful (model: {self.model})")
                return True
            else:
                print(f"[ERROR] Reranker connection failed")
                return False
        except Exception as e:
            print(f"[ERROR] Reranker connection error: {e}")
            return False
