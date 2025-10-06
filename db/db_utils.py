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
            print(f"✓ Connected to database: {self.database}")
            return self
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("✓ Database connection closed")

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
        model: str = "embeddinggemma:latest"
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
                print(f"✓ Ollama connection successful (model: {self.model})")
                print(f"  Embedding dimension: {len(test_embedding)}")
                return True
            else:
                print(f"✗ Ollama connection failed")
                return False
        except Exception as e:
            print(f"✗ Ollama connection error: {e}")
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
                print("✓ PostgreSQL is ready")
                return True
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"  Attempt {attempt + 1}/{max_attempts} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"✗ PostgreSQL not ready after {max_attempts} attempts")
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
                print(f"✗ Ollama not ready after {max_attempts} attempts")
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
