"""
Embedding Service - Generate vector embeddings for text chunks
Uses OpenAI text-embedding-3-small for cost-effectiveness and quality
"""

import asyncio
from typing import List
from openai import AsyncOpenAI
import numpy as np

from config.settings import Settings


class EmbeddingService:
    """Generate embeddings using OpenAI's embedding models"""

    def __init__(self, settings: Settings):
        self.settings = settings
        api_key = settings.resolved_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key or PAT token not found in settings")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.resolved_openai_base_url(),
            timeout=settings.request_timeout_seconds,
            max_retries=1
        )
        self.model = settings.openai_embedding_model
        self.dimensions = settings.openai_embedding_dimensions

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single batch

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        # OpenAI API supports batch embedding (up to 2048 texts)
        try:
            response = await asyncio.wait_for(
                self.client.embeddings.create(
                    input=texts,
                    model=self.model
                ),
                timeout=self.settings.request_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"Embedding request timed out after {self.settings.request_timeout_seconds} seconds"
            ) from exc

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text string to embed

        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
