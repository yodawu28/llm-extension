"""
Vector Store Service - Store and retrieve embeddings using FAISS
In-memory vector database for fast similarity search
"""

from typing import List, Tuple, Optional
import numpy as np
import faiss

from services.chunking_service import Chunk


class VectorStore:
    """In-memory vector store using FAISS for similarity search"""

    def __init__(self, embedding_dimension: int = 1536):
        """
        Initialize FAISS index

        Args:
            embedding_dimension: Dimension of embedding vectors (1536 for text-embedding-3-small)
        """
        self.dimension = embedding_dimension

        # Use cosine similarity via normalized vectors + inner product.
        self.index = faiss.IndexFlatIP(embedding_dimension)

        # Store chunks alongside vectors
        self.chunks: List[Chunk] = []

        # Track number of vectors
        self.num_vectors = 0

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks and their embeddings to the store

        Args:
            chunks: List of Chunk objects
            embeddings: Corresponding embedding vectors
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})")

        # Convert embeddings to numpy array with float32 (FAISS requirement)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)

        # Add to FAISS index
        self.index.add(embeddings_array)

        # Store chunks
        self.chunks.extend(chunks)
        self.num_vectors = len(self.chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for most similar chunks to query

        Args:
            query_embedding: Embedding vector of user's question
            top_k: Number of top results to return

        Returns:
            List of (Chunk, similarity_score) tuples, sorted by relevance
        """
        if self.num_vectors == 0:
            return []

        # Limit top_k to available chunks
        k = min(top_k, self.num_vectors)

        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)

        faiss.normalize_L2(query_array)

        # Search FAISS index (returns cosine similarity scores and indices)
        similarities, indices = self.index.search(query_array, k)

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.chunks):  # Validate index
                chunk = self.chunks[idx]
                results.append((chunk, float(similarity)))

        return results

    def clear(self) -> None:
        """Clear all vectors and chunks from store"""
        self.index.reset()
        self.chunks.clear()
        self.num_vectors = 0

    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        unique_pages = set(chunk.metadata.page_url for chunk in self.chunks)

        return {
            "total_vectors": self.num_vectors,
            "total_pages": len(unique_pages),
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP (cosine)"
        }
