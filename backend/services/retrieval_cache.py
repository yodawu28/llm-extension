from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import Lock
from typing import List, Optional

from schemas.requests import PageContent
from services.chunking_service import Chunk


@dataclass
class CachedCorpus:
    chunks: List[Chunk]
    embeddings: List[List[float]]
    chunk_stats: dict


class RetrievalCache:
    """Small in-memory cache for chunked corpora and repeated query embeddings."""

    def __init__(self, max_corpora: int = 12, max_query_embeddings: int = 128):
        self.max_corpora = max_corpora
        self.max_query_embeddings = max_query_embeddings
        self._corpora: OrderedDict[str, CachedCorpus] = OrderedDict()
        self._query_embeddings: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = Lock()

    def build_corpus_signature(self, pages: List[PageContent]) -> str:
        hasher = sha256()

        for page in sorted(pages, key=lambda item: item.url):
            hasher.update(page.url.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(page.title.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(page.source_type.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(page.markdown.encode("utf-8"))
            hasher.update(b"\0")

        return hasher.hexdigest()

    def get_corpus(self, signature: str) -> Optional[CachedCorpus]:
        with self._lock:
            cached = self._corpora.get(signature)
            if cached is None:
                return None

            self._corpora.move_to_end(signature)
            return cached

    def set_corpus(self, signature: str, cached: CachedCorpus) -> None:
        with self._lock:
            self._corpora[signature] = cached
            self._corpora.move_to_end(signature)

            while len(self._corpora) > self.max_corpora:
                self._corpora.popitem(last=False)

    def build_query_signature(self, question: str) -> str:
        normalized = question.strip()
        return sha256(normalized.encode("utf-8")).hexdigest()

    def get_query_embedding(self, signature: str) -> Optional[List[float]]:
        with self._lock:
            cached = self._query_embeddings.get(signature)
            if cached is None:
                return None

            self._query_embeddings.move_to_end(signature)
            return cached

    def set_query_embedding(self, signature: str, embedding: List[float]) -> None:
        with self._lock:
            self._query_embeddings[signature] = embedding
            self._query_embeddings.move_to_end(signature)

            while len(self._query_embeddings) > self.max_query_embeddings:
                self._query_embeddings.popitem(last=False)
