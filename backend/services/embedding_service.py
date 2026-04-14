"""
Embedding Service - Generate vector embeddings for text chunks
Uses OpenAI text-embedding-3-small for cost-effectiveness and quality
"""

import atexit
import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import numpy as np
from openai import AsyncOpenAI

from config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using OpenAI's embedding models"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: AsyncOpenAI | None = None
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._load_error: str | None = None
        atexit.register(self.close)

        if settings.embedding_provider == "local":
            self.model = settings.local_embedding_model
            self.dimensions = settings.local_embedding_dimensions
            logger.info(
                "Embedding service configured for local model=%s dimensions=%s",
                self.model,
                self.dimensions,
            )
            return

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
        logger.info(
            "Embedding client initialized provider=%s model=%s dimensions=%s base_url=%s timeout_seconds=%s",
            settings.embedding_provider,
            self.model,
            self.dimensions,
            settings.resolved_openai_base_url(),
            settings.request_timeout_seconds,
        )

    def _worker_script_path(self) -> str:
        return str(Path(__file__).with_name("local_embedding_worker.py"))

    def _close_unlocked(self) -> None:
        if self._process is None:
            return

        try:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            self._process.wait(timeout=2)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        finally:
            self._process = None

    def close(self) -> None:
        with self._process_lock:
            self._close_unlocked()

    def _read_pipe_line(self, pipe, timeout_seconds: float) -> str | None:
        if pipe is None:
            return None

        line_queue: queue.Queue[str | None] = queue.Queue(maxsize=1)

        def _reader():
            try:
                line_queue.put(pipe.readline())
            except Exception:
                line_queue.put(None)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        try:
            line = line_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

        if line is None:
            return None

        stripped = line.strip()
        return stripped or None

    def _read_stdout_line(self, timeout_seconds: float) -> str | None:
        if self._process is None:
            return None

        return self._read_pipe_line(self._process.stdout, timeout_seconds)

    def _read_worker_error(self) -> str | None:
        if self._process is None or self._process.stderr is None:
            return None

        return self._read_pipe_line(self._process.stderr, 0.2)

    def _start_local_worker(self) -> bool:
        env = os.environ.copy()
        env["LOCAL_EMBEDDING_MODEL"] = self.settings.local_embedding_model
        env["TOKENIZERS_PARALLELISM"] = "false"

        try:
            self._process = subprocess.Popen(
                [sys.executable, self._worker_script_path()],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env
            )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Could not start local embedding worker: %s", exc)
            self._process = None
            return False

        ready_line = self._read_stdout_line(self.settings.local_embedding_startup_timeout_seconds)
        if not ready_line:
            self._load_error = self._read_worker_error() or (
                f"Local embedding worker did not respond within "
                f"{self.settings.local_embedding_startup_timeout_seconds} seconds"
            )
            self._close_unlocked()
            return False

        try:
            ready_payload = json.loads(ready_line)
        except json.JSONDecodeError:
            self._load_error = f"Invalid local embedding worker bootstrap response: {ready_line}"
            self._close_unlocked()
            return False

        if not ready_payload.get("success"):
            self._load_error = ready_payload.get("error", "Local embedding worker failed to initialize")
            self._close_unlocked()
            return False

        dimensions = int(ready_payload.get("dimensions") or self.dimensions or 0)
        if dimensions > 0:
            self.dimensions = dimensions

        self._load_error = None
        logger.info(
            "Local embedding worker ready model=%s dimensions=%s",
            ready_payload.get("model", self.model),
            self.dimensions,
        )
        return True

    def _ensure_local_worker(self) -> bool:
        if self.settings.embedding_provider != "local":
            return False

        if not self.settings.local_embedding_use_subprocess:
            self._load_error = "Only subprocess mode is supported for local embeddings"
            return False

        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                return True

            self._close_unlocked()
            return self._start_local_worker()

    def _embed_texts_local_sync(self, texts: List[str]) -> List[List[float]]:
        if not self._ensure_local_worker():
            raise RuntimeError(self._load_error or "Local embedding worker is not available")

        payload = {"action": "embed", "texts": texts}

        try:
            with self._process_lock:
                if self._process is None or self._process.poll() is not None:
                    raise RuntimeError("Local embedding worker is not running")

                assert self._process.stdin is not None

                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
                response_line = self._read_stdout_line(
                    self.settings.local_embedding_inference_timeout_seconds
                )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Local embedding subprocess failed: %s", exc)
            self.close()
            raise RuntimeError(self._load_error) from exc

        if not response_line:
            self._load_error = self._read_worker_error() or (
                f"Local embedding worker timed out after "
                f"{self.settings.local_embedding_inference_timeout_seconds} seconds"
            )
            logger.warning(self._load_error)
            self.close()
            raise TimeoutError(self._load_error)

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            self._load_error = f"Invalid local embedding worker response: {response_line}"
            logger.warning(self._load_error)
            self.close()
            raise RuntimeError(self._load_error) from exc

        if not response.get("success"):
            self._load_error = response.get("error", "Unknown local embedding worker error")
            logger.warning("Local embedding worker unavailable: %s", self._load_error)
            self.close()
            raise RuntimeError(self._load_error)

        return response.get("embeddings", [])

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

        if self.settings.embedding_provider == "local":
            try:
                return await asyncio.to_thread(self._embed_texts_local_sync, texts)
            except Exception:
                logger.exception(
                    "Local embedding request failed model=%s texts=%s",
                    self.model,
                    len(texts),
                )
                raise

        # OpenAI API supports batch embedding (up to 2048 texts)
        try:
            assert self.client is not None
            response = await asyncio.wait_for(
                self.client.embeddings.create(
                    input=texts,
                    model=self.model
                ),
                timeout=self.settings.request_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "Embedding request timed out model=%s texts=%s timeout_seconds=%s",
                self.model,
                len(texts),
                self.settings.request_timeout_seconds,
            )
            raise TimeoutError(
                f"Embedding request timed out after {self.settings.request_timeout_seconds} seconds"
            ) from exc
        except Exception:
            logger.exception(
                "Embedding request failed provider=%s model=%s texts=%s base_url=%s",
                self.settings.embedding_provider,
                self.model,
                len(texts),
                self.settings.resolved_openai_base_url(),
            )
            raise

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
