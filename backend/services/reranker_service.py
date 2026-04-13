"""
Cross-encoder reranker executed in a dedicated subprocess.

This keeps sentence-transformers/torch isolated from the main API process so it
does not share OpenMP runtime state with FAISS on macOS.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Tuple

from config.settings import Settings
from services.chunking_service import Chunk

logger = logging.getLogger(__name__)


class RerankerService:
    """Optional cross-encoder reranker using a persistent worker process."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: subprocess.Popen[str] | None = None
        self._process_lock = threading.Lock()
        self._load_error: str | None = None
        atexit.register(self.close)

    def is_enabled(self) -> bool:
        return self.settings.reranker_enabled

    def get_model_name(self) -> str:
        return self.settings.reranker_model

    def get_top_k(self) -> int:
        return self.settings.reranker_top_k

    def get_status(self) -> dict:
        available = self._ensure_worker() if self.is_enabled() else False
        return {
            "enabled": self.is_enabled(),
            "available": available,
            "model": self.get_model_name(),
            "top_k": self.get_top_k(),
            "error": self._load_error
        }

    def _close_unlocked(self) -> None:
        """Close the worker process. Caller must hold _process_lock when using this helper."""
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

    def _build_pair_text(self, chunk: Chunk) -> str:
        return (
            f"Title: {chunk.metadata.page_title}\n"
            f"Source type: {chunk.metadata.source_type}\n"
            f"Chunk {chunk.metadata.chunk_index + 1} of {chunk.metadata.total_chunks}\n"
            f"Content:\n{chunk.text}"
        )

    def _normalize_score(self, score: float) -> float:
        if 0.0 <= score <= 1.0:
            return float(score)

        return float(1.0 / (1.0 + math.exp(-score)))

    def _worker_script_path(self) -> str:
        return str(Path(__file__).with_name("reranker_worker.py"))

    def _start_worker(self) -> bool:
        env = os.environ.copy()
        env["RERANKER_MODEL"] = self.settings.reranker_model
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
            logger.warning("Could not start reranker worker: %s", exc)
            self._process = None
            return False

        ready_line = self._read_stdout_line(self.settings.reranker_startup_timeout_seconds)

        if not ready_line:
            self._load_error = self._read_worker_error() or (
                f"Reranker worker did not respond within "
                f"{self.settings.reranker_startup_timeout_seconds} seconds"
            )
            self._close_unlocked()
            return False

        try:
            ready_payload = json.loads(ready_line)
        except json.JSONDecodeError:
            self._load_error = f"Invalid reranker worker bootstrap response: {ready_line}"
            self._close_unlocked()
            return False

        if not ready_payload.get("success"):
            self._load_error = ready_payload.get("error", "Reranker worker failed to initialize")
            self._close_unlocked()
            return False

        self._load_error = None
        return True

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

    def _ensure_worker(self) -> bool:
        if not self.is_enabled():
            return False

        if not self.settings.reranker_use_subprocess:
            self._load_error = "Only subprocess mode is supported for the reranker"
            return False

        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                return True

            self._close_unlocked()
            return self._start_worker()

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]]
    ) -> tuple[List[Tuple[Chunk, float]], dict]:
        if not candidates:
            return candidates, {
                "reranker_used": False,
                "reranker_model": self.get_model_name(),
                "reranker_candidates": 0
            }

        if not self._ensure_worker():
            return candidates, {
                "reranker_used": False,
                "reranker_model": self.get_model_name(),
                "reranker_candidates": 0
            }

        top_k = min(self.get_top_k(), len(candidates))
        candidate_slice = candidates[:top_k]
        remaining_candidates = candidates[top_k:]

        payload = {
            "action": "rerank",
            "query": query,
            "pairs": [self._build_pair_text(chunk) for chunk, _ in candidate_slice]
        }

        try:
            with self._process_lock:
                if self._process is None or self._process.poll() is not None:
                    raise RuntimeError("Reranker worker is not running")

                assert self._process.stdin is not None

                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
                response_line = self._read_stdout_line(
                    self.settings.reranker_inference_timeout_seconds
                )
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Cross-encoder subprocess rerank failed, falling back: %s", exc)
            self.close()
            return candidates, {
                "reranker_used": False,
                "reranker_model": self.get_model_name(),
                "reranker_candidates": 0
            }

        if not response_line:
            self._load_error = self._read_worker_error() or (
                f"Reranker worker timed out after "
                f"{self.settings.reranker_inference_timeout_seconds} seconds"
            )
            logger.warning("Cross-encoder subprocess returned no data, falling back")
            self.close()
            return candidates, {
                "reranker_used": False,
                "reranker_model": self.get_model_name(),
                "reranker_candidates": 0
            }

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError:
            self._load_error = f"Invalid reranker worker response: {response_line}"
            logger.warning(self._load_error)
            self.close()
            return candidates, {
                "reranker_used": False,
                "reranker_model": self.get_model_name(),
                "reranker_candidates": 0
            }

        if not response.get("success"):
            self._load_error = response.get("error", "Unknown reranker worker error")
            logger.warning("Cross-encoder subprocess unavailable, falling back: %s", self._load_error)
            self.close()
            return candidates, {
                "reranker_used": False,
                "reranker_model": self.get_model_name(),
                "reranker_candidates": 0
            }

        raw_scores = response.get("scores", [])
        reranked_slice = []

        for (chunk, prior_score), raw_score in zip(candidate_slice, raw_scores):
            cross_score = self._normalize_score(float(raw_score))
            final_score = (cross_score * 0.8) + (prior_score * 0.2)
            reranked_slice.append((chunk, final_score))

        reranked_slice.sort(key=lambda item: item[1], reverse=True)

        return reranked_slice + remaining_candidates, {
            "reranker_used": True,
            "reranker_model": self.get_model_name(),
            "reranker_candidates": len(candidate_slice)
        }
