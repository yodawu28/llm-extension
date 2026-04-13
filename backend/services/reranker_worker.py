"""
Dedicated subprocess worker for cross-encoder reranking.

Protocol:
- stdin: JSON lines
- stdout: JSON lines
"""

from __future__ import annotations

import json
import math
import os
import sys


def _bootstrap():
    model_name = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model = CrossEncoder(model_name)
    except Exception as exc:
        print(json.dumps({
            "success": False,
            "error": str(exc)
        }), flush=True)
        return None, model_name

    print(json.dumps({
        "success": True,
        "model": model_name
    }), flush=True)
    return model, model_name


def _normalize_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return float(score)

    return float(1.0 / (1.0 + math.exp(-score)))


def main():
    model, _ = _bootstrap()
    if model is None:
      return

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({
                "success": False,
                "error": f"Invalid JSON payload: {exc}"
            }), flush=True)
            continue

        if payload.get("action") != "rerank":
            print(json.dumps({
                "success": False,
                "error": "Unsupported action"
            }), flush=True)
            continue

        query = payload.get("query", "")
        pairs = payload.get("pairs", [])

        try:
            raw_scores = model.predict(
                [(query, pair_text) for pair_text in pairs],
                show_progress_bar=False
            )
            scores = [_normalize_score(float(score)) for score in raw_scores]
            print(json.dumps({
                "success": True,
                "scores": scores
            }), flush=True)
        except Exception as exc:
            print(json.dumps({
                "success": False,
                "error": str(exc)
            }), flush=True)


if __name__ == "__main__":
    main()
