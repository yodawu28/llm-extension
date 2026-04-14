"""
Dedicated subprocess worker for local text embeddings.

Protocol:
- stdin: JSON lines
- stdout: JSON lines
"""

from __future__ import annotations

import json
import os
import sys


def _bootstrap():
    model_name = os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        dimensions = int(model.get_sentence_embedding_dimension() or 0)
    except Exception as exc:
        print(json.dumps({
            "success": False,
            "error": str(exc)
        }), flush=True)
        return None, model_name, 0

    print(json.dumps({
        "success": True,
        "model": model_name,
        "dimensions": dimensions
    }), flush=True)
    return model, model_name, dimensions


def main():
    model, _, _ = _bootstrap()
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

        if payload.get("action") != "embed":
            print(json.dumps({
                "success": False,
                "error": "Unsupported action"
            }), flush=True)
            continue

        texts = payload.get("texts", [])

        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            print(json.dumps({
                "success": True,
                "embeddings": embeddings.tolist()
            }), flush=True)
        except Exception as exc:
            print(json.dumps({
                "success": False,
                "error": str(exc)
            }), flush=True)


if __name__ == "__main__":
    main()
