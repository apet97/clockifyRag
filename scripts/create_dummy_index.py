#!/usr/bin/env python3
"""Create minimal dummy index artifacts for offline benchmark smoke tests."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clockify_rag import config, build_bm25, compute_sha256
from clockify_rag.utils import resolve_corpus_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    texts = [
        "Clockify allows you to track time for tasks and projects using timers or manual entries.",
        "You can invite team members via the Team settings page and assign different workspace roles.",
        "Export detailed reports as CSV or PDF from the Reports tab for billing or compliance purposes.",
        "Use tags to categorize entries and filter dashboards for focused productivity insights.",
    ]

    chunks: list[dict] = []
    for idx, text in enumerate(texts):
        chunks.append(
            {
                "id": f"dummy-{idx}",
                "title": "Clockify Help",
                "section": f"Section {idx + 1}",
                "url": f"https://example.com/clockify/{idx}",
                "text": text,
            }
        )

    bm = build_bm25(chunks)

    vecs = np.array([np.roll(np.eye(config.EMB_DIM, dtype="float32")[0], idx) for idx in range(len(chunks))])
    if vecs.shape[1] < config.EMB_DIM:
        raise RuntimeError("Failed to create dummy embeddings with correct dimension")

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_n = (vecs / norms).astype("float32")

    chunks_path = REPO_ROOT / config.FILES["chunks"]
    emb_path = REPO_ROOT / config.FILES["emb"]
    bm25_path = REPO_ROOT / config.FILES["bm25"]
    meta_path = REPO_ROOT / config.FILES["index_meta"]

    _write_jsonl(chunks_path, chunks)
    np.save(emb_path, vecs_n)
    with bm25_path.open("w", encoding="utf-8") as handle:
        json.dump(bm, handle)

    kb_candidate, _, _ = resolve_corpus_path()
    kb_path = Path(kb_candidate)
    if not kb_path.is_absolute():
        kb_path = REPO_ROOT / kb_path

    if kb_path.exists():
        kb_sha = compute_sha256(str(kb_path))
    else:
        kb_sha = "dummy"

    index_meta = {
        "kb_sha256": kb_sha,
        "chunks": len(chunks),
        "emb_rows": int(vecs_n.shape[0]),
        "bm25_docs": len(bm["doc_lens"]),
        "gen_model": config.RAG_CHAT_MODEL,
        "emb_model": "all-MiniLM-L6-v2" if config.EMB_BACKEND == "local" else config.RAG_EMBED_MODEL,
        "emb_backend": config.EMB_BACKEND,
        "ann": "none",
        "mmr_lambda": config.MMR_LAMBDA,
        "chunk_chars": config.CHUNK_CHARS,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_version": "ci-dummy",
    }

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(index_meta, handle)


if __name__ == "__main__":
    main()
