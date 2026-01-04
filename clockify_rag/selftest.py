"""Self-test harness for the Clockify RAG system."""

from __future__ import annotations

import os
import sys

from . import config
from .api_client import get_llm_client
from .runtime import ensure_index_ready


def run_selftest() -> bool:
    """Lightweight production self-test.

    Checks:
    1. Model endpoint reachable at RAG_OLLAMA_URL / OLLAMA_URL.
    2. Required models visible (best-effort).
    3. Basic index presence.
    4. Retrieval path does not crash (strict mode only).
    """
    ok = True
    strict_mode = os.environ.get("SELFTEST_STRICT", "").lower() not in {"", "0", "false", "no"}
    client_mode = (os.environ.get("RAG_LLM_CLIENT") or "").strip().lower()

    if not client_mode:
        os.environ["RAG_LLM_CLIENT"] = "mock"
        client_mode = "mock"
        print("[SELFTEST] RAG_LLM_CLIENT not set; using mock client for offline self-test.")

    if client_mode == "mock":
        print("[SELFTEST] Skipping Ollama connectivity check (mock client).")
    else:
        try:
            client = get_llm_client()
            models = client.list_models()
            model_names = {m.get("name") or m.get("model") for m in models}
            required = {config.RAG_CHAT_MODEL, config.RAG_EMBED_MODEL}
            missing = {m for m in required if m and m not in model_names}
            print(f"[SELFTEST] Model endpoint OK at {config.RAG_OLLAMA_URL}")
            if missing:
                print(f"[SELFTEST] Warning: missing models (not fatal for selftest): {sorted(missing)}")
        except Exception as exc:
            msg = f"[SELFTEST] WARNING: model endpoint check failed: {exc}"
            if strict_mode:
                print(msg.replace("WARNING", "ERROR"))
                ok = False
            else:
                print(msg)

    index_ok = True
    index_files = [
        config.FILES["chunks"],
        config.FILES["bm25"],
        config.FILES["index_meta"],
    ]
    ann_path = config.FILES.get("faiss_index") if config.USE_ANN == "faiss" else None
    if ann_path:
        index_files.append(ann_path)

    missing_any = any(not os.path.exists(path) for path in index_files)
    if missing_any:
        print("[SELFTEST] WARNING: one or more index files are missing.")
        print("          Run the ingest command before using in production.")
        if strict_mode:
            index_ok = False
    else:
        print("[SELFTEST] Index files found.")

    if index_ok and ok and strict_mode:
        try:
            ensure_index_ready(retries=0)
            print("[SELFTEST] Retrieval path OK.")
        except Exception as exc:
            print(f"[SELFTEST] ERROR: retrieval check failed: {exc}")
            ok = False

    return ok and (index_ok or not strict_mode)


def main() -> int:
    """CLI entrypoint for selftest."""
    return 0 if run_selftest() else 1


if __name__ == "__main__":
    sys.exit(main())
