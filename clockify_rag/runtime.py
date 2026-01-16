"""Shared runtime helpers for CLI and API workflows."""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple

from . import config
from .answer import answer_once
from .api_client import get_llm_client
from .caching import get_query_cache
from .error_handlers import log_and_raise
from .exceptions import IndexLoadError
from .indexing import build, load_index
from .precomputed_cache import get_precomputed_cache
from .utils import _log_config_summary, resolve_corpus_path

logger = logging.getLogger(__name__)


def _extract_source_urls(chunks: list, result: dict) -> list[str]:
    """Resolve URLs from sources_used or chunk ids."""
    urls: set[str] = set()
    meta = result.get("metadata") or {}
    sources_used = result.get("sources_used") or meta.get("sources_used") or []
    for src in sources_used:
        if isinstance(src, str) and src.strip().lower().startswith("http"):
            urls.add(src.strip())

    if urls:
        return sorted(urls)

    id_to_chunk = {}
    for c in chunks:
        cid = c.get("id")
        if cid is not None:
            id_to_chunk[str(cid)] = c

    candidates: list = []
    candidates.extend(result.get("selected_chunk_ids") or [])
    candidates.extend(meta.get("source_chunk_ids") or [])

    for cand in candidates:
        chunk = id_to_chunk.get(str(cand).strip())
        if chunk:
            url = chunk.get("url") or chunk.get("source_url") or chunk.get("doc_url")
            if url:
                urls.add(url)

    return sorted(urls)


def ensure_index_ready(retries: int = 0) -> Tuple:
    """Ensure retrieval artifacts are present and return loaded index components."""
    kb_path, kb_exists, candidates = resolve_corpus_path()

    artifacts_ok = True
    missing_files = []
    for fname in [
        config.FILES["chunks"],
        config.FILES["emb"],
        config.FILES["meta"],
        config.FILES["bm25"],
        config.FILES["index_meta"],
    ]:
        if not os.path.exists(fname):
            artifacts_ok = False
            missing_files.append(fname)

    if not artifacts_ok:
        logger.info(
            "[rebuild] artifacts missing or invalid: building from %s... (missing: %s)",
            kb_path,
            ", ".join(missing_files),
        )
        if kb_exists:
            try:
                build(kb_path, retries=retries)
            except Exception as exc:
                log_and_raise(
                    IndexLoadError,
                    f"Failed to build index: {str(exc)}",
                    f"provide one of: {', '.join(candidates)}",
                )
        else:
            log_and_raise(
                IndexLoadError,
                f"{kb_path} not found",
                f"provide a valid knowledge base file to build the index (looked for: {', '.join(candidates)})",
            )

    result = load_index(kb_path)
    if isinstance(result, dict) and (result.get("meta") or {}).get("_stale"):
        if config.AUTO_REBUILD_ON_STALE:
            logger.info("[rebuild] index stale; rebuilding because AUTO_REBUILD_ON_STALE=1")
            if kb_exists:
                try:
                    build(kb_path, retries=retries)
                    result = load_index(kb_path)
                except Exception as exc:
                    log_and_raise(
                        IndexLoadError,
                        f"Failed to rebuild stale index: {str(exc)}",
                        f"check {kb_path} or provide one of: {', '.join(candidates)}",
                    )
            else:
                log_and_raise(
                    IndexLoadError,
                    f"{kb_path} not found while rebuilding stale index",
                    f"provide a valid knowledge base file to build the index (looked for: {', '.join(candidates)})",
                )
        else:
            logger.info("[rebuild] index stale; set AUTO_REBUILD_ON_STALE=1 to rebuild automatically")
    if result is None:
        logger.info("[rebuild] artifact validation failed: rebuilding...")
        if kb_exists:
            try:
                build(kb_path, retries=retries)
                result = load_index()
            except Exception as exc:
                log_and_raise(
                    IndexLoadError,
                    f"Failed to rebuild index: {str(exc)}",
                    f"check {kb_path} or provide one of: {', '.join(candidates)}",
                )
        else:
            log_and_raise(
                IndexLoadError,
                f"{kb_path} not found after validation failure",
                f"provide a valid knowledge base file to build the index (looked for: {', '.join(candidates)})",
            )

    if result is None:
        log_and_raise(
            IndexLoadError,
            "Failed to load artifacts after rebuild",
            "the index may be corrupted; try deleting index files and rebuilding",
        )

    if isinstance(result, dict):
        chunks = result["chunks"]
        vecs_n = result["vecs_n"]
        bm = result["bm"]
        hnsw = result.get("hnsw")
    elif isinstance(result, tuple):
        chunks, vecs_n, bm, hnsw = result
    else:
        log_and_raise(
            TypeError,
            f"load_index() must return dict or tuple, got {type(result)}",
            "contact support - this indicates a system error",
        )

    return chunks, vecs_n, bm, hnsw


def chat_repl(
    top_k=None,
    pack_top=None,
    threshold=None,
    use_rerank=False,
    debug=False,
    seed=None,
    num_ctx=None,
    num_predict=None,
    retries=None,
    use_json=False,
):
    """Stateless REPL loop used by the modern CLI."""
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
    if pack_top is None:
        pack_top = config.DEFAULT_PACK_TOP
    if threshold is None:
        threshold = config.DEFAULT_THRESHOLD
    if seed is None:
        seed = config.DEFAULT_SEED
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX
    if num_predict is None:
        num_predict = config.DEFAULT_NUM_PREDICT
    if retries is None:
        retries = config.DEFAULT_RETRIES

    _log_config_summary(
        use_rerank=use_rerank,
        pack_top=pack_top,
        seed=seed,
        threshold=threshold,
        top_k=top_k,
        num_ctx=num_ctx,
        num_predict=num_predict,
        retries=retries,
    )

    chunks, vecs_n, bm, hnsw = ensure_index_ready(retries=retries)

    query_cache = get_query_cache()
    query_cache.load()
    logger.info("Query cache loaded: %d entries", len(query_cache.cache))

    faq_cache = None
    if config.FAQ_CACHE_ENABLED and os.path.exists(config.FAQ_CACHE_PATH):
        try:
            faq_cache = get_precomputed_cache(config.FAQ_CACHE_PATH)
            if faq_cache.is_stale():
                logger.warning("FAQ cache signature mismatch; ignoring stale cache at %s", config.FAQ_CACHE_PATH)
                faq_cache = None
            else:
                logger.info("FAQ cache loaded: %d precomputed answers", faq_cache.size())
        except Exception as exc:
            logger.warning("Failed to load FAQ cache: %s", exc)

    warmup_on_startup()

    print("-" * 60)
    print("Clockify Support CLI - Type ':exit' to quit, ':debug' to toggle diagnostics")
    print("-" * 60)

    debug_enabled = debug

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not question:
            continue

        if question == ":exit":
            break
        if question == ":debug":
            debug_enabled = not debug_enabled
            print(f"Debug mode: {'ON' if debug_enabled else 'OFF'}")
            continue

        faq_result = None
        if faq_cache:
            faq_result = faq_cache.get(question, fuzzy=True)

        if faq_result:
            result = {
                "answer": faq_result["answer"],
                "refused": False,
                "confidence": faq_result.get("confidence"),
                "selected_chunks": faq_result.get("packed_chunks", []),
                "metadata": {
                    "used_tokens": 0,
                    "cache_type": "faq_precomputed",
                },
                "routing": {"action": "cache"},
            }
            if debug_enabled:
                print("[DEBUG] FAQ cache hit (precomputed answer)")
        else:
            result = answer_once(
                question,
                chunks,
                vecs_n,
                bm,
                top_k=top_k,
                pack_top=pack_top,
                threshold=threshold,
                use_rerank=use_rerank,
                hnsw=hnsw,
                seed=seed,
                num_ctx=num_ctx,
                num_predict=num_predict,
                retries=retries,
            )

        answer = result["answer"]
        metadata = result.get("metadata", {})
        selected_chunks = result.get("selected_chunks", [])

        if use_json:
            used_tokens = metadata.get("used_tokens")
            if used_tokens is None:
                used_tokens = len(selected_chunks)
            output = {
                "answer": answer,
                "citations": selected_chunks,
                "used_tokens": used_tokens,
                "topk": top_k,
                "packed": len(selected_chunks),
                "confidence": result.get("confidence"),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            intent = result.get("intent") or metadata.get("intent") or "unknown"
            role = result.get("user_role_inferred") or metadata.get("user_role_inferred") or "unknown"
            security = result.get("security_sensitivity") or metadata.get("security_sensitivity") or "medium"
            needs_human = bool(result.get("needs_human_escalation") or metadata.get("needs_human_escalation"))
            print(f"\nIntent: {intent} | Role: {role} | Security: {security} | Needs escalation: {needs_human}")
            print(f"\n{answer}")
            urls = _extract_source_urls(chunks, result)
            print("\nSources:")
            if urls:
                for url in urls:
                    print(f"- {url}")
            else:
                print("(none)")

        if debug_enabled:
            print(f"\n[DEBUG] Retrieved: {len(selected_chunks)} chunks")
            scores = metadata.get("scores", [])
            if scores:
                print(f"[DEBUG] Scores: {scores[:5]}")

    query_cache.save()
    logger.info("Query cache saved: %d entries", len(query_cache.cache))


def warmup_on_startup():
    """Warm-up embeddings, LLM, and FAISS on startup."""
    warmup_enabled = config.WARMUP_ENABLED
    if not warmup_enabled:
        logger.debug("Warm-up disabled via WARMUP=0")
        return

    logger.info("Warming up...")

    try:
        from .embedding import embed_texts

        embed_texts(["warmup query"], suppress_errors=True)
    except Exception as exc:
        logger.warning("Embedding warmup failed: %s", exc)

    try:
        client = get_llm_client()
        client.generate_text(
            prompt="Hi",
            model=config.RAG_CHAT_MODEL,
            options={"num_predict": 1, "seed": config.DEFAULT_SEED},
            timeout=(config.CHAT_CONNECT_T, config.CHAT_READ_T),
        )
    except Exception as exc:
        logger.warning("LLM warmup failed: %s", exc)

    if config.USE_ANN == "faiss":
        try:
            from .indexing import load_faiss_index

            _ = load_faiss_index()
            logger.debug("FAISS index preloaded")
        except Exception as exc:
            logger.warning("FAISS warmup failed: %s", exc)

    logger.info("Warmup complete")
