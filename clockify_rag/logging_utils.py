"""Helpers for structured query logging across CLI and API surfaces."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .caching import log_query
from .utils import sanitize_for_log

logger = logging.getLogger(__name__)


def _normalize_chunk_dict(chunk: Mapping[str, Any], rank: int) -> Dict[str, Any]:
    """Return a sanitized chunk dict for logging."""

    normalized = {
        "id": chunk.get("id") or chunk.get("chunk_id"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "rank": rank,
    }

    # Carry optional identifiers for easier debugging
    if "url" in chunk:
        normalized["url"] = chunk["url"]
    if "score" in chunk:
        normalized["score"] = chunk["score"]
    if "dense" in chunk:
        normalized["dense"] = chunk["dense"]
    if "bm25" in chunk:
        normalized["bm25"] = chunk["bm25"]
    if "hybrid" in chunk:
        normalized["hybrid"] = chunk["hybrid"]

    return normalized


def build_chunk_log_entries(
    chunks: Sequence[Mapping[str, Any]] | None,
    selected_chunks: Sequence[Any] | None,
    selected_chunk_ids: Sequence[Any] | None = None,
) -> List[Dict[str, Any]]:
    """Convert selected chunk references into structured log entries."""

    if not selected_chunks and not selected_chunk_ids:
        return []

    log_entries: List[Dict[str, Any]] = []
    total_chunks = len(chunks) if chunks is not None else 0
    chunk_id_lookup: Dict[str, Mapping[str, Any]] = {}

    if chunks is not None:
        for chunk in chunks:
            ident = chunk.get("id") or chunk.get("chunk_id")
            if ident is not None:
                chunk_id_lookup[str(ident)] = chunk

    seq_indices = list(selected_chunks or [])
    seq_ids = list(selected_chunk_ids or [])
    max_len = max(len(seq_indices), len(seq_ids))

    for rank in range(max_len):
        chunk_ref = seq_indices[rank] if rank < len(seq_indices) else None
        chunk_id = seq_ids[rank] if rank < len(seq_ids) else None

        if isinstance(chunk_ref, Mapping):
            entry = _normalize_chunk_dict(chunk_ref, rank)
            if chunk_id is not None:
                entry["id"] = chunk_id
            elif entry.get("id") is None and "chunk_id" in chunk_ref:
                entry["id"] = chunk_ref["chunk_id"]
            log_entries.append(entry)
            continue

        chunk_obj: Optional[Mapping[str, Any]] = None
        idx_value: Optional[int] = None

        if chunk_ref is not None:
            try:
                idx_value = int(chunk_ref)
            except (TypeError, ValueError):
                log_entries.append({"id": chunk_ref if chunk_id is None else chunk_id, "rank": rank})
                continue

        if idx_value is not None and 0 <= idx_value < total_chunks and chunks is not None:
            chunk_obj = chunks[idx_value]
        elif chunk_id is not None:
            chunk_obj = chunk_id_lookup.get(str(chunk_id))

        if chunk_obj is not None:
            entry_id = chunk_id
            if entry_id is None:
                entry_id = chunk_obj.get("id") or chunk_obj.get("chunk_id")
            if entry_id is None and idx_value is not None:
                entry_id = idx_value
            entry = {
                "id": entry_id,
                "title": chunk_obj.get("title"),
                "section": chunk_obj.get("section"),
                "rank": rank,
            }
            if "url" in chunk_obj:
                entry["url"] = chunk_obj["url"]
            log_entries.append(entry)
        else:
            fallback_id = chunk_id if chunk_id is not None else idx_value
            log_entries.append({"id": fallback_id, "rank": rank})

    return log_entries


def log_query_event(
    question: str,
    result: Mapping[str, Any],
    chunks: Sequence[Mapping[str, Any]] | None,
    latency_ms: Optional[float],
    *,
    channel: Optional[str] = None,
    disabled: bool = False,
) -> None:
    """Write a structured query log entry when logging is enabled."""

    if disabled:
        return

    if not question or not isinstance(result, Mapping):
        return

    retrieved_chunks = build_chunk_log_entries(
        chunks,
        result.get("selected_chunks"),
        result.get("selected_chunk_ids"),
    )
    timing = result.get("timing") or {}
    computed_latency = latency_ms if latency_ms is not None else timing.get("total_ms")

    metadata: Dict[str, Any] = {
        "confidence": result.get("confidence"),
        "routing": result.get("routing"),
        "result_metadata": result.get("metadata") or {},
        "timing": timing,
    }
    if channel:
        metadata["channel"] = channel

    try:
        safe_question = sanitize_for_log(question, max_length=500)
        log_query(
            safe_question,
            result.get("answer", ""),
            retrieved_chunks,
            float(computed_latency) if computed_latency is not None else 0.0,
            refused=bool(result.get("refused")),
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("Failed to log query event: %s", exc)
