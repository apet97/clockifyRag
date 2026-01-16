"""Answer generation module for Clockify RAG system.

Priority #6: Split monolithic CLI (ROI 7/10)

This module contains the complete answer generation pipeline:
- MMR diversification
- Optional LLM reranking
- Answer generation with confidence scoring
- Citation validation
- Complete answer_once workflow
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .config import (
    DEFAULT_TOP_K,
    DEFAULT_PACK_TOP,
    DEFAULT_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
    MMR_LAMBDA,
    MAX_CHUNKS_PER_ARTICLE,
    MAX_CHUNKS_PER_SECTION,
    REFUSAL_STR,
    STRICT_CITATIONS,
    get_llm_client_mode,
)
from .retrieval import (
    retrieve,
    rerank_with_llm,
    pack_snippets,
    coverage_ok,
    ask_llm,
)
from .exceptions import LLMError, LLMUnavailableError
from .confidence_routing import get_routing_action
from .metrics import MetricNames
from . import metrics as metrics_module
from .utils import sanitize_for_log

logger = logging.getLogger(__name__)


def parse_qwen_json(raw_text: str) -> Dict[str, Any]:
    """Parse Qwen JSON output for the ticket-answering contract with safe defaults."""

    allowed_intents = {
        "feature_howto",
        "troubleshooting",
        "account_security",
        "billing",
        "workspace_admin",
        "workspace_member",
        "data_privacy",
        "screenshots_troubleshooting",
        "other",
    }
    allowed_roles = {"admin", "manager", "regular_member", "external_client", "unknown"}
    allowed_security = {"low", "medium", "high"}

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
        elif len(lines) >= 2:
            cleaned = "\n".join(lines[1:]).replace("```", "").strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    def _get_str(key: str, default: str = "") -> str:
        val = data.get(key)
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, (int, float, bool)):
            return str(val)
        return default

    def _coerce_list(key: str) -> List[str]:
        src = data.get(key, [])
        if not isinstance(src, list):
            return []
        out: List[str] = []
        for item in src:
            if isinstance(item, (str, int, float)):
                text = str(item).strip()
                if text:
                    out.append(text)
        return out

    intent = _get_str("intent", "other").lower()
    if intent not in allowed_intents:
        intent = "other"

    user_role = _get_str("user_role_inferred", "unknown").lower()
    if user_role not in allowed_roles:
        user_role = "unknown"

    security = _get_str("security_sensitivity", "medium").lower()
    if security not in allowed_security:
        security = "medium"

    short_intent = _get_str("short_intent_summary", "")
    answer_style = _get_str("answer_style", "ticket_reply") or "ticket_reply"
    needs_human = data.get("needs_human_escalation")
    needs_human = bool(needs_human) if isinstance(needs_human, bool) else False

    sources_used = _coerce_list("sources_used")
    if not sources_used and "sources" in data:
        sources_used = _coerce_list("sources")

    reasoning_val = data.get("reasoning")
    reasoning = _get_str("reasoning", "") if isinstance(reasoning_val, str) else None

    raw_conf = data.get("confidence")
    confidence = None
    if raw_conf is not None:
        try:
            # Handle string, int, float - convert to float first, then validate range
            conf_float = float(raw_conf)
            if 0 <= conf_float <= 100:
                # Round to nearest int, clamping to valid range
                confidence = max(0, min(100, round(conf_float)))
        except (TypeError, ValueError):
            # Silently ignore unparseable confidence values
            pass

    answer_val = _get_str("answer", cleaned)

    return {
        "answer": answer_val,
        "intent": intent,
        "user_role_inferred": user_role,
        "security_sensitivity": security,
        "answer_style": answer_style,
        "short_intent_summary": short_intent,
        "sources_used": sources_used,
        "needs_human_escalation": needs_human,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def apply_mmr_diversification(
    selected: List[int], scores: Dict[str, Any], vecs_n: np.ndarray, pack_top: int
) -> List[int]:
    """Apply Maximal Marginal Relevance diversification to selected chunks.

    Args:
        selected: List of selected chunk indices
        scores: Dict with "dense" scores
        vecs_n: Normalized embedding vectors
        pack_top: Maximum number of chunks to select

    Returns:
        List of diversified chunk indices
    """
    mmr_selected = []
    cand = list(selected)

    # Always include the top dense score first for better recall
    if cand:
        top_dense_idx = max(cand, key=lambda j: scores["dense"][j])
        mmr_selected.append(top_dense_idx)
        cand.remove(top_dense_idx)

    # Then diversify the rest using vectorized MMR
    if cand and len(mmr_selected) < pack_top:
        # Convert to numpy arrays for vectorized operations
        cand_array = np.array(cand, dtype=np.int32)
        relevance_scores = np.array([scores["dense"][j] for j in cand], dtype=np.float32)

        # Get embedding vectors for candidates
        cand_vecs = vecs_n[cand_array]  # [num_candidates, emb_dim]

        # Iteratively select using MMR
        remaining_mask = np.ones(len(cand_array), dtype=bool)

        while np.any(remaining_mask) and len(mmr_selected) < pack_top:
            # Compute MMR scores for remaining candidates
            mmr_scores = MMR_LAMBDA * relevance_scores.copy()

            if len(mmr_selected) > 0:  # Only apply diversity when we have prior selections
                # Get vectors of all already-selected items
                selected_vecs = vecs_n[mmr_selected]  # [num_selected, emb_dim]

                # Compute similarity matrix: [num_candidates, num_selected]
                similarity_matrix = cand_vecs @ selected_vecs.T

                # Get max similarity for each candidate
                max_similarities = similarity_matrix.max(axis=1)

                # Update MMR scores with diversity penalty
                mmr_scores -= (1 - MMR_LAMBDA) * max_similarities

            # Mask out already-selected candidates
            mmr_scores[~remaining_mask] = -np.inf

            # Select candidate with highest MMR score
            best_idx = mmr_scores.argmax()
            selected_chunk_idx = cand_array[best_idx]

            mmr_selected.append(int(selected_chunk_idx))
            remaining_mask[best_idx] = False

    return mmr_selected


def _chunk_article_key(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata", {}) or {}
    for key in ("url", "source_url", "doc_url"):
        val = chunk.get(key) or meta.get(key)
        if val:
            return str(val)
    if chunk.get("article_id"):
        return f"article:{chunk['article_id']}"
    if meta.get("article_id"):
        return f"article:{meta['article_id']}"
    return str(chunk.get("doc_name") or chunk.get("id"))


def _chunk_section_key(chunk: Dict[str, Any], article_key: str) -> str:
    section = chunk.get("section") or chunk.get("subsection") or (chunk.get("metadata") or {}).get("section_type") or ""
    return f"{article_key}::{section}"


def apply_diversity_limits(selected: List[int], chunks: List[Dict[str, Any]]) -> List[int]:
    """Limit over-representation from the same article/section."""

    max_per_article = MAX_CHUNKS_PER_ARTICLE
    max_per_section = MAX_CHUNKS_PER_SECTION
    if max_per_article <= 0 and max_per_section <= 0:
        return selected

    filtered: List[int] = []
    article_counts: Dict[str, int] = {}
    section_counts: Dict[str, int] = {}

    for idx in selected:
        chunk = chunks[idx]
        article_key = _chunk_article_key(chunk)
        section_key = _chunk_section_key(chunk, article_key)

        if max_per_article > 0 and article_counts.get(article_key, 0) >= max_per_article:
            continue
        if max_per_section > 0 and section_counts.get(section_key, 0) >= max_per_section:
            continue

        filtered.append(idx)
        article_counts[article_key] = article_counts.get(article_key, 0) + 1
        section_counts[section_key] = section_counts.get(section_key, 0) + 1

    return filtered or selected


def apply_reranking(
    question: str,
    chunks: List[Dict],
    mmr_selected: List[int],
    scores: Dict[str, Any],
    use_rerank: bool,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
) -> Tuple[List[int], Dict, bool, str, float]:
    """Apply optional LLM reranking to MMR-selected chunks.

    Args:
        question: User question
        chunks: All chunks
        mmr_selected: List of MMR-selected chunk indices
        scores: Dict with relevance scores
        use_rerank: Whether to apply reranking
        seed, num_ctx, num_predict, retries: LLM parameters

    Returns:
        Tuple of (reranked_chunks, rerank_scores, rerank_applied, rerank_reason, timing)
    """
    rerank_scores = {}
    rerank_applied = False
    rerank_reason = "disabled"
    timing = 0.0

    if use_rerank:
        logger.debug(json.dumps({"event": "rerank_start", "candidates": len(mmr_selected)}))
        t0 = time.time()
        mmr_selected, rerank_scores, rerank_applied, rerank_reason = rerank_with_llm(
            question, chunks, mmr_selected, scores, seed=seed, num_ctx=num_ctx, num_predict=num_predict, retries=retries
        )
        timing = time.time() - t0
        logger.debug(json.dumps({"event": "rerank_done", "selected": len(mmr_selected), "scored": len(rerank_scores)}))

        # Add greppable rerank fallback log
        if not rerank_applied:
            logger.debug("info: rerank=fallback reason=%s", rerank_reason)

    return mmr_selected, rerank_scores, rerank_applied, rerank_reason, timing


def build_citation_details(chunks: List[Dict[str, Any]], packed_ids: List[Any]) -> List[Dict[str, Any]]:
    """Build rich citation objects from packed chunk IDs."""

    if not packed_ids:
        return []

    chunk_by_id = {str(c.get("id")): c for c in chunks}
    details: List[Dict[str, Any]] = []

    for cid in packed_ids:
        chunk = chunk_by_id.get(str(cid))
        if not chunk:
            continue
        meta = chunk.get("metadata") or {}
        details.append(
            {
                "id": str(chunk.get("id")),
                "title": chunk.get("title") or chunk.get("doc_name") or "",
                "url": chunk.get("url") or meta.get("source_url") or meta.get("url") or "",
                "section": chunk.get("section") or "",
                "breadcrumb": meta.get("breadcrumb") or "",
                "article_id": str(chunk.get("article_id") or meta.get("article_id") or ""),
            }
        )

    return details


def extract_citations(text: str) -> List[str]:
    """Extract citation IDs from answer text.

    Supports formats:
    - Single: [id_123], [123], [abc123-def]
    - Comma-separated: [id_a, id_b], [123, 456]
    - Mixed: [id_123, 456, abc-def]
    """
    import re

    # Match brackets containing citation IDs (single or comma-separated)
    # First, find all bracketed content: [...]
    bracket_pattern = r"\[([^\]]+)\]"
    bracket_matches = re.findall(bracket_pattern, text)

    citations = []
    for match in bracket_matches:
        # Split by comma and extract individual IDs
        # Match alphanumeric IDs with underscores and hyphens
        id_pattern = r"([a-zA-Z0-9_-]+)"
        ids = re.findall(id_pattern, match)
        citations.extend([id.strip() for id in ids if id.strip()])

    return citations


def validate_citations(answer: str, valid_chunk_ids: List) -> Tuple[bool, List[str], List[str]]:
    """Validate that citations in answer reference valid chunk IDs.

    Args:
        answer: Answer text with citations
        valid_chunk_ids: List of valid chunk IDs from context

    Returns:
        Tuple of (is_valid, valid_citations, invalid_citations)
    """
    extracted = extract_citations(answer)
    # Normalize to strings for comparison
    valid_set = set(str(cid) for cid in valid_chunk_ids)

    valid_citations = [cid for cid in extracted if cid in valid_set]
    invalid_citations = [cid for cid in extracted if cid not in valid_set]

    is_valid = len(invalid_citations) == 0
    return is_valid, valid_citations, invalid_citations


def generate_llm_answer(
    question: str,
    context_block: str,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
    packed_ids: Optional[List] = None,
    all_chunks: Optional[List[Dict]] = None,
    selected_indices: Optional[List[int]] = None,
    scores_dict: Optional[Dict[str, Any]] = None,
    article_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, float, Optional[int], Optional[str], Optional[List[str]], Dict[str, Any]]:
    """Generate answer from LLM with production-grade Qwen prompts.

    BREAKING CHANGE (v6.0): Qwen now returns structured JSON with confidence, reasoning, and sources.
    When using new prompts (packed_chunks provided), expects JSON output from LLM.
    Legacy prompts (no chunks) still return plain text.

    Args:
        question: User question
        context_block: Legacy packed context string (for backward compatibility)
        seed, num_ctx, num_predict, retries: LLM parameters
        packed_ids: List of chunk IDs included in context (for citation validation)
        all_chunks: Full chunk list (for extracting packed chunks)
        selected_indices: Selected chunk indices (for confidence computation)
        scores_dict: Score arrays from retrieval (for confidence computation)

    Returns:
        Tuple of (answer_text, timing, confidence, reasoning, sources_used, structured_meta)
        - answer_text: The LLM's answer (customer-ready Markdown)
        - timing: Time taken for LLM call
        - confidence: 0-100 score from LLM (if provided) or computed from retrieval scores
        - reasoning: Brief explanation from LLM (if provided)
        - sources_used: List of URLs/IDs the LLM says it used
        - structured_meta: Dict with intent, user_role_inferred, security_sensitivity, short_intent_summary,
          needs_human_escalation, answer_style
    """
    from .retrieval import compute_confidence_from_scores

    t0 = time.time()

    # Extract packed chunks if all data is provided (new code path)
    packed_chunks = None
    structured_prompt = False
    if all_chunks is not None and packed_ids is not None:
        # Build list of chunk dicts for the packed IDs
        chunk_id_to_chunk = {c["id"]: c for c in all_chunks}
        packed_chunks = [chunk_id_to_chunk[cid] for cid in packed_ids if cid in chunk_id_to_chunk]
        structured_prompt = True

    def _normalize_url(url: str) -> str:
        base = str(url).strip()
        if not base:
            return ""
        base = base.split("#")[0].rstrip("/")
        return base.lower()

    # Build map of valid URLs present in the provided context for source verification
    valid_url_map: dict[str, str] = {}
    if article_blocks:
        for blk in article_blocks:
            url_val = blk.get("url")
            norm = _normalize_url(url_val) if url_val else ""
            if norm:
                valid_url_map.setdefault(norm, str(url_val).strip())
    elif packed_chunks:
        for ch in packed_chunks:
            url_val = ch.get("url") or ch.get("source_url") or (ch.get("metadata") or {}).get("source_url")
            norm = _normalize_url(url_val) if url_val else ""
            if norm:
                valid_url_map.setdefault(norm, str(url_val).strip())

    # Call LLM with new or legacy prompts
    raw_response = ask_llm(
        question,
        context_block,
        seed=seed,
        num_ctx=num_ctx,
        num_predict=num_predict,
        retries=retries,
        chunks=article_blocks or packed_chunks,  # None for legacy, list of dicts for new
    ).strip()
    timing = time.time() - t0

    # Parse response based on prompt type
    answer = raw_response
    confidence = None
    reasoning = None
    sources_used: Optional[List[str]] = None
    intent = "other"
    user_role_inferred = "unknown"
    security_sensitivity = "medium"
    answer_style = "ticket_reply"
    short_intent_summary = ""
    needs_human_escalation = False
    client_mode = (get_llm_client_mode("") or "").lower()
    try:
        if structured_prompt:
            parsed = parse_qwen_json(raw_response)
            answer = parsed["answer"]
            intent = parsed.get("intent", intent)
            user_role_inferred = parsed.get("user_role_inferred", user_role_inferred)
            security_sensitivity = parsed.get("security_sensitivity", security_sensitivity)
            answer_style = parsed.get("answer_style", answer_style)
            short_intent_summary = parsed.get("short_intent_summary", short_intent_summary)
            needs_human_escalation = parsed.get("needs_human_escalation", needs_human_escalation)
            confidence = parsed.get("confidence", confidence)
            reasoning = parsed.get("reasoning", reasoning)
            parsed_sources = parsed.get("sources_used", sources_used)
            sources_used = parsed_sources if parsed_sources else None
            if sources_used and valid_url_map:
                verified_sources: list[str] = []
                for src in sources_used:
                    norm_src = _normalize_url(src)
                    match = valid_url_map.get(norm_src)
                    if not match:
                        for candidate_norm, canonical in valid_url_map.items():
                            if norm_src and (norm_src in candidate_norm or candidate_norm in norm_src):
                                match = canonical
                                break
                    if match:
                        verified_sources.append(match)
                sources_used = sorted(set(verified_sources)) if verified_sources else None
            logger.debug(f"Parsed Qwen JSON: confidence={confidence}, sources={len(sources_used or [])}")
        else:
            # Legacy prompt: treat output as plain text without parsing
            answer = raw_response
    except (json.JSONDecodeError, ValueError) as e:
        # JSON parsing failed - log and fall back to raw text
        logger.warning(f"Failed to parse Qwen JSON output: {e}. Using raw text as fallback.")
        logger.debug(f"Raw output (first 500 chars): {raw_response[:500]}")
        answer = raw_response
        # Compute fallback confidence from retrieval scores
        if scores_dict is not None and selected_indices is not None:
            confidence = compute_confidence_from_scores(scores_dict, selected_indices)
        # On parse failure, treat as legacy/plain output: skip citation synthesis entirely
        structured_prompt = False
        sources_used = None
        # Do not synthesize citations in fallback; return raw text as-is

    if confidence is None and scores_dict is not None and selected_indices is not None:
        confidence = compute_confidence_from_scores(scores_dict, selected_indices)

    # Citation validation
    if packed_ids:
        if client_mode == "mock":
            if confidence is None and structured_prompt:
                confidence = 100
            return (
                answer,
                timing,
                confidence,
                reasoning,
                sources_used,
                {
                    "intent": intent,
                    "user_role_inferred": user_role_inferred,
                    "security_sensitivity": security_sensitivity,
                    "short_intent_summary": short_intent_summary,
                    "needs_human_escalation": needs_human_escalation,
                    "answer_style": answer_style,
                },
            )

        # Skip noisy inline citation validation when sources_used are present
        has_citations = bool(extract_citations(answer))
        if sources_used:
            # Treat provided sources as canonical; avoid invalid-citation noise
            has_citations = True
        elif not has_citations and STRICT_CITATIONS and answer != REFUSAL_STR:
            logger.warning("Answer lacks citations in strict mode, refusing answer")
            answer = REFUSAL_STR
            confidence = None

    if client_mode == "mock" and structured_prompt and confidence is None:
        confidence = 100

    return (
        answer,
        timing,
        confidence,
        reasoning,
        sources_used,
        {
            "intent": intent,
            "user_role_inferred": user_role_inferred,
            "security_sensitivity": security_sensitivity,
            "short_intent_summary": short_intent_summary,
            "needs_human_escalation": needs_human_escalation,
            "answer_style": answer_style,
        },
    )


def answer_once(
    question: str,
    chunks: List[Dict],
    vecs_n: np.ndarray,
    bm: Dict,
    hnsw=None,
    top_k: int = DEFAULT_TOP_K,
    pack_top: int = DEFAULT_PACK_TOP,
    threshold: float = DEFAULT_THRESHOLD,
    use_rerank: bool = False,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
    faiss_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete answer generation pipeline.

    Args:
        question: User question
        chunks: List of all chunks
        vecs_n: Normalized embedding vectors
        bm: BM25 index
        hnsw: Optional HNSW index
        top_k: Number of candidates to retrieve
        pack_top: Number of chunks to pack in context
        threshold: Minimum similarity threshold
        use_rerank: Whether to apply LLM reranking
        seed, num_ctx, num_predict, retries: LLM parameters
        faiss_index_path: Path to FAISS index file

    Returns:
        Dict with answer and metadata
    """
    t_start = time.time()
    metrics = metrics_module.get_metrics()

    def _normalize_chunk_ids(seq: Optional[List]) -> List:
        if not seq:
            return []
        normalized: List = []
        for item in seq:
            if isinstance(item, np.generic):
                normalized.append(item.item())
            else:
                normalized.append(item)
        return normalized

    metrics.increment_counter(MetricNames.QUERIES_TOTAL)
    question_preview = sanitize_for_log(question, max_length=200)
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]
    logger.info(
        json.dumps(
            {
                "event": "rag.query.start",
                "question_hash": question_hash,
                "question_preview": question_preview,
                "top_k": top_k,
                "pack_top": pack_top,
            }
        )
    )

    # Retrieve
    t0 = time.time()
    selected, scores = retrieve(
        question, chunks, vecs_n, bm, top_k=top_k, hnsw=hnsw, retries=retries, faiss_index_path=faiss_index_path
    )
    retrieve_time = time.time() - t0

    # Check coverage
    if not coverage_ok(selected, scores["dense"], threshold):
        metrics.increment_counter(MetricNames.ERRORS_TOTAL, labels={"type": "coverage"})
        metrics.increment_counter(MetricNames.REFUSALS_TOTAL, labels={"reason": "coverage"})
        total_time = time.time() - t_start
        metrics.observe_histogram(MetricNames.QUERY_LATENCY, total_time * 1000)
        metrics.observe_histogram(MetricNames.RETRIEVAL_LATENCY, retrieve_time * 1000)
        logger.warning(
            json.dumps(
                {
                    "event": "rag.query.coverage_failure",
                    "question_hash": question_hash,
                    "selected": len(selected),
                    "threshold": threshold,
                }
            )
        )
        return {
            "answer": REFUSAL_STR,
            "refused": True,
            "confidence": None,
            "selected_chunks": [],
            "packed_chunks": [],
            "context_block": "",
            "timing": {
                "total_ms": (time.time() - t_start) * 1000,
                "retrieve_ms": retrieve_time * 1000,
                "mmr_ms": 0,
                "rerank_ms": 0,
                "llm_ms": 0,
            },
            "metadata": {"retrieval_count": len(selected), "coverage_check": "failed"},
            "routing": get_routing_action(None, refused=True, critical=False),
        }

    # Apply MMR diversification
    t0 = time.time()
    mmr_selected = apply_mmr_diversification(selected, scores, vecs_n, pack_top)
    mmr_time = time.time() - t0

    # Optional reranking
    mmr_selected, rerank_scores, rerank_applied, rerank_reason, rerank_time = apply_reranking(
        question,
        chunks,
        mmr_selected,
        scores,
        use_rerank,
        seed=seed,
        num_ctx=num_ctx,
        num_predict=num_predict,
        retries=retries,
    )

    mmr_selected = apply_diversity_limits(mmr_selected, chunks)

    # Pack snippets grouped by article
    context_block, packed_ids, used_tokens, article_blocks = pack_snippets(
        chunks, mmr_selected, pack_top=pack_top, num_ctx=num_ctx
    )
    citation_details = build_citation_details(chunks, packed_ids)

    def _llm_failure(reason: str, error: Exception) -> Dict[str, Any]:
        total_time = time.time() - t_start
        metrics.increment_counter(MetricNames.ERRORS_TOTAL, labels={"type": reason})
        metrics.increment_counter(MetricNames.REFUSALS_TOTAL, labels={"reason": reason})
        logger.error(
            json.dumps(
                {
                    "event": "rag.query.failure",
                    "reason": reason,
                    "question_hash": question_hash,
                    "message": str(error),
                }
            )
        )
        metrics.observe_histogram(MetricNames.QUERY_LATENCY, total_time * 1000)
        metrics.observe_histogram(MetricNames.RETRIEVAL_LATENCY, retrieve_time * 1000)
        return {
            "answer": REFUSAL_STR,
            "refused": True,
            "confidence": None,
            "selected_chunks": _normalize_chunk_ids(selected),
            "packed_chunks": _normalize_chunk_ids(mmr_selected),
            "context_block": context_block,
            "timing": {
                "total_ms": total_time * 1000,
                "retrieve_ms": retrieve_time * 1000,
                "mmr_ms": mmr_time * 1000,
                "rerank_ms": rerank_time * 1000,
                "llm_ms": 0,
            },
            "metadata": {
                "retrieval_count": len(selected),
                "packed_count": len(packed_ids),
                "used_tokens": used_tokens,
                "rerank_applied": rerank_applied,
                "rerank_reason": rerank_reason,
                "llm_error": reason,
                "llm_error_msg": str(error),
                "source_chunk_ids": _normalize_chunk_ids(packed_ids),
            },
            "routing": get_routing_action(None, refused=True, critical=True),
        }

    # Generate answer
    try:
        answer, llm_time, confidence, reasoning, sources_used, structured_meta = generate_llm_answer(
            question,
            context_block,
            seed=seed,
            num_ctx=num_ctx,
            num_predict=num_predict,
            retries=retries,
            packed_ids=packed_ids,
            all_chunks=chunks,
            selected_indices=selected,
            scores_dict=scores,
            article_blocks=article_blocks,
        )
    except LLMUnavailableError as exc:
        logger.error(f"LLM unavailable during answer generation: {exc}")
        return _llm_failure("llm_unavailable", exc)
    except LLMError as exc:
        logger.error(f"LLM error during answer generation: {exc}")
        return _llm_failure("llm_error", exc)

    total_time = time.time() - t_start
    metrics.observe_histogram(MetricNames.QUERY_LATENCY, total_time * 1000)
    metrics.observe_histogram(MetricNames.RETRIEVAL_LATENCY, retrieve_time * 1000)
    metrics.observe_histogram(MetricNames.LLM_LATENCY, llm_time * 1000)

    refused = answer == REFUSAL_STR
    if refused:
        metrics.increment_counter(MetricNames.ERRORS_TOTAL, labels={"type": "refused"})
        metrics.increment_counter(MetricNames.REFUSALS_TOTAL, labels={"reason": "llm"})
    logger.info(
        json.dumps(
            {
                "event": "rag.query.complete",
                "question_hash": question_hash,
                "refused": refused,
                "selected": len(selected),
                "packed": len(packed_ids),
                "confidence": confidence,
                "total_ms": round(total_time * 1000, 2),
                "llm_ms": round(llm_time * 1000, 2),
            }
        )
    )

    # OPTIMIZATION (Analysis Section 9.1 #4): Confidence-based routing
    # Auto-escalate low-confidence queries to human review
    routing = get_routing_action(confidence, refused=refused, critical=False)

    return {
        "answer": answer,
        "refused": refused,
        "confidence": confidence,
        "intent": structured_meta.get("intent"),
        "user_role_inferred": structured_meta.get("user_role_inferred"),
        "security_sensitivity": structured_meta.get("security_sensitivity"),
        "short_intent_summary": structured_meta.get("short_intent_summary"),
        "answer_style": structured_meta.get("answer_style"),
        "needs_human_escalation": structured_meta.get("needs_human_escalation"),
        "sources_used": sources_used,
        "citation_details": citation_details,
        "selected_chunks": _normalize_chunk_ids(selected),
        "packed_chunks": _normalize_chunk_ids(mmr_selected),
        "selected_chunk_ids": _normalize_chunk_ids(packed_ids),
        "context_block": context_block,
        "timing": {
            "total_ms": total_time * 1000,
            "retrieve_ms": retrieve_time * 1000,
            "mmr_ms": mmr_time * 1000,
            "rerank_ms": rerank_time * 1000,
            "llm_ms": llm_time * 1000,
        },
        "metadata": {
            "retrieval_count": len(selected),
            "packed_count": len(packed_ids),
            "used_tokens": used_tokens,
            "rerank_applied": rerank_applied,
            "rerank_reason": rerank_reason,
            "source_chunk_ids": _normalize_chunk_ids(packed_ids),
            "citation_details": citation_details,
            "reasoning": reasoning,  # LLM's explanation (new JSON format)
            "sources_used": sources_used,  # LLM's cited sources (new JSON format)
            **structured_meta,
        },
        "routing": routing,  # Add routing recommendation
    }


def answer_to_json(
    answer: str,
    citations: list,
    used_tokens: int | None,
    topk: int,
    packed: int,
    confidence: int | None = None,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    routing: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, Any]] = None,
    refused: bool = False,
) -> dict:
    """Convert answer and metadata to JSON structure.

    Args:
        answer: Generated answer text.
        citations: Sequence of citation identifiers (chunk IDs).
        used_tokens: Actual token budget consumed when packing context.
        topk: Retrieval depth requested.
        packed: Maximum number of snippets packed.
        confidence: LLM confidence score (0-100), if available.
    """
    from .config import KPI, EMB_BACKEND, USE_ANN, ALPHA_HYBRID

    budget_tokens = 0 if used_tokens is None else int(used_tokens)
    metadata_payload = metadata or {}
    routing_payload = routing or {}
    timing_payload = timing or {
        "retrieve_ms": KPI.retrieve_ms,
        "ann_ms": KPI.ann_ms,
        "rerank_ms": KPI.rerank_ms,
        "llm_ms": KPI.ask_ms,
        "total_ms": KPI.retrieve_ms + KPI.rerank_ms + KPI.ask_ms,
    }

    result: Dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "refused": refused,
        "metadata": metadata_payload,
        "routing": routing_payload,
        "timing": timing_payload,
        "debug": {
            "meta": {
                "used_tokens": budget_tokens,
                "topk": topk,
                "packed": packed,
                "emb_backend": EMB_BACKEND,
                "ann": USE_ANN,
                "alpha": ALPHA_HYBRID,
            },
            "timing": timing_payload,
        },
    }

    # Drop confidence key if None to keep payload tidy
    if confidence is None:
        result.pop("confidence", None)

    return result


__all__ = [
    "parse_qwen_json",
    "apply_mmr_diversification",
    "apply_reranking",
    "extract_citations",
    "validate_citations",
    "generate_llm_answer",
    "answer_once",
    "answer_to_json",
]
