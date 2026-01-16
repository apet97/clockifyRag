"""Async support for non-blocking Ollama API calls.

OPTIMIZATION (Analysis Section 9.1 #1): Async LLM calls for 2-4x concurrent throughput.
This module provides async versions of HTTP and LLM functions while maintaining
backward compatibility with the synchronous API.

Usage:
    # Async mode (requires asyncio event loop)
    import asyncio
    from clockify_rag.async_support import async_answer_once

    result = asyncio.run(async_answer_once(question, chunks, vecs_n, bm))

    # Synchronous mode (default, no changes needed)
    from clockify_rag.answer import answer_once

    result = answer_once(question, chunks, vecs_n, bm)
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .config import (
    RAG_CHAT_MODEL,
    RAG_EMBED_MODEL,
    EMB_CONNECT_T,
    EMB_READ_T,
    CHAT_CONNECT_T,
    CHAT_READ_T,
    DEFAULT_TOP_K,
    DEFAULT_PACK_TOP,
    DEFAULT_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
    REFUSAL_STR,
)
from .exceptions import LLMError, LLMUnavailableError
from .confidence_routing import get_routing_action
from .api_client import get_llm_client, ChatMessage, ChatCompletionOptions

logger = logging.getLogger(__name__)


async def async_embed_query(text: str, retries: int = 0) -> np.ndarray:
    """Async version of embed_query.

    Args:
        text: Text to embed
        retries: Number of retries

    Returns:
        Normalized embedding vector (numpy array)
    """
    client = get_llm_client()
    try:
        vector: List[float] = await asyncio.to_thread(
            client.create_embedding,
            text,
            RAG_EMBED_MODEL,
            (EMB_CONNECT_T, EMB_READ_T),
            retries,
        )
        embedding = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise LLMError(f"Embedding failed: {e}") from e


async def async_ask_llm(
    question: str,
    context_block: str,
    seed: int = DEFAULT_SEED,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    retries: int = DEFAULT_RETRIES,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Async version of ask_llm.

    Args:
        question: User question
        context_block: Context snippets
        seed, num_ctx, num_predict, retries: LLM parameters
        chunks: Optional list of chunk dicts for new Qwen prompts

    Returns:
        LLM response text
    """
    from .retrieval import get_system_prompt, USER_WRAPPER
    from .prompts import QWEN_SYSTEM_PROMPT, build_rag_user_prompt

    # Use new Qwen prompts if chunks provided, otherwise legacy prompts
    if chunks is not None:
        system_prompt = QWEN_SYSTEM_PROMPT
        user_prompt = build_rag_user_prompt(question, chunks)
    else:
        # Legacy format for backward compatibility
        system_prompt = get_system_prompt()
        user_prompt = USER_WRAPPER.format(snips=context_block, q=question)

    client = get_llm_client()
    messages: List[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options: ChatCompletionOptions = {
        "seed": seed,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }

    try:
        result = await asyncio.to_thread(
            client.chat_completion,
            messages,
            RAG_CHAT_MODEL,
            options,
            False,
            (CHAT_CONNECT_T, CHAT_READ_T),
            retries,
        )
        return result.get("message", {}).get("content", "")
    except LLMUnavailableError:
        raise
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise LLMError(f"LLM generation failed: {e}") from e


async def async_generate_llm_answer(
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
    """Async version of generate_llm_answer with confidence scoring and citation validation.

    Args:
        question: User question
        context_block: Packed context snippets
        seed, num_ctx, num_predict, retries: LLM parameters
        packed_ids: List of chunk IDs included in context (for citation validation)
        all_chunks: All chunks (for extracting packed chunks)
        selected_indices: Indices of selected chunks (for confidence computation)
        scores_dict: Retrieval scores (for confidence computation)

    Returns:
        Tuple of (answer_text, timing, confidence, reasoning, sources_used, structured_meta)
    """
    from .answer import parse_qwen_json

    # Extract packed chunks if all data is provided (new code path)
    packed_chunks = None
    if all_chunks is not None and packed_ids is not None:
        chunk_id_to_chunk = {c["id"]: c for c in all_chunks}
        packed_chunks = [chunk_id_to_chunk[cid] for cid in packed_ids if cid in chunk_id_to_chunk]

    t0 = time.time()
    raw_response = (
        await async_ask_llm(
            question,
            context_block,
            seed,
            num_ctx,
            num_predict,
            retries,
            chunks=article_blocks or packed_chunks,  # Pass chunks for new Qwen prompts
        )
    ).strip()
    timing = time.time() - t0

    # Parse JSON response with confidence
    confidence = None
    reasoning: Optional[str] = None
    sources_used: Optional[List[str]] = None
    intent = "other"
    user_role_inferred = "unknown"
    security_sensitivity = "medium"
    short_intent_summary = ""
    needs_human_escalation = False
    answer_style = "ticket_reply"
    answer = raw_response  # Default to raw response if parsing fails

    try:
        parsed = parse_qwen_json(raw_response)
        answer = parsed["answer"]
        intent = parsed.get("intent", intent)
        user_role_inferred = parsed.get("user_role_inferred", user_role_inferred)
        security_sensitivity = parsed.get("security_sensitivity", security_sensitivity)
        short_intent_summary = parsed.get("short_intent_summary", short_intent_summary)
        needs_human_escalation = parsed.get("needs_human_escalation", needs_human_escalation)
        answer_style = parsed.get("answer_style", answer_style)
        confidence = parsed.get("confidence", confidence)
        reasoning = parsed.get("reasoning")
        sources_used = parsed.get("sources_used")
    except (json.JSONDecodeError, ValueError):
        answer = raw_response

    # Compute confidence from scores (preferred method, overrides LLM JSON confidence)
    if scores_dict is not None and selected_indices is not None:
        from .retrieval import compute_confidence_from_scores

        confidence = compute_confidence_from_scores(scores_dict, selected_indices)

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


async def async_answer_once(
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
    """Async version of answer_once for non-blocking LLM calls.

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
        Dict with answer and metadata (same format as answer_once)
    """
    from .retrieval import retrieve, coverage_ok, pack_snippets
    from .answer import apply_mmr_diversification, apply_diversity_limits, build_citation_details

    t_start = time.time()

    # Retrieve (synchronous for now, retrieval is fast)
    t0 = time.time()
    selected, scores = retrieve(
        question, chunks, vecs_n, bm, top_k=top_k, hnsw=hnsw, retries=retries, faiss_index_path=faiss_index_path
    )
    retrieve_time = time.time() - t0

    # Check coverage
    if not coverage_ok(selected, scores["dense"], threshold):
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

    # Reranking (synchronous for now)
    rerank_time = 0.0
    rerank_applied = False
    rerank_reason = "disabled"
    if use_rerank:
        from .answer import apply_reranking

        t0 = time.time()
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

    # Pack snippets
    context_block, packed_ids, used_tokens, article_blocks = pack_snippets(
        chunks, mmr_selected, pack_top=pack_top, num_ctx=num_ctx
    )
    citation_details = build_citation_details(chunks, packed_ids)

    def _failure(reason: str, error: Exception) -> Dict[str, Any]:
        total_time = time.time() - t_start
        return {
            "answer": REFUSAL_STR,
            "refused": True,
            "confidence": None,
            "selected_chunks": selected,
            "packed_chunks": mmr_selected,
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
            },
            "routing": get_routing_action(None, refused=True, critical=True),
        }

    # Generate answer (async)
    try:
        answer, llm_time, confidence, reasoning, sources_used, structured_meta = await async_generate_llm_answer(
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
        logger.error(f"LLM unavailable during async answer generation: {exc}")
        return _failure("llm_unavailable", exc)
    except LLMError as exc:
        logger.error(f"LLM error during async answer generation: {exc}")
        return _failure("llm_error", exc)

    total_time = time.time() - t_start

    # Confidence-based routing
    refused = answer == REFUSAL_STR
    routing = get_routing_action(confidence, refused=refused, critical=False)

    return {
        "answer": answer,
        "refused": refused,
        "confidence": confidence,
        "intent": (structured_meta or {}).get("intent"),
        "user_role_inferred": (structured_meta or {}).get("user_role_inferred"),
        "security_sensitivity": (structured_meta or {}).get("security_sensitivity"),
        "short_intent_summary": (structured_meta or {}).get("short_intent_summary"),
        "answer_style": (structured_meta or {}).get("answer_style"),
        "needs_human_escalation": (structured_meta or {}).get("needs_human_escalation"),
        "sources_used": sources_used,
        "citation_details": citation_details,
        "selected_chunks": selected,
        "packed_chunks": mmr_selected,
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
            "citation_details": citation_details,
            "reasoning": reasoning,
            "sources_used": sources_used,
            **(structured_meta or {}),
        },
        "routing": routing,
    }


__all__ = [
    "async_embed_query",
    "async_ask_llm",
    "async_generate_llm_answer",
    "async_answer_once",
]
