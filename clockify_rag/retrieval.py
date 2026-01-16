"""Retrieval module for Clockify RAG system.

This module contains all retrieval-related functionality:
- Query expansion and embedding
- Hybrid retrieval (BM25 + dense + MMR)
- LLM-based reranking
- Snippet packing with token budget management
- Coverage checking
- Answer generation with LLM
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
import clockify_rag.config as config
from .api_client import ChatCompletionOptions, ChatMessage
from .embedding import embed_query as _embedding_embed_query
from .exceptions import LLMError, ValidationError
from .indexing import bm25_scores, get_faiss_index
from .utils import tokenize  # FIX (Error #17): Import tokenize from utils instead of duplicating
from .intent_classification import classify_intent, get_intent_metadata, adjust_scores_by_intent
from .prompts import QWEN_SYSTEM_PROMPT, build_rag_user_prompt

logger = logging.getLogger(__name__)

# ====== RETRIEVAL PROFILING ======
# FIX (Error #15): Add thread-safe lock for concurrent access to profiling state
_RETRIEVE_PROFILE_LOCK = __import__("threading").RLock()
RETRIEVE_PROFILE_LAST: dict = {}

# ====== RERANK CACHE ======
_RERANK_CACHE_LOCK = threading.Lock()
_RERANK_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _rerank_cache_enabled() -> bool:
    return config.RERANK_CACHE_MAX_ITEMS > 0 and config.RERANK_CACHE_TTL_SEC > 0


def _make_rerank_cache_key(question: str, selected: List[int], chunks) -> str:
    norm_q = " ".join(question.lower().split())
    hasher = hashlib.sha256()
    hasher.update(norm_q.encode("utf-8"))
    hasher.update(b"|")
    for idx in selected:
        cid = str(chunks[idx].get("id"))
        hasher.update(cid.encode("utf-8"))
        hasher.update(b",")
    return hasher.hexdigest()


def _rerank_cache_get(cache_key: str, selected: List[int], chunks) -> Optional[Tuple[List[int], Dict[int, float]]]:
    if not _rerank_cache_enabled():
        return None

    now = time.time()
    with _RERANK_CACHE_LOCK:
        entry = _RERANK_CACHE.get(cache_key)
        if not entry:
            return None
        if entry.get("expires_at", 0) <= now:
            _RERANK_CACHE.pop(cache_key, None)
            return None
        _RERANK_CACHE.move_to_end(cache_key)
        order_ids = entry.get("order_ids") or []
        scores_by_id = entry.get("scores") or {}

    cid_to_idx = {str(chunks[i].get("id")): i for i in selected}
    order = [cid_to_idx[cid] for cid in order_ids if cid in cid_to_idx]
    if not order:
        return None
    scores = {cid_to_idx[cid]: float(score) for cid, score in scores_by_id.items() if cid in cid_to_idx}
    return order, scores


def _rerank_cache_put(cache_key: str, order_ids: List[str], scores_by_id: Dict[str, float]) -> None:
    if not _rerank_cache_enabled():
        return

    with _RERANK_CACHE_LOCK:
        _RERANK_CACHE[cache_key] = {
            "order_ids": order_ids,
            "scores": scores_by_id,
            "expires_at": time.time() + config.RERANK_CACHE_TTL_SEC,
        }
        _RERANK_CACHE.move_to_end(cache_key)
        while len(_RERANK_CACHE) > config.RERANK_CACHE_MAX_ITEMS:
            _RERANK_CACHE.popitem(last=False)


def get_retrieve_profile():
    """Get retrieval profile data in a thread-safe manner.

    FIX (Error #15): Returns a copy to prevent race conditions when reading.

    Returns:
        Dict with profiling data (copy of current state)
    """
    with _RETRIEVE_PROFILE_LOCK:
        return dict(RETRIEVE_PROFILE_LAST)


# ====== PROMPTS ======
_SYSTEM_PROMPT_TEMPLATE = QWEN_SYSTEM_PROMPT


def get_system_prompt() -> str:
    """Return the canonical system prompt for the RAG assistant."""

    return _SYSTEM_PROMPT_TEMPLATE


SYSTEM_PROMPT = None  # Dynamically resolved via __getattr__


USER_WRAPPER = """SNIPPETS:
{snips}

QUESTION:
{q}

Respond with only a JSON object following this schema (no code fences or preamble):
{{
  "answer": "<customer-ready Markdown answer in the user's language>",
  "confidence": <0-100 integer>,
  "reasoning": "<2-4 sentences on how the context supports the answer>",
  "sources_used": ["<CONTEXT_BLOCK IDs used, e.g., \"1\" or \"2\">"]
}}
- If context is missing or insufficient, say so in the answer, keep guidance generic, set confidence <= 20, and return an empty sources_used list.
- Do not invent product features, settings, or APIs not present in the snippets or obvious general knowledge.
- Keep all narrative content inside the JSON 'answer' field.
- Do not wrap the JSON in markdown fences or add extra prose."""

RERANK_PROMPT = """You rank passages for a Clockify support answer. Score each 0.0–1.0 strictly.
Output JSON only: [{{"id":"<chunk_id>","score":0.82}}, ...].

QUESTION:
{q}

PASSAGES:
{passages}"""


# ====== CONFIDENCE COMPUTATION ======
def compute_confidence_from_scores(
    scores_dict: Dict[str, Any],
    selected_indices: List[int],
    threshold: float = 0.25,
) -> int:
    """Compute confidence score (0-100) from retrieval scores.

    This provides a more reliable confidence estimate than asking the LLM to guess.
    Confidence is based on:
    - Top hybrid score (main signal)
    - Score distribution (consistency)
    - Coverage (number of high-quality chunks)

    Args:
        scores_dict: Dictionary with 'hybrid', 'dense', 'bm25' score arrays
        selected_indices: List of selected chunk indices
        threshold: Minimum acceptable similarity threshold

    Returns:
        Integer confidence score from 0-100
    """
    if not selected_indices:
        return 0

    try:
        import numpy as np

        # Extract scores for selected indices
        hybrid_scores = np.asarray(scores_dict.get("hybrid", np.array([])), dtype="float32")
        if hybrid_scores.size == 0:
            return 50  # Fallback if no scores available

        selected_scores = [hybrid_scores[i] for i in selected_indices if i < len(hybrid_scores)]
        if not selected_scores:
            return 50

        # Clip to [0,1] for stability regardless of normalization/z-scores
        clipped = np.clip(np.array(selected_scores, dtype="float32"), 0.0, 1.0)
        top_score = float(np.max(clipped))

        # Map top score: below threshold → up to 40; above threshold → up to 100
        if top_score < threshold:
            base_confidence = int(top_score / max(threshold, 1e-6) * 40)
        else:
            normalized = (top_score - threshold) / max(1e-6, (1.0 - threshold))
            base_confidence = int(40 + normalized * 60)

        # Adjust based on score distribution (consistency boost)
        if len(clipped) >= 2:
            avg_score = float(np.mean(clipped))
            # If average is close to top, boost confidence (consistent results)
            consistency = avg_score / top_score if top_score > 0 else 0
            if consistency > 0.8:  # Very consistent
                base_confidence = min(100, base_confidence + 5)
            elif consistency < 0.5:  # Very inconsistent
                base_confidence = max(0, base_confidence - 10)

        # Adjust based on coverage (number of high-quality chunks)
        high_quality_count = sum(1 for s in clipped if s >= threshold)
        if high_quality_count >= 3:
            base_confidence = min(100, base_confidence + 5)
        elif high_quality_count < 2:
            base_confidence = max(0, base_confidence - 10)

        return max(0, min(100, base_confidence))

    except Exception as e:
        logger.warning(f"Error computing confidence from scores: {e}, using fallback")
        return 50  # Safe fallback


# ====== INPUT VALIDATION ======
def validate_query_length(question: str, max_length: Optional[int] = None) -> str:
    """Validate and sanitize user query to prevent DoS attacks.

    FIX (Error #5): Prevent unbounded user input from causing memory/CPU exhaustion.

    Args:
        question: User question
        max_length: Maximum allowed length (defaults to config.MAX_QUERY_LENGTH)

    Returns:
        Sanitized question string

    Raises:
        ValidationError: If query is empty or exceeds max length
    """
    if max_length is None:
        max_length = config.MAX_QUERY_LENGTH

    if not question:
        raise ValidationError("Query cannot be empty")

    if len(question) > max_length:
        raise ValidationError(
            f"Query too long ({len(question)} chars). "
            f"Maximum allowed: {max_length} chars. "
            f"Set MAX_QUERY_LENGTH env var to override."
        )

    # Additional sanitization: strip excessive whitespace
    question = " ".join(question.split())

    return question


def normalize_query(text: str) -> str:
    """Lightweight cleanup to make messy tickets retrievable without LLM pre-processing."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n")
    lines = cleaned.splitlines()
    filtered: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        # Drop common signatures/replies/auto-footers
        if lower.startswith(("thanks", "thank you", "best,", "regards,")) and filtered:
            continue
        if "sent from my iphone" in lower or "sent from my android" in lower:
            continue
        if stripped.startswith(">"):
            continue

        # Remove obviously noisy long tokens (e.g., pasted hashes/base64)
        if re.fullmatch(r"[A-Za-z0-9+/=]{60,}", stripped):
            continue

        filtered.append(stripped)

    normalized = " ".join(filtered)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return normalized or text.strip()


# ====== QUERY EXPANSION ======
QUERY_EXPANSIONS_ENV_VAR = "CLOCKIFY_QUERY_EXPANSIONS"
_DEFAULT_QUERY_EXPANSION_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "query_expansions.json"
_query_expansion_cache = None
_query_expansion_override = None


def set_query_expansion_path(path):
    """Override the query expansion configuration file path."""
    global _query_expansion_override
    if path is None:
        _query_expansion_override = None
    else:
        _query_expansion_override = pathlib.Path(path)
    reset_query_expansion_cache()


def reset_query_expansion_cache():
    """Clear cached query expansion data (useful for tests)."""
    global _query_expansion_cache
    _query_expansion_cache = None


def _resolve_query_expansion_path():
    import clockify_rag.config as config

    if _query_expansion_override is not None:
        return _query_expansion_override
    env_path = config.get_query_expansions_path()
    if env_path:
        return pathlib.Path(env_path)
    return _DEFAULT_QUERY_EXPANSION_PATH


def _read_query_expansion_file(path):
    import clockify_rag.config as config

    MAX_EXPANSION_FILE_SIZE = config.MAX_QUERY_EXPANSION_FILE_SIZE
    MAX_EXPANSION_ENTRIES = config.MAX_QUERY_EXPANSION_ENTRIES

    try:
        file_size = os.path.getsize(path)
        if file_size > MAX_EXPANSION_FILE_SIZE:
            raise ValueError(
                f"Query expansion file too large ({file_size} bytes, max {MAX_EXPANSION_FILE_SIZE}). "
                f"Set MAX_QUERY_EXPANSION_FILE_SIZE env var to override."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(f"Query expansion file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in query expansion file {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read query expansion file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Query expansion config must be a JSON object (file: {path})")

    normalized = {}
    for term, synonyms in data.items():
        if not isinstance(term, str):
            raise ValueError(f"Query expansion terms must be strings (file: {path})")
        if not isinstance(synonyms, list):
            raise ValueError(f"Query expansion entry for '{term}' must be a list (file: {path})")

        cleaned = []
        for syn in synonyms:
            syn_str = syn if isinstance(syn, str) else str(syn)
            syn_str = syn_str.strip()
            if syn_str:
                cleaned.append(syn_str)

        if cleaned:
            normalized[term.lower()] = cleaned

    # Validate entry count to prevent memory DoS from malicious expansion files
    if len(normalized) > MAX_EXPANSION_ENTRIES:
        raise ValueError(
            f"Query expansion file has {len(normalized)} entries, "
            f"max allowed is {MAX_EXPANSION_ENTRIES}. "
            f"Set MAX_QUERY_EXPANSION_ENTRIES env var to override."
        )

    return normalized


def load_query_expansion_dict(force_reload=False, suppress_errors=True):
    """Load query expansion dictionary from disk with optional caching."""
    global _query_expansion_cache

    if not force_reload and _query_expansion_cache is not None:
        return _query_expansion_cache

    path = _resolve_query_expansion_path()

    try:
        _query_expansion_cache = _read_query_expansion_file(path)
        return _query_expansion_cache
    except Exception as e:
        if suppress_errors:
            logger.warning(f"Failed to load query expansion config from {path}: {e}")
            _query_expansion_cache = {}
            return _query_expansion_cache
        else:
            raise


# FIX (Error #17): Removed duplicate tokenize function - now imported from utils.py


def approx_tokens(chars: int) -> int:
    """Estimate tokens: 1 token ≈ 4 chars."""
    return max(1, chars // 4)


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count actual tokens using tiktoken for supported models.

    Falls back to model-specific heuristics for Qwen and other models.

    Args:
        text: Text to count tokens for
        model: Model name (defaults to config.RAG_CHAT_MODEL)

    Returns:
        Estimated token count
    """
    from . import config

    model_name: str = model or config.RAG_CHAT_MODEL or ""

    # Try tiktoken for GPT models
    if "gpt" in model_name.lower():
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except (ImportError, KeyError):
            pass

    # For Qwen and other models, use improved heuristic
    # Qwen tokenizer tends to be more efficient than GPT for English (~3.5 chars/token)
    # but less efficient for CJK content (~1.5 chars/token)
    if "qwen" in model_name.lower():
        # Count CJK characters (Chinese, Japanese, Korean)
        import re

        cjk_pattern = r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
        cjk_chars = len(re.findall(cjk_pattern, text))
        non_cjk_chars = len(text) - cjk_chars

        # CJK: ~1.5 chars/token, non-CJK: ~3.5 chars/token
        # Use ceil to ensure we never underestimate (safer for budget enforcement)
        return math.ceil(cjk_chars / 1.5 + non_cjk_chars / 3.5)

    # Default fallback
    return approx_tokens(len(text))


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Truncate text to fit token budget, append ellipsis.

    FIX (Error #9): Handles edge case where budget is smaller than ellipsis tokens.
    """
    est_tokens = count_tokens(text)
    if est_tokens <= budget:
        return text

    ellipsis = "..."
    ellipsis_tokens = count_tokens(ellipsis)

    # FIX (Error #9): Guard against budget too small for ellipsis
    if budget < ellipsis_tokens:
        # Budget too small for ellipsis, truncate to budget without ellipsis
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if count_tokens(text[:mid]) <= budget:
                left = mid
            else:
                right = mid - 1
        return text[:left]

    # Normal case: budget >= ellipsis tokens
    # Binary search for optimal truncation point
    left, right = 0, len(text)
    target = budget - ellipsis_tokens

    while left < right:
        mid = (left + right + 1) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= target:
            left = mid
        else:
            right = mid - 1

    return text[:left] + ellipsis


def expand_query(question: str) -> str:
    """Expand query with domain-specific synonyms and acronyms.

    FIX (Error #5): Validates query length before expansion to prevent DoS.

    Returns expanded query string with original + synonym terms.
    Example: "How to track time?" → "How to track log record enter time hours duration?"
    """
    # FIX (Error #5): Validate input at entry point
    question = validate_query_length(question)
    max_length = config.MAX_QUERY_LENGTH

    if not question:
        return question

    expansions = load_query_expansion_dict()
    q_lower = question.lower()
    expanded_terms = []

    for term, synonyms in expansions.items():
        if re.search(r"\b" + re.escape(term) + r"\b", q_lower):
            for syn in synonyms:
                if syn not in expanded_terms:
                    expanded_terms.append(syn)

    if expanded_terms:
        max_extra = max_length - len(question) - 1
        if max_extra <= 0:
            return question

        expansion = " ".join(expanded_terms)
        if len(expansion) > max_extra:
            expansion = expansion[:max_extra].rsplit(" ", 1)[0].strip()
        if not expansion:
            return question
        return f"{question} {expansion}"

    return question


def normalize_scores_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize scores to have mean=0 and std=1.

    When all scores are identical (std=0), returns zeros since there's no
    discriminative signal between items. This preserves the invariant that
    normalized scores are centered around 0.

    Args:
        arr: Array of scores to normalize

    Returns:
        Normalized array with mean=0, std=1 (or zeros if no variance)
    """
    a = np.asarray(arr, dtype="float32")
    if a.size == 0:
        return a
    # Guard against NaN/inf to avoid warnings in degenerate cases
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    m, s = a.mean(), a.std()
    if s == 0:
        # All scores identical = no discriminative signal, return zeros
        # This maintains the invariant that normalized scores center around 0
        return np.zeros_like(a)
    return (a - m) / s


def embed_query(question: str, retries=0) -> np.ndarray:
    """Embed a query using the configured backend.

    Delegates to :mod:`clockify_rag.embedding` so the query vector shares the
    same dimensionality and normalization strategy as stored document
    embeddings, regardless of whether the backend is Ollama or the local
    SentenceTransformer.
    """

    return _embedding_embed_query(question, retries=retries)


class DenseScoreStore:
    """Container for dense similarity scores with optional lazy materialization."""

    __slots__ = ("_length", "_full", "_vecs", "_qv", "_cache")

    def __init__(
        self,
        length: int,
        *,
        full_scores: Optional[np.ndarray] = None,
        vecs: Optional[np.ndarray] = None,
        qv: Optional[np.ndarray] = None,
        initial: Optional[list] = None,
    ) -> None:
        self._length = int(length)
        self._full: Optional[np.ndarray] = None
        self._vecs = vecs
        self._qv = qv
        self._cache: dict = {}

        if full_scores is not None:
            self._full = np.asarray(full_scores, dtype="float32")
        elif initial:
            self._cache.update({int(idx): float(score) for idx, score in initial})

    def __len__(self) -> int:
        return self._length

    def _materialize_full(self) -> np.ndarray:
        if self._full is None:
            if self._vecs is None or self._qv is None:
                self._full = np.zeros(self._length, dtype="float32")
            else:
                self._full = self._vecs.dot(self._qv).astype("float32")
        return self._full

    def __getitem__(self, idx: int) -> float:
        idx = int(idx)
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        if self._full is not None:
            return float(self._full[idx])

        if idx not in self._cache:
            if self._vecs is None or self._qv is None:
                raise KeyError(idx)
            self._cache[idx] = float(self._vecs[idx].dot(self._qv))
        return self._cache[idx]

    def get(self, idx: int, default: Optional[float] = None) -> Optional[float]:
        try:
            return self[idx]
        except (IndexError, KeyError):
            return default

    def to_array(self) -> np.ndarray:
        return self._materialize_full().copy()


def retrieve(
    question: str, chunks, vecs_n, bm, top_k=None, hnsw=None, retries=0, faiss_index_path=None
) -> Tuple[List[int], Dict[str, Any]]:
    """Hybrid retrieval: dense + BM25 + dedup. Optionally uses FAISS/HNSW for fast K-NN.

    FIX (Error #5): Validates query length at entry point to prevent DoS attacks.

    OPTIMIZATION: Intent-based retrieval with dynamic alpha weighting (+8-12% accuracy).
    Adjusts BM25/dense balance based on query intent:
    - Procedural (how-to): alpha=0.65 (favor BM25 for keyword matching)
    - Factual (what/define): alpha=0.35 (favor dense for semantic understanding)
    - Pricing: alpha=0.70 (high BM25 for exact terms)
    - General: alpha=0.50 (balanced)

    Query expansion: Applies domain-specific synonym expansion for BM25 (keyword-based),
    uses original query for dense retrieval (embeddings already capture semantics).

    Returns:
        Tuple of (filtered_indices, scores_dict) where filtered_indices is list of int
        and scores_dict contains 'dense', 'bm25', 'hybrid' numpy arrays plus 'intent_metadata'.
    """
    # FIX (Error #1): Use centralized FAISS index getter instead of duplicate global state
    # FIX (Error #5): Validate query at entry point
    global RETRIEVE_PROFILE_LAST

    question = validate_query_length(normalize_query(question))

    # Use centralized config value if not specified
    requested_top_k = top_k
    if top_k is None:
        top_k = config.DEFAULT_TOP_K
        logger.debug(f"top_k not specified, using DEFAULT_TOP_K={config.DEFAULT_TOP_K}")

    # Enforce hard ceiling to prevent context overflow (safety cap for user-supplied values)
    # This protects against accidental or malicious large requests that would blow up context
    if top_k > config.MAX_TOP_K:
        logger.warning(f"top_k={top_k} exceeds MAX_TOP_K={config.MAX_TOP_K}, clamping to MAX_TOP_K")
        top_k = config.MAX_TOP_K

    logger.debug(
        f"Retrieval config: requested_top_k={requested_top_k}, effective_top_k={top_k}, max_allowed={config.MAX_TOP_K}"
    )

    # OPTIMIZATION: Classify query intent for specialized retrieval strategy (if enabled)
    intent_metadata = {}
    if config.USE_INTENT_CLASSIFICATION:
        intent_name, intent_config, intent_confidence = classify_intent(question)
        alpha_hybrid = intent_config.alpha_hybrid  # Use intent-specific alpha
        intent_metadata = get_intent_metadata(intent_name, intent_confidence)
    else:
        alpha_hybrid = config.ALPHA_HYBRID  # Use static alpha from config

    # Expand query for BM25 keyword matching
    expanded_question = expand_query(question)

    # Use original question for embedding
    qv_n = embed_query(question, retries=retries)

    # Get FAISS index from centralized source (indexing module)
    faiss_index = None
    if config.USE_ANN == "faiss":
        faiss_index = get_faiss_index(faiss_index_path)
        if faiss_index:
            # Defensive: skip FAISS if dimension mismatches current query vectors (e.g., toy tests)
            try:
                faiss_dim = getattr(faiss_index, "d", None)
                if faiss_dim is not None and faiss_dim != qv_n.shape[0]:
                    logger.info(
                        "info: ann=fallback reason=dim-mismatch faiss_d=%s q_dim=%s",
                        faiss_dim,
                        qv_n.shape[0],
                    )
                    faiss_index = None
            except Exception as e:
                logger.debug("FAISS dimension check failed: %s", e)
                faiss_index = None

        if faiss_index:
            # Only set nprobe for IVF indexes (not flat indexes)
            if hasattr(faiss_index, "nprobe"):
                faiss_index.nprobe = config.ANN_NPROBE
            logger.info("info: ann=faiss status=loaded nprobe=%d", config.ANN_NPROBE)
        elif faiss_index_path:
            logger.info("info: ann=fallback reason=missing-index")

    dense_scores_full = None
    candidate_idx: List[int] = []
    n_chunks = len(chunks)
    dot_elapsed = 0.0
    dense_computed = 0

    if faiss_index:
        # Only score FAISS candidates, don't compute full corpus
        distances, indices = faiss_index.search(
            qv_n.reshape(1, -1).astype("float32"),
            max(config.ANN_CANDIDATE_MIN, top_k * config.FAISS_CANDIDATE_MULTIPLIER),
        )
        # Filter indices and distances together to maintain alignment
        # (prevents misalignment when FAISS returns -1 sentinels)
        valid_pairs = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if 0 <= i < n_chunks]
        candidate_idx = [i for i, _ in valid_pairs]
        dense_from_ann = np.array([d for _, d in valid_pairs], dtype=np.float32)

        dense_scores = dense_from_ann
        dense_computed = len(candidate_idx)
        dot_elapsed = 0.0
    elif hnsw:
        _, cand = hnsw.knn_query(qv_n, k=max(config.ANN_CANDIDATE_MIN, top_k * config.FAISS_CANDIDATE_MULTIPLIER))
        candidate_idx = cand[0].tolist()
        dot_start = time.perf_counter()
        # Compute scores only for HNSW candidates to avoid full-matrix dot products
        if candidate_idx:
            dense_scores = np.array([float(vecs_n[idx].dot(qv_n)) for idx in candidate_idx], dtype=np.float32)
        else:
            dense_scores = np.array([], dtype=np.float32)
        dense_scores_full = None
        dot_elapsed = time.perf_counter() - dot_start
        dense_computed = len(candidate_idx)
    else:
        dot_start = time.perf_counter()
        dense_scores_full = vecs_n.dot(qv_n)
        dot_elapsed = time.perf_counter() - dot_start
        dense_computed = n_chunks
        dense_scores = dense_scores_full
        candidate_idx = np.arange(len(chunks)).tolist()

    if not candidate_idx and not faiss_index:
        max_candidates = max(config.ANN_CANDIDATE_MIN, top_k * config.FAISS_CANDIDATE_MULTIPLIER)
        dense_scores_full = vecs_n.dot(qv_n)
        if len(chunks) > max_candidates:
            top_indices = np.argsort(dense_scores_full)[::-1][:max_candidates]
            candidate_idx = top_indices.tolist()
            dense_scores = dense_scores_full[top_indices]
        else:
            candidate_idx = np.arange(len(chunks)).tolist()
            dense_scores = dense_scores_full

    candidate_idx_array = np.array(candidate_idx, dtype=np.int32)

    # Use expanded query for BM25
    bm_scores_full = bm25_scores(expanded_question, bm, top_k=top_k * 3)

    # Normalize once, then slice for candidates
    zs_bm_full = normalize_scores_zscore(bm_scores_full)
    zs_dense_full = None
    if dense_scores_full is not None:
        dense_scores_full = np.asarray(dense_scores_full, dtype="float32")
        zs_dense_full = normalize_scores_zscore(dense_scores_full)
        zs_dense = zs_dense_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")
    else:
        dense_scores = np.asarray(dense_scores, dtype="float32")
        zs_dense = normalize_scores_zscore(dense_scores)
    zs_bm = zs_bm_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")

    # OPTIMIZATION: Apply intent-based score boosting (if enabled)
    # Boosts chunks containing intent-specific keywords (e.g., pricing sections for pricing queries)
    # Note: When using FAISS, only BM25 scores are boosted (dense scores not fully materialized for performance)
    if config.USE_INTENT_CLASSIFICATION and intent_config.boost_factor != 1.0:
        # Build scores dict for boosting (include dense only if available)
        temp_scores = {"bm25": zs_bm_full}
        if zs_dense_full is not None:
            temp_scores["dense"] = zs_dense_full

        # Apply intent-specific boosting to relevant chunks
        temp_scores = adjust_scores_by_intent(chunks, temp_scores, intent_config)

        # Update normalized scores with boosted values
        zs_bm_full = temp_scores["bm25"]
        if zs_dense_full is not None:
            # Only update dense scores if they were fully materialized
            zs_dense_full = temp_scores["dense"]

        # Re-slice candidate scores from boosted full scores
        zs_bm = zs_bm_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")
        if zs_dense_full is not None:
            zs_dense = zs_dense_full[candidate_idx_array] if candidate_idx_array.size else np.array([], dtype="float32")

    # Hybrid scoring (OPTIMIZATION: use intent-specific alpha for +8-12% accuracy)
    def _apply_hub_penalty(scores: np.ndarray, idx_array: np.ndarray) -> np.ndarray:
        """Down-weight hub/category pages to keep specific answers prioritized."""

        if not scores.size or config.HUB_PAGE_SCORE_MULTIPLIER >= 1.0:
            return scores

        # Copy to avoid mutating inputs used elsewhere
        penalized = scores.copy()
        for pos, chunk_idx in enumerate(idx_array):
            meta = chunks[chunk_idx].get("metadata", {}) or {}
            if bool(meta.get("is_hub")):
                penalized[pos] = penalized[pos] * config.HUB_PAGE_SCORE_MULTIPLIER
        return penalized

    hybrid = alpha_hybrid * zs_bm + (1 - alpha_hybrid) * zs_dense
    hybrid_penalized = _apply_hub_penalty(hybrid, candidate_idx_array) if hybrid.size else hybrid
    if hybrid_penalized.size:
        top_positions = np.argsort(hybrid_penalized)[::-1][:top_k]
        top_idx = candidate_idx_array[top_positions]
    else:
        top_idx = np.array([], dtype=np.int32)

    # Deduplication (stable by article key to avoid cross-article collisions)
    seen = set()
    filtered = []
    for i in top_idx:
        try:
            dedup_key = (_article_key(chunks[i]), chunks[i].get("section"))
        except Exception:
            dedup_key = (chunks[i].get("title"), chunks[i].get("section"))
        key = dedup_key
        if key in seen:
            continue
        seen.add(key)
        filtered.append(i)

    # Reuse cached normalized scores for full hybrid (OPTIMIZATION: use intent-specific alpha)
    if zs_dense_full is not None:
        hybrid_full = alpha_hybrid * zs_bm_full + (1 - alpha_hybrid) * zs_dense_full
    else:
        hybrid_full = np.zeros(len(chunks), dtype="float32")
        for idx, score in zip(candidate_idx, hybrid):
            hybrid_full[idx] = score

    if hybrid_full.size and config.HUB_PAGE_SCORE_MULTIPLIER < 1.0:
        full_idx_array = np.arange(len(chunks), dtype=np.int32)
        hybrid_full = _apply_hub_penalty(hybrid_full, full_idx_array)

    if dense_scores_full is not None:
        dense_scores_store = DenseScoreStore(len(chunks), full_scores=dense_scores_full)
    else:
        dense_scores_store = DenseScoreStore(
            len(chunks), vecs=vecs_n, qv=qv_n, initial=list(zip(candidate_idx, dense_scores))
        )

    dense_total = n_chunks
    used_hnsw = bool(hnsw) and faiss_index is None
    dense_computed_total = dense_computed or (dense_total if (used_hnsw or not faiss_index) else 0)
    dense_reused = dense_total - dense_computed_total

    # FIX: Thread-safe update of profiling state
    global RETRIEVE_PROFILE_LAST
    profile_data = {
        "used_faiss": bool(faiss_index),
        "used_hnsw": used_hnsw,
        "candidates": int(len(candidate_idx)),
        "dense_total": int(dense_total),
        "dense_reused": int(dense_reused),
        "dense_computed": int(dense_computed_total),
        "dense_saved": int(dense_total - dense_computed_total),
        "dense_dot_time_ms": round(dot_elapsed * 1000, 3),
    }

    with _RETRIEVE_PROFILE_LOCK:
        RETRIEVE_PROFILE_LAST = profile_data

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "profile: retrieval ann=%s reused=%d computed=%d total=%d dot_ms=%.3f",
            "faiss" if faiss_index else ("hnsw" if used_hnsw else "linear"),
            profile_data["dense_reused"],
            profile_data["dense_computed"],
            dense_total,
            profile_data["dense_dot_time_ms"],
        )

    # OPTIMIZATION: Include intent metadata for logging and debugging (already populated above)
    return filtered, {
        "dense": dense_scores_store,
        "bm25": bm_scores_full,
        "hybrid": hybrid_full,
        "intent_metadata": intent_metadata,  # OPTIMIZATION: intent classification metadata (or empty dict if disabled)
    }


def rerank_with_llm(
    question: str,
    chunks,
    selected,
    scores,
    seed: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    retries: Optional[int] = None,
) -> Tuple:
    """Optional: rerank MMR-selected passages with LLM using API client.

    Returns: (order, scores, rerank_applied, rerank_reason)
    """
    if len(selected) <= 1:
        return selected, {}, False, "disabled"

    cache_key = None
    if _rerank_cache_enabled():
        cache_key = _make_rerank_cache_key(question, selected, chunks)
        cached = _rerank_cache_get(cache_key, selected, chunks)
        if cached:
            order, scores = cached
            return order, scores, True, "cache"

    # Build passage list
    passages_text = "\n\n".join(
        [f"[id={chunks[i]['id']}]\n{chunks[i]['text'][:config.RERANK_SNIPPET_MAX_CHARS]}" for i in selected]
    )
    if seed is None:
        seed = config.DEFAULT_SEED
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX
    if num_predict is None:
        num_predict = config.DEFAULT_NUM_PREDICT
    if retries is None:
        retries = config.DEFAULT_RETRIES

    from .api_client import chat_completion

    messages: List[ChatMessage] = [
        {"role": "user", "content": RERANK_PROMPT.format(q=question, passages=passages_text)}
    ]

    options: ChatCompletionOptions = {
        "temperature": 0,
        "seed": seed,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.05,
    }

    rerank_scores: Dict[int, float] = {}
    rerank_model = getattr(config, "RERANK_MODEL", "") or config.RAG_CHAT_MODEL

    try:
        response = chat_completion(
            messages=messages,
            model=rerank_model,
            options=options,
            timeout=(config.CHAT_CONNECT_T, config.RERANK_READ_T),
            retries=retries,
        )
        resp = response
        msg = (resp.get("message") or {}).get("content", "").strip()

        if not msg:
            logger.debug("info: rerank=fallback reason=empty")
            return selected, rerank_scores, False, "empty"

        # Try to parse strict JSON array
        try:
            ranked = json.loads(msg)
            if not isinstance(ranked, list):
                logger.debug("info: rerank=fallback reason=json")
                return selected, rerank_scores, False, "json"

            # Map back to indices
            cid_to_idx = {chunks[i]["id"]: i for i in selected}
            reranked = []
            for entry in ranked:
                idx = cid_to_idx.get(entry.get("id"))
                if idx is not None:
                    score = entry.get("score", 0)
                    rerank_scores[idx] = score
                    reranked.append((idx, score))

            if reranked:
                reranked.sort(key=lambda x: x[1], reverse=True)
                if cache_key:
                    order_ids = [str(chunks[idx]["id"]) for idx, _ in reranked]
                    scores_by_id = {str(chunks[idx]["id"]): float(score) for idx, score in rerank_scores.items()}
                    _rerank_cache_put(cache_key, order_ids, scores_by_id)
                return [idx for idx, _ in reranked], rerank_scores, True, ""
            else:
                logger.debug("info: rerank=fallback reason=empty")
                return selected, rerank_scores, False, "empty"
        except json.JSONDecodeError:
            logger.debug("info: rerank=fallback reason=json")
            return selected, rerank_scores, False, "json"
    except LLMError as e:
        # Handle LLM-specific errors from the API client
        error_type = type(e).__name__
        if "timeout" in str(e).lower():
            logger.debug("info: rerank=fallback reason=timeout")
            return selected, rerank_scores, False, "timeout"
        elif "connection" in str(e).lower():
            logger.debug("info: rerank=fallback reason=conn")
            return selected, rerank_scores, False, "conn"
        else:
            logger.debug(f"info: rerank=fallback reason=http error_type={error_type}")
            return selected, rerank_scores, False, "http"
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        # FIX (Error #8): More specific exception handling for expected errors
        logger.debug(f"info: rerank=fallback reason=error error_type={type(e).__name__}")
        return selected, rerank_scores, False, "error"
    except Exception as e:
        # FIX (Error #8): Unexpected errors logged at WARNING level for visibility
        logger.warning(f"Unexpected error in reranking: {type(e).__name__}: {e}", exc_info=True)
        return selected, rerank_scores, False, "unexpected"


def _fmt_snippet_header(chunk):
    """Format chunk header: [id | title | section] + optional URL."""
    hdr = f"[{chunk['id']} | {chunk['title']} | {chunk['section']}]"
    if chunk.get("url"):
        hdr += f"\n{chunk['url']}"
    return hdr


def _article_key(chunk: Dict[str, Any]) -> str:
    """Return a stable article identifier for grouping chunks."""
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


def _sort_article_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort chunks from the same article by structural position."""

    def _sort_key(ch: Dict[str, Any]):
        section_idx = ch.get("section_idx")
        chunk_idx = ch.get("chunk_idx")
        return (
            section_idx if isinstance(section_idx, int) else 10_000,
            chunk_idx if isinstance(chunk_idx, int) else 10_000,
        )

    try:
        return sorted(chunks, key=_sort_key)
    except Exception as e:
        logger.debug("Chunk sorting failed, returning unsorted: %s", e)
        return chunks


def pack_snippets(
    chunks,
    order,
    pack_top: Optional[int] = None,
    budget_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
):
    """Pack snippets by article, respecting token budget and hard caps.

    Groups retrieved chunks by their source article (URL or article_id), preserves
    retrieval order during selection, concatenates chunks within each article
    in document order, and then moves the top-ranked article to the end for
    recency bias. Returns a rendered context string, the list of chunk IDs
    actually included, the token count used, and a list of article-level blocks
    suitable for LLM prompting.
    """
    if pack_top is None:
        pack_top = config.DEFAULT_PACK_TOP
    if budget_tokens is None:
        budget_tokens = config.CTX_TOKEN_BUDGET
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX

    effective_budget = min(budget_tokens, int(num_ctx * 0.6))
    if effective_budget <= 0:
        return "", [], 0, []

    sep_text = "\n\n---\n\n"
    sep_tokens = count_tokens(sep_text)

    # Group chunks by article key in retrieval order
    article_order: List[str] = []
    article_chunks: Dict[str, List[Dict[str, Any]]] = {}
    best_article_key = None
    if order:
        try:
            best_article_key = _article_key(chunks[order[0]])
        except Exception as e:
            logger.debug("Failed to get best article key: %s", e)
            best_article_key = None
    for idx in order:
        chunk = chunks[idx]
        key = _article_key(chunk)
        if key not in article_order:
            article_order.append(key)
        article_chunks.setdefault(key, []).append(chunk)

    # Cap the number of articles we attempt to pack
    article_order = article_order[:pack_top]

    selected_blocks: List[Dict[str, Any]] = []
    used_tokens = 0

    # Selection phase: walk in rank order so best articles get budget first
    for art_pos, art_key in enumerate(article_order, start=1):
        if used_tokens >= effective_budget:
            break

        chunks_for_article = _sort_article_chunks(article_chunks.get(art_key, []))
        if not chunks_for_article:
            continue

        title = chunks_for_article[0].get("title") or chunks_for_article[0].get("doc_name") or "Untitled Article"
        url = (
            chunks_for_article[0].get("url")
            or chunks_for_article[0].get("source_url")
            or (chunks_for_article[0].get("metadata") or {}).get("source_url")
            or art_key
        )

        article_header = f"### Article: {title}\nURL: {url}\n\n"
        sep_cost = sep_tokens if selected_blocks else 0
        available_tokens = effective_budget - used_tokens - sep_cost
        if available_tokens <= 0:
            break

        body_parts: List[str] = []
        included_ids: List[Any] = []
        body_text = ""
        for chunk in chunks_for_article:
            addition = ("" if not body_parts else "\n\n") + chunk["text"]
            candidate_body = body_text + addition
            candidate_tokens = count_tokens(article_header + candidate_body)
            if candidate_tokens <= available_tokens:
                body_parts.append(chunk["text"])
                included_ids.append(chunk["id"])
                body_text = candidate_body
                continue

            remaining_for_body = available_tokens - count_tokens(article_header + body_text)
            truncated = truncate_to_token_budget(chunk["text"], max(0, remaining_for_body))
            if truncated:
                body_parts.append(truncated)
                included_ids.append(chunk["id"])
                body_text = (body_text + ("\n\n" if body_text else "") + truncated) if truncated else body_text
            break

        if not body_parts:
            continue

        article_body = "\n\n".join(body_parts)
        block_text = article_header + article_body
        block_tokens = count_tokens(block_text)
        needed_tokens = sep_cost + block_tokens
        if used_tokens + needed_tokens > effective_budget:
            break

        selected_blocks.append(
            {
                "article_key": art_key,
                "title": title,
                "url": url,
                "text": article_body,
                "chunk_ids": included_ids,
                "text_block": block_text,
            }
        )
        used_tokens += needed_tokens

    # Reorder phase: move best article to end for recency bias
    if best_article_key:
        for i, blk in enumerate(selected_blocks):
            if blk["article_key"] == best_article_key:
                best_blk = selected_blocks.pop(i)
                selected_blocks.append(best_blk)
                break

    # Render final packed text and metadata
    out_pieces: List[str] = []
    packed_ids: List[Any] = []
    article_blocks: List[Dict[str, Any]] = []
    for blk in selected_blocks:
        if out_pieces:
            out_pieces.append(sep_text)
        out_pieces.append(blk["text_block"])
        packed_ids.extend(blk["chunk_ids"])
        article_blocks.append(
            {
                "id": str(len(article_blocks) + 1),
                "title": blk["title"],
                "url": blk["url"],
                "text": blk["text"],
                "chunk_ids": blk["chunk_ids"],
            }
        )

    packed_text = "".join(out_pieces)
    used_tokens = count_tokens(packed_text)

    return packed_text, packed_ids, used_tokens, article_blocks


def derive_role_security_hints(question: str) -> Tuple[str, str]:
    """Derive lightweight role/security hints from the raw ticket text."""
    text = (question or "").lower()
    admin_markers = [
        "my team",
        "my users",
        "my employees",
        "i'm an admin",
        "im an admin",
        "i'm the owner",
        "i am the owner",
    ]
    security_markers = [
        "screenshot",
        "screenshots",
        "delete my account",
        "account deletion",
        "export data",
        "privacy",
        "gdpr",
        "retention",
    ]

    role_hint = "admin" if any(marker in text for marker in admin_markers) else "unknown"
    security_hint = "high" if any(marker in text for marker in security_markers) else "unknown"
    return role_hint, security_hint


def coverage_ok(selected, dense_scores, threshold):
    """Check coverage."""
    if len(selected) < config.COVERAGE_MIN_CHUNKS:
        return False
    highs = sum(1 for i in selected if dense_scores[i] >= threshold)
    return highs >= 2


def ask_llm(
    question: str,
    snippets_block: str,
    seed: Optional[int] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    retries: Optional[int] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call Ollama chat with Qwen using production-grade prompts.

    Returns plain text answer (not JSON). Confidence is computed separately
    from retrieval scores, not from LLM output.

    Args:
        question: User question
        snippets_block: Legacy formatted context string (used if chunks not provided)
        seed, num_ctx, num_predict, retries: LLM parameters
        chunks: Optional list of chunk dicts for new prompt format.
                If provided, uses QWEN_SYSTEM_PROMPT and build_rag_user_prompt().
                If None, falls back to legacy prompts (for backward compatibility).

    Returns:
        Plain text answer from LLM
    """
    if seed is None:
        seed = config.DEFAULT_SEED
    if num_ctx is None:
        num_ctx = config.DEFAULT_NUM_CTX
    if num_predict is None:
        num_predict = config.DEFAULT_NUM_PREDICT
    if retries is None:
        retries = config.DEFAULT_RETRIES

    from .api_client import chat_completion

    # Use new prompts if chunks provided, otherwise fall back to legacy
    if chunks is not None:
        role_hint, security_hint = derive_role_security_hints(question)
        system_prompt = QWEN_SYSTEM_PROMPT
        user_prompt = build_rag_user_prompt(question, chunks, role_hint=role_hint, security_hint=security_hint)
    else:
        # Legacy format for backward compatibility
        system_prompt = get_system_prompt()
        user_prompt = USER_WRAPPER.format(snips=snippets_block, q=question)

    messages: List[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    options: ChatCompletionOptions = {
        "temperature": 0,
        "seed": seed,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.05,
    }

    try:
        response = chat_completion(
            messages=messages,
            model=config.RAG_CHAT_MODEL,
            options=options,
            timeout=(config.CHAT_CONNECT_T, config.CHAT_READ_T),
            retries=retries,
        )
        msg = (response.get("message") or {}).get("content")
        if msg:
            return msg
        return str(response.get("response", ""))
    except LLMError:
        # Re-raise LLMError as is
        raise
    except Exception as e:
        raise LLMError(f"LLM call failed: {e} [hint: check RAG_OLLAMA_URL or increase CHAT timeouts]") from e


# ====== HYBRID SCORING ======
def hybrid_score(bm25_score: float, dense_score: float, alpha: float = 0.5) -> float:
    """Blend BM25 and dense scores: alpha * bm25_norm + (1 - alpha) * dense_norm."""
    return alpha * bm25_score + (1 - alpha) * dense_score


# ====== DYNAMIC PACKING ======
def pack_snippets_dynamic(
    chunk_ids: list, chunks: dict, budget_tokens: int | None = None, target_util: float = 0.75
) -> tuple:
    """Pack snippets with dynamic targeting. Returns (snippets, used_tokens, was_truncated)."""
    if budget_tokens is None:
        budget_tokens = config.CTX_TOKEN_BUDGET
    if not chunk_ids:
        return [], 0, False

    snippets: list[str] = []
    token_count = 0
    target = int(budget_tokens * target_util)

    for cid in chunk_ids:
        try:
            chunk = chunks[cid]
            snippet_tokens = max(1, len(chunk.get("text", "")) // 4)
            separator_tokens = 16
            new_total = token_count + snippet_tokens + separator_tokens

            if new_total > budget_tokens:
                if snippets:
                    return snippets + [{"id": "[TRUNCATED]", "text": "..."}], token_count, True
                else:
                    snippets.append(chunk)
                    return snippets, token_count + snippet_tokens, True

            snippets.append(chunk)
            token_count = new_total

            if token_count >= target:
                break
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            # Skip chunks with invalid data or missing indices
            logger.debug(f"Skipping chunk {cid}: {e}")
            continue

    return snippets, token_count, False


def __getattr__(name: str) -> str:
    """Dynamically resolve derived attributes such as ``SYSTEM_PROMPT``."""
    if name == "SYSTEM_PROMPT":
        return get_system_prompt()
    raise AttributeError(name)


__all__ = [
    "expand_query",
    "embed_query",
    "normalize_scores_zscore",
    "DenseScoreStore",
    "retrieve",
    "rerank_with_llm",
    "pack_snippets",
    "derive_role_security_hints",
    "coverage_ok",
    "ask_llm",
    "tokenize",
    "count_tokens",
    "truncate_to_token_budget",
    "RETRIEVE_PROFILE_LAST",
    "normalize_query",
    "get_system_prompt",
    "SYSTEM_PROMPT",
    "USER_WRAPPER",
    "RERANK_PROMPT",
    "hybrid_score",
    "pack_snippets_dynamic",
    "compute_confidence_from_scores",
]
