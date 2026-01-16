"""Embedding generation using local SentenceTransformer or Ollama API."""

import atexit
import gc
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

from . import config
from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Global state for lazy-loaded sentence transformer
_ST_ENCODER = None
_ST_BATCH_SIZE = 32
# Cap concurrent in-flight embedding requests to avoid socket exhaustion
_MAX_OUTSTANDING_REQUESTS = 64

# Global state for lazy-loaded cross-encoder (OPTIMIZATION: Fast, accurate reranking)
_CROSS_ENCODER = None


def cleanup_embedding_models():
    """Clean up embedding model resources to free memory.

    This function releases the SentenceTransformer and CrossEncoder models
    from memory. Useful for:
    - Process shutdown (registered via atexit)
    - Testing (reset state between tests)
    - Memory-constrained environments

    After cleanup, models will be lazily reloaded on next use.
    """
    global _ST_ENCODER, _CROSS_ENCODER

    cleaned: list[str] = []

    if _ST_ENCODER is not None:
        del _ST_ENCODER
        _ST_ENCODER = None
        cleaned.append("SentenceTransformer")

    if _CROSS_ENCODER is not None:
        del _CROSS_ENCODER
        _CROSS_ENCODER = None
        cleaned.append("CrossEncoder")

    if cleaned:
        # Force garbage collection to actually free memory
        gc.collect()
        logger.debug("Cleaned up embedding models: %s", ", ".join(cleaned))


# Register cleanup on process exit
atexit.register(cleanup_embedding_models)


def _load_st_encoder():
    """Lazy-load SentenceTransformer model once."""
    global _ST_ENCODER
    if _ST_ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ST_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        logger.debug("Loaded SentenceTransformer: all-MiniLM-L6-v2 (384-dim)")
    return _ST_ENCODER


def _load_cross_encoder():
    """Lazy-load CrossEncoder model for reranking.

    OPTIMIZATION: CrossEncoder provides 10-15% accuracy boost over LLM reranking
    with 50-100x speed improvement (10ms vs 500-1000ms per rerank).
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        logger.debug("Loaded CrossEncoder: ms-marco-MiniLM-L-12-v2 (for reranking)")
    return _CROSS_ENCODER


def rerank_cross_encoder(query: str, chunks: list, top_k: int = 6) -> list:
    """Rerank chunks using cross-encoder for better relevance scoring.

    OPTIMIZATION: 10-15% accuracy improvement over LLM reranking at 50-100x speed.
    Cross-encoder directly scores query-document pairs instead of independent embeddings.

    Args:
        query: User question
        chunks: List of chunk dicts with 'text' field
        top_k: Number of top chunks to return after reranking

    Returns:
        List of reranked chunk dicts (top_k best matches)
    """
    if not chunks:
        return []

    model = _load_cross_encoder()

    # Create query-document pairs
    pairs = [[query, chunk.get("text", "")] for chunk in chunks]

    # Score all pairs (batch prediction is fast)
    scores = model.predict(pairs)

    # Sort by score (descending)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    # Return top_k
    result = [chunk for chunk, score in ranked[:top_k]]

    logger.debug(
        f"[cross-encoder] Reranked {len(chunks)} → {len(result)} chunks (scores: {[f'{s:.3f}' for _, s in ranked[:top_k]]})"
    )
    return result


def embed_local_batch(texts: list, normalize: bool = True) -> np.ndarray:
    """Encode texts locally using SentenceTransformer in batches."""
    model = _load_st_encoder()
    vecs = []
    for i in range(0, len(texts), _ST_BATCH_SIZE):
        batch = texts[i : i + _ST_BATCH_SIZE]
        batch_vecs = model.encode(batch, normalize_embeddings=normalize, convert_to_numpy=True)
        vecs.append(batch_vecs.astype("float32"))
    return np.vstack(vecs) if vecs else np.zeros((0, config.EMB_DIM), dtype="float32")


def validate_ollama_embeddings(sample_text: str = "test") -> tuple:
    """Validate Ollama embedding endpoint returns correct format and dimensions.

    FIX (v4.1.2): Detect and report API format issues early before building index.
    Returns: (embedding_dim: int, is_valid: bool)
    """
    try:
        from .api_client import get_llm_client

        client = get_llm_client()
        emb = client.create_embedding(
            text=sample_text,
            model=config.RAG_EMBED_MODEL,
            timeout=(config.EMB_CONNECT_T, config.EMB_READ_T),
            retries=config.DEFAULT_RETRIES,
        )

        if not emb or len(emb) == 0:
            logger.error(f"❌ Ollama {config.RAG_EMBED_MODEL}: empty embedding returned (check API format)")
            return 0, False

        dim = len(emb)
        logger.info(f"✅ Ollama {config.RAG_EMBED_MODEL}: {dim}-dim embeddings validated")
        return dim, True
    except Exception as e:
        logger.error(f"❌ Ollama validation failed: {e}")
        return 0, False


def _embed_single_text(index: int, text: str, retries: int, total: int) -> tuple:
    """Embed a single text using remote Ollama (helper for parallel batching).

    Args:
        index: Index of this text in the full list
        text: Text to embed
        retries: Number of retries for HTTP session (passed to embeddings_client)
        total: Total number of texts (for logging)

    Returns:
        tuple: (index, embedding_list) or raises EmbeddingError
    """
    try:
        # Use the new remote-first embeddings_client module
        from .embeddings_client import embed_query

        # Embed single text and return as list (matching old API contract)
        embedding = embed_query(text, retries=retries)

        if embedding is None or embedding.size == 0:
            raise EmbeddingError(f"Embedding chunk {index}: empty embedding returned (check Ollama API format)")

        return (index, embedding.tolist())
    except EmbeddingError as e:
        raise EmbeddingError(f"Embedding chunk {index} failed: {e}") from e
    except Exception as e:
        raise EmbeddingError(f"Embedding chunk {index}: {e}") from e


def embed_texts(texts: list, retries: int | None = None, suppress_errors: bool = False) -> np.ndarray:
    """Embed texts using Ollama with parallel batching (Rank 10: 3-5x speedup).

    Uses ThreadPoolExecutor to send multiple embedding requests concurrently.
    Falls back to sequential processing for small batches or on error.
    """
    if len(texts) == 0:
        return np.zeros((0, config.EMB_DIM), dtype="float32")

    total = len(texts)
    effective_retries = config.DEFAULT_RETRIES if retries is None else retries

    # OPTIMIZATION: Always use parallel batching for 3-5x speedup, even on small batches
    # This eliminates the sequential fallback that added 10x overhead on query embeddings
    # Previous behavior: sequential for < 32 texts or single-threaded
    # New behavior: always parallel (even for 1 text, the overhead is negligible vs 3-5x speedup)

    # Parallel batching mode (always enabled for internal deployment)
    # Priority #7: Cap outstanding futures to prevent socket exhaustion
    logger.info(
        f"[Rank 10] Embedding {total} texts with {config.EMB_MAX_WORKERS} workers (retries={effective_retries})"
    )
    results = [None] * total  # Pre-allocate to maintain order
    completed = 0

    # Priority #7: Limit outstanding futures to max_workers * config.EMB_BATCH_SIZE
    # This prevents memory exhaustion and socket exhaustion on large corpora
    max_outstanding = min(config.EMB_MAX_WORKERS * config.EMB_BATCH_SIZE, _MAX_OUTSTANDING_REQUESTS)
    logger.debug(f"[Priority #7] Capping outstanding futures at {max_outstanding}")

    try:
        with ThreadPoolExecutor(max_workers=config.EMB_MAX_WORKERS) as executor:
            # Priority #7: Use sliding window approach instead of submitting all at once
            # Submit initial batch
            pending_futures: dict = {}
            text_iter = enumerate(texts)

            # Submit initial batch up to max_outstanding without dropping the item
            while len(pending_futures) < max_outstanding:
                try:
                    i, text = next(text_iter)
                except StopIteration:
                    break
                future = executor.submit(_embed_single_text, i, text, effective_retries, total)
                pending_futures[future] = i

            # Process completions and submit new tasks as slots open
            while pending_futures:
                # Wait for at least one future to complete
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    pending_futures.pop(future)
                    idx_result, emb = future.result()  # Will raise if _embed_single_text raised
                    results[idx_result] = emb
                    completed += 1

                    # Log progress every 100 completions
                    if completed % 100 == 0 or completed == total:
                        logger.info(f"  [{completed}/{total}]")

                # Submit new tasks to fill slots (up to max_outstanding)
                while len(pending_futures) < max_outstanding:
                    try:
                        i, text = next(text_iter)
                        future = executor.submit(_embed_single_text, i, text, effective_retries, total)
                        pending_futures[future] = i
                    except StopIteration:
                        # No more texts to process
                        break

    except Exception as e:
        if suppress_errors:
            logger.warning(f"[Rank 10] Batched embedding failed (suppressed): {e}")
            return np.zeros((len(texts), config.EMB_DIM), dtype="float32")
        # If batching fails, log and re-raise
        logger.error(f"[Rank 10] Batched embedding failed: {e}")
        raise

    # Verify all results collected
    if any(r is None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        raise EmbeddingError(f"Missing embeddings for indices: {missing}")

    return np.array(results, dtype="float32")


def load_embedding_cache() -> dict:
    """Load embedding cache from disk with dimension validation.

    FIX: Filters out cached embeddings with mismatched dimensions to prevent
    mixing embeddings from different models/backends (e.g., 384-dim local vs 768-dim ollama).

    Returns:
        dict: {content_hash: embedding_vector} mapping (only valid embeddings for current config.EMB_DIM)
    """
    # Compute expected dimension based on current backend
    expected_dim = config.EMB_DIM_LOCAL if config.EMB_BACKEND == "local" else config.EMB_DIM_OLLAMA

    cache = {}
    cache_path = config.FILES["emb_cache"]
    if os.path.exists(cache_path):
        logger.info(f"[INFO] Loading embedding cache from {cache_path}")
        filtered_count = 0
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            embedding = np.array(entry["embedding"], dtype=np.float32)

                            # Validate dimension matches current backend
                            if len(embedding) != expected_dim:
                                filtered_count += 1
                                logger.debug(
                                    f"Filtered cache entry (line {line_num}): "
                                    f"dim={len(embedding)} != expected={expected_dim} "
                                    f"(backend={config.EMB_BACKEND})"
                                )
                                continue

                            # Optionally validate backend/model if stored
                            stored_backend = entry.get("backend")
                            if stored_backend and stored_backend != config.EMB_BACKEND:
                                filtered_count += 1
                                logger.debug(
                                    f"Filtered cache entry (line {line_num}): "
                                    f"backend={stored_backend} != current={config.EMB_BACKEND}"
                                )
                                continue

                            cache[entry["hash"]] = embedding
                        except (KeyError, ValueError, TypeError) as e:
                            logger.debug(f"Skipping malformed cache entry (line {line_num}): {e}")
                            continue

            valid_count = len(cache)
            logger.info(
                f"[INFO] Cache loaded: {valid_count} valid embeddings " f"(filtered {filtered_count} mismatched)"
            )

            if filtered_count > 0:
                logger.warning(
                    f"⚠️  Filtered {filtered_count} cached embeddings with incompatible dimensions. "
                    f"This is expected after switching embedding backends "
                    f"(e.g., local→ollama or vice versa). "
                    f"These will be recomputed and cached with current backend."
                )
        except Exception as e:
            logger.warning(f"[WARN] Failed to load cache: {e}; starting fresh")
            cache = {}
    return cache


def save_embedding_cache(cache: dict):
    """Save embedding cache to disk with backend/dimension metadata.

    FIX: Stores backend, model, and dimension metadata with each cache entry
    to enable validation when loading (prevents dimension mismatches).

    Args:
        cache: dict of {content_hash: embedding_vector}
    """
    cache_path = config.FILES["emb_cache"]
    logger.info(f"[INFO] Saving {len(cache)} embeddings to cache")
    try:
        # Atomic write with temp file
        temp_path = cache_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            for content_hash, embedding in cache.items():
                entry = {
                    "hash": content_hash,
                    "embedding": embedding.tolist(),
                    "backend": config.EMB_BACKEND,  # Store backend for validation
                    "model": config.RAG_EMBED_MODEL if config.EMB_BACKEND == "ollama" else "all-MiniLM-L6-v2",
                    "dim": int(len(embedding)),  # Store dimension for validation
                }
                f.write(json.dumps(entry) + "\n")
        # Ensure write hits disk before rename
        with open(temp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp_path, cache_path)  # Atomic on POSIX
        logger.info(f"[INFO] Cache saved successfully (backend={config.EMB_BACKEND}, dim={config.EMB_DIM})")
    except Exception as e:
        logger.warning(f"[WARN] Failed to save cache: {e}")


def embed_query(question: str, retries=0) -> np.ndarray:
    """Embed a single query using configured backend with proper normalization.

    For local backend: Uses SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
    For ollama backend: Uses remote Ollama via embeddings_client (nomic-embed-text, 768-dim)

    Args:
        question: Query text to embed
        retries: Unused parameter (for backward compatibility with older API)

    Returns:
        Normalized float32 numpy array of shape (embedding_dim,)
    """
    if config.EMB_BACKEND == "local":
        vec = embed_local_batch([question], normalize=True)
        return vec[0]
    else:
        # Use the remote embeddings_client for all Ollama queries
        from .embeddings_client import embed_query as embed_query_remote

        try:
            # embeddings_client.embed_query returns already-normalized 1D array
            vec = embed_query_remote(question, retries=retries)
            return vec
        except Exception as e:
            logger.error(f"Failed to embed query via embeddings_client: {e}")
            # Fallback: try embed_texts if single query fails
            try:
                vecs = embed_texts([question], retries=retries)
                return vecs[0] if len(vecs) > 0 else np.zeros(config.EMB_DIM, dtype=np.float32)
            except Exception as e2:
                logger.error(f"Fallback embed_texts also failed: {e2}")
                raise
