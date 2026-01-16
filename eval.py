#!/usr/bin/env python3
"""Offline evaluation runner for the Clockify RAG system.

The script supports two modes of operation:

1. **Full RAG evaluation** – uses the production ``retrieve`` pipeline if the
   hybrid index (``chunks.jsonl`` + ``vecs_n.npy`` + ``bm25.json``) is present
   and importable. This measures the real retrieval stack (dense + BM25 + MMR).
2. **Lexical fallback** – if the hybrid index is unavailable, the script builds
   an in-memory BM25 index directly from ``knowledge_helpcenter.md`` using the same chunking
   heuristics as the main application. This path keeps CI lightweight
   (no heavy Torch/SentenceTransformer dependencies) while still validating the
   knowledge base coverage.

Both paths compute the same metrics:

* MRR@10  – position of the first relevant result
* Precision@5 – fraction of relevant results among the top 5 retrieved chunks
* NDCG@10 – position-aware relevance weighting for the top 10 results

The build fails (exit code 1) when any metric falls below the thresholds
defined in ``SUCCESS_THRESHOLDS``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

from clockify_rag.utils import resolve_corpus_path

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - handled at runtime
    BM25Okapi = None  # type: ignore[assignment]
SUCCESS_THRESHOLDS = {
    "mrr_at_10": 0.70,
    "precision_at_5": 0.35,
    "ndcg_at_10": 0.60,
}
FALLBACK_THRESHOLDS = {
    "mrr_at_10": 0.65,
    "precision_at_5": 0.35,
    "ndcg_at_10": 0.60,
}
WARN_FRACTION = 0.85

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOP_K = 12


def compute_mrr(retrieved_ids, relevant_ids):
    """Mean Reciprocal Rank - measures rank of first relevant result.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of ground truth relevant document IDs

    Returns:
        float: 1/rank of first relevant result, or 0 if no relevant results
    """
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def compute_precision_at_k(retrieved_ids, relevant_ids, k=5):
    """Precision@K - fraction of top K results that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        float: Precision@K score
    """
    if k == 0:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / k


def compute_ndcg_at_k(retrieved_ids, relevant_ids, k=10):
    """Normalized Discounted Cumulative Gain@K - position-aware metric.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of ground truth relevant document IDs
        k: Number of top results to consider

    Returns:
        float: NDCG@K score (0-1)
    """
    # DCG: sum of relevances discounted by log position
    dcg = sum(
        1.0 / np.log2(i + 2) if doc_id in relevant_ids else 0.0
        for i, doc_id in enumerate(retrieved_ids[:k])
    )

    # IDCG: DCG of perfect ranking (all relevant docs first)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))

    return dcg / idcg if idcg > 0 else 0.0


def _normalize_text(value: str) -> str:
    """Lowercase + collapse whitespace for robust key matching."""

    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 fallback."""

    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


class LexicalRetriever:
    """BM25-based lexical retriever used when the hybrid index is unavailable."""

    def __init__(self, chunks: list[dict]) -> None:
        if BM25Okapi is None:
            raise RuntimeError(
                "rank-bm25 is required for lexical evaluation. Install rank-bm25"
            )

        corpus_tokens = []
        for chunk in chunks:
            fields = " ".join(
                [
                    chunk.get("title", ""),
                    chunk.get("section", ""),
                    chunk.get("text", ""),
                ]
            )
            corpus_tokens.append(_tokenize(fields))
        self._bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return order.tolist()


def _load_chunks() -> list[dict]:
    """Load chunk metadata from disk or rebuild from the knowledge base."""

    if os.path.exists("chunks.jsonl"):
        chunks: list[dict] = []
        with open("chunks.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    knowledge_path, exists, candidates = resolve_corpus_path()
    if not exists:
        raise FileNotFoundError(
            f"Help corpus is missing. Provide one of: {', '.join(candidates)}"
        )

    try:
        from clockify_rag.chunking import build_chunks
    except ImportError as exc:
        raise RuntimeError(
            "clockify_rag.chunking is required to rebuild chunks. Install the "
            "project dependencies (numpy, requests, nltk)."
        ) from exc

    return build_chunks(knowledge_path)


def _build_chunk_lookup(chunks: list[dict]) -> tuple[dict[str, int], dict[str, set[int]], dict[str, set[int]]]:
    """Create helper mappings for resolving dataset references."""

    id_map: dict[str, int] = {}
    title_section_map: dict[str, set[int]] = defaultdict(set)
    title_map: dict[str, set[int]] = defaultdict(set)

    for idx, chunk in enumerate(chunks):
        cid = chunk.get("id")
        if cid is not None:
            id_map[str(cid)] = idx

        title = _normalize_text(chunk.get("title", ""))
        section = _normalize_text(chunk.get("section", ""))
        title_map[title].add(idx)
        title_section_map[f"{title}|{section}"].add(idx)

    return id_map, title_section_map, title_map


def _resolve_relevant_indices(
    example: dict,
    id_map: dict[str, int],
    title_section_map: dict[str, set[int]],
    title_map: dict[str, set[int]],
) -> set[int]:
    """Resolve dataset annotations into chunk indices."""

    relevant: set[int] = set()

    for cid in example.get("relevant_chunk_ids", []) or []:
        idx = id_map.get(str(cid))
        if idx is not None:
            relevant.add(idx)

    for chunk_ref in example.get("relevant_chunks", []) or []:
        title = _normalize_text(chunk_ref.get("title", ""))
        section = _normalize_text(chunk_ref.get("section", ""))
        key = f"{title}|{section}"
        if section and key in title_section_map:
            relevant.update(title_section_map[key])
            continue
        if title in title_map:
            relevant.update(title_map[title])

    return relevant


def evaluate(
    dataset_path="eval_datasets/clockify_v1.jsonl",
    verbose=False,
    llm_report=False,
    llm_output=None,
    min_mrr: float | None = None,
    min_precision: float | None = None,
    min_ndcg: float | None = None,
):
    """Run evaluation on dataset.

    Args:
        dataset_path: Path to evaluation dataset JSONL file
        verbose: Print per-query results if True
        min_mrr: Optional minimum MRR threshold (hybrid mode target)
        min_precision: Optional Precision@5 threshold (hybrid mode target)
        min_ndcg: Optional NDCG@10 threshold (hybrid mode target)

    Returns:
        dict: Evaluation metrics
    """
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Evaluation dataset not found: {dataset_path}")
        sys.exit(1)

    # Load chunk metadata (from built index or source markdown)
    try:
        chunks = _load_chunks()
    except Exception as exc:  # pragma: no cover - fatal setup error
        print(f"Error preparing chunks: {exc}")
        sys.exit(1)

    # Check for hybrid artifacts (Priority #4: fail fast if artifacts exist but hybrid unusable)
    hybrid_artifacts_exist = (
        os.path.exists("vecs_n.npy") and
        os.path.exists("bm25.json") and
        os.path.exists("chunks.jsonl")
    )
    faiss_available = os.path.exists("faiss.index")
    faiss_index_path = "faiss.index" if faiss_available else None

    # Try to load the production hybrid index; fallback to lexical BM25 only if NO artifacts exist
    rag_available = False
    retrieval_chunks = chunks
    retrieval_fn = None
    lexical_retriever = None
    vecs_n = None
    bm = None
    hnsw = None
    retrieval_mode = "unknown"

    if hybrid_artifacts_exist:
        # Hybrid artifacts exist - MUST use hybrid path (Priority #4)
        print("Hybrid artifacts detected - requiring hybrid retrieval path...")
        try:
            # Priority #12: Use modular retrieval from clockify_rag package
            from clockify_rag.retrieval import retrieve
            from clockify_rag.indexing import load_index

            print("Loading knowledge base index...")
            result = load_index()
            if result is None:
                print("❌ Error: Hybrid artifacts exist but load_index() returned None")
                print("   This indicates index corruption. Run 'make rebuild-all' to fix.")
                sys.exit(1)

            if isinstance(result, dict):
                retrieval_chunks = result["chunks"]
                vecs_n = result["vecs_n"]
                bm = result["bm"]
                hnsw = result.get("hnsw")
            else:
                retrieval_chunks, vecs_n, bm, hnsw = result

            retrieval_fn = lambda q: retrieve(
                q,
                retrieval_chunks,
                vecs_n,
                bm,
                top_k=TOP_K,
                hnsw=hnsw,
                faiss_index_path=faiss_index_path,
            )[0]
            rag_available = True
            retrieval_mode = f"Hybrid (FAISS={'enabled' if faiss_available else 'disabled'})"
            print(f"✅ Hybrid retrieval loaded successfully ({retrieval_mode})")

        except Exception as exc:
            print(f"❌ Error: Hybrid artifacts exist but failed to load hybrid index:")
            print(f"   {exc}")
            print("   Cannot fall back to lexical when hybrid artifacts present.")
            print("   Fix: Run 'make rebuild-all' or remove corrupted artifacts.")
            sys.exit(1)
    else:
        # No hybrid artifacts - use lexical fallback for lightweight CI
        print("No hybrid artifacts found - using lexical BM25 fallback...")
        print("(This is acceptable for CI; run 'make build' for full hybrid evaluation)")
        try:
            lexical_retriever = LexicalRetriever(retrieval_chunks)
            retrieval_fn = lexical_retriever.retrieve
            retrieval_mode = "Lexical BM25 (fallback)"
            print(f"✅ Lexical retrieval loaded successfully")
        except Exception as exc:
            print(f"Error building lexical index: {exc}")
            sys.exit(1)

    # Load evaluation dataset
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    id_map, title_section_map, title_map = _build_chunk_lookup(retrieval_chunks)

    print(f"Loaded {len(dataset)} evaluation queries")

    # Compute metrics
    mrr_scores = []
    precision_at_5_scores = []
    ndcg_at_10_scores = []

    skipped = 0
    llm_outputs: list[dict] = []
    answer_once_fn = None
    if llm_report and rag_available:
        from clockify_rag.answer import answer_once as answer_once_fn
    for i, example in enumerate(dataset):
        query = example["query"]
        relevant_ids = _resolve_relevant_indices(example, id_map, title_section_map, title_map)
        if not relevant_ids:
            print(f"Warning: Skipping query {i+1} ('{query}') - no matching relevant chunks found.")
            skipped += 1
            continue

        try:
            # Retrieve chunks using configured retrieval function
            retrieved_ids = list(retrieval_fn(query)) if retrieval_fn else []

            # Compute metrics
            mrr = compute_mrr(retrieved_ids, relevant_ids)
            precision_at_5 = compute_precision_at_k(retrieved_ids, relevant_ids, k=5)
            ndcg_at_10 = compute_ndcg_at_k(retrieved_ids, relevant_ids, k=10)

            mrr_scores.append(mrr)
            precision_at_5_scores.append(precision_at_5)
            ndcg_at_10_scores.append(ndcg_at_10)

            if verbose:
                print(f"\nQuery {i+1}: {query}")
                print(f"  MRR:         {mrr:.3f}")
                print(f"  Precision@5: {precision_at_5:.3f}")
                print(f"  NDCG@10:     {ndcg_at_10:.3f}")
                print(f"  Retrieved idx: {retrieved_ids[:5]}")
                print(f"  Relevant idx:  {sorted(relevant_ids)}")

            if llm_report and answer_once_fn:
                answer_payload = answer_once_fn(
                    query,
                    retrieval_chunks,
                    vecs_n,
                    bm,
                    hnsw=hnsw,
                    faiss_index_path=faiss_index_path,
                )
                llm_outputs.append(
                    {
                        "query": query,
                        "answer": answer_payload["answer"],
                        "confidence": answer_payload.get("confidence"),
                        "refused": answer_payload.get("refused"),
                        "metadata": answer_payload.get("metadata", {}),
                    }
                )

        except Exception as e:
            print(f"Error evaluating query '{query}': {e}")
            continue

    # Compute aggregate metrics (Priority #13: include retrieval metrics)
    results = {
        "dataset_size": len(dataset),
        "mrr_at_10": float(np.mean(mrr_scores)),
        "precision_at_5": float(np.mean(precision_at_5_scores)),
        "ndcg_at_10": float(np.mean(ndcg_at_10_scores)),
        "mrr_std": float(np.std(mrr_scores)),
        "precision_std": float(np.std(precision_at_5_scores)),
        "ndcg_std": float(np.std(ndcg_at_10_scores)),
        # Retrieval system metrics (Priority #13)
        "retrieval_mode": retrieval_mode,
        "hybrid_available": rag_available,
        "faiss_enabled": faiss_available,
        "queries_processed": len(mrr_scores),
        "queries_skipped": skipped,
    }

    strict_thresholds = {
        "mrr_at_10": min_mrr if min_mrr is not None else SUCCESS_THRESHOLDS["mrr_at_10"],
        "precision_at_5": min_precision if min_precision is not None else SUCCESS_THRESHOLDS["precision_at_5"],
        "ndcg_at_10": min_ndcg if min_ndcg is not None else SUCCESS_THRESHOLDS["ndcg_at_10"],
    }
    thresholds_in_use = strict_thresholds
    threshold_mode = "hybrid_strict"
    if not rag_available:
        thresholds_in_use = {
            metric: min(strict_thresholds[metric], FALLBACK_THRESHOLDS[metric]) for metric in strict_thresholds
        }
        threshold_mode = "lexical_fallback"

    results["thresholds_applied"] = thresholds_in_use
    results["threshold_mode"] = threshold_mode

    # Print results
    processed = len(mrr_scores)
    if processed == 0:
        print("Error: No evaluation queries were processed. Check dataset annotations.")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"RAG EVALUATION RESULTS")
    print("="*70)
    print(f"Retrieval Mode:  {results['retrieval_mode']}")
    if results['hybrid_available']:
        ann_status = "✅ FAISS enabled" if results['faiss_enabled'] else "⚠️  FAISS disabled (full-scan)"
        print(f"ANN Index:       {ann_status}")
    threshold_label = (
        "Hybrid (strict thresholds)" if threshold_mode == "hybrid_strict" else "Lexical BM25 fallback (lenient)"
    )
    print(f"Threshold mode:  {threshold_label}")
    print(
        "Gating:          "
        f"MRR@10≥{thresholds_in_use['mrr_at_10']:.2f} | "
        f"P@5≥{thresholds_in_use['precision_at_5']:.2f} | "
        f"NDCG@10≥{thresholds_in_use['ndcg_at_10']:.2f}"
    )
    print(f"Dataset size:    {results['dataset_size']}")
    print(f"Queries eval:    {results['queries_processed']}")
    if skipped:
        print(f"Skipped queries: {skipped} (missing relevance annotations)")
    print("-"*70)
    print(f"MRR@10:          {results['mrr_at_10']:.3f} (±{results['mrr_std']:.3f})")
    print(f"Precision@5:     {results['precision_at_5']:.3f} (±{results['precision_std']:.3f})")
    print(f"NDCG@10:         {results['ndcg_at_10']:.3f} (±{results['ndcg_std']:.3f})")
    print("="*70)

    # Interpretation
    print("\nINTERPRETATION:")
    warn_mrr = thresholds_in_use["mrr_at_10"] * WARN_FRACTION
    warn_precision = thresholds_in_use["precision_at_5"] * WARN_FRACTION
    warn_ndcg = thresholds_in_use["ndcg_at_10"] * WARN_FRACTION

    mrr_val = results["mrr_at_10"]
    if mrr_val >= thresholds_in_use["mrr_at_10"]:
        print(
            f"✅ MRR@10 ≥ {thresholds_in_use['mrr_at_10']:.2f}: Excellent - first relevant result typically in top 2"
        )
    elif mrr_val >= warn_mrr:
        print(f"⚠️  MRR@10 ≥ {warn_mrr:.2f}: Adequate - first relevant result usually appears near the top")
    else:
        print("❌ MRR@10 below target: Needs improvement - relevant results ranked too low")

    precision_val = results["precision_at_5"]
    if precision_val >= thresholds_in_use["precision_at_5"]:
        print(
            f"✅ Precision@5 ≥ {thresholds_in_use['precision_at_5']:.2f}: Relevant snippets dominate the top 5"
        )
    elif precision_val >= warn_precision:
        print(f"⚠️  Precision@5 ≥ {warn_precision:.2f}: Mixed relevance in top results")
    else:
        print("❌ Precision@5 below target: Too many irrelevant chunks in top results")

    ndcg_val = results["ndcg_at_10"]
    if ndcg_val >= thresholds_in_use["ndcg_at_10"]:
        print(f"✅ NDCG@10 ≥ {thresholds_in_use['ndcg_at_10']:.2f}: Relevant results are well ranked")
    elif ndcg_val >= warn_ndcg:
        print(f"⚠️  NDCG@10 ≥ {warn_ndcg:.2f}: Ranking acceptable but could be sharper")
    else:
        print("❌ NDCG@10 below target: Ranking quality is suboptimal")

    if llm_report and rag_available and llm_outputs:
        output_path = llm_output or os.path.join("eval_reports", "llm_answers.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in llm_outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        results["llm_report_path"] = output_path
        print(f"\n📝 Saved LLM answers to {output_path}")
    elif llm_report and not rag_available:
        print("⚠️  LLM report requested but hybrid index not available. Skipping answer generation.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system on ground truth dataset")
    parser.add_argument(
        "--dataset",
        default="eval_datasets/clockify_v1.jsonl",
        help="Path to evaluation dataset",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-query results")
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=SUCCESS_THRESHOLDS["mrr_at_10"],
        help="Minimum acceptable MRR@10 (default: %(default).2f)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=SUCCESS_THRESHOLDS["precision_at_5"],
        help="Minimum acceptable Precision@5 (default: %(default).2f)",
    )
    parser.add_argument(
        "--min-ndcg",
        type=float,
        default=SUCCESS_THRESHOLDS["ndcg_at_10"],
        help="Minimum acceptable NDCG@10 (default: %(default).2f)",
    )
    parser.add_argument(
        "--llm-report",
        action="store_true",
        help="Generate LLM answers for each query (uses answer_once and respects RAG_LLM_CLIENT)",
    )
    parser.add_argument(
        "--llm-output",
        default="eval_reports/llm_answers.jsonl",
        help="Path to save LLM answer report when --llm-report is used",
    )
    args = parser.parse_args()

    results = evaluate(
        dataset_path=args.dataset,
        verbose=args.verbose,
        llm_report=args.llm_report,
        llm_output=args.llm_output,
        min_mrr=args.min_mrr,
        min_precision=args.min_precision,
        min_ndcg=args.min_ndcg,
    )

    thresholds = results.get("thresholds_applied") or {
        "mrr_at_10": args.min_mrr,
        "precision_at_5": args.min_precision,
        "ndcg_at_10": args.min_ndcg,
    }
    threshold_mode = results.get("threshold_mode", "hybrid_strict")
    mode_label = "Lexical BM25 fallback" if threshold_mode == "lexical_fallback" else "Hybrid evaluation"

    success = all(results[key] >= threshold for key, threshold in thresholds.items())
    if not success:
        print(
            f"\nEvaluation thresholds not met ({mode_label}):" +
            " ".join(
                f"{key}={results[key]:.3f} < {threshold:.2f}"
                for key, threshold in thresholds.items()
                if results[key] < threshold
            )
        )
        sys.exit(1)

    sys.exit(0)
