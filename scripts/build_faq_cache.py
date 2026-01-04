#!/usr/bin/env python3
"""Build precomputed FAQ cache for instant responses.

OPTIMIZATION (Analysis Section 9.1 #3): Pre-generate answers for top FAQs.

Usage:
    python3 scripts/build_faq_cache.py config/sample_faqs.txt

    # With custom output path
    python3 scripts/build_faq_cache.py config/sample_faqs.txt --output faq_cache.json

    # With retrieval parameters
    python3 scripts/build_faq_cache.py config/sample_faqs.txt --top-k 15 --pack-top 8
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clockify_rag.indexing import load_index
from clockify_rag.precomputed_cache import build_faq_cache, load_faq_list
from clockify_rag.config import (
    DEFAULT_TOP_K,
    DEFAULT_PACK_TOP,
    DEFAULT_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build precomputed FAQ cache for instant responses")
    parser.add_argument("faq_file", help="Path to FAQ file (one question per line)")
    parser.add_argument("--output", default="faq_cache.json", help="Output cache file path (default: faq_cache.json)")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of candidates to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--pack-top",
        type=int,
        default=DEFAULT_PACK_TOP,
        help=f"Number of chunks to pack in context (default: {DEFAULT_PACK_TOP})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum similarity threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for LLM (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"LLM context window size (default: {DEFAULT_NUM_CTX})"
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=f"LLM max tokens to generate (default: {DEFAULT_NUM_PREDICT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retries for LLM calls (default: {DEFAULT_RETRIES})",
    )

    args = parser.parse_args()

    # Load FAQ questions
    logger.info(f"Loading FAQ questions from {args.faq_file}...")
    try:
        questions = load_faq_list(args.faq_file)
        logger.info(f"Loaded {len(questions)} FAQ questions")
    except FileNotFoundError:
        logger.error(f"FAQ file not found: {args.faq_file}")
        sys.exit(1)

    if not questions:
        logger.error("No questions found in FAQ file")
        sys.exit(1)

    # Load index
    logger.info("Loading RAG index...")
    try:
        index_data = load_index()
        if index_data is None:
            logger.error(
                "Failed to load index. Run 'python3 -m clockify_rag.cli_modern ingest --input knowledge_helpcenter.md --force' first."
            )
            sys.exit(1)

        chunks = index_data["chunks"]
        vecs_n = index_data["vecs_n"]
        bm = index_data["bm"]
        meta = index_data.get("meta") if isinstance(index_data, dict) else None
        logger.info(f"Index loaded: {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        sys.exit(1)

    # Build FAQ cache
    logger.info(f"Building FAQ cache with {len(questions)} questions...")
    try:
        kb_signature = None
        if isinstance(meta, dict):
            kb_signature = meta.get("kb_sha256") or meta.get("kb_sha") or meta.get("kb_signature")
            if kb_signature:
                kb_signature = str(kb_signature)

        cache = build_faq_cache(
            questions=questions,
            chunks=chunks,
            vecs_n=vecs_n,
            bm=bm,
            output_path=args.output,
            kb_signature=kb_signature,
            top_k=args.top_k,
            pack_top=args.pack_top,
            threshold=args.threshold,
            seed=args.seed,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            retries=args.retries,
        )

        logger.info(f"✅ FAQ cache built successfully: {cache.size()} entries")
        logger.info(f"   Cache saved to: {args.output}")
        logger.info(f"   To use: Set FAQ_CACHE_PATH={args.output} environment variable")

    except Exception as e:
        logger.error(f"Failed to build FAQ cache: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
