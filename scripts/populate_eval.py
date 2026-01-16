#!/usr/bin/env python3
"""Populate evaluation datasets with relevant chunk IDs.

This helper script loads the chunk corpus (``chunks.jsonl``) and an
evaluation dataset (``eval_datasets/*.jsonl``). For each query the script
uses lightweight keyword heuristics to surface likely relevant chunks and
optionally prompts the user to confirm or adjust the selection. The final
assignments are written back to JSONL so evaluation metrics can be
computed programmatically.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

# OPTIMIZATION (Analysis Section 10.3 #3): Import tokenize from utils to consolidate
from clockify_rag.utils import tokenize

# Basic English stop words to avoid noisy matches in keyword heuristics.
STOPWORDS: Set[str] = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "are",
    "was",
    "were",
    "how",
    "what",
    "can",
    "you",
    "your",
    "does",
    "did",
    "why",
    "who",
    "where",
    "when",
    "which",
    "will",
    "able",
    "into",
    "about",
    "using",
    "use",
    "set",
    "sets",
    "been",
    "more",
    "than",
    "such",
    "per",
    "any",
    "all",
    "each",
    "other",
    "should",
    "could",
    "would",
    "their",
    "them",
    "our",
    "ours",
    "its",
    "it's",
    "his",
    "her",
    "hers",
    "him",
    "she",
    "he",
    "they",
    "them",
    "been",
    "being",
    "also",
    "very",
    "much",
    "please",
    "need",
    "want",
    "like",
    "just",
    "does",
    "did",
    "make",
    "made",
    "give",
    "takes",
    "take",
    "takes",
    "some",
    "many",
    "few",
    "via",
    "etc",
    "etcetera",
}


@dataclass
class ChunkRecord:
    """Container with preprocessed information for a chunk."""

    raw: Dict[str, object]
    text_tokens: Set[str]
    meta_tokens: Set[str]
    search_blob: str

    def __post_init__(self) -> None:
        self.raw.setdefault("id", "")

    @property
    def id(self) -> str:
        return str(self.raw.get("id", ""))

    @property
    def title(self) -> str:
        return str(self.raw.get("title", ""))

    @property
    def section(self) -> str:
        return str(self.raw.get("section", ""))

    def preview(self, width: int = 140) -> str:
        """Return a short preview of the chunk content."""
        body = str(self.raw.get("text", "")).replace("\n", " ").strip()
        if not body:
            body = " ".join(
                part
                for part in (
                    self.raw.get("summary"),
                    self.raw.get("description"),
                    self.raw.get("url"),
                )
                if part
            )
        if not body:
            body = "(no text)"
        return textwrap.shorten(body, width=width, placeholder="…")


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> Tuple[List[ChunkRecord], Dict[str, ChunkRecord]]:
    chunks: List[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            text_tokens = set(tokenize(str(raw.get("text", ""))))
            meta_tokens = set(
                tokenize(
                    " ".join(
                        filter(
                            None,
                            (
                                str(raw.get("title", "")),
                                str(raw.get("section", "")),
                                str(raw.get("url", "")),
                                str(raw.get("breadcrumbs", "")),
                            ),
                        )
                    )
                )
            )
            search_blob = " ".join(
                filter(
                    None,
                    (
                        str(raw.get("title", "")),
                        str(raw.get("section", "")),
                        str(raw.get("url", "")),
                        str(raw.get("text", "")),
                    ),
                )
            ).lower()
            chunks.append(
                ChunkRecord(
                    raw=raw,
                    text_tokens=text_tokens,
                    meta_tokens=meta_tokens,
                    search_blob=search_blob,
                )
            )
    lookup = {chunk.id: chunk for chunk in chunks}
    return chunks, lookup


def extract_keywords(entry: Dict[str, object]) -> List[str]:
    fields: List[str] = [str(entry.get("query", ""))]
    tags = entry.get("tags") or []
    if isinstance(tags, (list, tuple)):
        fields.extend(str(tag) for tag in tags)
    notes = entry.get("notes")
    if isinstance(notes, str):
        fields.append(notes)
    keywords: List[str] = []
    for field in fields:
        for token in tokenize(field):
            if len(token) <= 2:
                continue
            if token in STOPWORDS:
                continue
            keywords.append(token)
    return keywords


def extract_phrases(entry: Dict[str, object], max_tokens: int = 4) -> Set[str]:
    phrases: Set[str] = set()
    sources: List[str] = [str(entry.get("query", ""))]
    tags = entry.get("tags") or []
    if isinstance(tags, (list, tuple)):
        sources.extend(str(tag) for tag in tags)
    notes = entry.get("notes")
    if isinstance(notes, str):
        sources.append(notes)

    for text in sources:
        tokens = [tok for tok in tokenize(text) if tok not in STOPWORDS]
        length = len(tokens)
        for size in range(2, min(max_tokens, length) + 1):
            for idx in range(0, length - size + 1):
                phrase = " ".join(tokens[idx : idx + size])
                if len(phrase) <= 3:  # avoid extremely small/overlapping phrases
                    continue
                phrases.add(phrase)
    return phrases


def score_chunks(
    chunks: Sequence[ChunkRecord], keywords: Sequence[str], phrases: Sequence[str]
) -> List[Tuple[ChunkRecord, float]]:
    scored: List[Tuple[ChunkRecord, float]] = []
    keyword_weights: Dict[str, int] = {}
    for keyword in keywords:
        keyword_weights[keyword] = keyword_weights.get(keyword, 0) + 1

    for chunk in chunks:
        score = 0.0
        if keyword_weights:
            for keyword, weight in keyword_weights.items():
                if keyword in chunk.meta_tokens:
                    score += 3.0 * weight
                if keyword in chunk.text_tokens:
                    score += 1.5 * weight
                if keyword in chunk.search_blob:
                    score += 0.5 * weight
        for phrase in phrases:
            if phrase in chunk.search_blob:
                score += 1.0 + 0.25 * len(phrase.split())
        if score > 0:
            scored.append((chunk, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def prompt_user(
    entry: Dict[str, object],
    candidates: Sequence[Tuple[ChunkRecord, float]],
    chunk_lookup: Dict[str, ChunkRecord],
    args: argparse.Namespace,
) -> List[str]:
    default_selection: List[str] = []
    for chunk, score in candidates[: args.auto_top]:
        if score >= args.score_threshold:
            default_selection.append(chunk.id)

    if args.auto:
        return default_selection

    query = str(entry.get("query", ""))
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    difficulty = entry.get("difficulty")
    tags = entry.get("tags")
    notes = entry.get("notes")
    if difficulty:
        print(f"Difficulty: {difficulty}")
    if tags:
        print(f"Tags: {', '.join(map(str, tags))}")
    if notes:
        print(f"Notes: {notes}")
    existing_ids = entry.get("relevant_chunk_ids") or []
    if existing_ids:
        print(f"Current chunk IDs: {', '.join(map(str, existing_ids))}")
    if candidates:
        print("\nHeuristic matches:")
        for idx, (chunk, score) in enumerate(candidates[: args.max_candidates], 1):
            title = chunk.title or "(no title)"
            section = chunk.section
            label = f"{title}"
            if section and section not in title:
                label += f" — {section}"
            preview = chunk.preview(args.preview_width)
            print(f"[{idx}] score={score:.2f} id={chunk.id}\n    {label}\n    {preview}")
    else:
        print("\nNo heuristic matches found. Enter chunk IDs manually or type 'skip'.")

    if default_selection:
        default_prompt = ", ".join(default_selection)
    else:
        default_prompt = "none"

    print(
        'Commands: enter comma/space separated numbers or chunk IDs, "show <id>" to preview,'
        ' "skip" to leave empty, or press Enter to accept the default.'
    )

    candidate_map: Dict[str, str] = {}
    for idx, (chunk, _) in enumerate(candidates[: args.max_candidates], 1):
        candidate_map[str(idx)] = chunk.id
        candidate_map[chunk.id] = chunk.id

    while True:
        raw = input(f"Selection [{default_prompt}]: ").strip()
        if not raw:
            return default_selection
        lower = raw.lower()
        if lower in {"skip", "none"}:
            return []
        if lower in {"help", "?"}:
            print(
                "Enter numbers (1-based) or chunk IDs separated by commas/spaces."
                " Example: '1 3 4' or 'clk_123, clk_456'."
            )
            continue
        if lower.startswith("show "):
            token = raw.split(maxsplit=1)[1]
            chunk_id = candidate_map.get(token) or token
            chunk = chunk_lookup.get(chunk_id)
            if not chunk:
                print(f"Unknown chunk reference: {token}")
                continue
            print("-" * 80)
            print(f"Chunk {chunk.id} — {chunk.title or '(no title)'}")
            if chunk.section:
                print(f"Section: {chunk.section}")
            text = str(chunk.raw.get("text", "")).strip()
            if not text:
                text = "(chunk has no text field)"
            print(textwrap.fill(text, width=args.preview_width))
            print("-" * 80)
            continue

        tokens = re.split(r"[\s,]+", raw)
        selected: List[str] = []
        unknown: List[str] = []
        for token in tokens:
            if not token:
                continue
            chunk_id = candidate_map.get(token) or token
            if chunk_id in chunk_lookup:
                if chunk_id not in selected:
                    selected.append(chunk_id)
            else:
                unknown.append(token)
        if unknown:
            print(f"Unrecognized chunk IDs: {', '.join(unknown)}")
            continue
        return selected


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate evaluation datasets with relevant chunk IDs.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval_datasets/clockify_v1.jsonl"),
        help="Path to the evaluation dataset JSONL file.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Path to the chunks JSONL file produced by the index build.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Path for the populated dataset. Defaults to <dataset> with "
            "'_populated' suffix inside the eval_datasets directory."
        ),
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Assign chunk IDs automatically using heuristics without prompting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run selection even if relevant_chunk_ids are already present.",
    )
    parser.add_argument(
        "--auto-top",
        type=int,
        default=3,
        help="Number of top-scoring heuristic matches to accept by default.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Maximum number of candidate chunks to display per query.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=3.0,
        help="Minimum heuristic score required for automatic selection.",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=120,
        help="Character width for chunk previews in interactive mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write any files; useful for reviewing selections.",
    )
    return parser.parse_args(argv)


def determine_output_path(dataset: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    parent = dataset.parent
    base = dataset.stem
    suffix = dataset.suffix
    populated_name = f"{base}_populated{suffix or '.jsonl'}"
    if parent.name != "eval_datasets":
        parent = Path("eval_datasets")
    return parent / populated_name


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    if not args.chunks.exists():
        print(f"Chunks file not found: {args.chunks}", file=sys.stderr)
        return 1

    dataset_records = load_jsonl(args.dataset)
    chunks, chunk_lookup = load_chunks(args.chunks)

    print(f"Loaded {len(dataset_records)} evaluation queries and {len(chunks)} chunks.")

    updated = 0
    for idx, entry in enumerate(dataset_records, 1):
        existing_ids = entry.get("relevant_chunk_ids") or []
        if existing_ids and not args.overwrite:
            continue
        keywords = extract_keywords(entry)
        phrases = extract_phrases(entry)
        candidates = score_chunks(chunks, keywords, phrases)
        selected_ids = prompt_user(entry, candidates, chunk_lookup, args)
        if selected_ids:
            entry["relevant_chunk_ids"] = selected_ids
        else:
            entry["relevant_chunk_ids"] = []
        updated += 1
        if args.auto:
            print(f"[{idx}/{len(dataset_records)}] {entry.get('query')!r} → {selected_ids}")

    output_path = determine_output_path(args.dataset, args.output)
    populated_count = sum(1 for record in dataset_records if record.get("relevant_chunk_ids"))
    print(f"Processed {updated} queries. {populated_count} now have relevant chunk IDs.")

    if args.dry_run:
        print("Dry run requested; skipping write.")
        return 0

    save_jsonl(output_path, dataset_records)
    print(f"Saved populated dataset to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
