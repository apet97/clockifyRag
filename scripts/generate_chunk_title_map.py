#!/usr/bin/env python3
"""Generate a mapping of article titles to chunk UUIDs from chunks.jsonl."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_chunks(chunks_path: Path):
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_title_map(chunks_iter):
    title_map = defaultdict(list)
    for chunk in chunks_iter:
        title = chunk.get("title")
        chunk_id = chunk.get("id")
        if not title or not chunk_id:
            continue
        title_map[title].append(chunk_id)
    return title_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chunks",
        nargs="?",
        default="chunks.jsonl",
        type=Path,
        help="Path to the chunks.jsonl file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="chunk_title_map.json",
        type=Path,
        help="Path to write the resulting mapping",
    )
    args = parser.parse_args()

    chunks_path = args.chunks
    if not chunks_path.exists():
        parser.error(f"Chunks file not found: {chunks_path}")

    title_map = build_title_map(load_chunks(chunks_path))
    output_path = args.output
    output_path.write_text(
        json.dumps(title_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote mapping for {len(title_map)} titles to {output_path}")


if __name__ == "__main__":
    main()
