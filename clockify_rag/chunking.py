"""Text parsing and chunking functions for knowledge base processing.

This module provides utilities to parse and chunk documents for the RAG system.
It includes heading-aware splitting, sentence-aware chunking, and overlap management.
"""

import logging
import pathlib
import re
import unicodedata
import hashlib
from typing import Any, Dict, List, Optional

from .config import CHUNK_CHARS, CHUNK_OVERLAP
from .utils import norm_ws, strip_noise

logger = logging.getLogger(__name__)

_FRONT_MATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*?)(?=\n---\s*\n|\Z)", re.S | re.M)
_HIGH_PRIORITY_SECTIONS = {"key points", "limits & gotchas", "canonical answer"}

# Rank 23: NLTK for sentence-aware chunking
try:
    import nltk

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

_NLTK_READY = False


def _ensure_nltk_punkt() -> bool:
    """Ensure NLTK punkt is available, honoring NLTK_AUTO_DOWNLOAD."""
    global _NLTK_READY
    if not _NLTK_AVAILABLE:
        return False
    if _NLTK_READY:
        return True
    try:
        nltk.data.find("tokenizers/punkt")
        _NLTK_READY = True
        return True
    except LookupError:
        from . import config

        if not config.NLTK_AUTO_DOWNLOAD:
            logger.warning("NLTK punkt not available and auto-download disabled; falling back to character chunking")
            return False
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            _NLTK_READY = True
            return True
        except Exception as e:
            logger.warning("NLTK punkt download failed: %s; falling back to character chunking", e)
            return False


def _coerce_front_matter_value(raw: str) -> Any:
    """Parse simple scalar or list values from YAML-like front matter."""
    val = raw.strip().strip('"').strip("'")
    lower = val.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]
    return val


def _parse_front_matter_block(block: str) -> Dict[str, Any]:
    """Parse a YAML-like front matter block without requiring PyYAML."""
    meta: Dict[str, Any] = {}
    current_list_key: Optional[str] = None

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("-") and current_list_key:
            meta.setdefault(current_list_key, [])
            meta[current_list_key].append(stripped.lstrip("-").strip().strip('"').strip("'"))
            continue

        if ":" in stripped:
            key, raw_val = stripped.split(":", 1)
            key = key.strip()
            raw_val = raw_val.strip()

            if raw_val == "":
                meta[key] = []
                current_list_key = key
            else:
                parsed_val = _coerce_front_matter_value(raw_val)
                meta[key] = parsed_val
                current_list_key = key if isinstance(parsed_val, list) else None

    return meta


def _parse_front_matter_articles(md_text: str) -> list:
    """Parse articles that use YAML front matter with UpdateHelpGPT metadata."""
    articles = []
    for match in _FRONT_MATTER_PATTERN.finditer(md_text.strip()):
        meta_block, body = match.groups()
        meta = _parse_front_matter_block(meta_block)
        if bool(meta.get("suppress_from_rag")):
            logger.debug("Skipping suppressed article id=%s", meta.get("id"))
            continue

        title = meta.get("title") or meta.get("short_title") or meta.get("id") or "Untitled Article"
        url = meta.get("source_url") or meta.get("url") or ""
        articles.append({"title": title, "url": url, "body": body.strip(), "meta": meta})

    return articles


def _clean_section_header(header: str) -> str:
    """Normalize section titles by stripping markdown hashes and whitespace."""
    return norm_ws(re.sub(r"^#+\s*", "", header))


def _section_importance(section: str) -> Optional[str]:
    """Return a hint for sections we want to emphasize during retrieval."""
    normalized = section.lower()
    if normalized in _HIGH_PRIORITY_SECTIONS:
        return "high"
    if "faq" in normalized:
        return "medium"
    return None


# ====== KB PARSING ======
def parse_articles(md_text: str) -> list:
    """Parse articles from markdown supporting front matter + legacy formats."""
    articles = _parse_front_matter_articles(md_text)
    if articles:
        return articles

    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("# [ARTICLE]"):
            title_line = re.sub(r"^#\s*\[ARTICLE\]\s*", "", lines[i]).strip()
            url = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("http"):
                url = lines[i + 1].strip()
                i += 2
            else:
                i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("# [ARTICLE]"):
                buf.append(lines[i])
                i += 1
            body = "\n".join(buf).strip()
            articles.append({"title": title_line, "url": url, "body": body, "meta": {}})
        else:
            i += 1

    if not articles:
        articles = [{"title": "KB", "url": "", "body": md_text, "meta": {}}]
    return articles


def split_by_headings(body: str) -> list:
    """Split by H2 headers."""
    parts = re.split(r"\n(?=## +)", body)
    return [p.strip() for p in parts if p.strip()]


def _iter_markdown_sources(md_path: pathlib.Path):
    """Yield (path, text) pairs from a file or directory of markdown."""
    if md_path.is_dir():
        files = sorted(p for p in md_path.rglob("*.md") if p.is_file())
        for path in files:
            yield path, path.read_text(encoding="utf-8", errors="ignore")
    else:
        yield md_path, md_path.read_text(encoding="utf-8", errors="ignore")


def sliding_chunks(text: str, maxc: Optional[int] = None, overlap: Optional[int] = None) -> list:
    """Advanced overlapping chunks with multiple strategies and semantic awareness.

    Uses hierarchical chunking strategies:
    1. Semantic boundaries (paragraphs, sections)
    2. Sentence-aware splitting with NLTK
    3. Character-based fallback

    Args:
        text: Text to chunk
        maxc: Maximum characters per chunk (defaults to config.CHUNK_CHARS)
        overlap: Overlap in characters (defaults to config.CHUNK_OVERLAP)

    Returns:
        List of text chunks
    """
    if maxc is None:
        maxc = CHUNK_CHARS
    if overlap is None:
        overlap = CHUNK_OVERLAP

    if len(text) <= maxc:
        return [text]

    text = strip_noise(text)
    # Normalize to NFKC
    text = unicodedata.normalize("NFKC", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Try semantic boundary chunking first (paragraphs, lists, etc.)
    chunks = semantic_boundary_chunking(text, maxc, overlap)

    # If semantic chunking returns chunks that are too small, refine with sentence-aware splitting
    refined_chunks = []
    for chunk in chunks:
        if len(chunk) <= maxc:
            refined_chunks.append(chunk)
        else:
            # For oversized semantic chunks, apply sentence-aware splitting
            sentence_chunks = sentence_aware_chunking(chunk, maxc, overlap)
            refined_chunks.extend(sentence_chunks)

    return refined_chunks


def semantic_boundary_chunking(text: str, maxc: int, overlap: int) -> list:
    """Split text using semantic boundaries like paragraphs, lists, and sections.

    Args:
        text: Text to chunk semantically
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of semantically-boundaried chunks
    """
    # Split on major semantic boundaries (paragraphs, sections, lists)
    # Preserve the separators to maintain context
    parts = re.split(r"(\n\s*\n|\n\s*[-*]\s+|\n\s*\d+\.\s+|\n#{2,}\s+)", text)

    # Merge separators back with the following content
    merged_parts = []
    separator_re = re.compile(r"^\n\s*\n$|^\n\s*[-*]\s+$|^\n\s*\d+\.\s+$|^\n#{2,}\s+$")
    i = 0
    while i < len(parts):
        current = parts[i]
        if separator_re.match(current or "") and i + 1 < len(parts):
            merged_parts.append((current or "") + (parts[i + 1] or ""))
            i += 2
            continue
        if current and current.strip():
            merged_parts.append(current)
        i += 1

    # Remove empty parts but keep the structure
    merged_parts = [part for part in merged_parts if part.strip()]

    # Group parts into chunks that respect semantic boundaries
    chunks = []
    current_chunk = ""

    for part in merged_parts:
        # If adding this part would exceed the limit
        if len(current_chunk) + len(part) > maxc and current_chunk:
            # Add current chunk to results
            chunks.append(current_chunk.strip())

            # Start a new chunk with overlap if possible
            if len(part) <= maxc:
                # If part fits in a single chunk, start fresh
                current_chunk = part
            else:
                # If part is too large, we'll handle it with sentence splitting later
                current_chunk = part
        else:
            # Add part to current chunk
            current_chunk += part

    # Add the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Now handle any chunks that are still too large with sentence-aware splitting
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= maxc:
            final_chunks.append(chunk)
        else:
            # Use sentence-aware splitting for oversized chunks
            sentence_chunks = sentence_aware_chunking(chunk, maxc, overlap)
            final_chunks.extend(sentence_chunks)

    return final_chunks


def sentence_aware_chunking(text: str, maxc: int, overlap: int) -> list:
    """Overlapping chunks with sentence-aware splitting (Rank 23).

    Uses NLTK sentence tokenization to avoid breaking sentences mid-way.
    Falls back to character-based chunking if NLTK is unavailable.

    Args:
        text: Text to chunk with sentence awareness
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of sentence-aware chunks
    """
    if not text.strip():
        return []

    out = []

    # Rank 23: Use sentence-aware chunking if NLTK is available
    if _ensure_nltk_punkt():
        try:
            sentences = nltk.sent_tokenize(text)

            # Build chunks by accumulating sentences
            current_chunk: list[str] = []
            current_len = 0

            for sent in sentences:
                sent_len = len(sent)

                # If single sentence exceeds maxc, fall back to character splitting
                if sent_len > maxc:
                    # Flush current chunk first
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            out.append(chunk_text)
                        current_chunk = []
                        current_len = 0

                    # Split long sentence by characters with consistent overlap
                    long_chunks = character_chunking(sent, maxc, overlap)
                    for chunk in long_chunks:
                        if chunk.strip():
                            out.append(chunk)
                    continue

                # Check if adding this sentence exceeds maxc
                potential_len = current_len + sent_len + (1 if current_chunk else 0)

                if potential_len > maxc:
                    # Flush current chunk
                    if current_chunk:
                        chunk_text = " ".join(current_chunk).strip()
                        if chunk_text:
                            out.append(chunk_text)

                    # Start new chunk with overlap (last N sentences that fit in overlap)
                    overlap_chars = 0
                    overlap_sents: list[str] = []
                    for prev_sent in reversed(current_chunk):
                        if overlap_chars + len(prev_sent) <= overlap:
                            overlap_sents.insert(0, prev_sent)
                            overlap_chars += len(prev_sent) + 1
                        else:
                            break

                    current_chunk = overlap_sents + [sent]
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sent)
                    current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1

            # Flush final chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    out.append(chunk_text)

            return out

        except Exception as e:
            # Fall back to character-based chunking if NLTK fails
            logger.warning(f"NLTK sentence tokenization failed: {e}, falling back to character chunking")

    # Fallback: Character-based chunking (original implementation)
    return character_chunking(text, maxc, overlap)


def character_chunking(text: str, maxc: int, overlap: int) -> list:
    """Basic character-based chunking as fallback.

    Args:
        text: Text to chunk character-wise
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of character-based chunks
    """
    if not text.strip():
        return []

    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + maxc, n)
        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return out


def yield_sentence_aware_chunk(text: str, maxc: int, overlap: int) -> list:
    """Helper to chunk overly long individual sentences.

    Args:
        text: Long sentence text to chunk
        maxc: Maximum characters per chunk
        overlap: Overlap in characters

    Returns:
        List of chunks from the long sentence
    """
    return character_chunking(text, maxc, overlap)


def build_chunks(md_path: str) -> list:
    """Parse and chunk markdown with enhanced metadata extraction.

    Args:
        md_path: Path to the markdown file to chunk

    Returns:
        List of chunk dictionaries with enhanced metadata
    """
    path_obj = pathlib.Path(md_path)
    chunks = []

    for source_path, raw in _iter_markdown_sources(path_obj):
        doc_name = source_path.stem

        for art in parse_articles(raw):
            meta = dict(art.get("meta") or {})
            article_id = str(meta.get("id") or meta.get("slug") or doc_name).strip()
            slug = re.sub(r"[^A-Za-z0-9_-]+", "-", meta.get("slug") or article_id or doc_name).strip("-") or doc_name
            meta.setdefault("slug", slug)
            title = norm_ws(meta.get("short_title") or meta.get("title") or art["title"])
            source_url = meta.get("source_url") or meta.get("url") or art.get("url")

            sects = split_by_headings(art["body"]) or [art["body"]]

            for sect_idx, sect in enumerate(sects):
                # Extract the section header/title from the content
                head = sect.splitlines()[0] if sect else art["title"]
                section_label = _clean_section_header(head)

                # Build breadcrumb-style hierarchy for disambiguation
                clean_title = title.replace(" - Clockify Help", "").strip()
                hierarchy = [clean_title] if clean_title else []
                if section_label and section_label.lower() != clean_title.lower():
                    hierarchy.append(section_label)

                subsection_headers = extract_subsection_headers(sect)
                if subsection_headers:
                    hierarchy.append(subsection_headers[0])

                breadcrumb = " > ".join(hierarchy)

                # Create chunks for this section
                text_chunks = sliding_chunks(sect)

                for chunk_idx, piece in enumerate(text_chunks):
                    enriched_text = f"Context: {breadcrumb}\n\n{piece}" if breadcrumb else piece
                    # Stable ID for repeatable citations across rebuilds
                    hash_source = f"{slug}|{sect_idx}|{chunk_idx}|{enriched_text}"
                    cid_hash = hashlib.sha1(hash_source.encode("utf-8")).hexdigest()[:8]
                    cid = f"{slug}_{sect_idx}_{chunk_idx}_{cid_hash}"

                    # Extract additional metadata
                    metadata = {**extract_metadata(piece), **meta}
                    if section_label:
                        metadata.setdefault("section_type", section_label)
                        importance = _section_importance(section_label)
                        if importance:
                            metadata["section_importance"] = importance
                    if breadcrumb:
                        metadata["breadcrumb"] = breadcrumb

                    chunk_obj = {
                        "id": cid,
                        "article_id": article_id,
                        "title": title,
                        "url": source_url,
                        "section": section_label,
                        "subsection": subsection_headers[0] if subsection_headers else "",
                        "text": enriched_text,
                        "doc_path": str(source_path),
                        "doc_name": doc_name,
                        "section_idx": sect_idx,
                        "chunk_idx": chunk_idx,
                        "char_count": len(enriched_text),
                        "word_count": len(enriched_text.split()),
                        "metadata": metadata,
                    }

                    chunks.append(chunk_obj)

    return chunks


def extract_subsection_headers(section_text: str) -> List[str]:
    """Extract H3 and H4 headers from section text.

    Args:
        section_text: Text of a section to parse for headers

    Returns:
        List of headers found in order of appearance
    """
    headers = []
    lines = section_text.splitlines()

    for line in lines:
        # Match H3 (###) and H4 (####) headers
        h3_match = re.match(r"^###\s+(.+)", line.strip())
        h4_match = re.match(r"^####\s+(.+)", line.strip())

        if h3_match:
            headers.append(h3_match.group(1).strip())
        elif h4_match:
            headers.append(h4_match.group(1).strip())

    return headers


def extract_metadata(text: str) -> Dict[str, str]:
    """Extract basic metadata from text content.

    Args:
        text: Text to extract metadata from

    Returns:
        Dictionary of extracted metadata
    """
    metadata: Dict[str, Any] = {}

    # Extract dates (common formats)
    date_patterns = [
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
        r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # MM-DD-YYYY
    ]

    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        if dates:
            metadata["dates"] = dates
            break

    # Extract URLs
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    if urls:
        metadata["urls"] = urls[:5]  # Limit to first 5 URLs

    # Extract email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    emails = re.findall(email_pattern, text)
    if emails:
        metadata["emails"] = emails[:5]  # Limit to first 5 emails

    return metadata
