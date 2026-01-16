"""Tests for article-level packing in retrieval."""

from clockify_rag.retrieval import pack_snippets


def test_pack_snippets_groups_by_article_and_orders_chunks():
    """Chunks from the same article should be grouped and ordered."""
    chunks = [
        {
            "id": "a-1",
            "text": "Alpha chunk one",
            "title": "Alpha",
            "url": "https://a",
            "section_idx": 0,
            "chunk_idx": 0,
        },
        {
            "id": "a-2",
            "text": "Alpha chunk two",
            "title": "Alpha",
            "url": "https://a",
            "section_idx": 0,
            "chunk_idx": 1,
        },
        {"id": "b-1", "text": "Beta chunk one", "title": "Beta", "url": "https://b", "section_idx": 0, "chunk_idx": 0},
    ]

    context_block, packed_ids, used_tokens, article_blocks = pack_snippets(chunks, [0, 1, 2], pack_top=2, num_ctx=500)

    assert "### Article: Alpha" in context_block
    assert "URL: https://a" in context_block
    # Alpha chunks should stay in document order
    assert context_block.index("Alpha chunk one") < context_block.index("Alpha chunk two")
    assert len(article_blocks) == 2
    # Best article is placed last for recency bias
    assert article_blocks[-1]["chunk_ids"] == ["a-1", "a-2"]
    assert "Beta chunk one" in context_block
    assert set(packed_ids) == {"a-1", "a-2", "b-1"}


def test_pack_snippets_respects_budget_by_trimming_articles():
    """Token budget should trim trailing articles first."""
    chunks = [
        {"id": "a-1", "text": "alpha " * 200, "title": "Alpha", "url": "https://a", "section_idx": 0, "chunk_idx": 0},
        {
            "id": "a-2",
            "text": "alpha two " * 200,
            "title": "Alpha",
            "url": "https://a",
            "section_idx": 0,
            "chunk_idx": 1,
        },
        {"id": "b-1", "text": "beta " * 200, "title": "Beta", "url": "https://b", "section_idx": 0, "chunk_idx": 0},
    ]

    context_block, packed_ids, used_tokens, article_blocks = pack_snippets(chunks, [0, 1, 2], pack_top=2, num_ctx=50)

    # Budget too small to include both articles; keep the highest-ranked article first
    assert len(article_blocks) == 1
    assert article_blocks[0]["url"] == "https://a"
    assert all(pid.startswith("a-") for pid in packed_ids)
    assert "Beta" not in context_block
