from clockify_rag.chunking import build_chunks


def test_build_chunks_supports_updatehelpgpt_front_matter(tmp_path):
    fixture = tmp_path / "help_corpus_front_matter.md"
    fixture.write_text(
        """---
id: clk-001
title: "Lock timesheets"
url: "https://clockify.me/help/timesheets/lock-timesheets"
tags: ["timesheets"]
detected_lang: "en"
---

## Key points
- Lock timesheets from the workspace settings.

---
id: clk-legacy-404
title: "Old article"
url: "https://clockify.me/help/legacy/removed"
suppress_from_rag: true
---

## Body
This should be skipped.
""",
        encoding="utf-8",
    )

    chunks = build_chunks(str(fixture))

    assert chunks, "Front-matter corpus should produce chunks"
    assert all(not (c.get("article_id") == "clk-legacy-404") for c in chunks), "Suppressed articles must be skipped"
    assert all(not str(c["section"]).startswith("#") for c in chunks), "Section labels should be cleaned"

    first = chunks[0]
    assert first["title"] == "Lock timesheets"
    assert first["url"].startswith("https://clockify.me/help/timesheets/lock-timesheets")
    assert first["metadata"]["id"] == "clk-001"
    assert first["metadata"]["detected_lang"] == "en"
    assert "timesheets" in first["metadata"].get("tags", [])
    assert first["metadata"].get("section_type") == first["section"]
    assert any(c["metadata"].get("section_importance") == "high" for c in chunks)
    assert any(c["metadata"].get("section_type") == "Key points" for c in chunks)
