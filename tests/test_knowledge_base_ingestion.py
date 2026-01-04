from clockify_rag.chunking import build_chunks
from clockify_rag.retrieval import _article_key


def test_build_chunks_from_helpcenter_file(tmp_path):
    article_path = tmp_path / "knowledge_helpcenter.md"
    article_path.write_text(
        """# [ARTICLE] Timesheet view
https://clockify.me/help/track-time-and-expenses/timesheet-view

## Timesheet view

Managers can review their team's time in Timesheet view.
""",
        encoding="utf-8",
    )

    chunks = build_chunks(str(article_path))

    assert chunks, "Knowledge base directory should produce chunks"
    first = chunks[0]
    assert first["title"] == "Timesheet view"
    assert first["url"] == "https://clockify.me/help/track-time-and-expenses/timesheet-view"
    assert first["doc_path"] == str(article_path)
    assert first["text"].startswith("Context: Timesheet view")
    assert _article_key(first) == first["url"]
