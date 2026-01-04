from pathlib import Path

from clockify_rag.utils import resolve_corpus_path


def test_prefers_primary_corpus_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "knowledge_helpcenter.md"
    corpus.write_text("# [ARTICLE] Sample\nhttps://example.com\n\nBody", encoding="utf-8")

    path, exists, candidates = resolve_corpus_path()

    assert exists is True
    assert Path(path).name == "knowledge_helpcenter.md"
    assert "knowledge_helpcenter.md" in candidates


def test_rejects_non_supported_preferred_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    preferred = tmp_path / "knowledge_full.md"
    preferred.write_text("legacy", encoding="utf-8")
    path, exists, candidates = resolve_corpus_path(str(preferred))

    assert exists is False
    assert Path(path).name == "knowledge_full.md"
    assert "knowledge_helpcenter.md" in candidates


def test_returns_first_candidate_when_none_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path, exists, candidates = resolve_corpus_path()

    assert Path(path).name == "knowledge_helpcenter.md"
    assert candidates[0] == "knowledge_helpcenter.md"
    if exists:
        assert Path(path).is_file()
    else:
        assert exists is False
