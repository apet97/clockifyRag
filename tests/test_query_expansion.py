"""Tests for query expansion functionality."""

import json
import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clockify_rag.retrieval import (
    QUERY_EXPANSIONS_ENV_VAR,
    expand_query,
    load_query_expansion_dict,
    reset_query_expansion_cache,
    set_query_expansion_path,
)


@pytest.fixture(autouse=True)
def restore_query_expansions():
    """Ensure each test starts with the default query expansion config."""
    set_query_expansion_path(None)
    reset_query_expansion_cache()
    yield
    set_query_expansion_path(None)
    reset_query_expansion_cache()


class TestQueryExpansion:
    """Test query expansion with synonyms and acronyms."""

    def test_expand_query_with_track(self):
        """Test that 'track' expands with synonyms."""
        query = "How to track time?"
        expanded = expand_query(query)

        # Should contain original query
        assert "How to track time?" in expanded
        # Should contain synonyms for 'track'
        assert any(syn in expanded for syn in ["log", "record", "enter", "add"])
        # Should contain synonyms for 'time'
        assert any(syn in expanded for syn in ["hours", "duration"])

    def test_expand_query_with_acronym(self):
        """Test that acronyms like 'SSO' are expanded."""
        query = "How to configure SSO?"
        expanded = expand_query(query)

        # Should contain original
        assert "How to configure SSO?" in expanded
        # Should contain expansion
        assert "single sign-on" in expanded or "single sign on" in expanded

    def test_expand_query_no_match(self):
        """Test that queries without matching terms are unchanged."""
        query = "What is the meaning of life?"
        expanded = expand_query(query)

        # Should return original query unchanged
        assert expanded == query

    def test_expand_query_empty(self):
        """Test that empty query raises ValidationError."""
        from clockify_rag.exceptions import ValidationError

        with pytest.raises(ValidationError, match="cannot be empty"):
            expand_query("")

    def test_expand_query_multiple_terms(self):
        """Test query with multiple matching terms."""
        query = "How to track billable time?"
        expanded = expand_query(query)

        # Should expand both 'track' and 'billable' and 'time'
        assert "How to track billable time?" in expanded
        assert any(syn in expanded for syn in ["log", "record"])
        assert any(syn in expanded for syn in ["chargeable", "invoiceable"])
        assert any(syn in expanded for syn in ["hours", "duration"])

    def test_expand_query_whole_word_only(self):
        """Test that partial word matches are not expanded."""
        # 'track' should not match 'attraction'
        query = "tourist attraction"
        expanded = expand_query(query)

        # Should not expand because 'track' is not a whole word
        assert expanded == query

    def test_expand_query_case_insensitive(self):
        """Test that expansion works regardless of case."""
        query = "How to TRACK TIME?"
        expanded = expand_query(query)

        # Should still expand despite uppercase
        assert any(syn in expanded for syn in ["log", "record", "enter"])

    def test_expand_query_with_reports(self):
        """Test expansion of 'report' term."""
        query = "Generate report"
        expanded = expand_query(query)

        assert "Generate report" in expanded
        assert any(syn in expanded for syn in ["summary", "analytics", "export"])

    def test_expand_query_mobile_terms(self):
        """Test expansion of mobile-related terms."""
        query = "Can I use mobile app offline?"
        expanded = expand_query(query)

        # Should expand 'mobile' and 'offline'
        assert any(syn in expanded for syn in ["phone", "smartphone", "app"])
        assert any(syn in expanded for syn in ["no internet", "no connection"])

    def test_expand_query_preserves_original(self):
        """Test that original query is always preserved."""
        query = "How to track time?"
        expanded = expand_query(query)

        # Original should be the first part
        assert expanded.startswith(query)

    def test_expand_query_uses_cli_override(self, tmp_path):
        """Query expansion honors CLI-style override via setter."""
        override_path = tmp_path / "custom_expansions.json"
        override_path.write_text(json.dumps({"support": ["helpdesk"]}))

        set_query_expansion_path(str(override_path))
        load_query_expansion_dict(force_reload=True, suppress_errors=False)

        expanded = expand_query("support ticket")
        assert "helpdesk" in expanded

    def test_expand_query_uses_env_override(self, tmp_path, monkeypatch):
        """Query expansion honors environment variable override."""
        override_path = tmp_path / "env_expansions.json"
        override_path.write_text(json.dumps({"timer": ["chronometer"]}))

        monkeypatch.setenv(QUERY_EXPANSIONS_ENV_VAR, str(override_path))
        reset_query_expansion_cache()
        set_query_expansion_path(None)

        expanded = expand_query("Start the timer")
        assert "chronometer" in expanded

    def test_invalid_override_raises_when_validation_requested(self, tmp_path):
        """Force reload raises when suppress_errors is False for invalid configs."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json")

        set_query_expansion_path(str(bad_path))

        with pytest.raises(ValueError):
            load_query_expansion_dict(force_reload=True, suppress_errors=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
