"""Tests for answer.py JSON parsing and type coercion."""

import json

from clockify_rag.answer import parse_qwen_json, extract_citations, validate_citations


class TestParseQwenJsonConfidence:
    """Tests for confidence parsing robustness."""

    def test_confidence_int(self):
        """Should parse integer confidence."""
        data = json.dumps({"answer": "test", "confidence": 85})
        result = parse_qwen_json(data)
        assert result["confidence"] == 85

    def test_confidence_float(self):
        """Should round float confidence to int."""
        data = json.dumps({"answer": "test", "confidence": 85.7})
        result = parse_qwen_json(data)
        assert result["confidence"] == 86  # rounded

    def test_confidence_float_edge_low(self):
        """Should handle low float values."""
        data = json.dumps({"answer": "test", "confidence": 0.4})
        result = parse_qwen_json(data)
        assert result["confidence"] == 0  # rounded down

    def test_confidence_float_edge_high(self):
        """Should clamp high values to 100."""
        data = json.dumps({"answer": "test", "confidence": 99.6})
        result = parse_qwen_json(data)
        assert result["confidence"] == 100  # rounded up

    def test_confidence_string_number(self):
        """Should parse string-encoded numbers."""
        data = json.dumps({"answer": "test", "confidence": "75"})
        result = parse_qwen_json(data)
        assert result["confidence"] == 75

    def test_confidence_string_float(self):
        """Should parse string-encoded floats."""
        data = json.dumps({"answer": "test", "confidence": "75.5"})
        result = parse_qwen_json(data)
        assert result["confidence"] == 76  # rounded

    def test_confidence_out_of_range_high(self):
        """Should reject confidence above 100."""
        data = json.dumps({"answer": "test", "confidence": 150})
        result = parse_qwen_json(data)
        assert result["confidence"] is None

    def test_confidence_out_of_range_negative(self):
        """Should reject negative confidence."""
        data = json.dumps({"answer": "test", "confidence": -5})
        result = parse_qwen_json(data)
        assert result["confidence"] is None

    def test_confidence_invalid_string(self):
        """Should handle non-numeric strings gracefully."""
        data = json.dumps({"answer": "test", "confidence": "high"})
        result = parse_qwen_json(data)
        assert result["confidence"] is None

    def test_confidence_none(self):
        """Should handle null confidence."""
        data = json.dumps({"answer": "test", "confidence": None})
        result = parse_qwen_json(data)
        assert result["confidence"] is None

    def test_confidence_missing(self):
        """Should handle missing confidence."""
        data = json.dumps({"answer": "test"})
        result = parse_qwen_json(data)
        assert result["confidence"] is None

    def test_confidence_bool_false(self):
        """Should handle boolean false gracefully."""
        data = json.dumps({"answer": "test", "confidence": False})
        result = parse_qwen_json(data)
        # False converts to 0.0 which is valid
        assert result["confidence"] == 0

    def test_confidence_bool_true(self):
        """Should handle boolean true gracefully."""
        data = json.dumps({"answer": "test", "confidence": True})
        result = parse_qwen_json(data)
        # True converts to 1.0 which is valid
        assert result["confidence"] == 1

    def test_confidence_list(self):
        """Should reject list as confidence."""
        data = json.dumps({"answer": "test", "confidence": [85]})
        result = parse_qwen_json(data)
        assert result["confidence"] is None

    def test_confidence_dict(self):
        """Should reject dict as confidence."""
        data = json.dumps({"answer": "test", "confidence": {"value": 85}})
        result = parse_qwen_json(data)
        assert result["confidence"] is None


class TestParseQwenJsonIntent:
    """Tests for intent parsing."""

    def test_valid_intent(self):
        """Should parse valid intent."""
        data = json.dumps({"answer": "test", "intent": "feature_howto"})
        result = parse_qwen_json(data)
        assert result["intent"] == "feature_howto"

    def test_invalid_intent_defaults(self):
        """Should default to 'other' for invalid intent."""
        data = json.dumps({"answer": "test", "intent": "invalid_intent"})
        result = parse_qwen_json(data)
        assert result["intent"] == "other"

    def test_intent_case_insensitive(self):
        """Should normalize intent to lowercase."""
        data = json.dumps({"answer": "test", "intent": "BILLING"})
        result = parse_qwen_json(data)
        assert result["intent"] == "billing"


class TestParseQwenJsonSourcesUsed:
    """Tests for sources_used parsing."""

    def test_sources_string_list(self):
        """Should parse list of strings."""
        data = json.dumps({"answer": "test", "sources_used": ["url1", "url2"]})
        result = parse_qwen_json(data)
        assert result["sources_used"] == ["url1", "url2"]

    def test_sources_mixed_types(self):
        """Should coerce mixed types to strings."""
        data = json.dumps({"answer": "test", "sources_used": ["url1", 123, 4.5]})
        result = parse_qwen_json(data)
        assert result["sources_used"] == ["url1", "123", "4.5"]

    def test_sources_empty_strings_filtered(self):
        """Should filter empty strings."""
        data = json.dumps({"answer": "test", "sources_used": ["url1", "", "  ", "url2"]})
        result = parse_qwen_json(data)
        assert result["sources_used"] == ["url1", "url2"]

    def test_sources_not_list(self):
        """Should return empty list for non-list input."""
        data = json.dumps({"answer": "test", "sources_used": "single_url"})
        result = parse_qwen_json(data)
        assert result["sources_used"] == []

    def test_sources_fallback_to_sources(self):
        """Should fallback to 'sources' key if sources_used empty."""
        data = json.dumps({"answer": "test", "sources": ["url1", "url2"]})
        result = parse_qwen_json(data)
        assert result["sources_used"] == ["url1", "url2"]


class TestParseQwenJsonCodeFence:
    """Tests for code fence handling."""

    def test_strips_code_fence(self):
        """Should strip markdown code fences."""
        data = '```json\n{"answer": "test", "confidence": 80}\n```'
        result = parse_qwen_json(data)
        assert result["answer"] == "test"
        assert result["confidence"] == 80

    def test_strips_code_fence_no_lang(self):
        """Should strip code fence without language marker."""
        data = '```\n{"answer": "test"}\n```'
        result = parse_qwen_json(data)
        assert result["answer"] == "test"


class TestExtractCitations:
    """Tests for citation extraction."""

    def test_single_citation(self):
        """Should extract single citation."""
        text = "This is from [chunk_123]."
        citations = extract_citations(text)
        assert "chunk_123" in citations

    def test_multiple_citations(self):
        """Should extract multiple citations."""
        text = "See [chunk_1] and [chunk_2]."
        citations = extract_citations(text)
        assert "chunk_1" in citations
        assert "chunk_2" in citations

    def test_comma_separated_citations(self):
        """Should extract comma-separated citations."""
        text = "See [chunk_1, chunk_2, chunk_3]."
        citations = extract_citations(text)
        assert "chunk_1" in citations
        assert "chunk_2" in citations
        assert "chunk_3" in citations

    def test_hyphenated_ids(self):
        """Should extract hyphenated IDs."""
        text = "Reference: [abc-123-def]"
        citations = extract_citations(text)
        assert "abc-123-def" in citations

    def test_no_citations(self):
        """Should return empty list when no citations."""
        text = "No citations here."
        citations = extract_citations(text)
        assert citations == []


class TestValidateCitations:
    """Tests for citation validation."""

    def test_all_valid(self):
        """Should validate when all citations are valid."""
        answer = "See [chunk_1] and [chunk_2]."
        valid_ids = ["chunk_1", "chunk_2", "chunk_3"]
        is_valid, valid, invalid = validate_citations(answer, valid_ids)
        assert is_valid is True
        assert set(valid) == {"chunk_1", "chunk_2"}
        assert invalid == []

    def test_some_invalid(self):
        """Should detect invalid citations."""
        answer = "See [chunk_1] and [fake_chunk]."
        valid_ids = ["chunk_1", "chunk_2"]
        is_valid, valid, invalid = validate_citations(answer, valid_ids)
        assert is_valid is False
        assert "chunk_1" in valid
        assert "fake_chunk" in invalid

    def test_handles_int_ids(self):
        """Should handle integer IDs in valid list."""
        answer = "See [123]."
        valid_ids = [123, 456]
        is_valid, valid, invalid = validate_citations(answer, valid_ids)
        assert is_valid is True
        assert "123" in valid
