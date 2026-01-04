"""Tests for Qwen JSON output contract validation."""

import json
import pytest

from clockify_rag.answer import parse_qwen_json


def test_parse_valid_qwen_json_ticket_contract():
    """Parse a well-formed ticket response with the new schema."""
    payload = {
        "intent": "troubleshooting",
        "user_role_inferred": "admin",
        "security_sensitivity": "high",
        "answer_style": "ticket_reply",
        "short_intent_summary": "User cannot see screenshots",
        "answer": "Please enable screenshots and check the retention policy.",
        "sources_used": ["https://example.com/help/screenshots"],
        "needs_human_escalation": False,
        "confidence": 92,
        "reasoning": "Based on the screenshots article.",
    }

    result = parse_qwen_json(json.dumps(payload))

    assert result["intent"] == "troubleshooting"
    assert result["user_role_inferred"] == "admin"
    assert result["security_sensitivity"] == "high"
    assert result["answer_style"] == "ticket_reply"
    assert result["short_intent_summary"] == "User cannot see screenshots"
    assert result["needs_human_escalation"] is False
    assert result["sources_used"] == ["https://example.com/help/screenshots"]
    assert result["confidence"] == 92
    assert "enable screenshots" in result["answer"]


def test_parse_defaults_when_fields_missing():
    """Missing optional fields should fall back to safe defaults."""
    minimal = {"answer": "No context found."}
    parsed = parse_qwen_json(json.dumps(minimal))

    assert parsed["intent"] == "other"
    assert parsed["user_role_inferred"] == "unknown"
    assert parsed["security_sensitivity"] == "medium"
    assert parsed["answer_style"] == "ticket_reply"
    assert parsed["short_intent_summary"] == ""
    assert parsed["needs_human_escalation"] is False
    assert parsed["sources_used"] == []
    assert parsed["confidence"] is None


def test_parse_coerces_sources_and_confidence():
    """Mixed-type sources and float confidence should be coerced safely."""
    payload = {
        "answer": "Use the timer",
        "sources_used": [1, " https://clockify.me/help/timer "],
        "confidence": 85.4,
    }
    parsed = parse_qwen_json(json.dumps(payload))

    assert parsed["sources_used"] == ["1", "https://clockify.me/help/timer"]
    assert parsed["confidence"] == 85


def test_confidence_out_of_range_returns_none():
    """Confidence outside 0-100 is ignored."""
    payload = {"answer": "Hi", "confidence": 150}
    parsed = parse_qwen_json(json.dumps(payload))
    assert parsed["confidence"] is None


def test_parse_invalid_json_raises_error():
    """Malformed JSON raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        parse_qwen_json("{answer: 'missing quotes'}")


def test_parse_non_dict_json_raises_error():
    """Non-object JSON raises ValueError."""
    with pytest.raises(ValueError):
        parse_qwen_json(json.dumps(["answer", "intent"]))
