"""Tests for role/security hint heuristics."""

from clockify_rag.retrieval import derive_role_security_hints


def test_admin_screenshots_hint_detected():
    """Admin + screenshots wording should trigger admin/high hints."""
    role_hint, security_hint = derive_role_security_hints("i cant see my teams screenshots and im an admin")
    assert role_hint == "admin"
    assert security_hint == "high"


def test_unknown_role_and_security_defaults():
    """Generic tickets should default to unknown/unknown."""
    role_hint, security_hint = derive_role_security_hints("i cant stop a time entry")
    assert role_hint == "unknown"
    assert security_hint == "unknown"
