import json

from clockify_rag.answer import answer_to_json


def test_answer_to_json_uses_measured_token_budget():
    measured_budget = 321
    citations = ["doc-1", "doc-2", "doc-3"]

    payload = answer_to_json(
        answer="Example answer",
        citations=citations,
        used_tokens=measured_budget,
        topk=12,
        packed=6,
    )

    # Ensure field exists and matches measured budget, not citation count
    assert payload["debug"]["meta"]["used_tokens"] == measured_budget
    assert payload["citations"] == citations

    # JSON serialization should preserve the measured budget value
    serialized = json.dumps(payload)
    assert str(measured_budget) in serialized
