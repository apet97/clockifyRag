import json

from clockify_rag import answer
from clockify_rag.answer import generate_llm_answer


def test_generate_llm_answer_filters_invalid_urls(monkeypatch):
    def fake_ask_llm(*args, **kwargs):
        return json.dumps(
            {
                "intent": "feature_howto",
                "user_role_inferred": "admin",
                "security_sensitivity": "low",
                "answer_style": "ticket_reply",
                "short_intent_summary": "Check timesheet view",
                "answer": "Use Timesheet view to see the team.",
                "sources_used": [
                    "https://clockify.me/help/track-time-and-expenses/timesheet-view#anchor",
                    "https://made-up.invalid/source",
                ],
                "needs_human_escalation": False,
            }
        )

    monkeypatch.setattr(answer, "ask_llm", fake_ask_llm)

    chunk = {
        "id": "c1",
        "title": "Timesheet view",
        "url": "https://clockify.me/help/track-time-and-expenses/timesheet-view",
        "text": "Context: Timesheet view\n\ncontent",
    }
    article_blocks = [
        {
            "id": "1",
            "title": "Timesheet view",
            "url": "https://clockify.me/help/track-time-and-expenses/timesheet-view",
            "text": "content",
            "chunk_ids": ["c1"],
        }
    ]

    answer_text, _, _, _, sources_used, _ = generate_llm_answer(
        "Messy ticket about timesheets",
        "ctx",
        packed_ids=["c1"],
        all_chunks=[chunk],
        article_blocks=article_blocks,
    )

    assert "Timesheet view" in answer_text
    assert sources_used == ["https://clockify.me/help/track-time-and-expenses/timesheet-view"]
