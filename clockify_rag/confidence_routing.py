"""Confidence-based routing for query escalation.

OPTIMIZATION (Analysis Section 9.1 #4): Auto-escalate low-confidence queries to human review.
This module provides confidence-based routing to reduce hallucination impact.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories for routing decisions."""

    HIGH = "high"  # 80-100: Auto-approve
    GOOD = "good"  # 60-79: Review if critical
    MEDIUM = "medium"  # 40-59: Manual review
    LOW = "low"  # 0-39: Escalate to human
    REFUSED = "refused"  # No answer provided


# Confidence thresholds from Analysis Section 4.2
CONFIDENCE_HIGH = 80
CONFIDENCE_GOOD = 60
CONFIDENCE_MEDIUM = 40
CONFIDENCE_ESCALATE = 40  # Escalate below this threshold


def classify_confidence(confidence: Optional[int], refused: bool = False) -> ConfidenceLevel:
    """Classify confidence score into level categories.

    Args:
        confidence: Confidence score (0-100) or None
        refused: Whether the answer was refused

    Returns:
        ConfidenceLevel enum value
    """
    if refused:
        return ConfidenceLevel.REFUSED

    if confidence is None:
        # No confidence score, treat as medium (requires review)
        return ConfidenceLevel.MEDIUM

    if confidence >= CONFIDENCE_HIGH:
        return ConfidenceLevel.HIGH
    elif confidence >= CONFIDENCE_GOOD:
        return ConfidenceLevel.GOOD
    elif confidence >= CONFIDENCE_MEDIUM:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


def should_escalate(confidence: Optional[int], refused: bool = False, critical: bool = False) -> bool:
    """Determine if query should be escalated to human review.

    Args:
        confidence: Confidence score (0-100) or None
        refused: Whether the answer was refused
        critical: Whether the query is marked as critical (lowers threshold)

    Returns:
        True if query should be escalated to human
    """
    # Always escalate refusals
    if refused:
        return True

    # No confidence score - escalate if critical
    if confidence is None:
        return critical

    # Escalate low confidence
    if confidence < CONFIDENCE_ESCALATE:
        return True

    # For critical queries, escalate medium confidence too
    if critical and confidence < CONFIDENCE_GOOD:
        return True

    return False


def get_routing_action(confidence: Optional[int], refused: bool = False, critical: bool = False) -> Dict[str, Any]:
    """Get recommended routing action based on confidence.

    Args:
        confidence: Confidence score (0-100) or None
        refused: Whether the answer was refused
        critical: Whether the query is marked as critical

    Returns:
        Dict with routing metadata:
        - action: "auto_approve", "review", "escalate"
        - level: ConfidenceLevel enum value
        - reason: Human-readable explanation
    """
    level = classify_confidence(confidence, refused)
    escalate = should_escalate(confidence, refused, critical)

    if escalate:
        action = "escalate"
        if refused:
            reason = "Answer was refused (insufficient information in KB)"
        elif confidence is None:
            reason = "No confidence score provided"
        elif critical:
            reason = f"Critical query with {'medium' if confidence >= CONFIDENCE_MEDIUM else 'low'} confidence ({confidence})"
        else:
            reason = f"Low confidence score ({confidence})"
    elif level == ConfidenceLevel.HIGH:
        action = "auto_approve"
        reason = f"High confidence ({confidence})"
    elif level == ConfidenceLevel.GOOD:
        if critical:
            action = "review"
            reason = f"Critical query requires review despite good confidence ({confidence})"
        else:
            action = "auto_approve"
            reason = f"Good confidence ({confidence})"
    else:  # MEDIUM
        action = "review"
        reason = f"Medium confidence ({confidence}) requires review"

    return {"action": action, "level": level.value, "confidence": confidence, "reason": reason, "escalated": escalate}


def log_routing_decision(question: str, routing: Dict[str, Any]) -> None:
    """Log routing decision for monitoring and analysis.

    Args:
        question: User question
        routing: Routing metadata from get_routing_action()
    """
    import json

    log_entry = {
        "event": "confidence_routing",
        "action": routing["action"],
        "level": routing["level"],
        "confidence": routing.get("confidence"),
        "reason": routing["reason"],
        "question_preview": question[:100],  # First 100 chars for context
    }

    logger.info(json.dumps(log_entry))


__all__ = [
    "ConfidenceLevel",
    "classify_confidence",
    "should_escalate",
    "get_routing_action",
    "log_routing_decision",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_GOOD",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_ESCALATE",
]
