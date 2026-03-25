"""Grader for Task 2 — Conflict Classification (medium).

Scoring:
    1.0  — exact match with the ground-truth violation type
    0.5  — related category (same constraint family)
    0.1  — valid category but unrelated to the actual violation
    0.0  — empty or unknown response
"""

from __future__ import annotations

from typing import Any

from models import Action

# Related-category groups for partial credit.
# resource_overload and capacity_exceeded both concern resource limits.
# deadline_violation and precedence_violation both concern temporal ordering.
_RELATED_GROUPS: list[set[str]] = [
    {"resource_overload", "capacity_exceeded"},   # resource-limit family
    {"deadline_violation", "precedence_violation"}, # temporal-ordering family
    {"availability_conflict"},                      # standalone
]

VALID_CATEGORIES = {
    "resource_overload",
    "deadline_violation",
    "precedence_violation",
    "availability_conflict",
    "capacity_exceeded",
}


class ConflictGrader:
    """Grade the agent's constraint-violation classification."""

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response = (
            action.response.strip().lower().replace(" ", "_").replace("-", "_")
        )
        expected: str = ground_truth.get("violation_type") or ""

        if not response:
            return 0.0

        # Exact match → full reward
        if response == expected:
            return 1.0

        # Not a recognised category → no credit
        if response not in VALID_CATEGORIES:
            return 0.0

        # Related-category partial credit (same constraint family)
        for group in _RELATED_GROUPS:
            if response in group and expected in group:
                return 0.5

        # Valid category, but from a different constraint family
        return 0.1
