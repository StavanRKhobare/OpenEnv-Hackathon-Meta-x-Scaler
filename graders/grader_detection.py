"""Grader for Task 1 — Feasibility Check (easy).

Scoring:
    1.0  — exact match: agent correctly identifies feasible/infeasible
    0.1  — agent responded but answer is wrong (partial signal)
    0.0  — empty or unparseable response
"""

from __future__ import annotations

from typing import Any

from models import Action


class FeasibilityGrader:
    """Grade whether the agent correctly determined schedule feasibility."""

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response = action.response.strip().lower()
        is_feasible: bool = ground_truth.get("is_feasible", False)
        expected = "feasible" if is_feasible else "infeasible"

        if not response:
            return 0.0

        # Normalise common variants the agent might produce
        feasible_words = {"feasible", "valid", "correct", "satisfiable", "yes"}
        infeasible_words = {
            "infeasible", "invalid", "incorrect", "unsatisfiable", "no",
            "violated", "conflict", "impossible",
        }

        if response in feasible_words:
            answer = "feasible"
        elif response in infeasible_words:
            answer = "infeasible"
        else:
            # Agent responded but we could not parse it — small partial signal
            return 0.1

        # Exact match → full score; wrong answer → small partial signal
        return 1.0 if answer == expected else 0.1
