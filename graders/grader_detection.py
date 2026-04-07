"""Grader for Task 1 — Feasibility Check (easy).

Scoring
-------
    1.0  — agent correctly identifies feasible / infeasible
    0.1  — agent responded but the answer was wrong (non-zero signal)
    0.0  — empty or completely unparseable response

The grader normalises common synonyms so agents that say "valid" instead of
"feasible" still receive full credit.

After each call, ``last_breakdown`` holds a dict describing the grading
decision; this is surfaced in the environment's ``info`` dict so training
loops can inspect the decision without parsing the float reward.
"""

from __future__ import annotations

from typing import Any

from models import Action

# Words treated as equivalent to "feasible"
_FEASIBLE_WORDS: frozenset[str] = frozenset(
    {"feasible", "valid", "correct", "satisfiable", "yes", "ok", "pass"}
)

# Words treated as equivalent to "infeasible"
_INFEASIBLE_WORDS: frozenset[str] = frozenset(
    {
        "infeasible", "invalid", "incorrect", "unsatisfiable", "no",
        "violated", "conflict", "fail", "impossible", "broken",
    }
)


class FeasibilityGrader:
    """Grade whether the agent correctly determined schedule feasibility."""

    def __init__(self) -> None:
        # Populated after each call to grade(); surfaced in env info dict.
        self.last_breakdown: dict[str, Any] = {}

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response: str = action.response.strip().lower()
        is_feasible: bool = ground_truth.get("is_feasible", False)
        expected: str = "feasible" if is_feasible else "infeasible"

        # Empty response → no signal
        if not response:
            self.last_breakdown = {
                "predicted": "",
                "expected": expected,
                "correct": False,
                "feedback": "Empty response — reply with 'feasible' or 'infeasible'.",
            }
            return 0.01

        # Normalise response to canonical form
        if response in _FEASIBLE_WORDS:
            predicted = "feasible"
        elif response in _INFEASIBLE_WORDS:
            predicted = "infeasible"
        else:
            # Recognisable attempt but could not be parsed cleanly
            self.last_breakdown = {
                "predicted": response,
                "expected": expected,
                "correct": False,
                "feedback": (
                    f"Could not parse '{response}'. "
                    "Use exactly 'feasible' or 'infeasible'."
                ),
            }
            return 0.1

        correct = predicted == expected
        self.last_breakdown = {
            "predicted": predicted,
            "expected": expected,
            "correct": correct,
            "feedback": (
                "Correct."
                if correct
                else f"Wrong — the schedule is {expected}, not {predicted}."
            ),
        }
        # Exact match → 0.99; wrong normalised answer → 0.1 (keeps gradient signal)
        return 0.99 if correct else 0.1
