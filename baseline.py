"""Baseline inference script for the Scheduling Optimisation Environment.

Runs GPT-4o-mini (or falls back to deterministic mock responses) against all
three tasks and prints a structured score report.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from environment import INSTANCE_BANK, SchedulingOptEnv
from graders.grader_classification import ConflictGrader
from graders.grader_detection import FeasibilityGrader
from graders.grader_fix import RepairGrader
from models import Action


def _get_openai_client():
    """Return an OpenAI client, or None if unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _llm_response(client, system_prompt: str, user_prompt: str) -> str:
    """Call GPT-4o-mini and return the response text."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error: {e}]")
        return ""


# ---------------------------------------------------------------------------
# Mock fallback responses (used when no API key is available)
# ---------------------------------------------------------------------------

# Ground-truth feasibility labels — index aligns with INSTANCE_BANK
_MOCK_FEASIBILITY: dict[int, str] = {
    0: "infeasible", 1: "infeasible", 2: "infeasible", 3: "infeasible",
    4: "infeasible", 5: "infeasible", 6: "infeasible", 7: "infeasible",
    8: "infeasible", 9: "infeasible", 10: "feasible", 11: "feasible",
}

# Ground-truth violation types for infeasible instances
_MOCK_CLASSIFICATION: dict[int, str] = {
    0: "resource_overload",
    1: "deadline_violation",
    2: "precedence_violation",
    3: "availability_conflict",
    4: "capacity_exceeded",
    5: "resource_overload",
    6: "deadline_violation",
    7: "precedence_violation",
    8: "availability_conflict",
    9: "capacity_exceeded",
}


def _mock_repair(instance_idx: int) -> str:
    """Return the known optimal schedule JSON for mock mode."""
    entry = INSTANCE_BANK[instance_idx]
    optimal = entry.get("optimal_schedule", {})
    if not optimal:
        # Return the proposed schedule unchanged as a safe fallback
        optimal = entry["instance"].get("proposed_schedule", {})
    return json.dumps(optimal)


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------


def run_baseline() -> dict[str, Any]:
    """Execute the baseline across all three tasks and return scores."""
    client = _get_openai_client()
    use_llm = client is not None
    mode = "GPT-4o-mini" if use_llm else "mock (no API key — oracle responses)"
    print(f"\n{'='*65}")
    print(f"  SchedulingOptEnv — Baseline Evaluation ({mode})")
    print(f"{'='*65}\n")

    results: dict[str, Any] = {"mode": mode, "tasks": {}}

    # ----- Task 1: Feasibility Check -----
    feas_grader = FeasibilityGrader()
    feas_scores: list[float] = []
    print("Task 1: Feasibility Check (easy)")
    for i, entry in enumerate(INSTANCE_BANK):
        instance_str = json.dumps(entry["instance"], indent=2)
        if use_llm:
            resp = _llm_response(
                client,
                (
                    "You are a scheduling expert. Determine if the proposed schedule "
                    "satisfies all constraints. Reply with ONLY 'feasible' or 'infeasible'."
                ),
                instance_str,
            )
        else:
            resp = _MOCK_FEASIBILITY.get(i, "infeasible")
        action = Action(response=resp, task_id="feasibility_check")
        score = feas_grader.grade(action, entry)
        feas_scores.append(score)
        status = "CORRECT" if score >= 0.95 else "wrong"
        expected = "feasible" if entry["is_feasible"] else "infeasible"
        print(
            f"  Instance {i:2d}: {status:7s} (score={score:.2f})  "
            f"expected={expected}  [{entry['description'][:45]}]"
        )

    avg_feas = sum(feas_scores) / len(feas_scores) if feas_scores else 0.0
    results["tasks"]["feasibility_check"] = {
        "average_score": round(avg_feas, 4),
        "num_instances": len(feas_scores),
        "scores": feas_scores,
    }
    print(f"  >> Average: {avg_feas:.3f}\n")

    # ----- Task 2: Conflict Classification -----
    conf_grader = ConflictGrader()
    conf_scores: list[float] = []
    infeasible_entries = [(i, e) for i, e in enumerate(INSTANCE_BANK) if not e["is_feasible"]]
    print("Task 2: Conflict Classification (medium)")
    for i, entry in infeasible_entries:
        instance_str = json.dumps(entry["instance"], indent=2)
        if use_llm:
            resp = _llm_response(
                client,
                (
                    "You are a scheduling expert. Identify the constraint violation type. "
                    "Reply with ONLY one of: resource_overload, deadline_violation, "
                    "precedence_violation, availability_conflict, capacity_exceeded."
                ),
                instance_str,
            )
        else:
            resp = _MOCK_CLASSIFICATION.get(i, "resource_overload")
        action = Action(response=resp, task_id="conflict_classification")
        score = conf_grader.grade(action, entry)
        conf_scores.append(score)
        status = "EXACT" if score >= 0.95 else ("partial" if score >= 0.45 else "wrong")
        print(
            f"  Instance {i:2d}: {status:7s} (score={score:.2f})  "
            f"expected={entry['violation_type']}"
        )

    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
    results["tasks"]["conflict_classification"] = {
        "average_score": round(avg_conf, 4),
        "num_instances": len(conf_scores),
        "scores": conf_scores,
    }
    print(f"  >> Average: {avg_conf:.3f}\n")

    # ----- Task 3: Schedule Repair -----
    repair_grader = RepairGrader()
    repair_scores: list[float] = []
    repairable = [
        (i, e) for i, e in enumerate(INSTANCE_BANK)
        if not e["is_feasible"] and e.get("optimal_schedule")
    ]
    print("Task 3: Schedule Repair (hard)")
    for i, entry in repairable:
        instance_str = json.dumps(entry["instance"], indent=2)
        if use_llm:
            resp = _llm_response(
                client,
                (
                    "You are a scheduling expert. Repair the infeasible schedule by "
                    "returning a JSON object with key 'assignments': a list of "
                    '{"job_id", "machine_id", "start_time"} dicts that satisfies all '
                    "constraints and minimises makespan. Return ONLY valid JSON."
                ),
                instance_str,
            )
        else:
            resp = _mock_repair(i)
        action = Action(response=resp, task_id="schedule_repair")
        score = repair_grader.grade(action, entry)
        repair_scores.append(score)
        print(
            f"  Instance {i:2d}: score={score:.2f}  "
            f"optimal_makespan={entry['optimal_makespan']}  "
            f"[{entry['description'][:45]}]"
        )

    avg_repair = sum(repair_scores) / len(repair_scores) if repair_scores else 0.0
    results["tasks"]["schedule_repair"] = {
        "average_score": round(avg_repair, 4),
        "num_instances": len(repair_scores),
        "scores": repair_scores,
    }
    print(f"  >> Average: {avg_repair:.3f}\n")

    # ----- Summary -----
    overall = (avg_feas + avg_conf + avg_repair) / 3
    results["overall_average"] = round(overall, 4)
    print(f"{'='*65}")
    print(f"  Overall Average Score: {overall:.3f}")
    print(f"{'='*65}\n")

    return results


if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as e:
        print(f"Baseline failed: {e}", file=sys.stderr)
        sys.exit(1)
