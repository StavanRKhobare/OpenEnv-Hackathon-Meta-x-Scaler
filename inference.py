"""Inference script for the Scheduling Optimisation Environment.

Runs an LLM agent (configured via environment variables) against all three
tasks and emits structured [START] / [STEP] / [END] logs for automated
evaluation.

Required environment variables:
    API_BASE_URL  — Base URL for the OpenAI-compatible API endpoint
    MODEL_NAME    — Model identifier (e.g. "gpt-4o-mini")
    HF_TOKEN      — API key / Hugging Face token

Usage:
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

from environment import INSTANCE_BANK, SchedulingOptEnv
from models import Action

# ---------------------------------------------------------------------------
# Configuration — all sourced from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    print(
        "[WARN] HF_TOKEN not set — falling back to oracle mock responses.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# OpenAI client (uses API_BASE_URL + HF_TOKEN as credentials)
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

USE_LLM: bool = bool(HF_TOKEN)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _llm(system: str, user: str) -> str:
    """Call the configured model and return the stripped response text."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[LLM error] {exc}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Oracle mock responses (used when HF_TOKEN is absent)
# ---------------------------------------------------------------------------

_MOCK_FEASIBILITY: dict[int, str] = {
    0: "infeasible", 1: "infeasible", 2: "infeasible", 3: "infeasible",
    4: "infeasible", 5: "infeasible", 6: "infeasible", 7: "infeasible",
    8: "infeasible", 9: "infeasible", 10: "feasible",  11: "feasible",
}

_MOCK_CLASSIFICATION: dict[int, str] = {
    0: "resource_overload",    1: "deadline_violation",
    2: "precedence_violation", 3: "availability_conflict",
    4: "capacity_exceeded",    5: "resource_overload",
    6: "deadline_violation",   7: "precedence_violation",
    8: "availability_conflict",9: "capacity_exceeded",
}


def _mock_repair(idx: int) -> str:
    entry = INSTANCE_BANK[idx]
    sched = entry.get("optimal_schedule") or entry["instance"].get("proposed_schedule", {})
    return json.dumps(sched)


# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------


def log_start(task_id: str, instance_id: int, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"task_id": task_id, "instance_id": instance_id}
    if extra:
        payload.update(extra)
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(
    task_id: str,
    instance_id: int,
    step: int,
    action: str,
    reward: float,
    done: bool,
    feedback: str = "",
) -> None:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "instance_id": instance_id,
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "done": done,
    }
    if feedback:
        payload["feedback"] = feedback
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(task_id: str, instance_id: int, final_reward: float) -> None:
    payload: dict[str, Any] = {
        "task_id": task_id,
        "instance_id": instance_id,
        "final_reward": round(final_reward, 4),
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# Per-task agent logic
# ---------------------------------------------------------------------------


def _agent_feasibility(instance_str: str, instance_idx: int) -> str:
    if not USE_LLM:
        return _MOCK_FEASIBILITY.get(instance_idx, "infeasible")
    return _llm(
        "You are a scheduling expert. Determine if the proposed schedule satisfies "
        "all constraints. Reply with ONLY 'feasible' or 'infeasible'. No extra text.",
        instance_str,
    )


def _agent_classification(instance_str: str, instance_idx: int) -> str:
    if not USE_LLM:
        return _MOCK_CLASSIFICATION.get(instance_idx, "resource_overload")
    return _llm(
        "You are a scheduling expert. Identify the single constraint violation type. "
        "Reply with ONLY one of: resource_overload, deadline_violation, "
        "precedence_violation, availability_conflict, capacity_exceeded. No extra text.",
        instance_str,
    )


def _agent_repair(instance_str: str, instance_idx: int) -> str:
    if not USE_LLM:
        return _mock_repair(instance_idx)
    return _llm(
        'You are a scheduling expert. Repair the infeasible schedule. Return ONLY a '
        'valid JSON object: {"assignments": [{"job_id": "...", "machine_id": "...", '
        '"start_time": <int>}, ...]}. No markdown, no explanation.',
        instance_str,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env: SchedulingOptEnv,
    task_id: str,
    instance_idx: int,
    instance_entry: dict[str, Any],
    max_steps: int,
    agent_fn,
) -> float:
    """Run one episode and return the final reward."""
    obs = env.reset(task_id=task_id)
    instance_str = json.dumps(instance_entry["instance"], indent=2)

    log_start(task_id, instance_idx)

    final_reward = 0.0
    for step_num in range(1, max_steps + 1):
        response = agent_fn(instance_str, instance_idx)
        action = Action(response=response, task_id=task_id)
        obs, reward, done, info = env.step(action)

        feedback = info.get("grading_breakdown", {}).get("feedback", "")
        log_step(task_id, instance_idx, step_num, response, reward, done, feedback)

        final_reward = reward
        if done:
            break

    log_end(task_id, instance_idx, final_reward)
    return final_reward


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main() -> None:
    env = SchedulingOptEnv()
    all_scores: dict[str, list[float]] = {
        "feasibility_check": [],
        "conflict_classification": [],
        "schedule_repair": [],
    }

    mode = f"LLM ({MODEL_NAME} @ {API_BASE_URL})" if USE_LLM else "oracle mock"
    print(json.dumps({"event": "eval_start", "mode": mode, "model": MODEL_NAME}), flush=True)

    # --- Task 1: Feasibility Check (all 12 instances) ---
    for i, entry in enumerate(INSTANCE_BANK):
        score = run_episode(
            env, "feasibility_check", i, entry,
            max_steps=3,
            agent_fn=_agent_feasibility,
        )
        all_scores["feasibility_check"].append(score)

    # --- Task 2: Conflict Classification (10 infeasible instances) ---
    for i, entry in enumerate(INSTANCE_BANK):
        if entry["is_feasible"]:
            continue
        score = run_episode(
            env, "conflict_classification", i, entry,
            max_steps=5,
            agent_fn=_agent_classification,
        )
        all_scores["conflict_classification"].append(score)

    # --- Task 3: Schedule Repair (10 infeasible instances) ---
    for i, entry in enumerate(INSTANCE_BANK):
        if entry["is_feasible"]:
            continue
        score = run_episode(
            env, "schedule_repair", i, entry,
            max_steps=8,
            agent_fn=_agent_repair,
        )
        all_scores["schedule_repair"].append(score)

    # --- Summary ---
    summary: dict[str, Any] = {}
    overall_scores: list[float] = []
    for task_id, scores in all_scores.items():
        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        summary[task_id] = {"average_score": avg, "num_instances": len(scores)}
        overall_scores.extend(scores)

    overall = round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0
    summary["overall_average"] = overall

    print(json.dumps({"event": "eval_end", "summary": summary}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
