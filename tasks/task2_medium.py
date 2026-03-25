"""Task 2 — Conflict Classification (Medium).

The agent observes an infeasible scheduling instance and must identify the
constraint violation type from the closed vocabulary:
    resource_overload, deadline_violation, precedence_violation,
    availability_conflict, capacity_exceeded

Grading:
    1.0  — exact match
    0.5  — related category (same constraint family)
    0.1  — valid category but wrong family
    0.0  — empty or unknown
Max steps per episode: 5.
Expected agent accuracy: ~60%.
"""

from __future__ import annotations

from typing import Any

from environment import INSTANCE_BANK, SchedulingOptEnv
from models import Action

TASK_ID = "conflict_classification"
MAX_STEPS = 5
DIFFICULTY = "medium"


def run_episode(env: SchedulingOptEnv, agent_fn: Any) -> dict[str, Any]:
    """Run a single conflict-classification episode.

    Args:
        env: An initialized SchedulingOptEnv instance.
        agent_fn: Callable receiving an Observation, returning a violation-type string.

    Returns:
        Episode summary dict.
    """
    obs = env.reset(task_id=TASK_ID)
    total_reward = 0.0
    steps = 0
    info: dict[str, Any] = {}

    for _ in range(MAX_STEPS):
        response = agent_fn(obs)
        action = Action(response=response, task_id=TASK_ID)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "task": TASK_ID,
        "difficulty": DIFFICULTY,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "info": info,
    }


def get_infeasible_instances() -> list[dict[str, Any]]:
    """Return only instances that have violations (for classification task)."""
    return [
        {
            "instance": entry["instance"],
            "violation_type": entry["violation_type"],
            "description": entry["description"],
        }
        for entry in INSTANCE_BANK
        if not entry["is_feasible"]
    ]
