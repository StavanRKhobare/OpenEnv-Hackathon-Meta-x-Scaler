"""Task 3 — Schedule Repair (Hard).

The agent observes an infeasible scheduling instance and must return a
corrected schedule (JSON) that:
    (a) is valid JSON with the required schema             — 0.4 pts
    (b) satisfies all scheduling constraints               — 0.4 pts
    (c) achieves a makespan within 30% of the known optimal— 0.2 pts

Partial progress: parseable JSON earns 0.2 base reward per step.
Max steps per episode: 8.
Expected agent accuracy: ~30%.
"""

from __future__ import annotations

from typing import Any

from environment import INSTANCE_BANK, SchedulingOptEnv
from models import Action

TASK_ID = "schedule_repair"
MAX_STEPS = 8
DIFFICULTY = "hard"


def run_episode(env: SchedulingOptEnv, agent_fn: Any) -> dict[str, Any]:
    """Run a single schedule-repair episode.

    Args:
        env: An initialized SchedulingOptEnv instance.
        agent_fn: Callable receiving an Observation, returning a JSON schedule string.

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


def get_repairable_instances() -> list[dict[str, Any]]:
    """Return instances that are infeasible and have known optimal schedules."""
    return [
        {
            "instance": entry["instance"],
            "optimal_schedule": entry["optimal_schedule"],
            "optimal_makespan": entry["optimal_makespan"],
            "violation_type": entry["violation_type"],
            "description": entry["description"],
        }
        for entry in INSTANCE_BANK
        if not entry["is_feasible"] and entry.get("optimal_schedule")
    ]
