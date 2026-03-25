"""Task 1 — Feasibility Check (Easy).

The agent observes a scheduling instance (jobs, machines, proposed assignments)
and must respond with "feasible" or "infeasible" to indicate whether all
scheduling constraints are satisfied.

Grading: exact match — 1.0 if correct, 0.1 if wrong, 0.0 if empty.
Max steps per episode: 3.
Expected agent accuracy: ~90%.
"""

from __future__ import annotations

from typing import Any

from environment import INSTANCE_BANK, SchedulingOptEnv
from graders.grader_detection import FeasibilityGrader
from models import Action

TASK_ID = "feasibility_check"
MAX_STEPS = 3
DIFFICULTY = "easy"


def run_episode(env: SchedulingOptEnv, agent_fn: Any) -> dict[str, Any]:
    """Run a single feasibility-check episode.

    Args:
        env: An initialized SchedulingOptEnv instance.
        agent_fn: A callable that receives an Observation and returns a
                  response string ("feasible" or "infeasible").

    Returns:
        Episode summary dict with total reward and step count.
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


def get_all_instances_with_answers() -> list[dict[str, Any]]:
    """Return instance bank entries relevant to feasibility check."""
    return [
        {
            "instance": entry["instance"],
            "is_feasible": entry["is_feasible"],
            "description": entry["description"],
        }
        for entry in INSTANCE_BANK
    ]
