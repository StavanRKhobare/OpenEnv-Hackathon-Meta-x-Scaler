"""Core Scheduling Optimisation Environment implementing the OpenEnv API contract."""

from __future__ import annotations

import copy
import json
from typing import Any

from models import Action, Observation

# ---------------------------------------------------------------------------
# Scheduling instance bank — 12 diverse instances with ground-truth metadata.
#
# Each entry carries:
#   instance        – dict, the scheduling problem (jobs + machines + proposed_schedule)
#   is_feasible     – bool, for Task 1 (feasibility check)
#   violation_type  – str | None, for Task 2 (conflict classification)
#   optimal_schedule – dict, the corrected schedule for Task 3 (schedule repair)
#   optimal_makespan – int, the minimum achievable makespan
#   description     – human-readable summary of the issue
# ---------------------------------------------------------------------------

INSTANCE_BANK: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ #
    # 0 — resource_overload: two jobs overlap on a single-capacity machine  #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P01",
            "jobs": [
                {"id": "J1", "duration": 4, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 2, "deadline": 20, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 2},
                    {"job_id": "J3", "machine_id": "M2", "start_time": 0},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "resource_overload",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M1", "start_time": 4},
                {"job_id": "J3", "machine_id": "M2", "start_time": 0},
            ]
        },
        "optimal_makespan": 7,
        "description": "J1[0,4) and J2[2,5) overlap on M1 (capacity=1) → resource_overload.",
    },
    # ------------------------------------------------------------------ #
    # 1 — deadline_violation: job finishes after its hard deadline          #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P02",
            "jobs": [
                {"id": "J1", "duration": 5, "deadline": 8, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 5},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "deadline_violation",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M1", "start_time": 5},
            ]
        },
        "optimal_makespan": 8,
        "description": "J1 starts at t=5 and finishes at t=10, violating deadline=8.",
    },
    # ------------------------------------------------------------------ #
    # 2 — precedence_violation: dependent job starts before its predecessor #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P03",
            "jobs": [
                {"id": "J1", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 3, "deadline": 20, "dependencies": ["J1"], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 5},
                    {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "precedence_violation",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M2", "start_time": 3},
            ]
        },
        "optimal_makespan": 6,
        "description": "J2 depends on J1; J2 starts at t=0 but J1 does not finish until t=8.",
    },
    # ------------------------------------------------------------------ #
    # 3 — availability_conflict: job scheduled outside machine hours        #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P04",
            "jobs": [
                {"id": "J1", "duration": 4, "deadline": 24, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 3, "deadline": 24, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 8, "available_end": 18},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 5},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 9},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "availability_conflict",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 8},
                {"job_id": "J2", "machine_id": "M1", "start_time": 12},
            ]
        },
        "optimal_makespan": 15,
        "description": "J1 starts at t=5, before M1's available window [8,18] → availability_conflict.",
    },
    # ------------------------------------------------------------------ #
    # 4 — capacity_exceeded: more concurrent jobs than machine capacity     #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P05",
            "jobs": [
                {"id": "J1", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 2, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J3", "machine_id": "M1", "start_time": 0},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "capacity_exceeded",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                {"job_id": "J3", "machine_id": "M1", "start_time": 3},
            ]
        },
        "optimal_makespan": 6,
        "description": "3 jobs start simultaneously on M1 (capacity=2); concurrent load=3 > 2.",
    },
    # ------------------------------------------------------------------ #
    # 5 — resource_overload: three-job pairwise overlap on one machine      #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P06",
            "jobs": [
                {"id": "J1", "duration": 5, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 4, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 2, "deadline": 20, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 1},
                    {"job_id": "J3", "machine_id": "M1", "start_time": 8},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "resource_overload",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M1", "start_time": 5},
                {"job_id": "J3", "machine_id": "M1", "start_time": 9},
            ]
        },
        "optimal_makespan": 11,
        "description": "J1[0,5) and J2[1,5) overlap on M1 (capacity=1) → resource_overload.",
    },
    # ------------------------------------------------------------------ #
    # 6 — deadline_violation: precedence chain forces late finish           #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P07",
            "jobs": [
                {"id": "J1", "duration": 4, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 5, "deadline": 20, "dependencies": ["J1"], "resource_req": 1},
                {"id": "J3", "duration": 4, "deadline": 12, "dependencies": ["J2"], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M3", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M2", "start_time": 4},
                    {"job_id": "J3", "machine_id": "M3", "start_time": 9},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "deadline_violation",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M2", "start_time": 4},
                {"job_id": "J3", "machine_id": "M3", "start_time": 9},
            ]
        },
        "optimal_makespan": 13,
        "description": "J3 finishes at t=13, violating its hard deadline of t=12.",
    },
    # ------------------------------------------------------------------ #
    # 7 — precedence_violation: job with two predecessors starts too early  #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P08",
            "jobs": [
                {"id": "J1", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 4, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 2, "deadline": 20, "dependencies": ["J1", "J2"], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M3", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                    {"job_id": "J3", "machine_id": "M3", "start_time": 2},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "precedence_violation",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                {"job_id": "J3", "machine_id": "M3", "start_time": 4},
            ]
        },
        "optimal_makespan": 6,
        "description": "J3 depends on J1 and J2; J3 starts at t=2 but J2 does not finish until t=4.",
    },
    # ------------------------------------------------------------------ #
    # 8 — availability_conflict: job overlaps maintenance window            #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P09",
            "jobs": [
                {"id": "J1", "duration": 3, "deadline": 24, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 2, "deadline": 24, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {
                    "id": "M1",
                    "capacity": 1,
                    "available_start": 0,
                    "available_end": 10,
                    "note": "M1 under maintenance [10, 15]; available again from t=15",
                },
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 9},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "availability_conflict",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                {"job_id": "J1", "machine_id": "M1", "start_time": 2},
            ]
        },
        "optimal_makespan": 5,
        "description": "J1 starts at t=9, extends into maintenance window [10,15] → availability_conflict.",
    },
    # ------------------------------------------------------------------ #
    # 9 — capacity_exceeded: four jobs on capacity-3 machine simultaneously #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P10",
            "jobs": [
                {"id": "J1", "duration": 2, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 2, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 2, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J4", "duration": 2, "deadline": 20, "dependencies": [], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 3, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J3", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J4", "machine_id": "M1", "start_time": 0},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "capacity_exceeded",
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M1", "start_time": 0},
                {"job_id": "J3", "machine_id": "M1", "start_time": 0},
                {"job_id": "J4", "machine_id": "M1", "start_time": 2},
            ]
        },
        "optimal_makespan": 4,
        "description": "4 jobs start simultaneously on M1 (capacity=3); concurrent load=4 > 3.",
    },
    # ------------------------------------------------------------------ #
    # 10 — feasible: 3-job, 2-machine schedule satisfying all constraints   #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P11",
            "jobs": [
                {"id": "J1", "duration": 4, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 3, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 5, "deadline": 20, "dependencies": ["J1"], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                    {"job_id": "J3", "machine_id": "M2", "start_time": 4},
                ]
            },
        },
        "is_feasible": True,
        "violation_type": None,
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                {"job_id": "J3", "machine_id": "M2", "start_time": 4},
            ]
        },
        "optimal_makespan": 9,
        "description": "Fully feasible 3-job schedule — all constraints satisfied.",
    },
    # ------------------------------------------------------------------ #
    # 11 — feasible: 5-job, 3-machine schedule with precedence and deadlines#
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P12",
            "jobs": [
                {"id": "J1", "duration": 3, "deadline": 30, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 2, "deadline": 30, "dependencies": [], "resource_req": 1},
                {"id": "J3", "duration": 4, "deadline": 30, "dependencies": [], "resource_req": 1},
                {"id": "J4", "duration": 3, "deadline": 30, "dependencies": ["J1", "J2"], "resource_req": 1},
                {"id": "J5", "duration": 2, "deadline": 30, "dependencies": ["J3"], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M3", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                    {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                    {"job_id": "J3", "machine_id": "M3", "start_time": 0},
                    {"job_id": "J4", "machine_id": "M1", "start_time": 3},
                    {"job_id": "J5", "machine_id": "M3", "start_time": 4},
                ]
            },
        },
        "is_feasible": True,
        "violation_type": None,
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M2", "start_time": 0},
                {"job_id": "J3", "machine_id": "M3", "start_time": 0},
                {"job_id": "J4", "machine_id": "M1", "start_time": 3},
                {"job_id": "J5", "machine_id": "M3", "start_time": 4},
            ]
        },
        "optimal_makespan": 6,
        "description": "Fully feasible 5-job, 3-machine schedule with precedence chain — all constraints satisfied.",
    },
]


class SchedulingOptEnv:
    """OpenEnv-compatible scheduling optimisation environment.

    API:
        reset(task_id)  → Observation
        step(action)    → (Observation, float, bool, dict)
        state()         → dict
    """

    def __init__(self) -> None:
        self._task_id: str = ""
        self._step: int = 0
        self._max_steps: int = 3
        self._instance_index: int = 0
        self._done: bool = True
        self._history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "feasibility_check") -> Observation:
        """Start a new episode for the given task.

        Cycles through the instance bank so repeated resets yield fresh
        scheduling problems.
        """
        self._task_id = task_id
        self._step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0

        # Determine max steps based on task difficulty
        step_limits = {
            "feasibility_check": 3,
            "conflict_classification": 5,
            "schedule_repair": 8,
        }
        self._max_steps = step_limits.get(task_id, 3)

        # Pick next instance (round-robin)
        instance_entry = INSTANCE_BANK[self._instance_index % len(INSTANCE_BANK)]
        self._instance_index += 1

        context = self._context_for_task(task_id)

        return Observation(
            schedule_instance=json.dumps(instance_entry["instance"], indent=2),
            task_id=task_id,
            context=context,
            step_number=self._step,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Process one agent action and return (obs, reward, done, info)."""
        if self._done:
            obs = Observation(
                schedule_instance="{}",
                task_id=self._task_id,
                context="Episode is over. Call reset() to start a new one.",
                step_number=self._step,
            )
            return obs, 0.0, True, {"error": "episode_already_done"}

        self._step += 1

        # Resolve the current instance
        instance_entry = INSTANCE_BANK[(self._instance_index - 1) % len(INSTANCE_BANK)]

        # Grade the action using the appropriate grader
        from graders.grader_detection import FeasibilityGrader
        from graders.grader_classification import ConflictGrader
        from graders.grader_fix import RepairGrader

        grader_map = {
            "feasibility_check": FeasibilityGrader(),
            "conflict_classification": ConflictGrader(),
            "schedule_repair": RepairGrader(),
        }
        grader = grader_map.get(self._task_id, FeasibilityGrader())
        reward = grader.grade(action, instance_entry)

        # Clamp reward to [0.0, 1.0] — hard invariant
        reward = max(0.0, min(1.0, reward))
        self._cumulative_reward += reward

        # Check termination: max steps reached or near-perfect reward
        done = self._step >= self._max_steps or reward >= 0.95
        self._done = done

        # Record history for state inspection
        self._history.append({
            "step": self._step,
            "action": action.response[:200],  # truncate for storage
            "reward": reward,
        })

        # Build next observation
        if done:
            obs = Observation(
                schedule_instance="{}",
                task_id=self._task_id,
                context="Episode complete." if reward >= 0.95 else "Max steps reached.",
                step_number=self._step,
            )
        else:
            obs = Observation(
                schedule_instance=json.dumps(instance_entry["instance"], indent=2),
                task_id=self._task_id,
                context=self._context_for_task(self._task_id),
                step_number=self._step,
            )

        info = {
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_remaining": max(0, self._max_steps - self._step),
            "instance_description": instance_entry["description"],
        }
        return obs, round(reward, 4), done, info

    def state(self) -> dict[str, Any]:
        """Return full current environment state."""
        if self._instance_index > 0:
            entry = INSTANCE_BANK[(self._instance_index - 1) % len(INSTANCE_BANK)]
            current_instance = entry["instance"].get("problem_id", "")
        else:
            current_instance = ""

        return {
            "task_id": self._task_id,
            "step": self._step,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "history": copy.deepcopy(self._history),
            "current_instance_id": current_instance,
            "instance_index": self._instance_index,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _context_for_task(task_id: str) -> str:
        contexts = {
            "feasibility_check": (
                "Analyse the scheduling instance. Respond with exactly 'feasible' if "
                "all constraints are satisfied, or 'infeasible' if any constraint is violated."
            ),
            "conflict_classification": (
                "The schedule contains a constraint violation. Classify it as exactly one of: "
                "resource_overload, deadline_violation, precedence_violation, "
                "availability_conflict, capacity_exceeded."
            ),
            "schedule_repair": (
                "The schedule is infeasible. Return a corrected schedule as a JSON object "
                'with key "assignments": a list of {"job_id", "machine_id", "start_time"} '
                "dicts that resolves all violations and minimises total makespan."
            ),
        }
        return contexts.get(task_id, "Analyse the scheduling instance.")

    @staticmethod
    def get_instance_bank() -> list[dict[str, Any]]:
        """Expose the instance bank for external use (e.g., baseline)."""
        return INSTANCE_BANK
