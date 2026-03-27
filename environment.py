"""Core Scheduling Optimisation Environment implementing the OpenEnv API contract.

Design principles
-----------------
* reset() always returns a valid Observation — never raises.
* step() clamps reward to [0.0, 1.0] unconditionally.
* Task-aware instance routing: conflict_classification and schedule_repair
  are shown only infeasible instances; feasibility_check sees all 12.
* Per-step contextual feedback: the context string and info['grading_breakdown']
  give the agent actionable signal on every step, enabling sample-efficient
  multi-step improvement within a single episode.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from graders.grader_classification import ConflictGrader
from graders.grader_detection import FeasibilityGrader
from graders.grader_fix import RepairGrader
from models import Action, Observation

# Grader singletons — one per task, reused across episodes.
_GRADERS: dict[str, Any] = {
    "feasibility_check": FeasibilityGrader(),
    "conflict_classification": ConflictGrader(),
    "schedule_repair": RepairGrader(),
}

# ---------------------------------------------------------------------------
# Scheduling instance bank — 12 diverse instances.
#
# Each entry:
#   instance         – dict exposed to the agent (jobs + machines + proposed_schedule)
#   is_feasible      – bool, ground-truth for Task 1
#   violation_type   – str | None, ground-truth for Task 2
#   optimal_schedule – dict, the repaired schedule for Task 3
#   optimal_makespan – int, minimum achievable makespan
#   description      – one-line human-readable summary
# ---------------------------------------------------------------------------

INSTANCE_BANK: list[dict[str, Any]] = [
    # ------------------------------------------------------------------ #
    # 0 — resource_overload                                                #
    # J1[0,4) and J2[2,5) overlap on M1 (capacity=1).                    #
    # Fix: sequence J2 after J1.                                           #
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
    # 1 — deadline_violation                                               #
    # J1 starts late (t=5, dur=5), finishes at t=10 > deadline=8.         #
    # Fix: schedule J1 first so it finishes at t=5 ≤ 8.                   #
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
    # 2 — precedence_violation                                             #
    # J2 depends on J1 (J1 finishes t=8) but J2 starts at t=0.           #
    # Fix: start J1 first, then J2 after J1 completes.                    #
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
    # 3 — availability_conflict                                            #
    # M1 available [8,18]. J1 starts at t=5, before the window opens.     #
    # Fix: shift J1 to start at t=8 (first valid slot).                   #
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
    # 4 — capacity_exceeded                                                #
    # 3 jobs on M1 simultaneously; capacity=2 → load=3 > 2.               #
    # Fix: stagger one job to start after the first batch finishes.        #
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
    # 5 — resource_overload (variant)                                      #
    # J1[0,5) and J2[1,5) overlap on M1 (capacity=1).                    #
    # Fix: run jobs sequentially.                                          #
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
    # 6 — deadline_violation (chain with avoidable idle time)              #
    # J1→J2→J3 chain. J1 starts at t=3 (wasted idle), making the chain   #
    # finish at t=15 > deadline=13. Fix: start J1 at t=0 → chain ends at  #
    # t=12 ≤ 13. NOTE: J3 duration is 3 (not 4) so the chain IS solvable. #
    # ------------------------------------------------------------------ #
    {
        "instance": {
            "problem_id": "P07",
            "jobs": [
                {"id": "J1", "duration": 4, "deadline": 20, "dependencies": [], "resource_req": 1},
                {"id": "J2", "duration": 5, "deadline": 20, "dependencies": ["J1"], "resource_req": 1},
                {"id": "J3", "duration": 3, "deadline": 13, "dependencies": ["J2"], "resource_req": 1},
            ],
            "machines": [
                {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M2", "capacity": 1, "available_start": 0, "available_end": 24},
                {"id": "M3", "capacity": 1, "available_start": 0, "available_end": 24},
            ],
            "proposed_schedule": {
                "assignments": [
                    {"job_id": "J1", "machine_id": "M1", "start_time": 3},
                    {"job_id": "J2", "machine_id": "M2", "start_time": 7},
                    {"job_id": "J3", "machine_id": "M3", "start_time": 12},
                ]
            },
        },
        "is_feasible": False,
        "violation_type": "deadline_violation",
        # Optimal: eliminate idle prefix → J1 starts at t=0, chain finishes at t=12 ≤ 13
        "optimal_schedule": {
            "assignments": [
                {"job_id": "J1", "machine_id": "M1", "start_time": 0},
                {"job_id": "J2", "machine_id": "M2", "start_time": 4},
                {"job_id": "J3", "machine_id": "M3", "start_time": 9},
            ]
        },
        "optimal_makespan": 12,
        "description": "J1 starts at t=3 (unnecessary idle); J3 finishes at t=15 > deadline=13.",
    },
    # ------------------------------------------------------------------ #
    # 7 — precedence_violation (fan-in: two predecessors)                  #
    # J3 depends on J1 and J2; J3 starts at t=2 but J2 finishes at t=4.  #
    # Fix: delay J3 start to t=4 (max predecessor finish time).           #
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
    # 8 — availability_conflict (maintenance window)                       #
    # M1 available only [0,10]. J1 starts at t=9, runs [9,12) → exceeds   #
    # the window. Fix: schedule J1 before the window closes.               #
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
                    "note": "M1 under maintenance t=[10,15]; use window [0,10] only.",
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
    # 9 — capacity_exceeded (four jobs on capacity-3 machine)              #
    # Concurrent load at t=0 is 4 > capacity=3.                           #
    # Fix: stagger the fourth job.                                         #
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
    # 10 — FEASIBLE: 3-job, 2-machine                                      #
    # All constraints satisfied in the proposed schedule.                  #
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
        "description": "Fully feasible 3-job, 2-machine schedule — all constraints satisfied.",
    },
    # ------------------------------------------------------------------ #
    # 11 — FEASIBLE: 5-job, 3-machine with fan-in precedence              #
    # All constraints satisfied in the proposed schedule.                  #
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
        "description": "Fully feasible 5-job, 3-machine schedule with fan-in precedence — all constraints satisfied.",
    },
]

# ---------------------------------------------------------------------------
# Task-specific instance pools (built once after INSTANCE_BANK is defined).
# This ensures task-appropriate instances are shown per task:
#   feasibility_check      → all 12 (mix of feasible and infeasible)
#   conflict_classification → 10 infeasible only (feasible has no violation)
#   schedule_repair         → 10 infeasible with known optimal repairs
# ---------------------------------------------------------------------------
_TASK_POOLS: dict[str, list[dict[str, Any]]] = {
    "feasibility_check": INSTANCE_BANK,
    "conflict_classification": [e for e in INSTANCE_BANK if not e["is_feasible"]],
    "schedule_repair": [
        e for e in INSTANCE_BANK if not e["is_feasible"] and e.get("optimal_schedule")
    ],
}


class SchedulingOptEnv:
    """OpenEnv-compatible scheduling optimisation environment.

    Public API (OpenEnv contract)
    -----------------------------
        reset(task_id: str)  → Observation
        step(action: Action) → (Observation, float, bool, dict)
        state()              → dict
    """

    def __init__(self) -> None:
        self._task_id: str = "feasibility_check"
        self._step: int = 0
        self._max_steps: int = 3
        # Per-task episode counters for round-robin cycling within each pool
        self._task_counters: dict[str, int] = {}
        # The instance used in the current episode (set by reset)
        self._current_instance: dict[str, Any] = {}
        self._done: bool = True
        self._history: list[dict[str, Any]] = []
        self._cumulative_reward: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "feasibility_check") -> Observation:
        """Start a new episode.

        Selects the next instance from the task-appropriate pool in round-robin
        order so that repeated resets present diverse scheduling problems.
        Always succeeds — never raises an exception.
        """
        self._task_id = task_id
        self._step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0

        step_limits: dict[str, int] = {
            "feasibility_check": 3,
            "conflict_classification": 5,
            "schedule_repair": 8,
        }
        self._max_steps = step_limits.get(task_id, 3)

        # Task-aware round-robin instance selection
        pool = _TASK_POOLS.get(task_id, INSTANCE_BANK)
        idx = self._task_counters.get(task_id, 0) % len(pool)
        self._current_instance = pool[idx]
        self._task_counters[task_id] = idx + 1

        return Observation(
            schedule_instance=json.dumps(self._current_instance["instance"], indent=2),
            task_id=task_id,
            context=self._build_context(task_id, step=0, last_reward=None),
            step_number=0,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Process one agent action.

        Returns (observation, reward, done, info).
        Reward is always clamped to [0.0, 1.0].
        """
        if self._done:
            return (
                Observation(
                    schedule_instance="{}",
                    task_id=self._task_id,
                    context="Episode is over. Call /reset to start a new episode.",
                    step_number=self._step,
                ),
                0.0,
                True,
                {"error": "episode_already_done"},
            )

        self._step += 1

        grader = _GRADERS.get(self._task_id, _GRADERS["feasibility_check"])

        reward: float = grader.grade(action, self._current_instance)
        reward = max(0.0, min(1.0, float(reward)))  # hard clamp — invariant
        self._cumulative_reward += reward

        # Capture grading breakdown for rich info dict
        breakdown: dict[str, Any] = getattr(grader, "last_breakdown", {})

        # Record step history (truncate long responses for storage efficiency)
        self._history.append({
            "step": self._step,
            "action": action.response[:300],
            "reward": round(reward, 4),
        })

        # Termination: max steps exhausted or near-perfect reward (≥0.95)
        done = self._step >= self._max_steps or reward >= 0.95
        self._done = done

        # Build next observation
        if done:
            best = max(h["reward"] for h in self._history)
            ctx = (
                "Episode complete — constraint satisfied."
                if reward >= 0.95
                else f"Max steps reached. Best reward this episode: {best:.2f}."
            )
            obs = Observation(
                schedule_instance="{}",
                task_id=self._task_id,
                context=ctx,
                step_number=self._step,
            )
        else:
            obs = Observation(
                schedule_instance=json.dumps(
                    self._current_instance["instance"], indent=2
                ),
                task_id=self._task_id,
                context=self._build_context(
                    self._task_id, step=self._step, last_reward=reward
                ),
                step_number=self._step,
            )

        info: dict[str, Any] = {
            "step_reward": round(reward, 4),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "steps_remaining": max(0, self._max_steps - self._step),
            "instance_description": self._current_instance.get("description", ""),
            "grading_breakdown": breakdown,
        }
        return obs, round(reward, 4), done, info

    def state(self) -> dict[str, Any]:
        """Return a snapshot of the full internal environment state."""
        return {
            "task_id": self._task_id,
            "step": self._step,
            "max_steps": self._max_steps,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "history": copy.deepcopy(self._history),
            "current_instance_id": (
                self._current_instance.get("instance", {}).get("problem_id", "")
            ),
            "current_instance_feasible": self._current_instance.get("is_feasible"),
            "task_counters": dict(self._task_counters),
            "instance_pool_sizes": {k: len(v) for k, v in _TASK_POOLS.items()},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(
        task_id: str, step: int, last_reward: float | None
    ) -> str:
        """Build a context string that adapts to the current step and last reward.

        On the first step (step=0) a clear task description is returned.
        On retry steps (step>0, last_reward<0.95) an informative hint is appended
        to guide the agent toward a better answer.
        """
        base_contexts: dict[str, str] = {
            "feasibility_check": (
                "Examine the proposed_schedule against all four constraint categories "
                "(machine capacity, job deadlines, precedence dependencies, machine "
                "availability windows). Reply with exactly 'feasible' if every constraint "
                "is satisfied, or 'infeasible' if any constraint is violated."
            ),
            "conflict_classification": (
                "The proposed_schedule is infeasible. Identify the PRIMARY constraint "
                "violation and reply with exactly one of: resource_overload, "
                "deadline_violation, precedence_violation, availability_conflict, "
                "capacity_exceeded."
            ),
            "schedule_repair": (
                "The proposed_schedule is infeasible. Return ONLY a JSON object with key "
                '"assignments": a list of {"job_id": str, "machine_id": str, '
                '"start_time": int} dicts that resolves ALL violations (capacity, '
                "deadlines, precedence, availability) and minimises total makespan."
            ),
        }
        ctx = base_contexts.get(task_id, "Analyse the scheduling instance.")

        # Add retry hint when the agent is wrong but still has steps remaining
        if step > 0 and last_reward is not None and last_reward < 0.95:
            hints: dict[str, str] = {
                "feasibility_check": (
                    " ← Previous answer was incorrect. "
                    "Re-examine all four constraint types carefully."
                ),
                "conflict_classification": (
                    " ← Previous classification was wrong. "
                    "Check whether jobs share a machine simultaneously (resource/capacity), "
                    "miss their deadlines, violate ordering, or run outside availability windows."
                ),
                "schedule_repair": (
                    " ← Previous repair had remaining violations. "
                    "Ensure no two jobs overlap on a capacity-1 machine, every job "
                    "finishes before its deadline, precedence order is respected, and "
                    "all jobs run within machine availability windows."
                ),
            }
            ctx += hints.get(task_id, "")

        return ctx

    @staticmethod
    def get_instance_bank() -> list[dict[str, Any]]:
        """Return the full instance bank (all 12 entries)."""
        return INSTANCE_BANK
