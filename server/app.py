"""FastAPI server exposing the Scheduling Optimisation Environment as an HTTP API.

Endpoints (OpenEnv-compatible):
    GET  /health    — liveness probe
    GET  /metadata  — environment name and description
    GET  /schema    — action / observation / state schemas
    POST /mcp       — JSON-RPC 2.0 stub (MCP compatibility)
    POST /reset     — start a new episode
    POST /step      — submit an action
    GET  /state     — current environment state
    GET  /tasks     — list available tasks
    POST /grader    — directly invoke a grader
    GET  /baseline  — run the oracle baseline

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so that environment, models, graders
# are importable whether this module is run directly or via uvicorn.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import SchedulingOptEnv
from graders.grader_classification import ConflictGrader
from graders.grader_detection import FeasibilityGrader
from graders.grader_fix import RepairGrader
from models import Action, Observation

app = FastAPI(
    title="Scheduling Optimisation Environment",
    description=(
        "OpenEnv-compatible environment for training AI agents on combinatorial "
        "scheduling optimisation problems."
    ),
    version="1.0.0",
)

# Single shared environment instance.
env = SchedulingOptEnv()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "feasibility_check"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


class GradeRequest(BaseModel):
    action: Action
    ground_truth: dict[str, Any]


class GradeResponse(BaseModel):
    score: float


# ---------------------------------------------------------------------------
# OpenEnv required endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check — returns 'healthy' status as required by OpenEnv runtime spec."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    """Environment metadata — name and description."""
    return {
        "name": "scheduling-opt-env",
        "description": (
            "A real-world AI agent training environment for combinatorial scheduling "
            "optimisation. Agents determine schedule feasibility, classify constraint "
            "violations, and repair infeasible schedules."
        ),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    """Return action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": (
                        "Agent answer: 'feasible'/'infeasible', a violation category, "
                        "or a JSON repair schedule."
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": "Task identifier this action is for.",
                },
            },
            "required": ["response", "task_id"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "schedule_instance": {
                    "type": "string",
                    "description": "JSON-encoded scheduling problem instance.",
                },
                "task_id": {"type": "string", "description": "Current task identifier."},
                "context": {
                    "type": "string",
                    "description": "Instructions or hints for the current step.",
                },
                "step_number": {
                    "type": "integer",
                    "description": "Current step number within the episode.",
                },
            },
            "required": ["schedule_instance", "task_id", "context", "step_number"],
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "done": {"type": "boolean"},
                "cumulative_reward": {"type": "number"},
            },
        },
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any] = {}) -> dict[str, Any]:
    """Minimal JSON-RPC 2.0 stub for MCP compatibility."""
    return {
        "jsonrpc": "2.0",
        "result": {
            "tools": [],
            "description": "Scheduling Optimisation Environment MCP endpoint",
        },
        "id": payload.get("id"),
    }


# ---------------------------------------------------------------------------
# Core environment endpoints
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()) -> Observation:
    """Reset the environment and start a new episode.

    Body: {"task_id": "feasibility_check" | "conflict_classification" | "schedule_repair"}
    Sending an empty body {} uses the default task_id.
    """
    valid_tasks = {"feasibility_check", "conflict_classification", "schedule_repair"}
    if req.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id. Choose from: {sorted(valid_tasks)}",
        )
    return env.reset(task_id=req.task_id)


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
    """Submit an action and advance the environment by one step.

    Body: {"response": "<answer>", "task_id": "<task_id>"}
    """
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state() -> dict[str, Any]:
    """Return the full current environment state."""
    return env.state()


@app.get("/tasks")
def tasks() -> list[dict[str, Any]]:
    """List available tasks with their action schemas."""
    return [
        {
            "task_id": "feasibility_check",
            "name": "Feasibility Check",
            "difficulty": "easy",
            "max_steps": 3,
            "action_schema": {
                "response": "feasible | infeasible",
                "task_id": "feasibility_check",
            },
        },
        {
            "task_id": "conflict_classification",
            "name": "Conflict Classification",
            "difficulty": "medium",
            "max_steps": 5,
            "action_schema": {
                "response": (
                    "resource_overload | deadline_violation | precedence_violation | "
                    "availability_conflict | capacity_exceeded"
                ),
                "task_id": "conflict_classification",
            },
        },
        {
            "task_id": "schedule_repair",
            "name": "Schedule Repair",
            "difficulty": "hard",
            "max_steps": 8,
            "action_schema": {
                "response": '{"assignments": [{"job_id": "J1", "machine_id": "M1", "start_time": 0}, ...]}',
                "task_id": "schedule_repair",
            },
        },
    ]


@app.post("/grader", response_model=GradeResponse)
def grader(req: GradeRequest) -> GradeResponse:
    """Directly invoke a grader with an action and ground truth.

    Body: {"action": {"response": "...", "task_id": "..."}, "ground_truth": {...}}
    """
    task_id = req.action.task_id
    grader_map = {
        "feasibility_check": FeasibilityGrader(),
        "conflict_classification": ConflictGrader(),
        "schedule_repair": RepairGrader(),
    }
    g = grader_map.get(task_id)
    if g is None:
        raise HTTPException(
            status_code=400, detail=f"No grader for task_id={task_id}"
        )
    score = g.grade(req.action, req.ground_truth)
    return GradeResponse(score=max(0.0, min(1.0, score)))


@app.get("/baseline")
def baseline() -> dict[str, Any]:
    """Trigger the baseline inference agent and return per-task scores."""
    try:
        from baseline import run_baseline
        return run_baseline()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline run failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the uvicorn server.

    Allows running via:
        uv run server
        python -m server.app
        python server/app.py
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scheduling Optimisation Environment server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 7860:
        main()
    else:
        main(host=args.host, port=args.port)
