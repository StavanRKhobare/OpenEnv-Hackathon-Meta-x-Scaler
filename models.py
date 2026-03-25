"""Pydantic v2 models for the Scheduling Optimisation Environment."""

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """What the agent sees at each step."""

    schedule_instance: str = Field(
        description="JSON-encoded scheduling problem instance to evaluate"
    )
    task_id: str = Field(description="Current task identifier")
    context: str = Field(description="Instructions or hints for the current step")
    step_number: int = Field(ge=0, description="Current step in the episode")


class Action(BaseModel):
    """What the agent submits as a response."""

    response: str = Field(
        description=(
            "Agent's answer: 'feasible'/'infeasible', a violation category, "
            "or a JSON repair schedule"
        )
    )
    task_id: str = Field(description="Task identifier this action is for")


class Reward(BaseModel):
    """Grading result returned to the agent."""

    score: float = Field(ge=0.0, le=1.0, description="Reward score in [0.0, 1.0]")
    feedback: str = Field(default="", description="Human-readable grading feedback")
