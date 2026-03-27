"""Task definitions for the Scheduling Optimisation Environment.

Exports
-------
    task1_easy   — Feasibility Check (binary classification, 3 steps)
    task2_medium — Conflict Classification (5-class, 5 steps)
    task3_hard   — Schedule Repair (JSON generation, 8 steps)
"""

from tasks import task1_easy, task2_medium, task3_hard

__all__ = ["task1_easy", "task2_medium", "task3_hard"]
