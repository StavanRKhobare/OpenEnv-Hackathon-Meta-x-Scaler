"""Grader for Task 3 — Schedule Repair (hard).

Scoring breakdown (additive, max 1.0):
    0.4  — response is valid JSON with the required "assignments" schema
    0.4  — the proposed schedule satisfies all constraints (no violations)
    0.2  — makespan is within 30% of the known optimal makespan

Partial-progress signal:
    0.2  — base reward if the response is at least parseable JSON, even before
           constraint checking.
"""

from __future__ import annotations

import json
from typing import Any

from models import Action


class RepairGrader:
    """Grade the agent's proposed schedule repair."""

    def grade(self, action: Action, ground_truth: dict[str, Any]) -> float:
        response = action.response.strip()

        if not response:
            return 0.0

        score = 0.0
        instance = ground_truth.get("instance", {})
        optimal_makespan: int = ground_truth.get("optimal_makespan", 1)

        # ------------------------------------------------------------------
        # Component 1: Is the response valid JSON with correct schema? (0.4)
        # We award 0.2 for parseable JSON and another 0.2 for correct schema.
        # ------------------------------------------------------------------
        parsed = self._parse_schedule(response)
        if parsed is None:
            # Not parseable JSON at all — no partial credit
            return 0.0

        # Parseable JSON → base partial credit
        score += 0.2

        assignments = parsed.get("assignments", [])
        if self._valid_schema(assignments, instance):
            score += 0.2  # correct schema → full 0.4 for component 1
        else:
            # Schema wrong — cap here, no constraint or optimality credit
            return round(max(0.0, min(1.0, score)), 4)

        # ------------------------------------------------------------------
        # Component 2: Does the schedule satisfy all constraints? (0.4)
        # Sub-components (each ~0.1):
        #   (a) no resource overload / capacity exceeded
        #   (b) all jobs finish by their deadlines
        #   (c) all precedence dependencies respected
        #   (d) all jobs within machine availability windows
        # ------------------------------------------------------------------
        constraint_score = self._check_constraints(assignments, instance)
        score += 0.4 * constraint_score

        # ------------------------------------------------------------------
        # Component 3: Makespan optimality (0.2)
        # Full 0.2 if makespan ≤ optimal × 1.30 (within 30% tolerance).
        # Partial 0.1 if makespan ≤ optimal × 1.60.
        # ------------------------------------------------------------------
        makespan = self._compute_makespan(assignments, instance)
        if makespan > 0 and optimal_makespan > 0:
            ratio = makespan / optimal_makespan
            if ratio <= 1.30:
                score += 0.2
            elif ratio <= 1.60:
                score += 0.1

        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_schedule(response: str) -> dict[str, Any] | None:
        """Attempt to parse the agent's response as JSON.

        Tries the raw response first, then looks for a JSON block inside
        markdown fences.
        """
        # Direct parse
        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to extract from ```json ... ``` fences
        import re
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find any {...} block in the response
        brace = re.search(r"(\{.*\})", response, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    @staticmethod
    def _valid_schema(
        assignments: list[Any], instance: dict[str, Any]
    ) -> bool:
        """Check that assignments is a list of dicts with the required keys."""
        if not isinstance(assignments, list) or len(assignments) == 0:
            return False
        required_keys = {"job_id", "machine_id", "start_time"}
        for a in assignments:
            if not isinstance(a, dict):
                return False
            if not required_keys.issubset(a.keys()):
                return False
            if not isinstance(a.get("start_time"), (int, float)):
                return False
        # Check all jobs are covered
        expected_jobs = {j["id"] for j in instance.get("jobs", [])}
        assigned_jobs = {a["job_id"] for a in assignments}
        return expected_jobs == assigned_jobs

    @staticmethod
    def _check_constraints(
        assignments: list[dict[str, Any]], instance: dict[str, Any]
    ) -> float:
        """Return fraction of constraint categories satisfied (0.0–1.0)."""
        jobs_by_id = {j["id"]: j for j in instance.get("jobs", [])}
        machines_by_id = {m["id"]: m for m in instance.get("machines", [])}
        assign_by_job = {a["job_id"]: a for a in assignments}

        checks_passed = 0
        total_checks = 4

        # (a) Resource / capacity: for each time point, count concurrent jobs per machine
        machine_timelines: dict[str, list[tuple[float, float]]] = {}
        for a in assignments:
            mid = a["machine_id"]
            st = float(a["start_time"])
            job = jobs_by_id.get(a["job_id"], {})
            dur = float(job.get("duration", 1))
            machine_timelines.setdefault(mid, []).append((st, st + dur))

        capacity_ok = True
        for mid, intervals in machine_timelines.items():
            cap = machines_by_id.get(mid, {}).get("capacity", 1)
            # Check every interval start for concurrent count
            for i, (s1, e1) in enumerate(intervals):
                concurrent = sum(1 for s2, e2 in intervals if s2 < e1 and e2 > s1)
                if concurrent > cap:
                    capacity_ok = False
                    break
            if not capacity_ok:
                break
        if capacity_ok:
            checks_passed += 1

        # (b) Deadlines: every job must finish by its deadline
        deadline_ok = True
        for a in assignments:
            job = jobs_by_id.get(a["job_id"], {})
            finish = a["start_time"] + job.get("duration", 0)
            if finish > job.get("deadline", float("inf")):
                deadline_ok = False
                break
        if deadline_ok:
            checks_passed += 1

        # (c) Precedence: job must start after all its dependencies finish
        precedence_ok = True
        for a in assignments:
            job = jobs_by_id.get(a["job_id"], {})
            for dep_id in job.get("dependencies", []):
                dep_assign = assign_by_job.get(dep_id)
                if dep_assign is None:
                    precedence_ok = False
                    break
                dep_job = jobs_by_id.get(dep_id, {})
                dep_finish = dep_assign["start_time"] + dep_job.get("duration", 0)
                if a["start_time"] < dep_finish:
                    precedence_ok = False
                    break
            if not precedence_ok:
                break
        if precedence_ok:
            checks_passed += 1

        # (d) Machine availability: job must run within machine window
        availability_ok = True
        for a in assignments:
            machine = machines_by_id.get(a["machine_id"], {})
            avail_start = machine.get("available_start", 0)
            avail_end = machine.get("available_end", float("inf"))
            job = jobs_by_id.get(a["job_id"], {})
            job_start = a["start_time"]
            job_end = job_start + job.get("duration", 0)
            if job_start < avail_start or job_end > avail_end:
                availability_ok = False
                break
        if availability_ok:
            checks_passed += 1

        return checks_passed / total_checks

    @staticmethod
    def _compute_makespan(
        assignments: list[dict[str, Any]], instance: dict[str, Any]
    ) -> int:
        """Return the makespan (latest finish time) of the proposed schedule."""
        jobs_by_id = {j["id"]: j for j in instance.get("jobs", [])}
        max_finish = 0
        for a in assignments:
            job = jobs_by_id.get(a["job_id"], {})
            finish = a["start_time"] + job.get("duration", 0)
            if finish > max_finish:
                max_finish = finish
        return int(max_finish)
