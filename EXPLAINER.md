# SchedulingOptEnv — Complete Project Explainer

### Everything You Need to Know, From Zero Background Required

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Does This Project Exist?](#2-why-does-this-project-exist)
3. [The Big Picture — A Bird's Eye View](#3-the-big-picture--a-birds-eye-view)
4. [Core Concept: What Is a Scheduling Problem?](#4-core-concept-what-is-a-scheduling-problem)
5. [Core Concept: What Is an AI Training Environment?](#5-core-concept-what-is-an-ai-training-environment)
6. [Core Concept: What Is Reinforcement Learning?](#6-core-concept-what-is-reinforcement-learning)
7. [Project Goals and Objectives](#7-project-goals-and-objectives)
8. [System Architecture — How Everything Fits Together](#8-system-architecture--how-everything-fits-together)
9. [Complete Flowchart of the Project](#9-complete-flowchart-of-the-project)
10. [File-by-File Breakdown](#10-file-by-file-breakdown)
11. [The Three Tasks — What the AI Must Learn to Do](#11-the-three-tasks--what-the-ai-must-learn-to-do)
12. [The 12 Problem Instances — The Puzzle Collection](#12-the-12-problem-instances--the-puzzle-collection)
13. [The Reward System — How the AI Gets Graded](#13-the-reward-system--how-the-ai-gets-graded)
14. [The API Server — The Communication Layer](#14-the-api-server--the-communication-layer)
15. [The Baseline — Testing With GPT-4o-mini](#15-the-baseline--testing-with-gpt-4o-mini)
16. [Data Models — The Language the System Speaks](#16-data-models--the-language-the-system-speaks)
17. [Methodology — How the System Was Built](#17-methodology--how-the-system-was-built)
18. [Constraint Theory — The Rules of Scheduling](#18-constraint-theory--the-rules-of-scheduling)
19. [Multi-Step Learning — How the AI Improves](#19-multi-step-learning--how-the-ai-improves)
20. [Scoring Deep Dive — Every Point Explained](#20-scoring-deep-dive--every-point-explained)
21. [End-to-End Walkthrough — One Full Example](#21-end-to-end-walkthrough--one-full-example)
22. [Glossary](#22-glossary)

---

## 1. What Is This Project?

In plain English: **this project is a gym for teaching AI how to fix broken work schedules.**

Think about a factory that needs to schedule 5 machines and 20 jobs. Some jobs can only run after other jobs finish. Some machines can only handle 2 jobs at once. Some machines are offline for maintenance between 3pm and 6pm. If someone hands the factory manager a schedule that violates any of these rules, the schedule is "broken" and production will fail.

This project creates a **training playground** where an AI can:
1. Look at a proposed schedule
2. Decide whether it is valid or broken
3. If broken, identify *what kind* of rule was broken
4. Fix the schedule so it works — and works efficiently

The AI learns by trial and error: it gets rewarded when it does well and gets a lower score when it does poorly. Over thousands of practice rounds, it gets better and better.

This was built for the **Meta × Scaler OpenEnv Hackathon** — a competition to build the best AI training environments.

---

## 2. Why Does This Project Exist?

### The Problem We Are Solving

Scheduling is everywhere in the real world:
- Hospital operating rooms scheduling surgeries
- Airlines scheduling flights and crews
- Cloud computing centers scheduling workloads on servers
- Factories scheduling machines and workers
- Construction projects scheduling tasks and equipment

These schedules are **extremely hard to make correctly** by hand. Even harder to *repair* when something goes wrong. And they can be incredibly costly when broken — a delayed surgery, a missed flight, a crashed server.

Human experts spend years learning to spot and fix scheduling conflicts. **Can we train an AI to do the same?**

### The Gap This Project Fills

Before environments like this one existed, there was no standardized, structured way to:
- Give an AI a scheduling problem to practice on
- Reward it in a nuanced, graduated way (not just "right" or "wrong")
- Let it try multiple times to get better within one session
- Compare different AI systems on the same benchmark

This project fills all four gaps.

---

## 3. The Big Picture — A Bird's Eye View

Here is the simplest possible description of what happens:

```
An AI agent receives a schedule (with possible problems)
          ↓
The agent analyzes it and gives an answer
          ↓
The system grades the answer and gives a score between 0 and 1
          ↓
The agent sees the score (and sometimes a hint)
          ↓
If the agent answered well enough OR ran out of attempts, the round ends
          ↓
A new schedule problem is presented, and the cycle repeats
```

The AI keeps doing this over and over, and over time learns patterns — what makes a schedule valid, what kinds of violations look like what, and how to reorganize a schedule to fix problems.

---

## 4. Core Concept: What Is a Scheduling Problem?

Imagine you are a production manager at a car factory. You have:

- **Jobs**: Tasks that need to be done (weld the door, install the engine, paint the body)
- **Machines**: Resources that do the work (welding station, engine bay, paint booth)
- **Rules** that must be obeyed:

| Rule Type | Real-World Meaning | Example |
|---|---|---|
| **Capacity** | A machine can only handle so many jobs at once | The paint booth can only paint 1 car at a time |
| **Deadline** | A job must finish by a certain time | The door must be installed before 2pm or the shift ends |
| **Precedence** | Some jobs must happen before others | You can't paint the car before you weld the body |
| **Availability** | Machines aren't available 24/7 | The welding station is offline from midnight to 6am |

A **proposed schedule** is someone's suggested plan: "Job A starts at 9am on Machine 2, Job B starts at 10am on Machine 1, ..."

The schedule is **feasible** (valid) if it obeys ALL these rules simultaneously.
The schedule is **infeasible** (broken) if it violates ANY rule.

In this project, the AI practices with 12 different scheduling scenarios.

---

## 5. Core Concept: What Is an AI Training Environment?

Think of it like a **video game for AI**.

A regular video game:
- Presents the player with a situation (a level)
- The player takes an action (jumps, shoots, moves)
- The game responds (the player moves, enemies react, score changes)
- The game continues until the player wins, dies, or quits

An AI training environment works the same way:
- **State**: The current situation (a scheduling problem)
- **Action**: What the AI "does" (its answer)
- **Reward**: A score for how good the answer was (0.0 to 1.0)
- **Episode**: One complete round from start to finish
- **Terminal**: When the round ends (correct answer, or ran out of turns)

This project implements the **OpenEnv standard** — a common API specification that makes AI training environments easy to plug into existing AI training systems. By following this standard, any AI agent that knows how to talk to an OpenEnv environment can instantly start learning from this project.

---

## 6. Core Concept: What Is Reinforcement Learning?

Reinforcement Learning (RL) is the branch of AI where an agent learns by doing.

**Analogy**: Teaching a dog a trick.
- Dog does the trick correctly → give a treat (reward signal)
- Dog does the trick wrong → no treat (low reward signal)
- Over time, the dog learns which actions lead to treats

For AI:
- AI produces an answer → system gives a score
- AI adjusts its "thinking" to increase future scores
- Over thousands of repetitions, AI gets better

The key insight is: **you don't have to tell the AI exactly what to do**. You just need to tell it *how good* what it did was. The AI figures out the rest.

This project provides those reward signals for scheduling tasks. Each grader (scorer) is carefully designed to give:
- **Dense rewards**: Partial credit for partial progress (not just 0 or 1)
- **Informative signals**: The score reflects *how close* the answer was, not just whether it was right

---

## 7. Project Goals and Objectives

### Primary Objective
Build a robust, standards-compliant AI training environment that teaches agents to reason about combinatorial scheduling constraints.

### Specific Goals

**Goal 1 — Progressive Difficulty**
Create three tasks that form a natural learning ladder:
- Task 1 (Easy): Binary classification — "is this schedule broken?"
- Task 2 (Medium): 5-way classification — "what kind of rule is broken?"
- Task 3 (Hard): Optimization — "fix the schedule and minimize total time"

**Goal 2 — Dense Reward Signals**
Reward functions that give partial credit so the AI always has something to learn, even when its answer is wrong.

**Goal 3 — Real-World Grounding**
Problems based on real industrial scheduling scenarios (factories, cloud computing, production planning) so trained agents transfer to practical settings.

**Goal 4 — Standardization**
Full compliance with the OpenEnv API specification so the environment works with any compatible AI training system.

**Goal 5 — Baseline Benchmarking**
Include a baseline evaluation using GPT-4o-mini so researchers can compare their AI agents to a known reference.

### Success Metrics
- All reward functions return values in [0.0, 1.0]
- Mock oracle baseline achieves 1.000 (perfect score) on all tasks
- Server starts and responds correctly to all API endpoints
- All 12 problem instances cover all 5 constraint violation types

---

## 8. System Architecture — How Everything Fits Together

The system has 5 main layers, each building on the one below it:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 5: EXTERNAL CLIENTS (AI agents, researchers, you!)       │
│  Send HTTP requests, receive JSON responses                      │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (port 7860)
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 4: API SERVER (server.py)                                │
│  FastAPI web server with 7 endpoints                             │
│  Translates HTTP requests into environment calls                 │
└────────────────────────┬────────────────────────────────────────┘
                         │  Python function calls
┌────────────────────────▼────────────────────────────────────────┐
│  LAYER 3: ENVIRONMENT (environment.py)                          │
│  The "game engine" — manages episodes, selects instances,        │
│  calls graders, builds feedback, tracks state                    │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
┌──────────▼──────────┐          ┌────────────▼──────────────────┐
│  LAYER 2a: TASKS    │          │  LAYER 2b: GRADERS            │
│  (tasks/ folder)    │          │  (graders/ folder)            │
│  task1_easy.py      │          │  grader_detection.py          │
│  task2_medium.py    │          │  grader_classification.py     │
│  task3_hard.py      │          │  grader_fix.py                │
└──────────┬──────────┘          └────────────┬──────────────────┘
           │                                  │
┌──────────▼──────────────────────────────────▼──────────────────┐
│  LAYER 1: DATA MODELS (models.py)                               │
│  Pydantic schemas for Observation, Action, Reward               │
│  The shared language all layers use to communicate              │
└─────────────────────────────────────────────────────────────────┘
```

**Additionally**, there is:
- `baseline.py` — sits alongside Layer 4, queries the API on behalf of GPT-4o-mini
- `openenv.yaml` — a configuration file describing the environment to external tools

---

## 9. Complete Flowchart of the Project

### Overall System Flow

```
                        ┌─────────────┐
                        │  AI AGENT   │
                        │  (external) │
                        └──────┬──────┘
                               │
                    ┌──────────▼──────────┐
                    │  POST /reset        │
                    │  {task_id: "..."}   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  SchedulingOptEnv.reset(task_id)     │
                    │                                      │
                    │  1. Validate task_id                 │
                    │  2. Select pool for this task        │
                    │  3. Pick next instance (round-robin) │
                    │  4. Set step=0, done=False           │
                    │  5. Build initial context/hint       │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  Return Observation                  │
                    │  {schedule_instance,                 │
                    │   task_id, context, step_number: 0} │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  AI reads the       │
                    │  schedule and       │
                    │  formulates answer  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  POST /step         │
                    │  {response: "...",  │
                    │   task_id: "..."}   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  SchedulingOptEnv.step(action)       │
                    │                                      │
                    │  1. Increment step counter           │
                    │  2. Select correct grader            │
                    │  3. Call grader.grade(action, inst)  │
                    │  4. Clamp reward to [0.0, 1.0]       │
                    │  5. Update cumulative reward         │
                    │  6. Log action to history            │
                    └──────────┬──────────────────────────┘
                               │
               ┌───────────────▼───────────────────┐
               │  Check termination conditions      │
               └────────┬──────────────┬────────────┘
                        │              │
              ┌─────────▼────┐  ┌──────▼──────────┐
              │ step >=      │  │  reward >= 0.95  │
              │ max_steps?   │  │  (near perfect)? │
              └─────────┬────┘  └──────┬───────────┘
                        │              │
                   YES  │         YES  │  NO (both conditions false)
                        │              │         │
              ┌─────────▼──────────────▼─┐       │
              │   done = True             │       │
              └─────────┬─────────────────┘       │
                        │                         │
              ┌─────────▼──────────┐   ┌──────────▼──────────────┐
              │  Final response:   │   │  Build retry hint       │
              │  observation,      │   │  (add encouragement +   │
              │  reward, done=True │   │  guidance to context)   │
              │  info (breakdown)  │   │  done = False           │
              └────────────────────┘   └──────────┬──────────────┘
                                                   │
                                        ┌──────────▼──────────────┐
                                        │  Return StepResponse    │
                                        │  observation, reward,   │
                                        │  done=False, info       │
                                        │                         │
                                        │  AI gets another try!   │
                                        └─────────────────────────┘
```

### Grading Flow (For Each Task)

```
TASK 1 GRADING (Feasibility Check)
───────────────────────────────────
Agent Answer
     │
     ▼
Lowercase + normalize synonyms
("valid" → "feasible", "no" → "infeasible")
     │
     ▼
Does it equal ground truth?
     │
  YES │   NO (but recognized word)   NO (empty/garbage)
     ▼              ▼                        ▼
  1.0            0.1                       0.0


TASK 2 GRADING (Conflict Classification)
──────────────────────────────────────────
Agent Answer
     │
     ▼
Lowercase + replace spaces/hyphens with underscores
     │
     ▼
Is it a valid category?
     │
  YES │                                    NO
     ▼                                      ▼
Exact match with ground truth?             0.0
     │
  YES │           NO
     ▼             ▼
  1.0    Same constraint family?
              │
           YES │          NO
              ▼            ▼
            0.5           0.1


TASK 3 GRADING (Schedule Repair)
──────────────────────────────────
Agent Answer (a text string)
     │
     ▼
Can we parse it as JSON?
     │
  YES (+0.20) │         NO (0.00 total, stop)
     ▼
Does it have the right structure?
(assignments list, all required fields, all jobs present)
     │
  YES (+0.20) │         NO (0.00 for rest of components)
     ▼
Check 4 constraint categories:
  ┌─ Capacity (machines not overloaded)       +0.10 if satisfied
  ├─ Deadlines (all jobs finish in time)      +0.10 if satisfied
  ├─ Precedence (job ordering respected)      +0.10 if satisfied
  └─ Availability (jobs within machine hours) +0.10 if satisfied
     │
     ▼
Compare makespan to optimal:
  ≤ 130% of optimal  →  +0.20
  ≤ 160% of optimal  →  +0.10
  > 160% of optimal  →  +0.00
     │
     ▼
Total score = sum of all components (max 1.0)
```

---

## 10. File-by-File Breakdown

### `models.py` — The Data Contracts

This file defines the **shapes of data** that flow through the system. Think of it as defining the forms everyone must fill out.

```
Observation (what the AI receives):
├── schedule_instance  ← The scheduling problem, encoded as text (JSON string)
├── task_id            ← Which task: "feasibility_check", "conflict_classification", or "schedule_repair"
├── context            ← Instructions and hints for the AI
└── step_number        ← Which attempt this is (0 = first attempt)

Action (what the AI submits):
├── response   ← The AI's answer as text
└── task_id    ← Which task this answer is for

Reward (what the system returns after grading):
├── score      ← A number from 0.0 to 1.0
└── feedback   ← Optional explanation of why the AI got that score
```

**Why Pydantic?** Pydantic is a Python library that automatically checks that the data has the right types and values. If the AI sends a number where text is expected, Pydantic catches the error immediately rather than letting it cause mysterious crashes later.

---

### `environment.py` — The Game Engine

This is the heart of the project. It does everything:

**1. Stores the 12 problem instances** (the INSTANCE_BANK list at the top of the file)

Each instance is a Python dictionary containing:
```
instance:
  problem_id        ← Unique ID like "P01", "P02"
  jobs:
    - id, duration, deadline, dependencies, resource_req
  machines:
    - id, capacity, available_start, available_end
  proposed_schedule:
    assignments:
      - job_id, machine_id, start_time   ← The (possibly broken) plan
is_feasible         ← True/False: is the proposed_schedule valid?
violation_type      ← What kind of rule is broken (or None if feasible)
optimal_schedule    ← The correct repaired schedule
optimal_makespan    ← How short the optimal schedule's total time is
description         ← Human-readable summary of the problem
```

**2. Organizes instances into task-specific pools**

Not every instance is appropriate for every task:
- Task 1 (Feasibility Check): Uses **all 12** instances (mix of valid and broken)
- Task 2 (Conflict Classification): Uses only the **10 broken** instances (can't classify a violation if there isn't one)
- Task 3 (Schedule Repair): Uses only the **10 broken** instances (nothing to repair if the schedule is already correct)

**3. Manages the episode lifecycle**

The `SchedulingOptEnv` class has three key methods:

`reset(task_id)`:
- Starts a new episode
- Picks the next instance from the appropriate pool using round-robin (cycles through them in order, wrapping back to start)
- Resets all counters and history
- Returns the first Observation

`step(action)`:
- Takes the AI's answer
- Calls the appropriate grader
- Calculates the reward
- Checks if the episode should end
- Builds feedback hints if the AI should try again
- Returns the result

`state()`:
- Returns a full snapshot of everything happening inside the environment right now
- Used for debugging and monitoring

**4. Tracks state**

```
_current_instance   ← The problem being worked on right now
_current_task_id    ← Which task is active
_step               ← How many actions the AI has taken this episode
_done               ← Whether the episode has ended
_history            ← List of recent actions (for debugging)
_task_counters      ← Round-robin counters for each task pool
_cumulative_reward  ← Running total of rewards this episode
```

---

### `server.py` — The Web Server

This file makes the environment accessible over the internet (or a local network) via HTTP. It uses **FastAPI** — a modern Python web framework.

Think of it as a **receptionist** that takes phone calls, routes them to the right department, and reports back the results.

The 7 endpoints it exposes:

| Endpoint | Method | What It Does |
|----------|--------|--------------|
| `/health` | GET | Quick check — "is the server alive?" → `{"status": "ok"}` |
| `/reset` | POST | Start a new episode. Send task name, get back a problem to solve |
| `/step` | POST | Submit an answer. Get back your score, feedback, and the next state |
| `/state` | GET | See the full current state of the environment |
| `/tasks` | GET | List all available tasks with their rules and settings |
| `/grader` | POST | Grade an answer directly without running a full episode |
| `/baseline` | GET | Run the built-in GPT-4o-mini evaluation |

Every endpoint accepts and returns **JSON** — a standard text format for structured data.

---

### `graders/grader_detection.py` — Task 1 Grader

**Job**: Score whether an AI correctly identified a schedule as valid or broken.

**Key design choice — synonym recognition**: Real AI systems don't always use the exact expected word. They might say "valid" instead of "feasible", or "impossible" instead of "infeasible". This grader handles that:

```
Recognized as "feasible": feasible, valid, correct, satisfiable, yes, ok, pass
Recognized as "infeasible": infeasible, invalid, incorrect, unsatisfiable,
                             no, violated, conflict, fail, impossible, broken
```

**Scoring logic**:
1. Normalize the AI's response to lowercase
2. Check if it contains any word from the feasible or infeasible synonym lists
3. Compare to the ground truth
4. Return 1.0 (correct), 0.1 (attempted but wrong), or 0.0 (no recognizable answer)

The 0.1 score for a wrong-but-recognized answer is intentional — it tells the AI "you understood what was being asked, you just got the answer wrong". This is more useful for learning than 0.0 which could mean "I don't understand the question".

---

### `graders/grader_classification.py` — Task 2 Grader

**Job**: Score how accurately an AI identified the *type* of constraint violation.

**The 5 violation categories**:
- `resource_overload` — Too many jobs running simultaneously on a machine that can only handle one
- `deadline_violation` — A job finishes too late (after its required completion time)
- `precedence_violation` — A job starts before a job it depends on has finished
- `availability_conflict` — A job is scheduled during a machine's maintenance/downtime window
- `capacity_exceeded` — Too many jobs running simultaneously on a machine with a specific capacity limit

**Why are there two "too many jobs" categories?** The distinction is:
- `resource_overload` = a machine with capacity=1 has 2+ concurrent jobs (completely overloaded)
- `capacity_exceeded` = a machine with capacity=N has N+1+ concurrent jobs (exceeds the specified limit)

In practice, they're very similar problems. That's why the grader groups them into the same "family" for partial credit.

**Constraint families** (for partial credit scoring):
```
Resource-limit family: resource_overload AND capacity_exceeded
  (both are about too many concurrent jobs)

Temporal-ordering family: deadline_violation AND precedence_violation
  (both are about time ordering being wrong)

Standalone: availability_conflict
  (unique — about machine availability windows)
```

**Scoring**:
- Exact match = 1.0
- Same family but wrong specific type = 0.5 (you were in the right neighborhood)
- Valid category but different family = 0.1 (you knew the format but missed the concept)
- Unrecognized response = 0.0

---

### `graders/grader_fix.py` — Task 3 Grader

**Job**: Score how good a repaired schedule is. This is the most complex grader.

**The repair task**: The AI must return a complete new schedule that:
1. Is valid JSON
2. Has the right structure (`{"assignments": [...]}`)
3. Obeys all 4 constraint types
4. Finishes as quickly as possible (minimal makespan)

**The 4-component scoring formula**:

```
Total Score = JSON Parseability Score
            + Schema Validity Score
            + Constraint Satisfaction Score
            + Makespan Optimality Score
```

In detail:

| Component | Max Points | How Earned |
|-----------|-----------|------------|
| JSON Parseable | 0.20 | The response is valid JSON (or extractable from surrounding text) |
| Schema Valid | 0.20 | Has `assignments` key, all jobs present, all fields correct |
| Constraints | 0.40 | 0.10 per satisfied category (capacity, deadlines, precedence, availability) |
| Makespan | 0.20 | 0.20 if within 30% of optimal; 0.10 if within 60%; 0.00 otherwise |
| **Total** | **1.00** | |

**Why partial credit at every level?** An AI that writes syntactically valid JSON but doesn't fully fix the schedule is still doing something useful — it learned the output format. An AI that fixes 3 of 4 constraints is closer to success than one that fixes none. Every bit of partial credit guides the AI toward improvement.

**Robust JSON parsing**: Real AI systems often wrap their JSON in text like "Here is my schedule: ```json {...} ```". The grader handles this with three strategies:
1. Try to parse the response directly as JSON
2. Strip markdown code fences (` ```json ` / ` ``` `) then parse
3. Find the outermost `{...}` block by counting braces, then parse that

---

### `tasks/task1_easy.py`, `task2_medium.py`, `task3_hard.py` — Task Runners

These are thin wrappers that make it easy to run an AI agent through an entire task without writing boilerplate code. They:
- Define task constants (ID, max steps, difficulty label)
- Provide a `run_episode(env, agent_fn)` function — give it an environment and an AI function, it runs one episode and returns a summary
- Provide helper functions to get all relevant instances with their ground-truth answers

They are mostly used by `baseline.py` for running evaluations.

---

### `baseline.py` — The Reference Evaluation

**Job**: Run a real AI (GPT-4o-mini) through all tasks and report scores.

**Two modes**:
1. **Real mode** (if `OPENAI_API_KEY` environment variable is set): Calls OpenAI's GPT-4o-mini model
2. **Mock mode** (no API key): Uses hardcoded perfect answers that match ground truth exactly

The mock mode is how the project was verified during development — it confirms the scoring system works correctly (perfect oracle answers should always score 1.000).

**How it works**:
1. Loop through all instances for each task
2. Format the scheduling problem as a text prompt
3. Ask GPT-4o-mini (or use mock answer)
4. Score the response with the appropriate grader
5. Report averages

**The prompts sent to GPT-4o-mini**:
- Task 1: "You are a scheduling expert. Determine if the proposed schedule satisfies all constraints. Reply with ONLY 'feasible' or 'infeasible'."
- Task 2: "You are a scheduling expert. Identify the constraint violation type. Reply with ONLY one of: resource_overload, deadline_violation, ..."
- Task 3: "You are a scheduling expert. Repair the infeasible schedule by returning a JSON object... Return ONLY valid JSON."

---

### `openenv.yaml` — The Environment Manifest

A configuration file in YAML format that describes the environment to external systems. It's like a "technical spec sheet" — it tells OpenEnv-compatible tools:
- What this environment is called
- What version it is
- What tasks it supports
- What the input/output format is
- What the difficulty levels are

External AI training orchestrators read this file to automatically configure themselves to work with this environment.

---

### `requirements.txt` — Dependencies

Lists all external Python packages needed to run the project:

| Package | What It Does |
|---------|-------------|
| `fastapi` | The web framework for building the API server |
| `uvicorn` | The actual server process that runs FastAPI (like a web server) |
| `pydantic` | Data validation and type checking (enforces data shapes) |
| `openai` | Python client for GPT-4o-mini (used by baseline.py) |
| `pyyaml` | Reads and writes YAML files (for openenv.yaml) |
| `httpx` | Modern async HTTP client (for making requests inside the system) |

---

## 11. The Three Tasks — What the AI Must Learn to Do

### Task 1: Feasibility Check (Easy)

**What the AI sees**: A complete scheduling scenario with jobs, machines, and a proposed assignment of jobs to machines at specific times.

**What the AI must answer**: One word — "feasible" or "infeasible"

**Why it's "easy"**: Binary choice. The AI only needs to spot whether *any* rule is broken, not figure out which specific rule or how to fix it.

**Max attempts**: 3 per episode

**Example scenario**:
```
Machine M1 has capacity 1 (can only run 1 job at a time)
Job J1 is assigned to M1 from time 0 to time 4
Job J2 is assigned to M1 from time 2 to time 5

Problem: J1 and J2 overlap on M1 from time 2 to time 4!
Correct answer: "infeasible"
```

---

### Task 2: Conflict Classification (Medium)

**What the AI sees**: A scheduling scenario that is known to be infeasible.

**What the AI must answer**: Which of the 5 categories of violation is present:
- `resource_overload`
- `deadline_violation`
- `precedence_violation`
- `availability_conflict`
- `capacity_exceeded`

**Why it's "medium"**: The AI must not only detect a problem exists but diagnose the *specific type* of problem. This requires understanding the semantics of each constraint type.

**Max attempts**: 5 per episode

**Example scenario**:
```
Job J1 must complete before 8am (deadline = 8)
Job J1 is assigned to start at 5am with a duration of 6 hours
Finish time = 5 + 6 = 11am > 8am deadline

Correct answer: "deadline_violation"
```

---

### Task 3: Schedule Repair (Hard)

**What the AI sees**: A scheduling scenario with a known violation.

**What the AI must answer**: A complete JSON object that specifies a new, corrected schedule where all constraints are satisfied and makespan is minimized.

**Why it's "hard"**: The AI must:
1. Understand what's broken (like Task 2)
2. Design a new schedule that fixes all violations
3. Optimize for minimal total time (makespan)
4. Return a syntactically correct JSON in the exact right format

**Max attempts**: 8 per episode (more attempts because this is genuinely harder)

**Example answer format**:
```json
{
  "assignments": [
    {"job_id": "J1", "machine_id": "M1", "start_time": 0},
    {"job_id": "J2", "machine_id": "M1", "start_time": 4},
    {"job_id": "J3", "machine_id": "M2", "start_time": 0}
  ]
}
```

---

## 12. The 12 Problem Instances — The Puzzle Collection

The project includes 12 hand-crafted scheduling scenarios. Here is every one of them:

### Infeasible Instances (10)

| # | Violation Type | The Problem Explained |
|---|---|---|
| P01 | Resource Overload | Machine M1 can only handle 1 job. J1 runs 0→4, J2 runs 2→5. They overlap from 2→4, breaking the capacity rule. |
| P02 | Deadline Violation | J1 must finish by time 8. It's scheduled to start at 5 with duration 5, so it finishes at time 10. Two units late. |
| P03 | Precedence Violation | J2 depends on J1 (J1 must finish before J2 starts). J1 finishes at time 8 but J2 is scheduled to start at time 0. J2 starts 8 time units too early. |
| P04 | Availability Conflict | Machine M1 is only available from time 8 to time 18. J1 is scheduled to start at time 5 — before the machine is even turned on. |
| P05 | Capacity Exceeded | Machine M1 has capacity 2 (can handle 2 concurrent jobs). Three jobs (J1, J2, J3) all run at the same time. 3 > 2. |
| P06 | Resource Overload | Machine M1 has capacity 1. J1 runs 0→5, J2 runs 1→5. They overlap from 1→5. |
| P07 | Deadline Violation | A chain of jobs (J1→J2→J3) where each must finish before the next starts. The chain's total duration means J3 finishes at time 15, but its deadline is 13. |
| P08 | Precedence Violation | J3 depends on both J1 and J2. J2 finishes at time 4 but J3 starts at time 2 — before J2 is done. |
| P09 | Availability Conflict | Machine M1 is only available from time 0 to time 10. J1 starts at time 9 but has duration 3, so it finishes at time 12 — extending beyond the machine's available window. |
| P10 | Capacity Exceeded | Machine M1 has capacity 3. Four jobs all run simultaneously. 4 > 3. |

### Feasible Instances (2)

| # | Description |
|---|---|
| P11 | A 3-job, 2-machine schedule where everything is correct. Precedence respected, no overlaps, all deadlines met. Used to teach the AI that not all schedules are broken. |
| P12 | A 5-job, 3-machine schedule with fan-in precedence (two jobs must finish before a third can start). All constraints satisfied. More complex feasible example. |

---

## 13. The Reward System — How the AI Gets Graded

### Design Philosophy

Traditional AI evaluation often uses binary scores: **right = 1, wrong = 0**. This project uses **dense rewards** — scores that reflect *how close* the answer is, not just whether it's right.

Why? Because binary rewards make it hard for the AI to learn. If the AI is 99% of the way to the correct answer and still gets 0, it can't tell if it's making progress. Dense rewards create a gradient — a direction of improvement the AI can follow.

### Score Scale (Always 0.0 to 1.0)

```
1.0  ─── Perfect answer
0.9  ─┐
0.8  ─┤  Very good (within 30% of optimal for repairs)
0.7  ─┘
0.5  ─── Partial credit (right category family, or 2 of 4 constraints fixed)
0.2  ─── Minimal credit (JSON valid, schema valid, no constraints)
0.1  ─── Attempted but wrong (recognized the format, gave wrong answer)
0.0  ─── No recognizable answer
```

### Termination Conditions

An episode ends when either:
1. The AI reaches the maximum number of steps (3, 5, or 8 depending on task)
2. The AI earns a reward of **0.95 or higher** (considered "good enough")

Why 0.95 and not 1.0? Because for the repair task, getting exactly 1.0 requires hitting exact optimality, but 0.95 means the AI produced a valid, nearly-optimal schedule. That's a success worth stopping for.

---

## 14. The API Server — The Communication Layer

The server makes the environment accessible as a **web service** — any program on any computer (or the internet) can talk to it using standard HTTP requests.

### Starting the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860 --reload
```

This starts listening on port 7860. The `--reload` flag makes the server automatically restart when you change code files (useful during development).

### How HTTP Communication Works (Simplified)

```
CLIENT                              SERVER (port 7860)
  │                                         │
  │──── POST /reset ───────────────────────►│
  │     {"task_id": "schedule_repair"}      │
  │                                         │ (runs env.reset())
  │◄─── Observation ────────────────────────│
  │     {schedule_instance: "...",          │
  │      task_id: "schedule_repair",        │
  │      context: "Fix the schedule...",    │
  │      step_number: 0}                    │
  │                                         │
  │──── POST /step ────────────────────────►│
  │     {"response": "{\"assignments\": …}",│
  │      "task_id": "schedule_repair"}      │
  │                                         │ (runs env.step())
  │◄─── StepResponse ───────────────────────│
  │     {observation: {...},                │
  │      reward: 0.8,                       │
  │      done: false,                       │
  │      info: {grading_breakdown: {...}}}  │
  │                                         │
  │  ... (more steps until done=true) ...   │
```

### The Info Dictionary

Every step response includes an `info` dictionary with the detailed grading breakdown. For Task 3, this looks like:

```json
{
  "grading_breakdown": {
    "json_parseable": true,
    "schema_valid": true,
    "constraints": {
      "capacity": true,
      "deadlines": false,
      "precedence": true,
      "availability": true
    },
    "constraints_satisfied": 3,
    "makespan": 14,
    "optimal_makespan": 12,
    "makespan_ratio": 1.167,
    "within_30pct": true
  }
}
```

This lets an AI agent (or a developer) see exactly *why* a particular score was given.

---

## 15. The Baseline — Testing With GPT-4o-mini

The baseline evaluation answers the question: **"How well does an off-the-shelf AI do on these tasks?"**

GPT-4o-mini is OpenAI's fast, cheap, capable language model. By running it through all tasks, we establish a **reference point** — a bar that new AI systems can be compared against.

### Mock Baseline (No API Key)

During development (and to verify the grading system), the project includes a "mock oracle" — hardcoded perfect answers. When the baseline script runs without an OpenAI API key, it uses these perfect answers.

The fact that the mock baseline scores 1.000 on everything is a **sanity check**: it proves the grading system is correct (perfect answers should always score perfectly).

### Running the Baseline

```bash
python baseline.py
```

Output:
```
================================================================
  SchedulingOptEnv — Baseline Evaluation
================================================================

Task 1: Feasibility Check (easy)
  Instance  0: CORRECT (score=1.00)  ...
  Instance  1: CORRECT (score=1.00)  ...
  ...
  >> Average: 1.000

Task 2: Conflict Classification (medium)
  Instance  0: EXACT   (score=1.00)  ...
  ...
  >> Average: 1.000

Task 3: Schedule Repair (hard)
  Instance  0: score=1.00  optimal_makespan=7  ...
  ...
  >> Average: 1.000

================================================================
  Overall Average Score: 1.000
================================================================
```

---

## 16. Data Models — The Language the System Speaks

Pydantic models define the **contract** — the agreed-upon shape of data that flows between components. Every piece of data in the system is validated against these shapes.

### Why This Matters

Without data contracts, you get silent failures. A missing field causes a confusing error 10 steps later. With Pydantic:
- Missing required fields are caught immediately at the boundary
- Wrong data types are caught and clear error messages are generated
- The code is self-documenting — you can read the model and know exactly what data is expected

### Validation Rules

```python
Observation:
  schedule_instance: str          # required, no default
  task_id: str                    # required
  context: str = ""               # optional, defaults to empty string
  step_number: int = 0            # optional, must be ≥ 0, defaults to 0

Action:
  response: str                   # required, the AI's answer
  task_id: str                    # required

Reward:
  score: float                    # required, range [0.0, 1.0]
  feedback: Optional[str] = None  # optional explanation
```

---

## 17. Methodology — How the System Was Built

### Step 1 — Define the Problem Domain
Chose scheduling optimization as the domain because:
- It has clear, verifiable constraints
- Multiple difficulty levels are natural (detect → classify → fix)
- Real-world relevance (factory, cloud, hospital scheduling)
- Human-interpretable (easy to understand if an AI's answer is right)

### Step 2 — Design the Instance Bank
Hand-crafted 12 instances covering:
- All 5 constraint violation types (at least 2 instances each)
- 2 feasible instances (so the AI learns to say "valid" sometimes)
- Varying complexity (2-job simple cases to 5-job, 3-machine complex cases)
- Known optimal solutions for all infeasible instances (for grading Task 3)

### Step 3 — Design the Reward Functions
For each task:
- Identified what partial credit makes sense
- Designed the scoring formula to be dense (not binary)
- Added synonym normalization to handle natural language variation
- Tested with oracle answers to verify correctness

### Step 4 — Implement the Environment
- Core environment logic in `environment.py`
- Separate grader modules for clean separation of concerns
- Task runner modules for easy evaluation
- Singleton grader pattern (create once, reuse) for efficiency

### Step 5 — Wrap in FastAPI Server
- Standard OpenEnv-compatible API endpoints
- JSON request/response throughout
- Error handling (400 for invalid task IDs, etc.)

### Step 6 — Write the Baseline
- GPT-4o-mini integration with system prompts
- Mock oracle fallback for offline testing
- Formatted output showing per-instance scores

### Step 7 — Package for Deployment
- `requirements.txt` for dependency management
- `openenv.yaml` for environment metadata
- Port 7860 for Hugging Face Spaces compatibility
- `.claude/launch.json` for Claude Code IDE integration

---

## 18. Constraint Theory — The Rules of Scheduling

### Operations Research Background

This project draws from a field called **Operations Research (OR)** — the mathematical study of how to make optimal decisions under constraints.

The specific problem type is called **Resource-Constrained Project Scheduling (RCPS)** — scheduling tasks with:
- Precedence relationships (some tasks depend on others)
- Resource limits (machines, workers, equipment)
- Time constraints (deadlines)

This class of problem is known to be **NP-Hard** — meaning no algorithm can solve the general case in polynomial time. For small instances (like those in this project), optimal solutions can be found by hand or with specialized algorithms. For large real-world instances, we need good heuristics — which is exactly what we want to train AI agents to produce.

### The Four Constraint Types

**Capacity Constraint** (machine resource limits):
```
At any instant t, the number of jobs running on machine M
must not exceed M.capacity

Formally: |{j : start_j ≤ t < start_j + duration_j, machine_j = M}| ≤ capacity_M
```

**Deadline Constraint** (temporal hard limits):
```
Every job j must finish at or before its deadline:
start_j + duration_j ≤ deadline_j
```

**Precedence Constraint** (dependency ordering):
```
If job j depends on job k, then j cannot start until k finishes:
start_j ≥ start_k + duration_k

For multiple dependencies, j must wait for ALL predecessors:
start_j ≥ max(start_k + duration_k) for all k in dependencies(j)
```

**Availability Constraint** (machine operational windows):
```
Every job assigned to machine M must run entirely within M's available window:
start_j ≥ M.available_start  AND  start_j + duration_j ≤ M.available_end
```

### Makespan

Makespan is the **total project completion time** — how long from the start of the first job to the finish of the last job.

```
Makespan = max(start_j + duration_j)  for all jobs j
```

Minimizing makespan is the optimization objective for Task 3. A schedule that satisfies all constraints but takes 20 time units is worse than one that takes 14 time units (if 14 is achievable).

### Why These Constraints?

These four categories cover the most common real-world scheduling constraints:
- **Capacity**: Shared resource limits (one printer, one surgeon, one CPU core)
- **Deadlines**: Customer commitments, legal requirements, shift end times
- **Precedence**: Physical dependencies (can't install software before buying computer)
- **Availability**: Maintenance windows, working hours, shift schedules

---

## 19. Multi-Step Learning — How the AI Improves

### The Episode Structure

Each episode is not just one question-answer pair. The AI gets **multiple attempts**:

```
Episode Start
    │
    ▼  Attempt 1 (step 0)
AI reads problem, gives answer
System grades it, returns score + feedback
    │
    ▼  Attempt 2 (step 1) — if not done
AI reads same problem + hint from previous attempt
AI tries again with the benefit of knowing its previous score
System grades, returns score + feedback
    │
    ▼  ... (up to max_steps attempts)
Episode End
```

This mimics how **humans** learn — you don't just see a problem once. You see it, try, get feedback, refine your thinking, try again.

### Context Hints

After a failed or imperfect attempt, the environment adds hints to the context:

- Task 1 hint: "Your answer did not match the expected format or was incorrect. Reconsider the schedule — check each constraint carefully."
- Task 2 hint: "Your classification was not correct. Look more carefully at the constraint type that is violated."
- Task 3 hint: "Your repaired schedule was not fully satisfactory. Check all four constraint types and ensure minimal makespan."

These hints guide the AI toward better answers without *giving away* the answer.

### Early Termination

If the AI scores ≥ 0.95, the episode ends immediately — there's no point giving more attempts when the AI has essentially solved the problem. This makes training more efficient.

---

## 20. Scoring Deep Dive — Every Point Explained

### Task 1 Scoring (max 1.0)

| Situation | Score | Reason |
|-----------|-------|--------|
| Correct answer ("feasible" when feasible) | 1.0 | Perfect |
| Correct answer ("infeasible" when infeasible) | 1.0 | Perfect |
| Wrong answer (said "feasible" but it was infeasible) | 0.1 | Tried but failed |
| Wrong answer (said "infeasible" but it was feasible) | 0.1 | Tried but failed |
| Empty response or total gibberish | 0.0 | Didn't engage |

### Task 2 Scoring (max 1.0)

| Situation | Score | Reason |
|-----------|-------|--------|
| Exact correct category | 1.0 | Perfect diagnosis |
| resource_overload instead of capacity_exceeded (or vice versa) | 0.5 | Right family, wrong specific |
| deadline_violation instead of precedence_violation (or vice versa) | 0.5 | Right family, wrong specific |
| availability_conflict instead of anything (or vice versa) | 0.1 | Valid category, wrong family |
| resource_overload instead of deadline_violation | 0.1 | Different family |
| Unrecognized text | 0.0 | Didn't understand the format |

### Task 3 Scoring (max 1.0)

**Component: JSON Parseability (max 0.20)**

| Situation | Points |
|-----------|--------|
| Response is valid JSON | +0.20 |
| JSON is buried in text with code fences (```json ... ```) | +0.20 (after extraction) |
| JSON is buried in prose but extractable | +0.20 (after extraction) |
| Response is not JSON at all | +0.00 |

**Component: Schema Validity (max 0.20)**

| Situation | Points |
|-----------|--------|
| Has `assignments` key, all jobs present, all fields correct | +0.20 |
| JSON valid but missing `assignments` key | +0.00 |
| JSON valid but missing some jobs | +0.00 |
| JSON valid but `start_time` is negative | +0.00 |

**Component: Constraint Satisfaction (max 0.40)**

For each of the 4 constraint types:
| Situation | Points |
|-----------|--------|
| All capacity constraints satisfied | +0.10 |
| All deadline constraints satisfied | +0.10 |
| All precedence constraints satisfied | +0.10 |
| All availability constraints satisfied | +0.10 |

**Component: Makespan Optimality (max 0.20)**

| Situation | Points |
|-----------|--------|
| Makespan ≤ optimal × 1.30 (within 30% of best) | +0.20 |
| Makespan ≤ optimal × 1.60 (within 60% of best) | +0.10 |
| Makespan > optimal × 1.60 (more than 60% worse) | +0.00 |

**Example**: If the optimal makespan is 10:
- Your makespan = 12 (20% worse) → +0.20
- Your makespan = 14 (40% worse) → +0.10
- Your makespan = 17 (70% worse) → +0.00

---

## 21. End-to-End Walkthrough — One Full Example

Let's trace through a complete interaction for **Task 2 (Conflict Classification)**, using Problem Instance P02 (deadline violation).

### The Problem

```json
{
  "problem_id": "P02",
  "jobs": [
    {"id": "J1", "duration": 5, "deadline": 8, "dependencies": [], "resource_req": 1}
  ],
  "machines": [
    {"id": "M1", "capacity": 1, "available_start": 0, "available_end": 20}
  ],
  "proposed_schedule": {
    "assignments": [
      {"job_id": "J1", "machine_id": "M1", "start_time": 5}
    ]
  }
}
```

**Reading this**: Job J1 needs to be done by time 8 (deadline=8). It takes 5 time units (duration=5). But it's scheduled to start at time 5. So it finishes at 5+5=10, which is past the deadline of 8.

### Step 1 — AI Calls `/reset`

```http
POST /reset
{"task_id": "conflict_classification"}
```

**Server processes this**:
1. Validates "conflict_classification" is a known task ID ✓
2. Calls `env.reset("conflict_classification")`
3. Environment selects the next instance from the 10-instance infeasible pool (let's say it picks P02)
4. Serializes the problem instance to a JSON string
5. Builds the context string: "You are analyzing a scheduling conflict. Identify the type of constraint violation..."
6. Returns Observation

**Response**:
```json
{
  "schedule_instance": "{\"problem_id\": \"P02\", \"jobs\": [...], ...}",
  "task_id": "conflict_classification",
  "context": "You are analyzing a scheduling conflict...",
  "step_number": 0
}
```

### Step 2 — AI Analyzes and Calls `/step`

The AI reads the schedule, reasons about it, and determines the deadline is violated.

```http
POST /step
{"response": "deadline_violation", "task_id": "conflict_classification"}
```

**Server processes this**:
1. Calls `env.step(action)`
2. Environment increments step counter to 1
3. Retrieves the ConflictGrader singleton
4. Calls `grader.grade(action, current_instance)`

**Grader logic**:
1. Normalize response: `"deadline_violation"` → already lowercase with underscores ✓
2. Is it in valid_categories? Yes ✓
3. Does it match ground_truth `violation_type = "deadline_violation"`? YES ✓
4. Return score = 1.0
5. Set `last_breakdown = {predicted: "deadline_violation", expected: "deadline_violation", exact_match: True, ...}`

**Back in environment**:
1. Reward = 1.0 (clamped to [0.0, 1.0] → still 1.0)
2. Cumulative reward updated: 0.0 + 1.0 = 1.0
3. Action logged to history
4. Check termination: reward (1.0) ≥ 0.95 → `done = True`
5. Build final Observation (context includes success message)

**Response**:
```json
{
  "observation": {
    "schedule_instance": "{...}",
    "task_id": "conflict_classification",
    "context": "Well done! Constraint violation correctly identified.",
    "step_number": 1
  },
  "reward": 1.0,
  "done": true,
  "info": {
    "grading_breakdown": {
      "predicted": "deadline_violation",
      "expected": "deadline_violation",
      "exact_match": true,
      "score": 1.0,
      "feedback": "Exact match: deadline_violation"
    }
  }
}
```

Episode complete! The AI correctly identified the deadline violation and earned a perfect score in one attempt.

---

### What If The AI Was Wrong?

Let's say the AI answered `"resource_overload"` instead.

**Grader logic**:
1. Is it in valid_categories? Yes ✓
2. Does it match ground_truth "deadline_violation"? NO
3. Is "resource_overload" in the same family as "deadline_violation"? NO (resource family vs. temporal family)
4. Return score = 0.1

**Back in environment**:
1. Reward = 0.1
2. Check termination: step(1) < max_steps(5) AND reward(0.1) < 0.95 → `done = False`
3. Build retry context: "Your classification was not correct. Look more carefully at the constraint type that is violated."

**Response**:
```json
{
  "observation": {
    "context": "...original problem...\n\nYour classification was not correct. Look more carefully...",
    "step_number": 1
  },
  "reward": 0.1,
  "done": false,
  "info": {...}
}
```

The AI gets another chance, now knowing its first answer was wrong.

---

## 22. Glossary

| Term | Definition |
|------|-----------|
| **Action** | The AI's response to an observation (its answer to the problem) |
| **Agent** | An AI system that interacts with the environment |
| **API** | Application Programming Interface — a way for programs to communicate |
| **Availability Conflict** | A job is scheduled outside a machine's operational hours |
| **Baseline** | A reference evaluation using a known AI (GPT-4o-mini) |
| **Capacity** | How many jobs a machine can handle simultaneously |
| **Capacity Exceeded** | More jobs run on a machine than its capacity allows |
| **Constraint** | A rule that a valid schedule must obey |
| **Cumulative Reward** | Total score accumulated across all steps of an episode |
| **Deadline** | The latest time by which a job must finish |
| **Deadline Violation** | A job finishes after its required completion time |
| **Dense Reward** | A reward signal that gives partial credit, not just 0 or 1 |
| **Dependencies** | Other jobs that must complete before a given job can start |
| **Done** | Boolean flag indicating whether an episode has ended |
| **Duration** | How long a job takes to complete |
| **Episode** | One complete interaction from reset to termination |
| **Feasible** | A schedule that satisfies all constraints |
| **FastAPI** | Python web framework used to build the HTTP server |
| **Grader** | A module that scores an AI's response |
| **Ground Truth** | The known correct answer, used for comparison during grading |
| **HTTP** | HyperText Transfer Protocol — the language of web requests |
| **Infeasible** | A schedule that violates at least one constraint |
| **Instance** | A specific scheduling problem scenario |
| **JSON** | JavaScript Object Notation — a text format for structured data |
| **Machine** | A resource that processes jobs (in this project: abstract, not physical) |
| **Makespan** | Total project duration — the time from start of first job to end of last job |
| **Markov Decision Process (MDP)** | Mathematical framework for sequential decision-making under uncertainty |
| **Observation** | The information the AI receives about the current state |
| **OpenEnv** | A standard API specification for AI training environments |
| **Optimal Schedule** | The schedule that satisfies all constraints AND minimizes makespan |
| **Partial Credit** | Receiving some score for an answer that is partially correct |
| **Pydantic** | Python library for data validation and type checking |
| **Precedence** | The requirement that certain jobs complete before others start |
| **Precedence Violation** | A job starts before all of its required predecessors have finished |
| **Reinforcement Learning (RL)** | AI learning paradigm where agents learn from reward signals |
| **Repair** | Modifying a broken schedule to satisfy all constraints |
| **Resource Overload** | A machine with capacity 1 has multiple concurrent jobs |
| **Reward** | A numeric score (0.0 to 1.0) indicating how good an action was |
| **Round-Robin** | Cycling through items in order, wrapping back to the start |
| **Schedule** | A complete plan assigning jobs to machines at specific times |
| **Start Time** | When a job begins execution on its assigned machine |
| **State** | The full internal description of the environment at a given moment |
| **Step** | One action taken by the AI within an episode |
| **Terminal State** | A state from which no further actions are possible (episode over) |
| **Uvicorn** | The ASGI server process that hosts the FastAPI application |
| **Violation Type** | The category of constraint that is broken in an infeasible schedule |

---

*This document covers every aspect of the SchedulingOptEnv project — from the highest-level concepts to the exact logic of every function. If something is still unclear after reading this, the source code files listed throughout are the definitive reference.*
