# Robot Arm Services (RAServ)

LLM-powered task planner and executor for robot arm control. Part of the Robot Arm project — works with [Robot Arm Simulator (RASim)](https://github.com/icorbal/robot-arm-sim) for physics simulation.

## Overview

Accepts natural language tasks (e.g., "put the blue box on top of the red one"), plans gripper movements via an LLM, executes them on RASim, and verifies completion — all in an automated loop.

## Quick Start

```bash
git clone https://github.com/icorbal/robot-arm-services.git
cd robot-arm-services
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your OpenAI API key

# Start RASim first (separate terminal)
cd ../robot-arm-sim && MUJOCO_GL=egl python run.py

# Start RAServ
source .env && python run.py
```

## Prerequisites

- Python 3.11+
- [RASim](https://github.com/icorbal/robot-arm-sim) running (default: `http://localhost:8100`)
- OpenAI API key (GPT-5.4 recommended for camera perception; GPT-4o works for scene-state mode)

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |

**Never commit API keys.** Use environment variables or `.env` (gitignored).

### Settings (`configs/settings.yaml`)

```yaml
rasim:
  url: "http://localhost:8100"
llm:
  provider: "openai"
  model: "gpt-5.4"
  api_key_env: "OPENAI_API_KEY"
executor:
  max_iterations: 10
  step_delay: 0.5
perception:
  mode: "camera"  # "camera" for LLM-based perception, "scene_state" for direct JSON
```

## REST API

Default: `http://localhost:8200`

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/version` | Git commit + timestamp |
| POST | `/task` | Execute a natural language task |
| POST | `/task/cancel` | Cancel running task |
| GET | `/task/status` | Check if a task is running |
| GET | `/snapshot` | On-demand perception snapshot |

### POST /task

```bash
curl -X POST http://localhost:8200/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "put the blue box on top of the red one"}'
```

**Status values:** `completed` | `max_iterations_reached` | `safety_abort` | `interaction_zone_violation` | `cancelled` | `error`

### GET /snapshot

On-demand perception snapshot at the current arm position (does **not** move the arm). Use for mid-task situation assessment.

## Architecture

```
User: "put the blue box on top of the red one"
                    │
                    ▼
            ┌───────────────┐
            │    RAServ      │
            │               │
            │  ┌──────────┐ │
            │  │ Perceiver │ │  observation images / scene state
            │  └──────────┘ │ ◄──────────────────────────► ┌──────────┐
            │       │       │                               │  RASim   │
            │  ┌──────────┐ │  POST /execute                │ (MuJoCo) │
            │  │ Planner  │ │ ────────────────────────────► │          │
            │  └──────────┘ │                               │          │
            │       │       │  GET /interaction-zone/check  │          │
            │  ┌──────────┐ │ ◄────────────────────────────►│          │
            │  │ Verifier │ │                               └──────────┘
            │  └──────────┘ │
            │       │       │
            │   done? ──no──► loop
            │       │       │
            │      yes      │
            └───────┼───────┘
                    ▼
              Result + Log
```

### Execution Loop

1. **Pre-flight** — Verify all props are in the interaction zone
2. **Perceive** — Scan scene via camera (Phase 1: quick left/right observation)
3. **Plan** — LLM plans next step given scene + task + history
4. **Execute** — Send commands to RASim
5. **Grab verify** — If `grip_close` was issued, targeted observation checks the object was actually picked up
6. **Safety checks:**
   - Props fallen off table → `safety_abort`
   - Props outside interaction zone → `interaction_zone_violation`
7. **Verify** — LLM checks task completion
8. **Loop** — On failure, escalate to refined perception (Phase 2) and retry

### Perception Pipeline

A single camera is mounted on the arm's end-effector. Perception is adaptive:

**Phase 1 — Scan (default):**
- Arm moves to left/right observation poses (~23° angular separation, ~19cm baseline)
- LLM identifies all objects and estimates pixel coordinates in both views
- DLT triangulation computes 3D positions (~1-2cm accuracy with GPT-5.4)
- Sufficient for most gripping tasks

**Phase 2 — Targeted refinement (on-demand):**
- Triggered when a grab fails or task verification fails
- For each object, camera is re-aimed directly at it via `POST /observe/targeted`
- High-res (1024×768) left/right images captured for that specific object
- Sub-cm triangulation accuracy

**Grab verification:**
- After every `grip_close`, a targeted observation checks the object's previous position
- If the object is still there → grab missed → escalate to Phase 2 and replan
- If the object is gone → grab confirmed → continue execution

### Perception Modes

| Mode | Source | Accuracy | Cost |
|------|--------|----------|------|
| `scene_state` | RASim JSON (ground truth) | Perfect | Free |
| `camera` | Multi-pose images → LLM → triangulation | 1-2cm (GPT-5.4) | API calls |

## Safety Systems

### Interaction Zone
The interaction zone is a rectangular area on the table visible from the observation poses and reachable by the arm. Enforced at three levels:

1. **Scene loading** — RASim rejects props outside the zone
2. **Pre-flight** — Executor verifies all props before starting a task
3. **Post-step** — Executor checks after every command execution

### Fallen Object Detection
Coordinate-based check after each step. Objects below the table surface or outside workspace bounds trigger an immediate abort.

## Key Design Decisions

### Programmatic Spatial Verification
The verifier pre-computes spatial relationships (stacking, ordering, distances) programmatically and injects them as authoritative facts into the LLM prompt. The LLM interprets whether facts satisfy the task — it never does coordinate math.

### Adaptive Perception
Phase 1 scan is fast and cheap (1 LLM call). Phase 2 targeted refinement costs more but achieves sub-cm accuracy. The executor only escalates to Phase 2 when needed — grab failures or task verification failures. This typically saves 4+ LLM calls per task.

### LLM Pixel Detection
GPT-5.4 achieves ~14px mean pixel accuracy at 1024×768 resolution. The multi-pose approach (arm moves to left and right observation poses) provides a ~19cm baseline with ~23° angular separation, yielding 1-2cm triangulation accuracy.

## Project Structure

```
robot-arm-services/
├── src/
│   ├── llm_adapter.py       # Abstract LLM provider + OpenAI implementation
│   ├── planner.py            # LLM task planner (scene → commands)
│   ├── verifier.py           # LLM completion checker + spatial analysis
│   ├── executor.py           # Plan-execute-verify loop + grab verification + safety
│   ├── perception.py         # Adaptive two-phase perception: LLM detection + DLT triangulation
│   └── api.py                # FastAPI REST endpoints
├── prompts/
│   ├── planner.txt           # Planner prompt (scene-state mode)
│   ├── planner_camera.txt    # Planner prompt (camera mode)
│   ├── verifier.txt          # Verifier prompt (scene-state mode)
│   ├── verifier_camera.txt   # Verifier prompt (camera mode)
│   ├── perceiver.txt         # Perceiver prompt (scene scan — all objects)
│   └── perceiver_targeted.txt # Perceiver prompt (targeted — single object)
├── configs/
│   └── settings.yaml
├── tests/
│   ├── test_triangulation.py # DLT triangulation math
│   ├── test_integration.py   # Unit tests (mocked LLM + RASim)
│   └── test_e2e.py           # E2e (mocked LLM + real MuJoCo)
├── .env.example
├── run.py
└── requirements.txt
```

## Tests

```bash
# Core tests (no API key needed)
python -m pytest tests/test_triangulation.py tests/test_integration.py -v

# E2e (requires RASim's MuJoCo)
python -m pytest tests/test_e2e.py -v -s
```

## AI Disclaimer

This project was built with significant AI assistance (architecture, code, docs). Human-directed, AI-implemented.

## License

MIT
