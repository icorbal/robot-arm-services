# Robot Arm Services (RAServ)

LLM-powered task planner and executor for robot arm control. Part of the Robot Arm project — works with [Robot Arm Simulator (RASim)](https://github.com/icorbal/robot-arm-sim) for physics simulation.

## Overview

Accepts natural language tasks (e.g., "put the blue box on top of the red one"), plans gripper movements via an LLM, executes them on RASim, and verifies completion — all in an automated loop.

**Stages:**
- **1a:** Scene state perception + planning + execution (complete)
- **1b:** Stereo camera vision with LLM-based perception (current)

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
  model: "gpt-4o"
  api_key_env: "OPENAI_API_KEY"
executor:
  max_iterations: 10
  step_delay: 0.5
perception:
  mode: "scene_state"  # "scene_state" | "camera"
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
| GET | `/snapshot` | On-demand stereo perception snapshot |

### POST /task

```bash
curl -X POST http://localhost:8200/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "put the blue box on top of the red one"}'
```

**Status values:** `completed` | `max_iterations_reached` | `safety_abort` | `interaction_zone_violation` | `cancelled` | `error`

### GET /snapshot

On-demand stereo perception at the current arm position (does **not** move the arm). Use for mid-task situation assessment.

Returns: stereo images (base64), camera params, scene state, interaction zone check.

## Architecture

```
User: "put the blue box on top of the red one"
                    │
                    ▼
            ┌───────────────┐
            │    RAServ      │
            │               │
            │  ┌──────────┐ │
            │  │ Perceiver │ │  stereo images / scene state
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
2. **Perceive** — Get scene state (direct JSON or stereo camera + LLM)
3. **Plan** — LLM plans next step given scene + task + history
4. **Execute** — Send commands to RASim
5. **Safety checks:**
   - Props fallen off table → `safety_abort`
   - Props outside interaction zone → `interaction_zone_violation`
6. **Verify** — LLM checks task completion
7. **Loop** or return result

### Perception Modes

| Mode | Source | Accuracy | Cost |
|------|--------|----------|------|
| `scene_state` | RASim JSON (ground truth) | Perfect | Free |
| `camera` | Stereo images → LLM → triangulation | 1-3cm (GPT-5.4) | API calls |

**Camera perception pipeline:**
- Stereo images captured from gripper-mounted cameras (20cm baseline, 60° FOV)
- LLM (GPT-5.4) identifies objects, matches across views, estimates pixel coordinates
- DLT triangulation computes 3D world positions from stereo correspondences
- LLM also provides: orientation assessment, grip strategy, spatial relationships

**On-demand snapshots** can be taken at any point during execution via `GET /snapshot` — the arm does not need to move to the observation pose.

## Safety Systems

### Interaction Zone
The interaction zone is a rectangular area on the table visible by **both** stereo cameras and reachable by the arm. It's enforced at three levels:

1. **Scene loading** — RASim rejects props outside the zone
2. **Pre-flight** — Executor verifies all props before starting a task
3. **Post-step** — Executor checks after every command execution

### Fallen Object Detection
Coordinate-based check after each step. Objects below the table surface or outside workspace bounds trigger an immediate abort.

## Key Design Decisions

### Programmatic Spatial Verification
The verifier pre-computes spatial relationships (stacking, ordering, distances) programmatically and injects them as authoritative facts into the LLM prompt. The LLM interprets whether facts satisfy the task — it never does coordinate math.

### LLM Pixel Detection (Camera Mode)
GPT-5.4 achieves ~14px pixel accuracy at 1024×768 resolution, yielding 1-2cm triangulation accuracy with a 20cm stereo baseline. Side-by-side composite images work best — the LLM sees both views in one image and handles object matching, orientation assessment, and grip strategy without any color-based heuristics.

## Project Structure

```
robot-arm-services/
├── src/
│   ├── llm_adapter.py          # Abstract LLM provider + OpenAI implementation
│   ├── planner.py              # LLM task planner (scene → commands)
│   ├── verifier.py             # LLM completion checker + spatial analysis
│   ├── executor.py             # Plan-execute-verify loop + safety + IZ checks
│   ├── perception.py           # Stereo perception: LLM detection + DLT triangulation
│   ├── perception_cv.py        # CV color segmentation fallback
│   ├── perception_multiview.py # Multi-view perception (arm repositioning)
│   └── api.py                  # FastAPI REST endpoints
├── prompts/
│   ├── planner.txt
│   ├── verifier.txt
│   └── perceiver.txt
├── configs/
│   └── settings.yaml
├── tests/
│   ├── test_triangulation.py        # DLT triangulation math (4 tests)
│   ├── test_integration.py          # Unit tests (mocked LLM + RASim)
│   ├── test_e2e.py                  # E2e (mocked LLM + real MuJoCo)
│   ├── test_stereo_investigation.py # FOV/resolution/noise analysis
│   ├── test_baseline_sweep.py       # Baseline vs accuracy sweep
│   ├── test_llm_pixel_gpt54.py      # GPT-5.4 vs GPT-4o comparison
│   ├── test_gpt54_sweep.py          # Multi-config prompt/resolution sweep
│   ├── test_gpt54_full_perception.py # Full perception (orient, grip, stacking)
│   └── test_interaction_space.py    # Camera frustum intersection computation
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

# Stereo vision tests (requires running RASim + OPENAI_API_KEY)
python tests/test_gpt54_full_perception.py
```

## License

MIT
