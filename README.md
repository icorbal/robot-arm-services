# Robot Arm Services (RAServ)

LLM-powered task planner and executor for robot arm control. Part of the Robot Arm project — works with [Robot Arm Simulator (RASim)](https://github.com/icorbal/robot-arm-sim) for physics simulation.

## Overview

Accepts natural language tasks (e.g., "put the blue box on top of the red one"), plans gripper movements via an LLM, executes them on RASim, and verifies completion — all in an automated loop.

**Stage 1a** (current): Uses scene state data directly from the simulator (no camera/vision pipeline).

## Quick Start

```bash
# Clone and set up
git clone https://github.com/icorbal/robot-arm-services.git
cd robot-arm-services
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure (copy and edit)
cp .env.example .env
# Edit .env with your OpenAI API key

# Start RASim first (separate terminal)
cd ../robot-arm-sim && python run.py

# Start RAServ
source .env && python run.py
```

## Prerequisites

- Python 3.11+
- [RASim](https://github.com/icorbal/robot-arm-sim) running (default: `http://localhost:8100`)
- OpenAI API key with access to gpt-4o (or your configured model)

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for the LLM planner/verifier |

**Never commit API keys.** Use environment variables or a `.env` file (gitignored).

### Settings File

Edit `configs/settings.yaml`:

```yaml
rasim:
  url: "http://localhost:8100"   # RASim endpoint
llm:
  provider: "openai"             # LLM provider (currently: openai)
  model: "gpt-4o"                # Model to use
  api_key_env: "OPENAI_API_KEY"  # Env var name for the API key
executor:
  max_iterations: 10             # Max plan-execute-verify loops
  step_delay: 0.5                # Seconds between steps
```

## Usage

```bash
# Default (port 8200)
python run.py

# Custom config
python run.py --port 8200 --config configs/settings.yaml --log-level DEBUG
```

### Sending a task

```bash
curl -X POST http://localhost:8200/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "put the blue box on top of the red one"}'
```

## REST API

Default: `http://localhost:8200`

### GET /health
```json
{"status": "ok"}
```

### GET /version

Returns the service version (git commit hash and timestamp).

```json
{"service": "raserv", "commit": "65d890e", "committed": "2026-03-10 16:00:00 +0100"}
```

### POST /task

Execute a natural language task through the plan-execute-verify loop.

**Request:**
```json
{
  "prompt": "put the blue box on top of the red one",
  "max_iterations": 10
}
```

**Response:**
```json
{
  "status": "completed",
  "task": "put the blue box on top of the red one",
  "iterations": 3,
  "final_scene_state": { "arm": { ... }, "props": [ ... ] },
  "verification": {
    "completed": true,
    "reason": "green_box IS on top of red_box (a_bottom=0.4838, b_top=0.4819, xy_dist=0.0035)",
    "confidence": 1.0
  },
  "log": [ ... ]
}
```

**Status values:** `completed` | `max_iterations_reached` | `safety_abort` | `cancelled` | `error`

### POST /task/cancel

Cancel the currently running task.

### GET /task/status

Check if a task is currently running.

## Architecture

```
User: "put the blue box on top of the red one"
                    │
                    ▼
            ┌──────────────┐
            │    RAServ     │
            │              │
            │  ┌─────────┐ │    GET /scene-state     ┌──────────┐
            │  │ Planner  │ │ ◄─────────────────────► │  RASim   │
            │  └─────────┘ │                          │ (MuJoCo) │
            │       │      │    POST /execute          │          │
            │       ▼      │ ─────────────────────►   │          │
            │  ┌─────────┐ │                          └──────────┘
            │  │Verifier │ │
            │  └─────────┘ │
            │       │      │
            │   done? ──no──► loop
            │       │      │
            │      yes     │
            └──────┼───────┘
                   ▼
              Result + Log
```

**Loop:** Get scene state → LLM plans next step → Execute on RASim → Get updated state → LLM verifies → Repeat until done.

## Key Design Decisions

### Programmatic Spatial Verification

GPT-4o cannot reliably compare coordinate values or do spatial arithmetic. The verifier pre-computes all spatial relationships programmatically (stacking, ordering, fallen objects) and injects authoritative facts into the LLM prompt. The LLM only interprets whether these facts satisfy the task — it never does the math itself.

This covers:
- **Stacking** — `_compute_spatial_facts()` checks Z-alignment and XY proximity between all prop pairs
- **Ordering** — Props are pre-sorted by each axis so the LLM can compare color sequences directly
- **Safety** — Fallen/out-of-bounds detection is purely coordinate-based

### Planner Failure Recovery

The planner receives execution history including verification failure reasons. It tracks completed sub-goals and avoids repeating them. For stacking tasks, it recognizes that objects already on the table are valid bases and focuses on the next incomplete sub-goal.

## Project Structure

```
robot-arm-services/
├── src/
│   ├── llm_adapter.py   # Abstract LLM provider + OpenAI implementation
│   ├── planner.py       # LLM-based task planner (scene state → commands)
│   ├── verifier.py      # LLM-based completion checker with programmatic spatial analysis
│   ├── executor.py      # Plan-execute-verify orchestration loop + safety checks
│   └── api.py           # FastAPI REST endpoints (incl. /version)
├── prompts/
│   ├── planner.txt      # System prompt: coordinate system, commands, planning rules
│   └── verifier.txt     # System prompt: spatial verification rules, ordering checks
├── configs/
│   └── settings.yaml    # RASim URL, LLM config, executor params
├── tests/
│   ├── test_integration.py  # Unit tests (mocked LLM + mocked RASim)
│   └── test_e2e.py          # E2e test (mocked LLM + real MuJoCo physics)
├── .env.example         # Template for environment variables
├── run.py               # Entry point
└── requirements.txt
```

## Tests

```bash
# Run all tests (unit + e2e)
python -m pytest tests/ -v

# Unit tests only (no external dependencies)
python -m pytest tests/test_integration.py -v

# E2e test (requires RASim's venv with MuJoCo installed)
python -m pytest tests/test_e2e.py -v -s
```

### Test Coverage

- **Unit tests (6):** Planner, verifier, and executor with fully mocked dependencies
- **E2e tests (2):** Starts RASim as a subprocess with real MuJoCo physics, runs the executor with a deterministic mock LLM, verifies the blue box actually ends up stacked on the red box

All tests run in ~2 seconds on a Raspberry Pi 5.

## LLM Adapter

The LLM adapter is designed for easy extension:

```python
from src.llm_adapter import LLMAdapter

class MyCustomAdapter(LLMAdapter):
    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        # Your implementation — must return parsed JSON
        ...

    async def close(self) -> None:
        ...
```

Register new providers in `create_llm_adapter()` in `llm_adapter.py`.

## Prompts

System prompts in `prompts/` use template variables (`{surface_height}`, `{scene_state}`, etc.) that are filled at runtime from the actual scene configuration. Edit these to tune LLM behavior.

## License

MIT
