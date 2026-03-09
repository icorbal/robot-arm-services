# Robot Arm Services (RAServ)

LLM-powered task planner and executor for robot arm control. Part of the Robot Arm project — works with [Robot Arm Simulator (RASim)](https://github.com/iagocorbal/robot-arm-sim) for physics simulation.

## Overview

Accepts natural language tasks (e.g., "put the blue box on top of the red one"), plans gripper movements via an LLM, executes them on RASim, and verifies completion — all in an automated loop.

**Stage 1a** (current): Uses scene state data directly from the simulator (no camera/vision pipeline).

## Quick Start

```bash
# Clone and set up
git clone https://github.com/iagocorbal/robot-arm-services.git
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
- [RASim](https://github.com/iagocorbal/robot-arm-sim) running (default: `http://localhost:8100`)
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
  "iterations": 6,
  "final_scene_state": { "arm": { ... }, "props": [ ... ] },
  "verification": {
    "completed": true,
    "reason": "Blue box is stacked on top of the red box",
    "confidence": 0.95
  },
  "log": [
    {
      "iteration": 1,
      "phase": "executed",
      "step_description": "Open gripper and position above blue box",
      "commands": [ ... ],
      "results": [ ... ],
      "verification": { "completed": false, "reason": "..." }
    }
  ]
}
```

**Status values:** `completed` | `max_iterations_reached` | `error`

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

## Project Structure

```
robot-arm-services/
├── src/
│   ├── llm_adapter.py   # Abstract LLM provider + OpenAI implementation
│   ├── planner.py       # LLM-based task planner (scene state → commands)
│   ├── verifier.py      # LLM-based completion checker
│   ├── executor.py      # Plan-execute-verify orchestration loop
│   └── api.py           # FastAPI REST endpoints
├── prompts/
│   ├── planner.txt      # System prompt: coordinate system, commands, planning rules
│   └── verifier.txt     # System prompt: spatial verification rules
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
