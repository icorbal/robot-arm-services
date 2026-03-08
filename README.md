# Robot Arm Services (RAServ)

LLM-powered task planner and executor for the Robot Arm Simulator (RASim). Accepts natural language tasks, plans gripper movements via an LLM, executes them on RASim, and verifies completion.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

## Prerequisites

- [Robot Arm Simulator (RASim)](../robot-arm-sim/) must be running on port 8100
- OpenAI API key with access to gpt-4o (or configured model)

## Usage

```bash
# Start RASim first (in another terminal)
cd ../robot-arm-sim && python run.py

# Start RAServ
python run.py

# Custom config
python run.py --port 8200 --config configs/settings.yaml
```

## API

### `GET /health`
Health check.
```json
{"status": "ok"}
```

### `POST /task`
Execute a natural language task:
```bash
curl -X POST http://localhost:8200/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "put the blue box on top of the red one"}'
```

Response:
```json
{
  "status": "completed",
  "task": "put the blue box on top of the red one",
  "iterations": 5,
  "final_scene_state": { ... },
  "verification": {
    "completed": true,
    "reason": "Blue box is positioned on top of the red box",
    "confidence": 0.95
  },
  "log": [
    {
      "iteration": 1,
      "phase": "executed",
      "step_description": "Open gripper and approach blue box",
      "commands": [...],
      "results": [...],
      "verification": { ... }
    }
  ]
}
```

## Architecture

```
User Task (natural language)
    │
    ▼
┌──────────┐    scene state    ┌──────────┐
│  RAServ  │ ◄───────────────► │  RASim   │
│          │    commands        │ (MuJoCo) │
└──────────┘                   └──────────┘
    │
    ▼
Plan-Execute-Verify Loop:
1. Get scene state from RASim
2. LLM plans next gripper commands
3. Execute commands on RASim
4. Get updated state
5. LLM verifies if task is complete
6. Loop until done or max iterations
```

### Components

- **llm_adapter.py** — Abstract LLM provider with OpenAI implementation
- **planner.py** — LLM-based task planning (scene → commands)
- **verifier.py** — LLM-based completion verification
- **executor.py** — Orchestrates the plan-execute-verify loop
- **api.py** — FastAPI REST endpoints

## Configuration

Edit `configs/settings.yaml`:
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
```

## Tests

```bash
pytest tests/ -v
```

Tests use mocked RASim API and LLM responses to test the full pipeline without external dependencies.

## Prompts

System prompts are in `prompts/`:
- **planner.txt** — Explains coordinate system, gripper capabilities, output format
- **verifier.txt** — Spatial relationship checking rules
