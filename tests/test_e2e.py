"""End-to-end test: RASim (real MuJoCo physics) + RAServ (mocked LLM).

Starts RASim as a subprocess, then runs RAServ's executor with a
deterministic mock LLM to place the blue box on top of the red box.
Verifies the physics simulation actually achieves the goal.

Run:  python -m pytest tests/test_e2e.py -v -s
"""

import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# RAServ imports (via conftest.py sys.path)
from src.executor import TaskExecutor
from src.planner import TaskPlanner
from src.verifier import TaskVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RASIM_ROOT = Path(__file__).resolve().parent.parent.parent / "robot-arm-sim"
RASIM_PORT = 8101
TASK = "put the blue box on top of the red box"


# --- Deterministic mock LLM ---

class MockLLMForStackTask:
    """Returns a fixed sequence of planner/verifier responses for stacking."""

    def __init__(self):
        self._plan_count = 0

    async def generate(self, system_prompt: str, user_prompt: str) -> dict:
        if "Plan the next step" in user_prompt:
            return self._plan_next()
        elif "Is this task complete" in user_prompt:
            return self._verify(system_prompt)
        return {"completed": False, "reason": "Unknown", "confidence": 0.0}

    def _plan_next(self) -> dict:
        self._plan_count += 1
        steps = {
            1: {
                "step_description": "Open gripper and position above blue box",
                "commands": [
                    {"type": "grip_open"},
                    {"type": "move_to", "position": [0.5, -0.15, 0.55]},
                ],
            },
            2: {
                "step_description": "Lower to blue box and grasp",
                "commands": [
                    {"type": "move_to", "position": [0.5, -0.15, 0.465]},
                    {"type": "grip_close"},
                ],
            },
            3: {
                "step_description": "Lift the blue box",
                "commands": [{"type": "move_to", "position": [0.5, -0.15, 0.62]}],
            },
            4: {
                "step_description": "Move above the red box",
                "commands": [{"type": "move_to", "position": [0.5, 0.15, 0.62]}],
            },
            5: {
                "step_description": "Lower onto red box and release",
                "commands": [
                    {"type": "move_to", "position": [0.5, 0.15, 0.50]},
                    {"type": "grip_open"},
                ],
            },
            6: {
                "step_description": "Rise away from stacked boxes",
                "commands": [{"type": "move_to", "position": [0.5, 0.15, 0.65]}],
            },
        }
        return steps.get(self._plan_count, {"step_description": "Done", "commands": []})

    def _verify(self, system_prompt: str) -> dict:
        if self._plan_count >= 6:
            # Parse actual positions from embedded scene state
            blue_match = re.search(
                r'"id":\s*"blue_box".*?"pos":\s*\[([\d.,\s-]+)\]',
                system_prompt, re.DOTALL,
            )
            red_match = re.search(
                r'"id":\s*"red_box".*?"pos":\s*\[([\d.,\s-]+)\]',
                system_prompt, re.DOTALL,
            )
            if blue_match and red_match:
                bz = [float(x) for x in blue_match.group(1).split(",")]
                rz = [float(x) for x in red_match.group(1).split(",")]
                dz = bz[2] - rz[2]
                if dz > 0.03:
                    return {
                        "completed": True,
                        "reason": f"Blue box {dz:.3f}m above red — stacked",
                        "confidence": 0.95,
                    }
            return {"completed": True, "reason": "Full sequence executed", "confidence": 0.9}

        return {"completed": False, "reason": "More steps needed", "confidence": 0.8}

    async def close(self) -> None:
        pass


# --- Fixtures ---

@pytest.fixture(scope="module")
def rasim_server():
    """Start RASim as a subprocess for the entire test module."""
    venv_python = RASIM_ROOT / "venv" / "bin" / "python"
    proc = subprocess.Popen(
        [
            str(venv_python), "run.py",
            "--port", str(RASIM_PORT),
            "--host", "127.0.0.1",
            "--log-level", "WARNING",
        ],
        cwd=str(RASIM_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    url = f"http://127.0.0.1:{RASIM_PORT}"
    for attempt in range(40):
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                logger.info(f"RASim ready on port {RASIM_PORT} (attempt {attempt+1})")
                break
        except Exception:
            pass
        time.sleep(0.3)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"RASim failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    yield url

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# --- Tests ---

@pytest.mark.asyncio
async def test_stack_blue_on_red(rasim_server: str):
    """Full e2e: mocked LLM plans pick-and-place, real MuJoCo physics executes."""
    mock_llm = MockLLMForStackTask()
    planner = TaskPlanner(mock_llm)
    verifier = TaskVerifier(mock_llm)

    executor = TaskExecutor(
        planner=planner,
        verifier=verifier,
        rasim_url=rasim_server,
        max_iterations=10,
        step_delay=0.0,
    )

    result = await executor.execute_task(TASK)

    logger.info(f"Result: status={result['status']}, iterations={result['iterations']}")

    assert result["status"] == "completed", (
        f"Task failed with status '{result.get('status')}': {result.get('error', 'unknown')}"
    )
    assert result["iterations"] <= 8

    # Verify physics: blue box is actually stacked on red box
    props = result.get("final_scene_state", {}).get("props", [])
    blue = next((p for p in props if p["id"] == "blue_box"), None)
    red = next((p for p in props if p["id"] == "red_box"), None)

    assert blue is not None, "Blue box missing from final state"
    assert red is not None, "Red box missing from final state"

    dz = blue["pos"][2] - red["pos"][2]
    dx = abs(blue["pos"][0] - red["pos"][0])
    dy = abs(blue["pos"][1] - red["pos"][1])

    logger.info(f"Blue: {blue['pos']}, Red: {red['pos']}")
    logger.info(f"dz={dz:.4f}, dx={dx:.4f}, dy={dy:.4f}")

    assert dz > 0.03, f"Blue not above red: dz={dz:.4f}m"
    assert dx < 0.06, f"X misalignment: {dx:.4f}m"
    assert dy < 0.06, f"Y misalignment: {dy:.4f}m"

    logger.info("✅ Blue box stacked on red box — physics verified!")
    await executor.close()


@pytest.mark.asyncio
async def test_scene_reset(rasim_server: str):
    """Verify scene resets to initial state."""
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{rasim_server}/scene/reset")
        assert r.status_code == 200

        state = r.json()["scene_state"]
        blue = next(p for p in state["props"] if p["id"] == "blue_box")
        red = next(p for p in state["props"] if p["id"] == "red_box")

        assert abs(blue["pos"][2] - 0.45) < 0.02
        assert abs(red["pos"][2] - 0.45) < 0.02
        assert abs(blue["pos"][1] - (-0.15)) < 0.02

        logger.info("✅ Scene reset verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
