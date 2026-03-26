"""LLM-based task planner - plans gripper commands from scene state and task description."""

import json
import logging
from pathlib import Path
from typing import Any

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _slim_scene_state(state: dict) -> dict:
    """Strip scene state to only what the LLM needs for planning.

    Removes: arm joint positions, gripper rotation, workspace bounds
    (those are already in the prompt template as named values).
    """
    arm = state.get("arm", {})
    slim = {
        "arm": {
            "gripper_pos": arm.get("gripper_pos"),
            "grip_open": arm.get("grip_open"),
        },
        "props": [
            {
                "id": p["id"],
                "type": p["type"],
                "color": p.get("color", ""),
                "pos": p["pos"],
                "size": p["size"],
            }
            for p in state.get("props", [])
        ],
    }
    return slim


class TaskPlanner:
    """Plans robot arm movements using an LLM."""

    def __init__(self, llm: LLMAdapter, prompt_path: str | Path | None = None):
        self._llm = llm
        self._prompt_template = self._load_prompt(prompt_path or PROMPT_DIR / "planner.txt")
        logger.info("TaskPlanner initialized")

    def _load_prompt(self, path: str | Path) -> str:
        """Load system prompt template."""
        path = Path(path)
        with open(path) as f:
            return f.read()

    async def plan_next_step(
        self, scene_state: dict[str, Any], task: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Plan the next step to accomplish the task.

        Args:
            scene_state: Current scene state from RASim
            task: Natural language task description
            history: List of previous execution log entries

        Returns:
            Dict with "step_description" and "commands" list
        """
        workspace = scene_state.get("workspace", {})
        bounds = workspace.get("bounds", [0.1, 0.9, -0.4, 0.4])
        surface_height = workspace.get("surface_height", 0.42)

        # Strip scene state to only what the LLM needs (saves ~30% tokens)
        slim_state = _slim_scene_state(scene_state)

        # Compress history: only keep last verification failure reason
        # The current scene state IS the result of all previous actions —
        # the LLM doesn't need the full command log.
        history_text = "None (first step)"
        if history:
            # Find last verification failure
            for entry in reversed(history):
                verif = entry.get("verification", {})
                if not verif.get("completed", True):
                    reason = verif.get("reason", "")
                    desc = entry.get("step_description", "")
                    if reason:
                        history_text = f"Last attempt: {desc} → {reason}"
                    break
                elif verif.get("completed"):
                    # Last step succeeded — note completed sub-goals
                    history_text = f"Completed so far: {verif.get('reason', '')}"
                    break

        system_prompt = (
            self._prompt_template
            .replace("{surface_height}", str(surface_height))
            .replace("{x_min}", str(bounds[0]))
            .replace("{x_max}", str(bounds[1]))
            .replace("{y_min}", str(bounds[2]))
            .replace("{y_max}", str(bounds[3]))
            .replace("{scene_state}", json.dumps(slim_state, separators=(",", ":")))
            .replace("{task}", task)
            .replace("{history}", history_text)
        )

        logger.info(f"Planning next step for task: {task}")
        result = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt="Plan the next phase.",
        )

        # Validate response format
        if "commands" not in result:
            logger.error(f"Invalid planner response: {result}")
            raise ValueError("Planner did not return 'commands' field")

        commands = result["commands"]
        step_desc = result.get("step_description", "Unknown step")
        logger.info(f"Planned step: {step_desc} ({len(commands)} commands)")

        return {
            "step_description": step_desc,
            "commands": commands,
        }
