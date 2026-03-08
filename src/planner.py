"""LLM-based task planner - plans gripper commands from scene state and task description."""

import json
import logging
from pathlib import Path
from typing import Any

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


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
        self, scene_state: dict[str, Any], task: str
    ) -> dict[str, Any]:
        """Plan the next step to accomplish the task.

        Args:
            scene_state: Current scene state from RASim
            task: Natural language task description

        Returns:
            Dict with "step_description" and "commands" list
        """
        workspace = scene_state.get("workspace", {})
        bounds = workspace.get("bounds", [0.1, 0.9, -0.4, 0.4])
        surface_height = workspace.get("surface_height", 0.42)

        system_prompt = self._prompt_template.format(
            surface_height=surface_height,
            x_min=bounds[0],
            x_max=bounds[1],
            y_min=bounds[2],
            y_max=bounds[3],
            scene_state=json.dumps(scene_state, indent=2),
            task=task,
        )

        logger.info(f"Planning next step for task: {task}")
        result = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=f"Plan the next step for: {task}",
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
