"""LLM-based task verification - checks if a task has been completed."""

import json
import logging
from pathlib import Path
from typing import Any

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class TaskVerifier:
    """Verifies task completion using an LLM."""

    def __init__(self, llm: LLMAdapter, prompt_path: str | Path | None = None):
        self._llm = llm
        self._prompt_template = self._load_prompt(prompt_path or PROMPT_DIR / "verifier.txt")
        logger.info("TaskVerifier initialized")

    def _load_prompt(self, path: str | Path) -> str:
        """Load system prompt template."""
        path = Path(path)
        with open(path) as f:
            return f.read()

    async def verify(
        self, scene_state: dict[str, Any], task: str
    ) -> dict[str, Any]:
        """Verify if the task has been completed.

        Args:
            scene_state: Current scene state from RASim
            task: Original task description

        Returns:
            Dict with "completed" (bool), "reason" (str), "confidence" (float)
        """
        workspace = scene_state.get("workspace", {})
        surface_height = workspace.get("surface_height", 0.42)

        system_prompt = (
            self._prompt_template
            .replace("{surface_height}", str(surface_height))
            .replace("{scene_state}", json.dumps(scene_state, indent=2))
            .replace("{task}", task)
        )

        logger.info(f"Verifying task completion: {task}")
        result = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=f"Is this task complete? {task}",
        )

        completed = result.get("completed", False)
        reason = result.get("reason", "No reason provided")
        confidence = result.get("confidence", 0.0)

        logger.info(
            f"Verification result: completed={completed}, "
            f"confidence={confidence:.2f}, reason={reason}"
        )

        return {
            "completed": bool(completed),
            "reason": reason,
            "confidence": float(confidence),
        }
