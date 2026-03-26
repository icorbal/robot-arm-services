"""LLM-based task verification - checks if a task has been completed.

Includes a programmatic spatial analysis step that computes object
relationships from coordinates before sending to the LLM, so the LLM
doesn't have to do arithmetic.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# Tolerance for spatial relationship checks (meters)
_POS_TOLERANCE = 0.03
_STACK_Z_TOLERANCE = 0.02


def _get_prop_height(prop: dict[str, Any]) -> float:
    """Get the full height of a prop (2 × half-height)."""
    ptype = prop.get("type", "box")
    size = prop.get("size", [0.03, 0.03, 0.03])
    if ptype == "box":
        return size[2] * 2  # size is half-extents
    elif ptype == "cylinder":
        return size[1] * 2  # [radius, half-height]
    elif ptype == "sphere":
        return size[0] * 2  # [radius]
    return 0.06  # fallback


def _compute_spatial_facts(scene_state: dict[str, Any]) -> str:
    """Compute spatial relationships between all props programmatically.

    Returns a compact summary of computed facts for the LLM.
    """
    props = scene_state.get("props", [])
    workspace = scene_state.get("workspace", {})
    surface_height = workspace.get("surface_height", 0.42)

    if not props:
        return "No props in scene."

    lines = []

    # Per-prop: compact one-liner
    for p in props:
        pos = p.get("pos", [0, 0, 0])
        height = _get_prop_height(p)
        bottom_z = pos[2] - height / 2
        on_table = abs(bottom_z - surface_height) < _POS_TOLERANCE
        lines.append(f"{p['id']}: pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                     f"{'on_table' if on_table else 'stacked/lifted'}")

    # Only report confirmed stacks (skip negatives — absence = not stacked)
    stacks = []
    for a in props:
        for b in props:
            if a["id"] == b["id"]:
                continue
            a_pos = a.get("pos", [0, 0, 0])
            b_pos = b.get("pos", [0, 0, 0])
            b_top_z = b_pos[2] + _get_prop_height(b) / 2
            a_bottom_z = a_pos[2] - _get_prop_height(a) / 2
            xy_dist = ((a_pos[0] - b_pos[0]) ** 2 + (a_pos[1] - b_pos[1]) ** 2) ** 0.5

            if abs(a_bottom_z - b_top_z) < _STACK_Z_TOLERANCE and xy_dist < _POS_TOLERANCE:
                stacks.append(f"✅ {a['id']} ON {b['id']}")

    if stacks:
        lines.append("Stacks: " + "; ".join(stacks))
    else:
        lines.append("Stacks: none")

    return "\n".join(lines)


class TaskVerifier:
    """Verifies task completion using an LLM with programmatic spatial pre-analysis."""

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
        self, scene_state: dict[str, Any], task: str,
        images: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Verify if the task has been completed.

        Computes spatial facts programmatically first, then asks the LLM
        to interpret them against the task description.

        Args:
            scene_state: Current scene state from RASim
            task: Original task description
            images: Optional camera images (left/right observation poses) for visual verification

        Returns:
            Dict with "completed" (bool), "reason" (str), "confidence" (float)
        """
        workspace = scene_state.get("workspace", {})
        surface_height = workspace.get("surface_height", 0.42)

        # Compute spatial facts programmatically
        spatial_facts = _compute_spatial_facts(scene_state)
        logger.debug(f"Spatial facts:\n{spatial_facts}")

        system_prompt = (
            self._prompt_template
            .replace("{surface_height}", str(surface_height))
            .replace("{spatial_facts}", spatial_facts)
            .replace("{task}", task)
        )

        user_msg = "Verify completion."
        if images:
            user_msg += " Images attached."

        logger.info(f"Verifying task completion: {task}")
        result = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_msg,
            images=images,
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
