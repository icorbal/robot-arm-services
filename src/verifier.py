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

    Returns a human-readable summary of computed facts for the LLM.
    """
    props = scene_state.get("props", [])
    workspace = scene_state.get("workspace", {})
    surface_height = workspace.get("surface_height", 0.42)
    bounds = workspace.get("bounds", [0.1, 0.9, -0.4, 0.4])

    if not props:
        return "No props in scene."

    lines = ["## Computed Spatial Facts (programmatic — trust these over your own math)"]

    # Per-prop position summary
    lines.append("\n### Object Positions")
    for p in props:
        pid = p["id"]
        pos = p.get("pos", [0, 0, 0])
        height = _get_prop_height(p)
        top_z = pos[2] + height / 2
        bottom_z = pos[2] - height / 2
        on_table = abs(bottom_z - surface_height) < _POS_TOLERANCE
        lines.append(
            f"- **{pid}**: center=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
            f"height={height:.4f}, top_z={top_z:.4f}, bottom_z={bottom_z:.4f}, "
            f"on_table={'YES' if on_table else 'NO'}"
        )

    # Check stacking relationships (A on top of B)
    lines.append("\n### Stacking Relationships")
    found_stacks = False
    for a in props:
        for b in props:
            if a["id"] == b["id"]:
                continue
            a_pos = a.get("pos", [0, 0, 0])
            b_pos = b.get("pos", [0, 0, 0])
            b_height = _get_prop_height(b)
            b_top_z = b_pos[2] + b_height / 2
            a_bottom_z = a_pos[2] - _get_prop_height(a) / 2

            # Check if A is on top of B:
            # - A's bottom ≈ B's top
            # - A and B are close in X/Y
            xy_dist = ((a_pos[0] - b_pos[0]) ** 2 + (a_pos[1] - b_pos[1]) ** 2) ** 0.5
            z_match = abs(a_bottom_z - b_top_z) < _STACK_Z_TOLERANCE

            if z_match and xy_dist < _POS_TOLERANCE:
                lines.append(f"- ✅ **{a['id']}** IS on top of **{b['id']}** "
                             f"(a_bottom={a_bottom_z:.4f}, b_top={b_top_z:.4f}, "
                             f"xy_dist={xy_dist:.4f})")
                found_stacks = True
            elif xy_dist < _POS_TOLERANCE and abs(a_pos[2] - b_pos[2]) > 0.02:
                # Close in XY but not matching stack height — near miss
                lines.append(f"- ❌ {a['id']} is NOT on top of {b['id']} "
                             f"(a_bottom={a_bottom_z:.4f}, b_top={b_top_z:.4f}, "
                             f"gap={abs(a_bottom_z - b_top_z):.4f}, "
                             f"xy_dist={xy_dist:.4f})")

    if not found_stacks:
        lines.append("- No objects are stacked on each other.")

    # Axis ordering (useful for arrangement/sorting tasks)
    lines.append("\n### Axis Ordering")
    for axis_name, axis_idx, direction in [
        ("X-axis (left=low, right=high)", 0, 1),
        ("Y-axis (left=high, right=low for viewer)", 1, -1),
        ("Y-axis ascending", 1, 1),
    ]:
        sorted_props = sorted(props, key=lambda p: p.get("pos", [0,0,0])[axis_idx], reverse=(direction == -1))
        order_str = " → ".join(
            f"{p['color']}({p.get('pos',[0,0,0])[axis_idx]:.4f})"
            for p in sorted_props
        )
        color_order = ", ".join(p["color"] for p in sorted_props)
        lines.append(f"- {axis_name}: {order_str}")
        lines.append(f"  Color order: [{color_order}]")

    # Check for fallen/out-of-bounds props
    x_min, x_max, y_min, y_max = bounds
    fallen = []
    for p in props:
        pos = p.get("pos", [0, 0, 0])
        if (pos[2] < surface_height - 0.05 or
                pos[0] < x_min - 0.1 or pos[0] > x_max + 0.1 or
                pos[1] < y_min - 0.1 or pos[1] > y_max + 0.1):
            fallen.append(p["id"])
    if fallen:
        lines.append(f"\n### ⚠️ Fallen/Out-of-bounds: {', '.join(fallen)}")

    # Gripper state
    arm = scene_state.get("arm", {})
    grip_open = arm.get("grip_open", True)
    grip_pos = arm.get("gripper_pos", [0, 0, 0])
    lines.append(f"\n### Gripper: {'OPEN' if grip_open else 'CLOSED'} at "
                 f"({grip_pos[0]:.4f}, {grip_pos[1]:.4f}, {grip_pos[2]:.4f})")

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
        self, scene_state: dict[str, Any], task: str
    ) -> dict[str, Any]:
        """Verify if the task has been completed.

        Computes spatial facts programmatically first, then asks the LLM
        to interpret them against the task description.

        Args:
            scene_state: Current scene state from RASim
            task: Original task description

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
