"""Multi-view perception pipeline.

Uses the arm's observation poses (left/right/center) instead of stereo cameras
to get a much wider baseline for triangulation, making the pipeline more
tolerant of LLM pixel coordinate errors.

With stereo snake eyes (4cm baseline): 28px disparity, 30px error = useless
With multi-view poses (15cm baseline): 250px disparity, 30px error = ~5cm accuracy
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import base64
import httpx
import numpy as np

from .llm_adapter import LLMAdapter
from .perception import triangulate_point, _DEFAULT_SIZES

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class MultiViewPerceiver:
    """Multi-view perception using arm poses for wider baseline triangulation."""

    def __init__(
        self,
        llm: LLMAdapter,
        image_width: int = 640,
        image_height: int = 480,
    ):
        self._llm = llm
        self._width = image_width
        self._height = image_height
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(
            f"MultiViewPerceiver initialized ({image_width}x{image_height})"
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def perceive(
        self,
        rasim_url: str,
        scene_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perceive the scene using multi-view observation.

        Steps:
        1. Capture images from left and right observation poses
        2. Ask LLM to identify objects and pixel coordinates in each view
        3. Triangulate 3D positions using the wider baseline
        4. Return scene state dict
        """
        url = rasim_url.rstrip("/")

        # 1. Multi-view observation
        logger.info("Capturing multi-view observations (left + right poses)")
        resp = await self._client.post(
            f"{url}/observe",
            json={
                "poses": ["left", "right"],
                "width": self._width,
                "height": self._height,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        views = data["views"]
        if len(views) < 2 or not all(v.get("success") for v in views):
            logger.error("Failed to capture multi-view observations")
            return {"arm": {}, "props": [], "workspace": {}}

        left_view = views[0]
        right_view = views[1]

        img_left = base64.b64decode(left_view["image_b64"])
        img_right = base64.b64decode(right_view["image_b64"])

        cam_left = left_view["camera"]
        cam_right = right_view["camera"]

        baseline = np.linalg.norm(
            np.array(cam_left["position"]) - np.array(cam_right["position"])
        )
        logger.info(f"Multi-view baseline: {baseline:.3f}m ({baseline*100:.1f}cm)")

        # 2. Ask LLM to identify objects
        logger.info("Sending multi-view images to LLM for object detection")

        system_prompt = f"""You are analyzing two images from a robot arm camera at different observation poses.
The camera moved between shots, providing different viewpoints of the same table scene.

## Image Details
- Resolution: {self._width} x {self._height} pixels
- First image: LEFT pose (camera rotated ~29° left)
- Second image: RIGHT pose (camera rotated ~29° right)
- Objects are colored geometric shapes: boxes, cylinders, spheres

## Your Task
Identify ALL objects on the table. For EACH object provide:
1. ID (e.g., "red_box")
2. Color and type
3. Pixel coordinates [u, v] of object CENTER in the FIRST (left pose) image
4. Pixel coordinates [u, v] of object CENTER in the SECOND (right pose) image

## Pixel Coordinate System
- Origin (0, 0) is TOP-LEFT corner
- u increases rightward (0 to {self._width})
- v increases downward (0 to {self._height})

## Important
- Objects will appear at VERY different positions between the two views
  (100-300 pixels difference in both u and v)
- Match objects by their color and shape, not by position
- Be as precise as possible with coordinates

## Output - ONLY valid JSON:
{{"objects": [{{"id": "red_box", "color": "red", "type": "box", "left_px": [u, v], "right_px": [u, v]}}]}}"""

        llm_result = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=(
                "Analyze these two images from different observation poses. "
                "First image is the LEFT pose view, second is the RIGHT pose view. "
                "Identify all objects and their precise pixel coordinates in each image."
            ),
            images=[img_left, img_right],
        )

        objects = llm_result.get("objects", [])
        logger.info(f"LLM detected {len(objects)} objects")

        # 3. Triangulate 3D positions
        props = []
        for obj in objects:
            obj_id = obj.get("id", "unknown")
            left_px = obj.get("left_px")
            right_px = obj.get("right_px")

            if not left_px or not right_px:
                logger.warning(f"Object {obj_id} missing pixel coordinates")
                continue

            try:
                world_pos = triangulate_point(
                    left_px, right_px,
                    cam_left, cam_right,
                    self._width, self._height,
                )
                logger.info(
                    f"Object {obj_id}: left_px={left_px}, right_px={right_px} "
                    f"→ world=[{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]"
                )
            except Exception as e:
                logger.error(f"Triangulation failed for {obj_id}: {e}")
                continue

            obj_type = obj.get("type", "box")
            size = _DEFAULT_SIZES.get(obj_type, [0.03, 0.03, 0.03])

            props.append({
                "id": obj_id,
                "type": obj_type,
                "color": obj.get("color", "unknown"),
                "pos": [round(float(world_pos[0]), 4),
                        round(float(world_pos[1]), 4),
                        round(float(world_pos[2]), 4)],
                "size": size,
            })

        # Get current arm state
        resp = await self._client.get(f"{url}/scene-state")
        resp.raise_for_status()
        scene = resp.json()

        return {
            "arm": scene.get("arm", {}),
            "props": props,
            "workspace": scene.get("workspace", {}),
        }
