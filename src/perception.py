"""Multi-pose perception pipeline.

Uses a single camera mounted on the robot arm's end-effector.  The arm moves
to different observation poses (left / right) so the camera captures the scene
from two viewpoints with a wide angular baseline (~58°).  An LLM identifies
objects and their pixel coordinates in each view; DLT triangulation then
computes 3D world positions.
"""

import base64
import json
import logging
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# Known object sizes catalog — we know what props exist from the scene config,
# so we can fill in sizes that vision can't estimate well.
_DEFAULT_SIZES: dict[str, list[float]] = {
    "box": [0.03, 0.03, 0.03],
    "sphere": [0.03],
    "cylinder": [0.02, 0.04],
}


def triangulate_point(
    px_left: list[float],
    px_right: list[float],
    cam_left: dict,
    cam_right: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Triangulate a 3D world point from two-view pixel coordinates using DLT.

    Args:
        px_left: [u, v] pixel coordinates in the first view.
        px_right: [u, v] pixel coordinates in the second view.
        cam_left: Camera params dict with fovy, position, rotation_matrix.
        cam_right: Camera params dict with fovy, position, rotation_matrix.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        3D world position as numpy array [x, y, z].
    """
    # Build intrinsic matrix K from vertical FOV
    fovy_left = math.radians(cam_left["fovy"])
    fy_left = (height / 2.0) / math.tan(fovy_left / 2.0)
    fx_left = fy_left  # square pixels
    cx, cy = width / 2.0, height / 2.0

    fovy_right = math.radians(cam_right["fovy"])
    fy_right = (height / 2.0) / math.tan(fovy_right / 2.0)
    fx_right = fy_right

    K_left = np.array([
        [fx_left, 0, cx],
        [0, fy_left, cy],
        [0, 0, 1],
    ])
    K_right = np.array([
        [fx_right, 0, cx],
        [0, fy_right, cy],
        [0, 0, 1],
    ])

    # MuJoCo camera convention: cam_xmat is world-to-camera rotation.
    # The rotation matrix rows are the camera axes in world frame.
    # MuJoCo camera looks along -Z in camera frame, with Y pointing down.
    # We need to build P = K @ [R | t] where R, t transform world → camera.
    #
    # cam_xmat from MuJoCo gives the camera frame axes in world coordinates:
    #   row 0 = camera X axis in world (right)
    #   row 1 = camera Y axis in world (down in MuJoCo convention)
    #   row 2 = camera Z axis in world (forward = -optical axis)
    #
    # For OpenCV-style projection (Z forward, Y down), we need to negate Z:
    # R_cv = [row0; row1; -row2] of the cam_xmat interpreted as world-to-cam.
    #
    # Actually, MuJoCo cam_xmat is stored column-major as a flat 9-vector,
    # reshaped to 3x3 it gives the rotation matrix where columns are
    # camera-frame basis vectors in world coords. So R_world_to_cam = cam_xmat^T.
    # But MuJoCo camera convention: Y up, -Z is viewing direction.
    # OpenCV convention: Y down, Z is viewing direction.
    # So we need to flip Y and Z: R_cv = diag(1,-1,-1) @ R_world_to_cam

    def build_projection(K: np.ndarray, cam: dict) -> np.ndarray:
        R_world = np.array(cam["rotation_matrix"])  # 3x3, columns = cam axes in world
        pos = np.array(cam["position"])  # camera position in world

        # World-to-camera rotation
        R_w2c = R_world.T

        # MuJoCo to OpenCV convention: flip Y and Z
        flip = np.diag([1.0, -1.0, -1.0])
        R_cv = flip @ R_w2c

        # Translation: t = -R @ camera_position
        t = -R_cv @ pos

        # Projection matrix: P = K @ [R | t]
        Rt = np.hstack([R_cv, t.reshape(3, 1)])
        return K @ Rt

    P_left = build_projection(K_left, cam_left)
    P_right = build_projection(K_right, cam_right)

    # DLT triangulation: solve Ax = 0 where A is 4x4
    u_l, v_l = px_left
    u_r, v_r = px_right

    A = np.array([
        u_l * P_left[2, :] - P_left[0, :],
        v_l * P_left[2, :] - P_left[1, :],
        u_r * P_right[2, :] - P_right[0, :],
        v_r * P_right[2, :] - P_right[1, :],
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]  # dehomogenize

    return X


class Perceiver:
    """Multi-pose perception using a single end-effector camera.

    The arm moves to different observation poses (left, right) to capture the
    scene from two viewpoints.  An LLM identifies objects and pixel
    coordinates; triangulation computes 3D positions.
    """

    def __init__(
        self,
        llm: LLMAdapter,
        image_width: int = 640,
        image_height: int = 480,
        prompt_path: str | Path | None = None,
        observation_poses: list[str] | None = None,
    ):
        self._llm = llm
        self._width = image_width
        self._height = image_height
        self._prompt = self._load_prompt(prompt_path or PROMPT_DIR / "perceiver.txt")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._poses = observation_poses or ["left", "right"]
        logger.info(
            f"Perceiver initialized ({image_width}x{image_height}, "
            f"poses={self._poses})"
        )

    def _load_prompt(self, path: str | Path) -> str:
        path = Path(path)
        with open(path) as f:
            return f.read()

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._client.aclose()

    async def _capture_views(
        self, rasim_url: str
    ) -> list[dict[str, Any]]:
        """Move to observation poses and capture images + camera params.

        Returns:
            List of view dicts, each with 'image' (bytes), 'camera' (dict),
            and 'pose' (str).
        """
        url = rasim_url.rstrip("/")

        resp = await self._client.post(
            f"{url}/observe",
            json={
                "poses": self._poses,
                "width": self._width,
                "height": self._height,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        views = []
        for i, view in enumerate(data.get("views", [])):
            if not view.get("success"):
                logger.warning(f"Observation pose {self._poses[i]} failed")
                continue
            views.append({
                "pose": self._poses[i],
                "image": base64.b64decode(view["image_b64"]),
                "camera": view["camera"],
            })

        return views

    async def perceive(
        self,
        rasim_url: str,
        scene_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perceive the scene using multi-pose observation.

        Steps:
        1. Move arm to observation poses and capture images
        2. Ask LLM to identify objects and pixel coordinates in each view
        3. Triangulate 3D positions from the two viewpoints
        4. Return scene state dict compatible with planner

        Args:
            rasim_url: Base URL of the RASim service.
            scene_config: Optional scene config for known object sizes.

        Returns:
            Scene state dict with arm, props, workspace keys.
        """
        url = rasim_url.rstrip("/")

        # 1. Capture views from multiple poses
        views = await self._capture_views(rasim_url)

        if len(views) < 2:
            logger.error(
                f"Need at least 2 views for triangulation, got {len(views)}"
            )
            # Fall back to scene-state API
            resp = await self._client.get(f"{url}/scene-state")
            resp.raise_for_status()
            return resp.json()

        view_a, view_b = views[0], views[1]

        baseline = np.linalg.norm(
            np.array(view_a["camera"]["position"])
            - np.array(view_b["camera"]["position"])
        )
        logger.info(
            f"Observation baseline: {baseline:.3f}m ({baseline*100:.1f}cm) "
            f"[{view_a['pose']} → {view_b['pose']}]"
        )

        # 2. Ask LLM to identify objects and pixel coordinates
        logger.info("Sending observation images to LLM for object detection")
        system_prompt = self._prompt.replace(
            "{width}", str(self._width)
        ).replace(
            "{height}", str(self._height)
        ).replace(
            "{pose_a}", view_a["pose"]
        ).replace(
            "{pose_b}", view_b["pose"]
        )

        llm_result = await self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=(
                f"Analyze these two images captured from observation poses "
                f"'{view_a['pose']}' and '{view_b['pose']}'. "
                f"Identify all objects on the table and their pixel coordinates "
                f"in each image."
            ),
            images=[view_a["image"], view_b["image"]],
        )

        objects = llm_result.get("objects", [])
        logger.info(f"LLM detected {len(objects)} objects")

        # 3. Triangulate 3D positions
        props = []
        for obj in objects:
            obj_id = obj.get("id", "unknown")
            px_a = obj.get("left_px")
            px_b = obj.get("right_px")

            if not px_a or not px_b:
                logger.warning(
                    f"Object {obj_id} missing pixel coordinates, skipping"
                )
                continue

            try:
                world_pos = triangulate_point(
                    px_a, px_b,
                    view_a["camera"], view_b["camera"],
                    self._width, self._height,
                )
                logger.info(
                    f"Object {obj_id}: pose_a={px_a}, pose_b={px_b} "
                    f"→ world=[{world_pos[0]:.4f}, {world_pos[1]:.4f}, "
                    f"{world_pos[2]:.4f}]"
                )
            except Exception as e:
                logger.error(f"Triangulation failed for {obj_id}: {e}")
                continue

            # Look up size from known catalog
            obj_type = obj.get("type", "box")
            size = _DEFAULT_SIZES.get(obj_type, [0.03, 0.03, 0.03])

            props.append({
                "id": obj_id,
                "type": obj_type,
                "color": obj.get("color", "unknown"),
                "pos": [
                    round(float(world_pos[0]), 4),
                    round(float(world_pos[1]), 4),
                    round(float(world_pos[2]), 4),
                ],
                "size": size,
            })

        # Get current arm state + workspace from RASim
        resp = await self._client.get(f"{url}/scene-state")
        resp.raise_for_status()
        scene = resp.json()

        return {
            "arm": scene.get("arm", {}),
            "props": props,
            "workspace": scene.get("workspace", {}),
        }

    async def get_camera_images(
        self, rasim_url: str
    ) -> tuple[bytes, bytes]:
        """Capture observation images from two poses (for verifier use).

        Returns:
            Tuple of (pose_a_image_bytes, pose_b_image_bytes).
        """
        views = await self._capture_views(rasim_url)
        if len(views) < 2:
            raise RuntimeError(
                f"Need 2 views, got {len(views)}"
            )
        return views[0]["image"], views[1]["image"]
