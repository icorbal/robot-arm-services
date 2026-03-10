"""Tests for stereo triangulation math."""

import math

import numpy as np
import pytest

from src.perception import triangulate_point


def _make_camera_params(position: list[float], look_at: list[float], fovy: float = 60.0) -> dict:
    """Create camera params dict given position and look-at point.

    Builds a rotation matrix where the camera looks from `position` toward
    `look_at`, with Y-down (MuJoCo convention: cam_xmat columns are
    camera axes in world frame).
    """
    pos = np.array(position)
    target = np.array(look_at)

    # Camera -Z axis points from camera to target (viewing direction)
    forward = target - pos
    forward = forward / np.linalg.norm(forward)

    # Camera Z axis is opposite to viewing direction (MuJoCo convention)
    cam_z = -forward

    # World up hint
    up_hint = np.array([0.0, 0.0, 1.0])

    # Camera X axis (right)
    cam_x = np.cross(up_hint, cam_z)
    if np.linalg.norm(cam_x) < 1e-6:
        # Looking straight down, use different up
        up_hint = np.array([1.0, 0.0, 0.0])
        cam_x = np.cross(up_hint, cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)

    # Camera Y axis (down in MuJoCo convention = up in world when looking down)
    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y / np.linalg.norm(cam_y)

    # cam_xmat: columns are cam_x, cam_y, cam_z in world frame
    rot_mat = np.column_stack([cam_x, cam_y, cam_z])

    return {
        "name": "test_cam",
        "fovy": fovy,
        "position": pos.tolist(),
        "rotation_matrix": rot_mat.tolist(),
    }


def _project_point(world_point: np.ndarray, cam: dict, width: int, height: int) -> list[float]:
    """Project a 3D world point to pixel coordinates using camera params."""
    fovy = math.radians(cam["fovy"])
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    R_world = np.array(cam["rotation_matrix"])
    pos = np.array(cam["position"])

    R_w2c = R_world.T
    flip = np.diag([1.0, -1.0, -1.0])
    R_cv = flip @ R_w2c
    t = -R_cv @ pos

    # Project
    p_cam = R_cv @ world_point + t
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return [float(u), float(v)]


class TestTriangulation:
    """Test stereo triangulation with known geometry."""

    def test_basic_triangulation(self):
        """Two cameras looking at a known 3D point should triangulate correctly."""
        width, height = 640, 480
        world_point = np.array([0.4, 0.0, 0.45])

        # Two cameras above, looking down at the table
        cam_left = _make_camera_params([0.3, 0.15, 0.7], [0.4, 0.0, 0.42])
        cam_right = _make_camera_params([0.3, -0.15, 0.7], [0.4, 0.0, 0.42])

        # Project the 3D point to both cameras
        px_left = _project_point(world_point, cam_left, width, height)
        px_right = _project_point(world_point, cam_right, width, height)

        # Triangulate back
        result = triangulate_point(px_left, px_right, cam_left, cam_right, width, height)

        np.testing.assert_allclose(result, world_point, atol=0.01,
                                   err_msg=f"Expected {world_point}, got {result}")

    def test_off_center_point(self):
        """Triangulate a point that's not centered between cameras."""
        width, height = 640, 480
        world_point = np.array([0.6, 0.2, 0.45])

        cam_left = _make_camera_params([0.3, 0.15, 0.7], [0.45, 0.0, 0.42])
        cam_right = _make_camera_params([0.3, -0.15, 0.7], [0.45, 0.0, 0.42])

        px_left = _project_point(world_point, cam_left, width, height)
        px_right = _project_point(world_point, cam_right, width, height)

        result = triangulate_point(px_left, px_right, cam_left, cam_right, width, height)

        np.testing.assert_allclose(result, world_point, atol=0.01,
                                   err_msg=f"Expected {world_point}, got {result}")

    def test_multiple_points(self):
        """Triangulate several points at different locations."""
        width, height = 640, 480
        cam_left = _make_camera_params([0.35, 0.1, 0.75], [0.4, 0.0, 0.42])
        cam_right = _make_camera_params([0.35, -0.1, 0.75], [0.4, 0.0, 0.42])

        test_points = [
            [0.3, 0.05, 0.45],
            [0.5, -0.1, 0.45],
            [0.4, 0.0, 0.48],  # slightly elevated (stacked object)
            [0.7, 0.3, 0.45],
        ]

        for pt in test_points:
            world_pt = np.array(pt)
            px_l = _project_point(world_pt, cam_left, width, height)
            px_r = _project_point(world_pt, cam_right, width, height)
            result = triangulate_point(px_l, px_r, cam_left, cam_right, width, height)
            np.testing.assert_allclose(
                result, world_pt, atol=0.01,
                err_msg=f"Point {pt}: expected {world_pt}, got {result}"
            )

    def test_pixel_noise_tolerance(self):
        """Triangulation should tolerate small pixel noise (~2px)."""
        width, height = 640, 480
        world_point = np.array([0.4, 0.0, 0.45])

        cam_left = _make_camera_params([0.3, 0.1, 0.7], [0.4, 0.0, 0.42])
        cam_right = _make_camera_params([0.3, -0.1, 0.7], [0.4, 0.0, 0.42])

        px_left = _project_point(world_point, cam_left, width, height)
        px_right = _project_point(world_point, cam_right, width, height)

        # Add ±2 pixel noise
        px_left_noisy = [px_left[0] + 2, px_left[1] - 1]
        px_right_noisy = [px_right[0] - 1, px_right[1] + 2]

        result = triangulate_point(
            px_left_noisy, px_right_noisy, cam_left, cam_right, width, height
        )

        # With noise, expect ~2cm accuracy
        error = np.linalg.norm(result - world_point)
        assert error < 0.05, f"Error {error:.4f}m too large with pixel noise"
