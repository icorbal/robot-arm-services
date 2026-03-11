"""Systematic investigation of stereo triangulation pipeline.

Tests different FOV and resolution combos to isolate error sources:
- LLM pixel accuracy vs CV ground truth
- Impact of FOV (60°, 90°, 120°, 160°)
- Impact of resolution (320x240, 640x480, 1024x768, 1280x960)
- Triangulation accuracy for each config

Requires RASim running on localhost:8100.
"""

import asyncio
import base64
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.perception import triangulate_point
from src.perception_cv import detect_objects_cv

RASIM_URL = "http://localhost:8100"

# Known ground truth from scene config (blocks.yaml)
GROUND_TRUTH = {
    "red_box": [0.3, 0.05, 0.45],
    "blue_box": [0.3, -0.15, 0.45],
    "green_box": [0.4, 0.0, 0.45],
    "yellow_cylinder": [0.35, 0.15, 0.46],
}

# Color mapping for CV detection matching
COLOR_TO_ID = {
    "red": "red_box",
    "blue": "blue_box",
    "green": "green_box",
    "yellow": "yellow_cylinder",
}

# Test configurations
FOVS = [60, 90, 120, 160]
RESOLUTIONS = [(320, 240), (640, 480), (1024, 768)]


async def modify_camera_fov(fov: float):
    """Modify camera FOV by reloading the model with new XML.

    Since we can't hot-swap camera params in MuJoCo, we modify the XML
    and call /reset. But the XML is built from configs...

    Alternative: We'll just test with the current FOV (90°) and use
    the projection math to simulate what different FOVs would give us.
    """
    pass  # See approach below


async def capture_stereo(client: httpx.AsyncClient, width: int, height: int):
    """Capture stereo pair and camera params."""
    # First move to observe pose
    resp = await client.post(
        f"{RASIM_URL}/execute",
        json={"commands": [{"type": "observe"}]},
        timeout=60.0,
    )
    resp.raise_for_status()
    scene_state = resp.json().get("scene_state", {})

    # Capture images
    resp_left = await client.get(
        f"{RASIM_URL}/eye/left",
        params={"width": width, "height": height},
    )
    resp_left.raise_for_status()

    resp_right = await client.get(
        f"{RASIM_URL}/eye/right",
        params={"width": width, "height": height},
    )
    resp_right.raise_for_status()

    # Get camera params
    resp_params = await client.get(f"{RASIM_URL}/eye/params")
    resp_params.raise_for_status()
    cam_params = resp_params.json()

    return resp_left.content, resp_right.content, cam_params, scene_state


def project_world_to_pixel(world_point, cam, width, height):
    """Project a 3D world point to pixel coordinates."""
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

    p_cam = R_cv @ np.array(world_point) + t
    if p_cam[2] <= 0:
        return None  # Behind camera
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return [float(u), float(v)]


def compute_theoretical_disparity(cam_left, cam_right, width, height, world_point):
    """Compute what the pixel disparity should be between left and right cameras."""
    px_l = project_world_to_pixel(world_point, cam_left, width, height)
    px_r = project_world_to_pixel(world_point, cam_right, width, height)
    if px_l is None or px_r is None:
        return None
    du = abs(px_l[0] - px_r[0])
    dv = abs(px_l[1] - px_r[1])
    return {"left": px_l, "right": px_r, "du": du, "dv": dv}


def simulate_fov_disparity(cam_left, cam_right, width, height, target_fov):
    """Simulate what disparity would be at a different FOV.

    Creates modified camera params with the target FOV and computes
    theoretical pixel positions.
    """
    cam_l_mod = dict(cam_left)
    cam_r_mod = dict(cam_right)
    cam_l_mod["fovy"] = target_fov
    cam_r_mod["fovy"] = target_fov

    results = {}
    for obj_id, world_pos in GROUND_TRUTH.items():
        disp = compute_theoretical_disparity(cam_l_mod, cam_r_mod, width, height, world_pos)
        if disp:
            results[obj_id] = disp
    return results


def measure_noise_tolerance(cam_left, cam_right, width, height, fov, noise_levels):
    """Measure how much pixel noise the triangulation can tolerate at a given FOV."""
    cam_l = dict(cam_left)
    cam_r = dict(cam_right)
    cam_l["fovy"] = fov
    cam_r["fovy"] = fov

    results = []
    for noise_px in noise_levels:
        errors = []
        for obj_id, world_pos in GROUND_TRUTH.items():
            px_l = project_world_to_pixel(world_pos, cam_l, width, height)
            px_r = project_world_to_pixel(world_pos, cam_r, width, height)
            if px_l is None or px_r is None:
                continue

            # Add noise
            np.random.seed(42)
            px_l_noisy = [px_l[0] + np.random.uniform(-noise_px, noise_px),
                          px_l[1] + np.random.uniform(-noise_px, noise_px)]
            px_r_noisy = [px_r[0] + np.random.uniform(-noise_px, noise_px),
                          px_r[1] + np.random.uniform(-noise_px, noise_px)]

            try:
                result = triangulate_point(px_l_noisy, px_r_noisy, cam_l, cam_r, width, height)
                error = np.linalg.norm(result - np.array(world_pos))
                errors.append(error)
            except Exception:
                errors.append(float('inf'))

        avg_error = np.mean(errors) if errors else float('inf')
        max_error = np.max(errors) if errors else float('inf')
        results.append({
            "noise_px": noise_px,
            "avg_error_m": round(float(avg_error), 4),
            "max_error_m": round(float(max_error), 4),
            "avg_error_cm": round(float(avg_error) * 100, 2),
            "max_error_cm": round(float(max_error) * 100, 2),
        })
    return results


async def test_cv_detection_accuracy(img_left, img_right, cam_left, cam_right, width, height):
    """Test CV color segmentation accuracy against ground truth."""
    left_detections = detect_objects_cv(img_left)
    right_detections = detect_objects_cv(img_right)

    print(f"\n  CV detected: {len(left_detections)} left, {len(right_detections)} right")

    results = {}
    for det in left_detections:
        obj_id = COLOR_TO_ID.get(det["color"])
        if obj_id and obj_id in GROUND_TRUTH:
            gt_px = project_world_to_pixel(GROUND_TRUTH[obj_id], cam_left, width, height)
            if gt_px:
                error = math.sqrt((det["px"][0] - gt_px[0])**2 + (det["px"][1] - gt_px[1])**2)
                results[obj_id] = {
                    "cv_px": det["px"],
                    "gt_px": [round(gt_px[0], 1), round(gt_px[1], 1)],
                    "error_px": round(error, 1),
                }
                print(f"    {obj_id}: CV={det['px']}, GT={results[obj_id]['gt_px']}, err={error:.1f}px")

    # Now triangulate using CV detections
    print("\n  CV Triangulation results:")
    left_by_color = {d["color"]: d for d in left_detections}
    right_by_color = {d["color"]: d for d in right_detections}

    for color, obj_id in COLOR_TO_ID.items():
        if color in left_by_color and color in right_by_color:
            px_l = left_by_color[color]["px"]
            px_r = right_by_color[color]["px"]
            try:
                world_pos = triangulate_point(px_l, px_r, cam_left, cam_right, width, height)
                gt = np.array(GROUND_TRUTH[obj_id])
                error = np.linalg.norm(world_pos - gt)
                print(f"    {obj_id}: pos={[round(x, 4) for x in world_pos.tolist()]}, "
                      f"gt={gt.tolist()}, err={error*100:.1f}cm")
            except Exception as e:
                print(f"    {obj_id}: triangulation failed: {e}")

    return results


async def main():
    print("=" * 70)
    print("STEREO TRIANGULATION INVESTIGATION")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Capture at current config (640x480, fov=90)
        print("\n[1] Capturing stereo pair at 640x480...")
        img_left, img_right, cam_params, scene_state = await capture_stereo(client, 640, 480)
        cam_left = cam_params["left"]
        cam_right = cam_params["right"]

        print(f"  Camera FOV: {cam_left['fovy']}°")
        print(f"  Left pos:  {[round(x, 4) for x in cam_left['position']]}")
        print(f"  Right pos: {[round(x, 4) for x in cam_right['position']]}")
        baseline = np.linalg.norm(
            np.array(cam_left["position"]) - np.array(cam_right["position"])
        )
        print(f"  Baseline: {baseline*100:.2f}cm")

        # Save images for inspection
        out_dir = Path(__file__).parent.parent / "test_output"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "left_640x480.png").write_bytes(img_left)
        (out_dir / "right_640x480.png").write_bytes(img_right)
        print(f"  Images saved to {out_dir}")

        # 2. Theoretical disparity analysis at different FOVs
        print("\n[2] THEORETICAL DISPARITY ANALYSIS")
        print("-" * 50)
        for fov in FOVS:
            print(f"\n  FOV = {fov}°:")
            disparities = simulate_fov_disparity(cam_left, cam_right, 640, 480, fov)
            for obj_id, disp in disparities.items():
                print(f"    {obj_id}: du={disp['du']:.1f}px, dv={disp['dv']:.1f}px "
                      f"(left={[round(x,1) for x in disp['left']]}, "
                      f"right={[round(x,1) for x in disp['right']]})")

        # 3. Noise tolerance at different FOVs
        print("\n[3] NOISE TOLERANCE (triangulation error vs pixel noise)")
        print("-" * 50)
        noise_levels = [1, 2, 5, 10, 20, 50, 100]
        for fov in FOVS:
            print(f"\n  FOV = {fov}°:")
            noise_results = measure_noise_tolerance(
                cam_left, cam_right, 640, 480, fov, noise_levels
            )
            for nr in noise_results:
                print(f"    noise={nr['noise_px']:3d}px → avg_err={nr['avg_error_cm']:6.1f}cm, "
                      f"max_err={nr['max_error_cm']:6.1f}cm")

        # 4. CV detection accuracy at current resolution
        print("\n[4] CV DETECTION ACCURACY (640x480, fov=90°)")
        print("-" * 50)
        await test_cv_detection_accuracy(img_left, img_right, cam_left, cam_right, 640, 480)

        # 5. Test different resolutions with CV
        print("\n[5] CV DETECTION AT DIFFERENT RESOLUTIONS")
        print("-" * 50)
        for w, h in RESOLUTIONS:
            print(f"\n  Resolution: {w}x{h}")
            img_l, img_r, params, _ = await capture_stereo(client, w, h)
            (out_dir / f"left_{w}x{h}.png").write_bytes(img_l)
            (out_dir / f"right_{w}x{h}.png").write_bytes(img_r)
            await test_cv_detection_accuracy(img_l, img_r, params["left"], params["right"], w, h)

        # 6. Actual scene state from sim (true ground truth check)
        print("\n[6] ACTUAL SIM STATE (ground truth verification)")
        print("-" * 50)
        resp = await client.get(f"{RASIM_URL}/scene-state")
        resp.raise_for_status()
        state = resp.json()
        for prop in state.get("props", []):
            print(f"  {prop['id']}: pos={[round(x, 4) for x in prop['pos']]}")
            if prop['id'] in GROUND_TRUTH:
                gt = GROUND_TRUTH[prop['id']]
                diff = np.linalg.norm(np.array(prop['pos']) - np.array(gt))
                if diff > 0.01:
                    print(f"    ⚠️  Differs from config! Δ={diff*100:.1f}cm "
                          f"(config={gt})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings to check:
1. Does CV detect objects accurately? (< 5px error = good)
2. What's the theoretical disparity at each FOV?
   - Need disparity >> expected pixel error for reliable triangulation
   - LLM error is 80-200px, so disparity must be >> 200px
   - CV error should be < 5px
3. At what noise level does triangulation break?
4. Best config = largest disparity relative to detection noise
""")


if __name__ == "__main__":
    asyncio.run(main())
