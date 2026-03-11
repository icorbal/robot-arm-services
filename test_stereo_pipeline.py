#!/usr/bin/env python3
"""Test script for the stereo vision pipeline.

Computes ground-truth pixel projections from known object positions,
then compares against LLM-returned pixel coordinates and measures
triangulation accuracy.

Tests multiple configurations:
- 640x480 vs 1024x768 vs 1280x960 resolution
- 90° vs 160° FOV (requires modifying camera in RASim)
"""

import asyncio
import base64
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Add RAServ src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.perception import triangulate_point

RASIM_URL = os.environ.get("RASIM_URL", "http://localhost:8100")
RESULTS_DIR = Path(__file__).parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True)


def project_point_to_pixel(
    world_pos: list[float],
    cam_params: dict,
    width: int,
    height: int,
) -> list[float]:
    """Project a 3D world point to pixel coordinates using camera params.

    This is the ground-truth projection — no LLM involved.
    """
    fovy = math.radians(cam_params["fovy"])
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    fx = fy
    cx, cy = width / 2.0, height / 2.0

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    R_world = np.array(cam_params["rotation_matrix"])
    pos = np.array(cam_params["position"])
    R_w2c = R_world.T
    flip = np.diag([1.0, -1.0, -1.0])
    R_cv = flip @ R_w2c
    t = -R_cv @ pos
    Rt = np.hstack([R_cv, t.reshape(3, 1)])
    P = K @ Rt

    X = np.array([*world_pos, 1.0])
    px_h = P @ X
    u = px_h[0] / px_h[2]
    v = px_h[1] / px_h[2]
    return [float(u), float(v)]


async def get_scene_and_cameras(client: httpx.AsyncClient, width: int, height: int):
    """Get scene state, camera params, and stereo images."""
    # Move to observation pose
    resp = await client.post(
        f"{RASIM_URL}/execute",
        json={"commands": [{"type": "observe"}]},
        timeout=60.0,
    )
    resp.raise_for_status()
    scene_state = resp.json()["scene_state"]

    # Get camera params
    resp = await client.get(f"{RASIM_URL}/eye/params")
    resp.raise_for_status()
    cam_params = resp.json()

    # Capture images
    resp_l = await client.get(f"{RASIM_URL}/eye/left", params={"width": width, "height": height})
    resp_l.raise_for_status()
    resp_r = await client.get(f"{RASIM_URL}/eye/right", params={"width": width, "height": height})
    resp_r.raise_for_status()

    return scene_state, cam_params, resp_l.content, resp_r.content


def compute_ground_truth(scene_state, cam_params, width, height):
    """Compute ground-truth pixel positions for all props."""
    results = []
    for prop in scene_state["props"]:
        pos = prop["pos"]
        left_px = project_point_to_pixel(pos, cam_params["left"], width, height)
        right_px = project_point_to_pixel(pos, cam_params["right"], width, height)

        # Verify triangulation with ground-truth pixels
        tri_pos = triangulate_point(
            left_px, right_px,
            cam_params["left"], cam_params["right"],
            width, height,
        )

        error = np.linalg.norm(np.array(pos) - tri_pos)

        results.append({
            "id": prop["id"],
            "actual_pos": pos,
            "left_px_gt": [round(left_px[0], 1), round(left_px[1], 1)],
            "right_px_gt": [round(right_px[0], 1), round(right_px[1], 1)],
            "disparity_u": round(right_px[0] - left_px[0], 1),
            "disparity_v": round(right_px[1] - left_px[1], 1),
            "tri_pos": [round(float(x), 4) for x in tri_pos],
            "tri_error_m": round(float(error), 6),
        })

    return results


async def test_llm_detection(
    img_left: bytes,
    img_right: bytes,
    width: int,
    height: int,
    cam_params: dict,
    ground_truth: list[dict],
    config_name: str,
):
    """Send stereo images to the LLM and compare results."""
    from src.llm_adapter import create_llm_adapter

    prompt_path = Path(__file__).parent / "prompts" / "perceiver.txt"
    with open(prompt_path) as f:
        system_prompt = f.read().replace("{width}", str(width)).replace("{height}", str(height))

    llm = create_llm_adapter(
        provider="openai",
        model="gpt-4o",
    )

    try:
        result = await llm.generate(
            system_prompt=system_prompt,
            user_prompt=(
                "Analyze these two stereo camera images (left eye, then right eye). "
                "Identify all objects on the table and their pixel coordinates in each image."
            ),
            images=[img_left, img_right],
        )
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None

    objects = result.get("objects", [])
    print(f"  LLM detected {len(objects)} objects (ground truth: {len(ground_truth)})")

    llm_results = []
    for obj in objects:
        obj_id = obj.get("id", "unknown")
        left_px = obj.get("left_px", [0, 0])
        right_px = obj.get("right_px", [0, 0])

        # Find matching ground truth
        gt_match = None
        for gt in ground_truth:
            if gt["id"] == obj_id or obj.get("color", "") in gt["id"]:
                gt_match = gt
                break

        # Triangulate from LLM pixels
        try:
            tri_pos = triangulate_point(
                left_px, right_px,
                cam_params["left"], cam_params["right"],
                width, height,
            )
            tri_pos_list = [round(float(x), 4) for x in tri_pos]
        except Exception as e:
            tri_pos_list = None
            tri_pos = None

        entry = {
            "id": obj_id,
            "llm_left_px": left_px,
            "llm_right_px": right_px,
            "llm_disparity_u": round(right_px[0] - left_px[0], 1),
            "llm_disparity_v": round(right_px[1] - left_px[1], 1),
            "tri_pos": tri_pos_list,
        }

        if gt_match:
            entry["gt_left_px"] = gt_match["left_px_gt"]
            entry["gt_right_px"] = gt_match["right_px_gt"]
            entry["gt_disparity_u"] = gt_match["disparity_u"]
            entry["gt_disparity_v"] = gt_match["disparity_v"]
            entry["actual_pos"] = gt_match["actual_pos"]

            # Pixel error
            l_err = np.linalg.norm(np.array(left_px) - np.array(gt_match["left_px_gt"]))
            r_err = np.linalg.norm(np.array(right_px) - np.array(gt_match["right_px_gt"]))
            entry["pixel_error_left"] = round(float(l_err), 1)
            entry["pixel_error_right"] = round(float(r_err), 1)

            # 3D position error
            if tri_pos is not None:
                pos_err = np.linalg.norm(np.array(gt_match["actual_pos"]) - tri_pos)
                entry["position_error_m"] = round(float(pos_err), 4)

        llm_results.append(entry)
        print(f"  {obj_id}: left={left_px} right={right_px} "
              f"disp_u={entry['llm_disparity_u']} disp_v={entry['llm_disparity_v']}")
        if gt_match:
            print(f"    GT:  left={gt_match['left_px_gt']} right={gt_match['right_px_gt']} "
                  f"disp_u={gt_match['disparity_u']} disp_v={gt_match['disparity_v']}")
            if "position_error_m" in entry:
                print(f"    3D error: {entry['position_error_m']:.4f}m "
                      f"(LLM: {entry['tri_pos']} vs actual: {entry['actual_pos']})")

    return llm_results


async def main():
    print("=" * 70)
    print("STEREO VISION PIPELINE TEST")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test configurations
        configs = [
            {"name": "640x480_fov90", "width": 640, "height": 480},
            {"name": "1024x768_fov90", "width": 1024, "height": 768},
            {"name": "1280x960_fov90", "width": 1280, "height": 960},
        ]

        all_results = {}

        for cfg in configs:
            name = cfg["name"]
            w, h = cfg["width"], cfg["height"]
            print(f"\n{'─' * 60}")
            print(f"Config: {name} ({w}x{h})")
            print(f"{'─' * 60}")

            scene_state, cam_params, img_left, img_right = await get_scene_and_cameras(client, w, h)

            # Save images
            with open(RESULTS_DIR / f"left_{name}.png", "wb") as f:
                f.write(img_left)
            with open(RESULTS_DIR / f"right_{name}.png", "wb") as f:
                f.write(img_right)

            # Ground truth
            gt = compute_ground_truth(scene_state, cam_params, w, h)
            print("\nGround Truth:")
            for obj in gt:
                print(f"  {obj['id']}: left={obj['left_px_gt']} right={obj['right_px_gt']} "
                      f"disp_u={obj['disparity_u']} disp_v={obj['disparity_v']} "
                      f"tri_error={obj['tri_error_m']:.6f}m")

            # LLM test
            print("\nLLM Detection:")
            llm_results = await test_llm_detection(
                img_left, img_right, w, h, cam_params, gt, name,
            )

            all_results[name] = {
                "ground_truth": gt,
                "llm_results": llm_results,
                "cam_params": cam_params,
            }

        # Save full results
        results_file = RESULTS_DIR / "stereo_test_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nFull results saved to: {results_file}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for name, data in all_results.items():
            print(f"\n{name}:")
            gt_errors = [obj["tri_error_m"] for obj in data["ground_truth"]]
            print(f"  Ground truth triangulation errors: "
                  f"max={max(gt_errors):.6f}m, mean={np.mean(gt_errors):.6f}m")

            if data["llm_results"]:
                pos_errors = [r.get("position_error_m", float("inf"))
                              for r in data["llm_results"] if "position_error_m" in r]
                if pos_errors:
                    print(f"  LLM triangulation errors: "
                          f"max={max(pos_errors):.4f}m, mean={np.mean(pos_errors):.4f}m")
                px_errors_l = [r.get("pixel_error_left", 0) for r in data["llm_results"]
                               if "pixel_error_left" in r]
                px_errors_r = [r.get("pixel_error_right", 0) for r in data["llm_results"]
                               if "pixel_error_right" in r]
                if px_errors_l:
                    print(f"  Mean pixel error: left={np.mean(px_errors_l):.1f}px, "
                          f"right={np.mean(px_errors_r):.1f}px")


if __name__ == "__main__":
    asyncio.run(main())
