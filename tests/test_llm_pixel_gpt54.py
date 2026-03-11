"""Test GPT-5.4 pixel detection accuracy with 20cm baseline + 60° FOV.

Captures real stereo images, sends to GPT-5.4 for pixel detection,
compares against CV ground truth, and measures triangulation accuracy.
"""

import asyncio
import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.perception import triangulate_point
from src.perception_cv import detect_objects_cv

RASIM_URL = "http://localhost:8100"

# Ground truth from scene config
GROUND_TRUTH = {
    "red_box": [0.3, 0.05, 0.45],
    "blue_box": [0.3, -0.15, 0.45],
    "green_box": [0.4, 0.0, 0.45],
    "yellow_cylinder": [0.35, 0.15, 0.46],
}

COLOR_TO_ID = {
    "red": "red_box",
    "blue": "blue_box",
    "green": "green_box",
    "yellow": "yellow_cylinder",
}

PERCEIVER_PROMPT = """You are analyzing two stereo camera images from a robot arm looking down at a table.
The images are from LEFT and RIGHT cameras mounted on the gripper, separated horizontally.

## Image Details
- Resolution: {width} x {height} pixels
- Camera FOV: 60° vertical
- The cameras are separated along the vertical axis of the image (the disparity is primarily in the v/y coordinate)

## Pixel Coordinate System
- Origin (0, 0) is the TOP-LEFT corner of the image
- u increases rightward (0 to {width})
- v increases downward (0 to {height})

## Your Task
Identify ALL colored objects on the table. For each object, provide:
1. Its ID (e.g., "red_box", "blue_box", "green_box", "yellow_cylinder")
2. Color and type (box or cylinder)
3. Pixel coordinates [u, v] of the object's CENTER in the LEFT image (first image)
4. Pixel coordinates [u, v] of the object's CENTER in the RIGHT image (second image)

## Critical Notes
- Objects will appear at DIFFERENT vertical positions between left and right images
- The horizontal (u) position should be very similar between images
- The vertical (v) position will differ by 50-200+ pixels between images
- Be EXTREMELY precise — even 10 pixels of error significantly affects 3D reconstruction
- Look at the actual center of each colored shape, not its edge

## Output Format — ONLY valid JSON, no other text:
{{"objects": [{{"id": "red_box", "color": "red", "type": "box", "left_px": [u, v], "right_px": [u, v]}}]}}"""


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
        return None
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return [float(u), float(v)]


async def run_llm_test(model_name: str, img_left: bytes, img_right: bytes,
                       cam_left: dict, cam_right: dict, width: int, height: int):
    """Send images to an LLM and evaluate pixel accuracy."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = PERCEIVER_PROMPT.replace("{width}", str(width)).replace("{height}", str(height))

    b64_left = base64.b64encode(img_left).decode("ascii")
    b64_right = base64.b64encode(img_right).decode("ascii")

    print(f"\n  Calling {model_name}...")
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze these stereo images. First image is LEFT camera, second is RIGHT camera. Identify all objects and their precise pixel coordinates."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_left}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_right}"}},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=2048,
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        usage = response.usage
        print(f"  Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")

    except Exception as e:
        print(f"  ❌ API call failed: {e}")
        await client.close()
        return None

    await client.close()

    objects = result.get("objects", [])
    print(f"  Detected {len(objects)} objects")
    print(f"  Raw response: {json.dumps(result, indent=2)}")

    # Evaluate each object
    print(f"\n  {'Object':<20} {'LLM Left':>16} {'GT Left':>16} {'Err L':>8} {'LLM Right':>16} {'GT Right':>16} {'Err R':>8}")
    print(f"  {'─'*104}")

    pixel_errors = []
    triangulation_results = []

    for obj in objects:
        obj_id = obj.get("id", "unknown")
        left_px = obj.get("left_px")
        right_px = obj.get("right_px")

        if not left_px or not right_px:
            continue

        # Find ground truth ID
        gt_id = obj_id
        if gt_id not in GROUND_TRUTH:
            # Try matching by color
            color = obj.get("color", "")
            gt_id = COLOR_TO_ID.get(color, obj_id)

        if gt_id not in GROUND_TRUTH:
            print(f"  {obj_id:<20} — no ground truth match")
            continue

        # Project ground truth to pixel coords
        gt_left = project_world_to_pixel(GROUND_TRUTH[gt_id], cam_left, width, height)
        gt_right = project_world_to_pixel(GROUND_TRUTH[gt_id], cam_right, width, height)

        if gt_left is None or gt_right is None:
            continue

        err_l = math.sqrt((left_px[0] - gt_left[0])**2 + (left_px[1] - gt_left[1])**2)
        err_r = math.sqrt((right_px[0] - gt_right[0])**2 + (right_px[1] - gt_right[1])**2)
        pixel_errors.extend([err_l, err_r])

        print(f"  {gt_id:<20} [{left_px[0]:>6.1f},{left_px[1]:>6.1f}] [{gt_left[0]:>6.1f},{gt_left[1]:>6.1f}] {err_l:>6.1f}px "
              f"[{right_px[0]:>6.1f},{right_px[1]:>6.1f}] [{gt_right[0]:>6.1f},{gt_right[1]:>6.1f}] {err_r:>6.1f}px")

        # Triangulate
        try:
            world_pos = triangulate_point(left_px, right_px, cam_left, cam_right, width, height)
            gt_world = np.array(GROUND_TRUTH[gt_id])
            error_3d = np.linalg.norm(world_pos - gt_world)
            triangulation_results.append({
                "id": gt_id,
                "pos": [round(float(x), 4) for x in world_pos],
                "gt": GROUND_TRUTH[gt_id],
                "error_cm": round(error_3d * 100, 2),
            })
        except Exception as e:
            triangulation_results.append({
                "id": gt_id, "error": str(e),
            })

    # Summary
    if pixel_errors:
        arr = np.array(pixel_errors)
        print(f"\n  Pixel error stats: mean={arr.mean():.1f}px, median={np.median(arr):.1f}px, "
              f"p90={np.percentile(arr, 90):.1f}px, max={arr.max():.1f}px")

    print(f"\n  Triangulation results:")
    for tr in triangulation_results:
        if "error" in tr:
            print(f"    {tr['id']}: FAILED — {tr['error']}")
        else:
            marker = "✅" if tr["error_cm"] < 3 else "⚡" if tr["error_cm"] < 5 else "❌"
            print(f"    {tr['id']}: pos={tr['pos']}, gt={tr['gt']}, err={tr['error_cm']}cm {marker}")

    return {
        "model": model_name,
        "pixel_errors": pixel_errors,
        "triangulation": triangulation_results,
    }


async def main():
    print("=" * 80)
    print("GPT-5.4 vs GPT-4o PIXEL DETECTION TEST")
    print("Config: 20cm gripper baseline, 60° FOV, 640x480")
    print("=" * 80)

    width, height = 640, 480

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Move to observe pose
        print("\nMoving to observe pose...")
        resp = await client.post(
            f"{RASIM_URL}/execute",
            json={"commands": [{"type": "observe"}]},
            timeout=60.0,
        )
        resp.raise_for_status()

        # Capture images
        print("Capturing stereo pair...")
        resp_left = await client.get(f"{RASIM_URL}/eye/left", params={"width": width, "height": height})
        resp_right = await client.get(f"{RASIM_URL}/eye/right", params={"width": width, "height": height})
        resp_left.raise_for_status()
        resp_right.raise_for_status()
        img_left = resp_left.content
        img_right = resp_right.content

        # Get camera params
        resp_params = await client.get(f"{RASIM_URL}/eye/params")
        resp_params.raise_for_status()
        cam_params = resp_params.json()
        cam_left = cam_params["left"]
        cam_right = cam_params["right"]

        baseline = np.linalg.norm(
            np.array(cam_left["position"]) - np.array(cam_right["position"])
        )
        print(f"Camera FOV: {cam_left['fovy']}°, Baseline: {baseline*100:.1f}cm")

        # Save images
        out_dir = Path(__file__).parent.parent / "test_output"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "stereo_left_20cm_60fov.png").write_bytes(img_left)
        (out_dir / "stereo_right_20cm_60fov.png").write_bytes(img_right)

        # CV baseline
        print("\n" + "─" * 80)
        print("CV COLOR SEGMENTATION (baseline)")
        print("─" * 80)
        left_det = detect_objects_cv(img_left)
        right_det = detect_objects_cv(img_right)
        print(f"  Detected: {len(left_det)} left, {len(right_det)} right")

        left_by_color = {d["color"]: d for d in left_det}
        right_by_color = {d["color"]: d for d in right_det}

        for color, gt_id in COLOR_TO_ID.items():
            if color in left_by_color and color in right_by_color:
                px_l = left_by_color[color]["px"]
                px_r = right_by_color[color]["px"]
                gt_l = project_world_to_pixel(GROUND_TRUTH[gt_id], cam_left, width, height)
                gt_r = project_world_to_pixel(GROUND_TRUTH[gt_id], cam_right, width, height)
                if gt_l and gt_r:
                    err_l = math.sqrt((px_l[0]-gt_l[0])**2 + (px_l[1]-gt_l[1])**2)
                    err_r = math.sqrt((px_r[0]-gt_r[0])**2 + (px_r[1]-gt_r[1])**2)
                    print(f"  {gt_id}: left_err={err_l:.1f}px, right_err={err_r:.1f}px")

                try:
                    world_pos = triangulate_point(px_l, px_r, cam_left, cam_right, width, height)
                    gt = np.array(GROUND_TRUTH[gt_id])
                    err = np.linalg.norm(world_pos - gt)
                    marker = "✅" if err*100 < 3 else "⚡" if err*100 < 5 else "❌"
                    print(f"    → 3D: {[round(x,4) for x in world_pos.tolist()]}, err={err*100:.1f}cm {marker}")
                except Exception as e:
                    print(f"    → 3D: FAILED — {e}")

        # LLM tests
        models = ["gpt-5.4"]
        for model in models:
            print(f"\n{'─' * 80}")
            print(f"LLM: {model.upper()}")
            print("─" * 80)
            await run_llm_test(model, img_left, img_right, cam_left, cam_right, width, height)


if __name__ == "__main__":
    asyncio.run(main())
