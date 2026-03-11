"""Test GPT-5.4 as full perception engine.

The LLM handles:
- Object identification and matching across views
- Pixel coordinate estimation
- Orientation assessment
- Grip strategy recommendation

We only handle: triangulation math.

Tests:
1. Side-by-side at 1024x768 (best config from sweep)
2. Two-image approach with better prompting
3. Scene with duplicate colors (modify scene)
4. Overlapping/stacked objects
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.perception import triangulate_point

RASIM_URL = "http://localhost:8100"
MODEL = "gpt-5.4"

GROUND_TRUTH = {
    "red_box": [0.3, 0.05, 0.45],
    "blue_box": [0.3, -0.15, 0.45],
    "green_box": [0.4, 0.0, 0.45],
    "yellow_cylinder": [0.35, 0.15, 0.46],
}

OUT_DIR = Path(__file__).parent.parent / "test_output"

FULL_PERCEPTION_PROMPT = """You are the vision system for a robotic arm. You receive stereo camera images and must provide complete scene understanding.

## Setup
- Two cameras (LEFT and RIGHT) mounted on the robot gripper, looking down at a table
- Camera baseline is vertical in the image (objects shift vertically between views, NOT horizontally)
- Resolution per view: {width} x {height} pixels
- Pixel origin (0,0) = top-left, u = rightward, v = downward

## Your Responsibilities

### 1. Object Detection & Matching
- Identify ALL objects on the table (ignore the robot arm/gripper hardware)
- Match the SAME object across both views — use shape, color, relative position, and context
- Give each object a unique descriptive ID (e.g., "red_cube_1", "small_blue_cylinder")
- If two objects share the same color, distinguish them (e.g., "red_cube_near", "red_cube_far")

### 2. Pixel Coordinates
- For each object, report its CENTER pixel [u, v] in BOTH views
- Be extremely precise — even 10px error significantly affects 3D reconstruction
- The u-coordinate should be nearly identical between views (within ~5px)
- The v-coordinate will differ significantly (50-250px) due to stereo baseline

### 3. Object Properties
- type: box, cylinder, sphere, or other
- color: primary color name
- approximate_size: small/medium/large relative to other objects
- orientation: describe any visible rotation or tilt (e.g., "rotated ~30° clockwise", "upright", "tilted")

### 4. Grip Strategy
For each object, suggest how the gripper should approach:
- approach_direction: "top_down", "angled", etc.
- grip_axis: which axis to grip along (e.g., "along_x", "along_y", "any" for symmetric)
- notes: any special considerations (e.g., "rotated, align gripper to match", "near edge of table", "partially occluded by X")

### 5. Scene Understanding
- Describe spatial relationships (what's next to what, any stacking, any clustering)
- Flag any objects that might be difficult to grasp (near edges, partially hidden, etc.)
- Note if any object appears in only one view (partially occluded or at frame edge)

## Output Format — ONLY valid JSON:
{{
  "scene_description": "Brief overview of the scene",
  "objects": [
    {{
      "id": "descriptive_unique_id",
      "type": "box",
      "color": "red",
      "approximate_size": "small",
      "orientation": "rotated ~20° clockwise when viewed from above",
      "left_px": [u, v],
      "right_px": [u, v],
      "grip_strategy": {{
        "approach_direction": "top_down",
        "grip_axis": "along_longer_side",
        "notes": "Slight rotation, gripper should align to ~20°"
      }},
      "visibility": "fully_visible_both_views"
    }}
  ],
  "spatial_relationships": ["red_cube is left of green_cube", "yellow_cylinder is near the edge"],
  "warnings": ["blue_box appears to be at the edge of right camera view"]
}}"""


TWO_IMAGE_PROMPT = """You are the vision system for a robotic arm. You will receive TWO images:
1. First image: LEFT camera view
2. Second image: RIGHT camera view

These are stereo cameras separated vertically. The SAME objects appear in both images but shifted VERTICALLY (not horizontally).

Resolution: {width} x {height} pixels per image.
Pixel origin (0,0) = top-left, u = rightward, v = downward.

## Critical Rules for Stereo Matching
- The u-coordinate (horizontal) for the same object should be nearly IDENTICAL in both views (within ~5px)
- The v-coordinate (vertical) will DIFFER by 50-250px between views
- Match objects by their appearance, shape, color, and HORIZONTAL position
- If an object's u-coordinate is very different between views, you may have mismatched objects

## Your Task
For each object on the table (ignore the robot arm):
1. Identify it with a unique ID
2. Report center pixel [u, v] in the LEFT image and [u, v] in the RIGHT image
3. Describe its type, color, orientation
4. Suggest grip strategy

## Output — ONLY valid JSON:
{{
  "scene_description": "...",
  "objects": [
    {{
      "id": "descriptive_id",
      "type": "box|cylinder|sphere",
      "color": "color_name",
      "orientation": "description of rotation/tilt",
      "left_px": [u, v],
      "right_px": [u, v],
      "grip_strategy": {{
        "approach_direction": "top_down",
        "grip_axis": "along_x|along_y|any",
        "notes": "..."
      }},
      "visibility": "fully_visible_both_views|left_only|right_only|partial"
    }}
  ],
  "warnings": []
}}"""


def project_world_to_pixel(world_point, cam, width, height):
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


def make_side_by_side(left_bytes, right_bytes):
    left = Image.open(io.BytesIO(left_bytes))
    right = Image.open(io.BytesIO(right_bytes))
    w, h = left.size
    combined = Image.new("RGB", (w * 2 + 20, h + 30), (0, 0, 0))
    combined.paste(left, (0, 30))
    combined.paste(right, (w + 20, 30))
    draw = ImageDraw.Draw(combined)
    draw.text((w // 2 - 30, 5), "LEFT", fill=(255, 255, 255))
    draw.text((w + 20 + w // 2 - 30, 5), "RIGHT", fill=(255, 255, 255))
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


async def call_gpt54(system_prompt, user_content):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=4096,
        )
        content = response.choices[0].message.content
        usage = response.usage
        result = json.loads(content)
        await client.close()
        return result, usage.prompt_tokens, usage.completion_tokens
    except Exception as e:
        await client.close()
        print(f"  ❌ API error: {e}")
        return None, 0, 0


def evaluate_and_print(result, cam_left, cam_right, width, height):
    """Full evaluation of perception result."""
    if result is None:
        print("  No result to evaluate")
        return

    # Print scene description
    desc = result.get("scene_description", "")
    if desc:
        print(f"\n  Scene: {desc}")

    warnings = result.get("warnings", [])
    if warnings:
        print(f"  Warnings: {warnings}")

    relationships = result.get("spatial_relationships", [])
    if relationships:
        print(f"  Relationships: {relationships}")

    objects = result.get("objects", [])
    print(f"\n  {'ID':<25} {'Color':<8} {'Type':<10} {'Orient':<30} {'Grip':<15} {'Vis'}")
    print(f"  {'─'*110}")

    for obj in objects:
        oid = obj.get("id", "?")[:24]
        color = obj.get("color", "?")[:7]
        otype = obj.get("type", "?")[:9]
        orient = obj.get("orientation", "?")[:29]
        grip = obj.get("grip_strategy", {})
        grip_str = grip.get("approach_direction", "?")[:14] if isinstance(grip, dict) else "?"
        vis = obj.get("visibility", "?")[:20]
        print(f"  {oid:<25} {color:<8} {otype:<10} {orient:<30} {grip_str:<15} {vis}")

    # Triangulation
    print(f"\n  Triangulation:")
    print(f"  {'ID':<25} {'Left px':<16} {'Right px':<16} {'3D pos':<30} {'Error':>8}")
    print(f"  {'─'*100}")

    for obj in objects:
        oid = obj.get("id", "?")
        left_px = obj.get("left_px")
        right_px = obj.get("right_px")

        if not left_px or not right_px:
            print(f"  {oid:<25} — missing pixel data")
            continue

        if any(x is None for x in left_px) or any(x is None for x in right_px):
            print(f"  {oid:<25} — None in pixel data")
            continue

        try:
            world_pos = triangulate_point(left_px, right_px, cam_left, cam_right, width, height)
            pos_str = f"[{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]"

            # Try to match to ground truth by finding closest
            best_gt_id = None
            best_dist = float('inf')
            for gt_id, gt_pos in GROUND_TRUTH.items():
                dist = np.linalg.norm(world_pos - np.array(gt_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_gt_id = gt_id

            err_str = f"{best_dist*100:.1f}cm"
            marker = "✅" if best_dist < 0.03 else "⚡" if best_dist < 0.05 else "❌" if best_dist < 0.20 else "💥"

            # Check if triangulated position is reasonable (on or near table)
            if abs(world_pos[2] - 0.45) > 0.15 or abs(world_pos[0]) > 1.0 or abs(world_pos[1]) > 0.5:
                marker = "💥"
                err_str = "off-table"

            print(f"  {oid:<25} {str(left_px):<16} {str(right_px):<16} {pos_str:<30} {err_str:>8} {marker}")

        except Exception as e:
            print(f"  {oid:<25} {str(left_px):<16} {str(right_px):<16} {'FAILED':<30} {str(e)[:20]}")

    # Print grip strategies in detail
    print(f"\n  Grip strategies:")
    for obj in objects:
        oid = obj.get("id", "?")
        grip = obj.get("grip_strategy", {})
        if isinstance(grip, dict) and grip:
            print(f"    {oid}: approach={grip.get('approach_direction', '?')}, "
                  f"axis={grip.get('grip_axis', '?')}, notes={grip.get('notes', '-')}")


async def capture(client, width, height):
    resp = await client.post(
        f"{RASIM_URL}/execute",
        json={"commands": [{"type": "observe"}]},
        timeout=60.0,
    )
    resp.raise_for_status()

    resp_l = await client.get(f"{RASIM_URL}/eye/left", params={"width": width, "height": height})
    resp_r = await client.get(f"{RASIM_URL}/eye/right", params={"width": width, "height": height})
    resp_l.raise_for_status()
    resp_r.raise_for_status()

    resp_p = await client.get(f"{RASIM_URL}/eye/params")
    resp_p.raise_for_status()
    p = resp_p.json()

    return resp_l.content, resp_r.content, p["left"], p["right"]


async def main():
    print("=" * 90)
    print("GPT-5.4 FULL PERCEPTION TEST")
    print("LLM handles: identification, matching, orientation, grip strategy")
    print("Math handles: triangulation only")
    print("=" * 90)

    OUT_DIR.mkdir(exist_ok=True)
    width, height = 1024, 768
    total_in, total_out = 0, 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Verify sim
        resp = await client.get(f"{RASIM_URL}/eye/params")
        resp.raise_for_status()
        p = resp.json()
        bl = np.linalg.norm(np.array(p["left"]["position"]) - np.array(p["right"]["position"]))
        print(f"Camera: FOV={p['left']['fovy']}°, baseline={bl*100:.1f}cm, res={width}x{height}\n")

        img_l, img_r, cam_l, cam_r = await capture(client, width, height)
        prompt_sbs = FULL_PERCEPTION_PROMPT.replace("{width}", str(width)).replace("{height}", str(height))

        print("(Tests 1-2 already completed, skipping to Test 3)\n")

        # ─── Test 3: Duplicate color scene ───
        print(f"\n{'━' * 90}")
        print("TEST 3: Duplicate color scene (two red boxes)")
        print("━" * 90)

        # Load a scene with two red boxes
        dup_scene = {
            "name": "duplicate_colors",
            "description": "Scene with two red boxes to test matching",
            "workspace": {"bounds": [0.1, 0.9, -0.4, 0.4], "surface_height": 0.42},
            "props": [
                {"id": "red_box_1", "type": "box", "color": [1, 0, 0, 1], "color_name": "red",
                 "size": [0.03, 0.03, 0.03], "position": [0.3, 0.05, 0.45], "mass": 0.1},
                {"id": "red_box_2", "type": "box", "color": [1, 0, 0, 1], "color_name": "red",
                 "size": [0.03, 0.03, 0.03], "position": [0.4, -0.05, 0.45], "mass": 0.1},
                {"id": "green_box", "type": "box", "color": [0, 1, 0, 1], "color_name": "green",
                 "size": [0.03, 0.03, 0.03], "position": [0.35, 0.0, 0.45], "mass": 0.1},
                {"id": "blue_cylinder", "type": "cylinder", "color": [0, 0, 1, 1], "color_name": "blue",
                 "size": [0.02, 0.04], "position": [0.35, 0.1, 0.46], "mass": 0.08},
            ],
        }

        resp = await client.post(f"{RASIM_URL}/scene/load", json=dup_scene, timeout=60.0)
        resp.raise_for_status()
        print("  Loaded duplicate-color scene")

        # Update ground truth
        dup_gt = {
            "red_box_1": [0.3, 0.05, 0.45],
            "red_box_2": [0.4, -0.05, 0.45],
            "green_box": [0.35, 0.0, 0.45],
            "blue_cylinder": [0.35, 0.1, 0.46],
        }
        # Temporarily override
        global GROUND_TRUTH
        old_gt = GROUND_TRUTH
        GROUND_TRUTH = dup_gt

        img_l2, img_r2, cam_l2, cam_r2 = await capture(client, width, height)
        img_sbs2 = make_side_by_side(img_l2, img_r2)
        (OUT_DIR / "full_perc_dup_sbs.png").write_bytes(img_sbs2)
        b64_sbs2 = base64.b64encode(img_sbs2).decode()

        user_dup = [
            {"type": "text", "text": "Analyze this stereo image pair. Note: there may be multiple objects of the same color. Identify each one uniquely and match correctly across views."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sbs2}"}},
        ]

        result3, ti, to = await call_gpt54(prompt_sbs, user_dup)
        total_in += ti
        total_out += to
        print(f"  Tokens: {ti} in, {to} out")
        evaluate_and_print(result3, cam_l2, cam_r2, width, height)

        # ─── Test 4: Stacked objects ───
        print(f"\n{'━' * 90}")
        print("TEST 4: Stacked objects (green on red)")
        print("━" * 90)

        stack_scene = {
            "name": "stacked",
            "description": "Stacked objects to test occlusion handling",
            "workspace": {"bounds": [0.1, 0.9, -0.4, 0.4], "surface_height": 0.42},
            "props": [
                {"id": "red_box_base", "type": "box", "color": [1, 0, 0, 1], "color_name": "red",
                 "size": [0.04, 0.04, 0.04], "position": [0.35, 0.0, 0.46], "mass": 0.15},
                {"id": "green_box_top", "type": "box", "color": [0, 1, 0, 1], "color_name": "green",
                 "size": [0.03, 0.03, 0.03], "position": [0.35, 0.0, 0.51], "mass": 0.1},
                {"id": "blue_box", "type": "box", "color": [0, 0, 1, 1], "color_name": "blue",
                 "size": [0.03, 0.03, 0.03], "position": [0.25, 0.08, 0.45], "mass": 0.1},
                {"id": "yellow_cylinder", "type": "cylinder", "color": [1, 1, 0, 1], "color_name": "yellow",
                 "size": [0.02, 0.04], "position": [0.45, -0.05, 0.46], "mass": 0.08},
            ],
        }

        resp = await client.post(f"{RASIM_URL}/scene/load", json=stack_scene, timeout=60.0)
        resp.raise_for_status()
        print("  Loaded stacked scene")

        GROUND_TRUTH = {
            "red_box_base": [0.35, 0.0, 0.46],
            "green_box_top": [0.35, 0.0, 0.51],
            "blue_box": [0.25, 0.08, 0.45],
            "yellow_cylinder": [0.45, -0.05, 0.46],
        }

        img_l3, img_r3, cam_l3, cam_r3 = await capture(client, width, height)
        img_sbs3 = make_side_by_side(img_l3, img_r3)
        (OUT_DIR / "full_perc_stack_sbs.png").write_bytes(img_sbs3)
        b64_sbs3 = base64.b64encode(img_sbs3).decode()

        user_stack = [
            {"type": "text", "text": "Analyze this stereo image pair. Some objects may be stacked on top of each other. Identify all objects including any that are partially occluded."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sbs3}"}},
        ]

        result4, ti, to = await call_gpt54(prompt_sbs, user_stack)
        total_in += ti
        total_out += to
        print(f"  Tokens: {ti} in, {to} out")
        evaluate_and_print(result4, cam_l3, cam_r3, width, height)

        # Restore original scene
        resp = await client.post(
            f"{RASIM_URL}/execute",
            json={"commands": [{"type": "observe"}]},  # just to keep sim alive
            timeout=60.0,
        )

        GROUND_TRUTH = old_gt

    print(f"\n{'━' * 90}")
    print(f"Total API: {total_in} tokens in, {total_out} tokens out")
    print("━" * 90)


if __name__ == "__main__":
    asyncio.run(main())
