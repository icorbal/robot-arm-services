"""Comprehensive GPT-5.4 pixel detection sweep.

Tests multiple configs:
- Resolutions: 640x480, 1024x768, 1280x960
- Prompt variations: basic, grid-overlay hint, single-image-per-call
- Multiple trials per config for consistency check
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.perception import triangulate_point
from src.perception_cv import detect_objects_cv

RASIM_URL = "http://localhost:8100"
MODEL = "gpt-5.4"

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

OUT_DIR = Path(__file__).parent.parent / "test_output"


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


def add_grid_overlay(img_bytes: bytes, step: int = 100) -> bytes:
    """Add a pixel grid with coordinates to help the LLM."""
    img = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Draw grid lines
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 128), width=1)
        draw.text((x + 2, 2), str(x), fill=(255, 255, 255))
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 128), width=1)
        draw.text((2, y + 2), str(y), fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_side_by_side(left_bytes: bytes, right_bytes: bytes) -> bytes:
    """Combine left and right images side-by-side with labels."""
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


PROMPTS = {
    "basic": """You are analyzing two stereo camera images from a robot arm looking down at a table.
The images are from LEFT and RIGHT cameras separated along the image's vertical axis.

Image resolution: {width} x {height} pixels.
Pixel origin (0,0) = top-left. u = rightward, v = downward.

Objects are colored geometric shapes (boxes, cylinders) on a beige table.
The robot arm/gripper is also visible — ignore it.

For each object, provide its center pixel [u, v] in BOTH images.
The u-coordinate should be SIMILAR between images. The v-coordinate will DIFFER (vertical disparity from stereo baseline).

Output ONLY valid JSON:
{{"objects": [{{"id": "red_box", "color": "red", "type": "box", "left_px": [u, v], "right_px": [u, v]}}]}}""",

    "grid": """You are analyzing two stereo camera images from a robot arm looking down at a table.
Grid lines are overlaid every 100 pixels with coordinate labels to help you locate objects precisely.

Image resolution: {width} x {height} pixels.
Pixel origin (0,0) = top-left. u = rightward, v = downward.

Objects are colored geometric shapes (boxes, cylinders) on a beige table.
The robot arm/gripper is also visible — ignore it.

IMPORTANT: Use the grid lines as reference! Count from the nearest grid line to estimate precise coordinates.
The u-coordinate should be VERY SIMILAR between left and right images (within 5px).
The v-coordinate will DIFFER significantly (50-250px) due to the stereo baseline.

Output ONLY valid JSON:
{{"objects": [{{"id": "red_box", "color": "red", "type": "box", "left_px": [u, v], "right_px": [u, v]}}]}}""",

    "single_then_match": """You are analyzing a single camera image from a robot arm looking down at a table.
Image resolution: {width} x {height} pixels.
Pixel origin (0,0) = top-left. u = rightward, v = downward.

Identify ALL colored objects on the table. For each, provide its center pixel [u, v].
Ignore the robot arm/gripper hardware visible in the image.

Output ONLY valid JSON:
{{"objects": [{{"id": "red_box", "color": "red", "type": "box", "px": [u, v]}}]}}""",

    "side_by_side": """You are analyzing a single image showing LEFT and RIGHT stereo camera views side by side.
The left half (columns 0 to {half_w}) is the LEFT camera view.
The right half (columns {offset} to {full_w}) is the RIGHT camera view.
Each individual view has resolution {width} x {height}.

Objects are colored geometric shapes on a beige table. Ignore the robot arm.

For each object, report its center pixel coordinates WITHIN each view:
- left_px: [u, v] measured from the LEFT view's top-left corner (0,0)
- right_px: [u, v] measured from the RIGHT view's top-left corner (0,0)

The u-coordinates should be similar between views. The v-coordinates will differ (stereo disparity).

Output ONLY valid JSON:
{{"objects": [{{"id": "red_box", "color": "red", "type": "box", "left_px": [u, v], "right_px": [u, v]}}]}}""",
}


async def call_gpt54(system_prompt, user_content, tag=""):
    """Call GPT-5.4 and return parsed JSON."""
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
            max_completion_tokens=2048,
        )
        content = response.choices[0].message.content
        usage = response.usage
        cost_in = usage.prompt_tokens
        cost_out = usage.completion_tokens
        result = json.loads(content)
        await client.close()
        return result, cost_in, cost_out
    except Exception as e:
        await client.close()
        print(f"    ❌ API error{' ('+tag+')' if tag else ''}: {e}")
        return None, 0, 0


def evaluate_result(result, cam_left, cam_right, width, height, label=""):
    """Evaluate LLM detection result against ground truth."""
    if result is None:
        print(f"    {label}: No result")
        return None

    objects = result.get("objects", [])
    pixel_errors = []
    tri_results = []

    for obj in objects:
        obj_id = obj.get("id", "unknown")
        left_px = obj.get("left_px") or obj.get("px")  # handle single-image mode
        right_px = obj.get("right_px")

        gt_id = obj_id
        if gt_id not in GROUND_TRUTH:
            color = obj.get("color", "")
            gt_id = COLOR_TO_ID.get(color, obj_id)
        if gt_id not in GROUND_TRUTH:
            continue

        gt_l = project_world_to_pixel(GROUND_TRUTH[gt_id], cam_left, width, height)
        if gt_l is None:
            continue

        if left_px and all(x is not None for x in left_px):
            err_l = math.sqrt((left_px[0] - gt_l[0])**2 + (left_px[1] - gt_l[1])**2)
            pixel_errors.append(err_l)
        else:
            continue

        if right_px and cam_right and all(x is not None for x in right_px):
            gt_r = project_world_to_pixel(GROUND_TRUTH[gt_id], cam_right, width, height)
            if gt_r:
                err_r = math.sqrt((right_px[0] - gt_r[0])**2 + (right_px[1] - gt_r[1])**2)
                pixel_errors.append(err_r)

            # Triangulate (even if gt_r is off-screen, we can still try)
            if left_px and right_px:
                try:
                    world_pos = triangulate_point(left_px, right_px, cam_left, cam_right, width, height)
                    gt_w = np.array(GROUND_TRUTH[gt_id])
                    err_3d = np.linalg.norm(world_pos - gt_w)
                    tri_results.append({"id": gt_id, "error_cm": round(err_3d * 100, 2)})
                except Exception:
                    tri_results.append({"id": gt_id, "error_cm": 999})

    if not pixel_errors:
        return None

    arr = np.array(pixel_errors)
    tri_ok = sum(1 for t in tri_results if t["error_cm"] < 5)
    tri_total = len(tri_results)

    summary = {
        "px_mean": round(float(arr.mean()), 1),
        "px_median": round(float(np.median(arr)), 1),
        "px_max": round(float(arr.max()), 1),
        "tri_ok": tri_ok,
        "tri_total": tri_total,
        "tri_details": tri_results,
    }

    status = f"✅ {tri_ok}/{tri_total}" if tri_ok == tri_total else f"⚡ {tri_ok}/{tri_total}" if tri_ok > 0 else f"❌ 0/{tri_total}"

    print(f"    {label}: px_mean={summary['px_mean']:.0f}, px_med={summary['px_median']:.0f}, "
          f"px_max={summary['px_max']:.0f} | tri {status} "
          + " ".join(f"{t['id'].split('_')[0]}={t['error_cm']}cm" for t in tri_results))

    return summary


async def capture_at_resolution(client, width, height):
    """Capture stereo pair at given resolution."""
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
    params = resp_p.json()

    return resp_l.content, resp_r.content, params["left"], params["right"]


async def main():
    print("=" * 90)
    print(f"GPT-5.4 COMPREHENSIVE PIXEL DETECTION SWEEP")
    print("=" * 90)

    OUT_DIR.mkdir(exist_ok=True)
    total_tokens_in = 0
    total_tokens_out = 0
    all_results = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Verify sim
        resp = await client.get(f"{RASIM_URL}/eye/params")
        resp.raise_for_status()
        p = resp.json()
        bl = np.linalg.norm(np.array(p["left"]["position"]) - np.array(p["right"]["position"]))
        print(f"Camera: FOV={p['left']['fovy']}°, baseline={bl*100:.1f}cm\n")

        resolutions = [(640, 480), (1024, 768)]
        n_trials = 3

        for width, height in resolutions:
            print(f"\n{'━' * 90}")
            print(f"RESOLUTION: {width}x{height}")
            print(f"{'━' * 90}")

            img_l, img_r, cam_l, cam_r = await capture_at_resolution(client, width, height)

            # Save images
            (OUT_DIR / f"left_{width}x{height}.png").write_bytes(img_l)
            (OUT_DIR / f"right_{width}x{height}.png").write_bytes(img_r)

            # Prepare image variants
            img_l_grid = add_grid_overlay(img_l)
            img_r_grid = add_grid_overlay(img_r)
            img_sbs = make_side_by_side(img_l, img_r)
            (OUT_DIR / f"sbs_{width}x{height}.png").write_bytes(img_sbs)

            b64_l = base64.b64encode(img_l).decode()
            b64_r = base64.b64encode(img_r).decode()
            b64_l_grid = base64.b64encode(img_l_grid).decode()
            b64_r_grid = base64.b64encode(img_r_grid).decode()
            b64_sbs = base64.b64encode(img_sbs).decode()

            # Test 1: Basic prompt, two images, N trials
            print(f"\n  [A] Basic prompt, 2 separate images × {n_trials} trials:")
            prompt_a = PROMPTS["basic"].replace("{width}", str(width)).replace("{height}", str(height))
            for i in range(n_trials):
                user_content = [
                    {"type": "text", "text": "First image = LEFT camera. Second image = RIGHT camera. Find all colored objects."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_l}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_r}"}},
                ]
                result, ti, to = await call_gpt54(prompt_a, user_content, f"trial {i+1}")
                total_tokens_in += ti
                total_tokens_out += to
                s = evaluate_result(result, cam_l, cam_r, width, height, f"trial {i+1}")
                if s:
                    all_results.append({"config": f"{width}x{height}_basic", "trial": i+1, **s})

            # Test 2: Grid overlay prompt
            print(f"\n  [B] Grid overlay prompt × {n_trials} trials:")
            prompt_b = PROMPTS["grid"].replace("{width}", str(width)).replace("{height}", str(height))
            for i in range(n_trials):
                user_content = [
                    {"type": "text", "text": "Images have grid overlays. Use grid lines for precise coordinate estimation. First = LEFT, second = RIGHT."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_l_grid}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_r_grid}"}},
                ]
                result, ti, to = await call_gpt54(prompt_b, user_content, f"grid {i+1}")
                total_tokens_in += ti
                total_tokens_out += to
                s = evaluate_result(result, cam_l, cam_r, width, height, f"grid {i+1}")
                if s:
                    all_results.append({"config": f"{width}x{height}_grid", "trial": i+1, **s})

            # Test 3: Single image per call (avoid confusion between L/R)
            print(f"\n  [C] Single-image-per-call (L then R separately) × {n_trials} trials:")
            prompt_c = PROMPTS["single_then_match"].replace("{width}", str(width)).replace("{height}", str(height))
            for i in range(n_trials):
                # Left image
                user_l = [
                    {"type": "text", "text": "This is the LEFT camera image. Find all colored objects and their center pixel coordinates."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_l}"}},
                ]
                result_l, ti, to = await call_gpt54(prompt_c, user_l, f"single-L {i+1}")
                total_tokens_in += ti
                total_tokens_out += to

                # Right image
                user_r = [
                    {"type": "text", "text": "This is the RIGHT camera image. Find all colored objects and their center pixel coordinates."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_r}"}},
                ]
                result_r, ti, to = await call_gpt54(prompt_c, user_r, f"single-R {i+1}")
                total_tokens_in += ti
                total_tokens_out += to

                # Combine by matching color
                if result_l and result_r:
                    left_objs = {o.get("color"): o for o in result_l.get("objects", [])}
                    right_objs = {o.get("color"): o for o in result_r.get("objects", [])}
                    combined = {"objects": []}
                    for color in left_objs:
                        if color in right_objs:
                            combined["objects"].append({
                                "id": left_objs[color].get("id", f"{color}_obj"),
                                "color": color,
                                "type": left_objs[color].get("type", "box"),
                                "left_px": left_objs[color]["px"],
                                "right_px": right_objs[color]["px"],
                            })
                    s = evaluate_result(combined, cam_l, cam_r, width, height, f"single {i+1}")
                    if s:
                        all_results.append({"config": f"{width}x{height}_single", "trial": i+1, **s})

            # Test 4: Side-by-side composite image
            print(f"\n  [D] Side-by-side composite image × {n_trials} trials:")
            sbs_w = width * 2 + 20
            prompt_d = (PROMPTS["side_by_side"]
                .replace("{width}", str(width))
                .replace("{height}", str(height))
                .replace("{half_w}", str(width - 1))
                .replace("{offset}", str(width + 20))
                .replace("{full_w}", str(sbs_w - 1)))
            for i in range(n_trials):
                user_content = [
                    {"type": "text", "text": "This image shows LEFT and RIGHT stereo views side by side. Report coordinates within each view's local frame."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_sbs}"}},
                ]
                result, ti, to = await call_gpt54(prompt_d, user_content, f"sbs {i+1}")
                total_tokens_in += ti
                total_tokens_out += to
                s = evaluate_result(result, cam_l, cam_r, width, height, f"sbs {i+1}")
                if s:
                    all_results.append({"config": f"{width}x{height}_sbs", "trial": i+1, **s})

    # Final summary
    print(f"\n{'━' * 90}")
    print("FINAL SUMMARY")
    print(f"{'━' * 90}")
    print(f"Total API usage: {total_tokens_in} tokens in, {total_tokens_out} tokens out\n")

    # Group by config
    configs = {}
    for r in all_results:
        c = r["config"]
        if c not in configs:
            configs[c] = []
        configs[c].append(r)

    print(f"{'Config':<25} {'Trials':>7} {'Avg px_mean':>12} {'Avg px_med':>11} "
          f"{'Tri<5cm':>8} {'Tri total':>10}")
    print("─" * 80)

    for config, trials in sorted(configs.items()):
        n = len(trials)
        avg_mean = np.mean([t["px_mean"] for t in trials])
        avg_med = np.mean([t["px_median"] for t in trials])
        total_ok = sum(t["tri_ok"] for t in trials)
        total_tri = sum(t["tri_total"] for t in trials)
        pct = (total_ok / total_tri * 100) if total_tri > 0 else 0

        print(f"{config:<25} {n:>7} {avg_mean:>10.1f}px {avg_med:>9.1f}px "
              f"{total_ok:>5}/{total_tri:<4} {pct:>6.0f}%")

    # Save raw results
    with open(OUT_DIR / "gpt54_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to {OUT_DIR / 'gpt54_sweep_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
