"""Compute the interaction space: the table region visible in BOTH cameras.

Projects camera frustum corners onto the table plane to find the overlap rectangle.
"""

import asyncio
import math
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RASIM_URL = "http://localhost:8100"
TABLE_Z = 0.42  # table surface height


def frustum_corners_on_table(cam, width, height, table_z=TABLE_Z):
    """Project the four image corners onto the table plane.

    Returns the 4 world [x, y] positions where the corner rays hit z=table_z.
    """
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

    # Inverse: world_point = R_cv_inv @ (cam_point - t)
    R_cv_inv = np.linalg.inv(R_cv)

    corners_px = [
        [0, 0],           # top-left
        [width, 0],       # top-right
        [width, height],  # bottom-right
        [0, height],      # bottom-left
    ]

    table_points = []
    for u, v in corners_px:
        # Ray in camera frame: direction = K_inv @ [u, v, 1]
        dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        # Transform to world frame
        dir_world = R_cv_inv @ dir_cam
        # Ray: P = pos + t_param * dir_world
        # Solve for z = table_z: pos[2] + t_param * dir_world[2] = table_z
        if abs(dir_world[2]) < 1e-10:
            continue
        t_param = (table_z - pos[2]) / dir_world[2]
        if t_param < 0:
            continue  # ray points away from table
        hit = pos + t_param * dir_world
        table_points.append(hit[:2].tolist())

    return table_points


def compute_visible_rect(cam_left, cam_right, width, height, table_z=TABLE_Z):
    """Compute the intersection of both camera views on the table.

    Returns the axis-aligned bounding box that's visible in BOTH cameras.
    """
    left_corners = frustum_corners_on_table(cam_left, width, height, table_z)
    right_corners = frustum_corners_on_table(cam_right, width, height, table_z)

    if len(left_corners) < 4 or len(right_corners) < 4:
        print("WARNING: Not all frustum corners hit the table!")
        return None

    # AABB of each camera's footprint
    left_arr = np.array(left_corners)
    right_arr = np.array(right_corners)

    left_min_x, left_min_y = left_arr.min(axis=0)
    left_max_x, left_max_y = left_arr.max(axis=0)
    right_min_x, right_min_y = right_arr.min(axis=0)
    right_max_x, right_max_y = right_arr.max(axis=0)

    # Intersection
    inter_min_x = max(left_min_x, right_min_x)
    inter_max_x = min(left_max_x, right_max_x)
    inter_min_y = max(left_min_y, right_min_y)
    inter_max_y = min(left_max_y, right_max_y)

    if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y:
        print("WARNING: No overlap between cameras!")
        return None

    return {
        "left_footprint": left_corners,
        "right_footprint": right_corners,
        "left_aabb": [left_min_x, left_max_x, left_min_y, left_max_y],
        "right_aabb": [right_min_x, right_max_x, right_min_y, right_max_y],
        "intersection": [inter_min_x, inter_max_x, inter_min_y, inter_max_y],
    }


async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Move to observe pose first
        resp = await client.post(
            f"{RASIM_URL}/execute",
            json={"commands": [{"type": "observe"}]},
            timeout=60.0,
        )
        resp.raise_for_status()

        # Get camera params at observe pose
        resp = await client.get(f"{RASIM_URL}/eye/params")
        resp.raise_for_status()
        params = resp.json()

        cam_l = params["left"]
        cam_r = params["right"]

        bl = np.linalg.norm(np.array(cam_l["position"]) - np.array(cam_r["position"]))
        print(f"Camera: FOV={cam_l['fovy']}°, baseline={bl*100:.1f}cm")
        print(f"Left pos:  {[round(x, 4) for x in cam_l['position']]}")
        print(f"Right pos: {[round(x, 4) for x in cam_r['position']]}")

        for res_name, w, h in [("640x480", 640, 480), ("1024x768", 1024, 768)]:
            print(f"\n{'='*60}")
            print(f"Resolution: {res_name}")
            print(f"{'='*60}")

            result = compute_visible_rect(cam_l, cam_r, w, h)
            if result is None:
                print("  No valid intersection!")
                continue

            print(f"\n  Left camera footprint on table:")
            for i, pt in enumerate(result["left_footprint"]):
                print(f"    corner {i}: x={pt[0]:.4f}, y={pt[1]:.4f}")
            print(f"  Left AABB: x=[{result['left_aabb'][0]:.4f}, {result['left_aabb'][1]:.4f}], "
                  f"y=[{result['left_aabb'][2]:.4f}, {result['left_aabb'][3]:.4f}]")

            print(f"\n  Right camera footprint on table:")
            for i, pt in enumerate(result["right_footprint"]):
                print(f"    corner {i}: x={pt[0]:.4f}, y={pt[1]:.4f}")
            print(f"  Right AABB: x=[{result['right_aabb'][0]:.4f}, {result['right_aabb'][1]:.4f}], "
                  f"y=[{result['right_aabb'][2]:.4f}, {result['right_aabb'][3]:.4f}]")

            inter = result["intersection"]
            print(f"\n  ✅ INTERACTION SPACE (both cameras see):")
            print(f"     x: [{inter[0]:.4f}, {inter[1]:.4f}] ({(inter[1]-inter[0])*100:.1f}cm)")
            print(f"     y: [{inter[2]:.4f}, {inter[3]:.4f}] ({(inter[3]-inter[2])*100:.1f}cm)")
            print(f"     area: {(inter[1]-inter[0])*(inter[3]-inter[2])*10000:.0f} cm²")

            # Also check: what's the arm's reachable workspace?
            resp2 = await client.get(f"{RASIM_URL}/scene-state")
            resp2.raise_for_status()
            ws = resp2.json().get("workspace", {})
            ws_bounds = ws.get("bounds", [0.1, 0.9, -0.4, 0.4])
            print(f"\n  Arm workspace bounds: x=[{ws_bounds[0]}, {ws_bounds[1]}], y=[{ws_bounds[2]}, {ws_bounds[3]}]")

            # Final interaction space = intersection of camera overlap AND arm reach
            final_min_x = max(inter[0], ws_bounds[0])
            final_max_x = min(inter[1], ws_bounds[1])
            final_min_y = max(inter[2], ws_bounds[2])
            final_max_y = min(inter[3], ws_bounds[3])

            print(f"\n  🎯 FINAL INTERACTION SPACE (cameras ∩ arm reach):")
            print(f"     x: [{final_min_x:.4f}, {final_max_x:.4f}] ({(final_max_x-final_min_x)*100:.1f}cm)")
            print(f"     y: [{final_min_y:.4f}, {final_max_y:.4f}] ({(final_max_y-final_min_y)*100:.1f}cm)")
            print(f"     area: {(final_max_x-final_min_x)*(final_max_y-final_min_y)*10000:.0f} cm²")

            # Add margin (shrink by 2cm each side for safety)
            margin = 0.02
            safe_min_x = final_min_x + margin
            safe_max_x = final_max_x - margin
            safe_min_y = final_min_y + margin
            safe_max_y = final_max_y - margin

            print(f"\n  🛡️  SAFE INTERACTION SPACE (with 2cm margin):")
            print(f"     x: [{safe_min_x:.4f}, {safe_max_x:.4f}] ({(safe_max_x-safe_min_x)*100:.1f}cm)")
            print(f"     y: [{safe_min_y:.4f}, {safe_max_y:.4f}] ({(safe_max_y-safe_min_y)*100:.1f}cm)")
            print(f"     area: {(safe_max_x-safe_min_x)*(safe_max_y-safe_min_y)*10000:.0f} cm²")

            # Check scene objects
            print(f"\n  Scene objects vs interaction space:")
            for prop in resp2.json().get("props", []):
                px, py = prop["pos"][0], prop["pos"][1]
                inside = safe_min_x <= px <= safe_max_x and safe_min_y <= py <= safe_max_y
                status = "✅ inside" if inside else "❌ OUTSIDE"
                print(f"    {prop['id']}: [{px:.4f}, {py:.4f}] — {status}")


if __name__ == "__main__":
    asyncio.run(main())
