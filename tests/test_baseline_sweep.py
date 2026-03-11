"""Test triangulation accuracy vs baseline width.

Simulates different camera separations to find the sweet spot.
Uses actual CV detection errors (~10-15px) as the noise model.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.perception import triangulate_point

GROUND_TRUTH = {
    "red_box": [0.3, 0.05, 0.45],
    "blue_box": [0.3, -0.15, 0.45],
    "green_box": [0.4, 0.0, 0.45],
    "yellow_cylinder": [0.35, 0.15, 0.46],
}

# Observe pose: gripper at roughly [0.15, 0, 0.76] looking down
OBSERVE_POS_CENTER = np.array([0.1543, 0.0, 0.762])

W, H = 640, 480


def make_camera(pos, fovy=60.0):
    """Make a camera looking straight down from pos."""
    # Camera axes in world frame (looking down)
    # cam X = world X (right)
    # cam Y = world -Y (MuJoCo Y-down convention)
    # cam Z = world Z (up = away from scene, which is -viewing direction)
    # Actually for looking straight down: viewing dir = -Z world
    # cam -Z = viewing direction = [0, 0, -1] → cam Z = [0, 0, 1]
    # cam Y (MuJoCo down) = [0, -1, 0] in world
    # cam X = [1, 0, 0] in world
    # cam_xmat columns = cam axes in world
    rot_mat = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return {
        "fovy": fovy,
        "position": pos.tolist(),
        "rotation_matrix": rot_mat,
    }


def project(world_pt, cam):
    fovy = math.radians(cam["fovy"])
    fy = (H / 2.0) / math.tan(fovy / 2.0)
    fx = fy
    cx, cy = W / 2.0, H / 2.0
    R_world = np.array(cam["rotation_matrix"])
    pos = np.array(cam["position"])
    R_w2c = R_world.T
    flip = np.diag([1.0, -1.0, -1.0])
    R_cv = flip @ R_w2c
    t = -R_cv @ pos
    p_cam = R_cv @ np.array(world_pt) + t
    if p_cam[2] <= 0:
        return None
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return [float(u), float(v)]


def test_config(baseline_cm, fov, noise_px, n_trials=50):
    """Test triangulation accuracy for a given config with random noise."""
    half_b = baseline_cm / 200.0  # half baseline in meters
    cam_left = make_camera(OBSERVE_POS_CENTER + np.array([0, -half_b, 0]), fov)
    cam_right = make_camera(OBSERVE_POS_CENTER + np.array([0, half_b, 0]), fov)

    all_errors = []
    rng = np.random.RandomState(42)

    for _ in range(n_trials):
        for obj_id, world_pos in GROUND_TRUTH.items():
            px_l = project(world_pos, cam_left)
            px_r = project(world_pos, cam_right)
            if px_l is None or px_r is None:
                continue

            # Add Gaussian noise (more realistic than uniform)
            px_l_n = [px_l[0] + rng.normal(0, noise_px),
                      px_l[1] + rng.normal(0, noise_px)]
            px_r_n = [px_r[0] + rng.normal(0, noise_px),
                      px_r[1] + rng.normal(0, noise_px)]

            try:
                result = triangulate_point(px_l_n, px_r_n, cam_left, cam_right, W, H)
                error = np.linalg.norm(result - np.array(world_pos))
                all_errors.append(error)
            except Exception:
                pass

    if not all_errors:
        return None

    arr = np.array(all_errors)
    return {
        "mean_cm": round(float(arr.mean()) * 100, 2),
        "p50_cm": round(float(np.percentile(arr, 50)) * 100, 2),
        "p90_cm": round(float(np.percentile(arr, 90)) * 100, 2),
        "max_cm": round(float(arr.max()) * 100, 2),
    }


# Also compute disparity for reference
def get_disparity(baseline_cm, fov):
    half_b = baseline_cm / 200.0
    cam_left = make_camera(OBSERVE_POS_CENTER + np.array([0, -half_b, 0]), fov)
    cam_right = make_camera(OBSERVE_POS_CENTER + np.array([0, half_b, 0]), fov)
    disparities = []
    for world_pos in GROUND_TRUTH.values():
        px_l = project(world_pos, cam_left)
        px_r = project(world_pos, cam_right)
        if px_l and px_r:
            disparities.append(abs(px_l[1] - px_r[1]))  # v-disparity
    return round(np.mean(disparities), 1) if disparities else 0


def main():
    baselines = [4, 6, 8, 10, 15, 20, 30]
    fovs = [60, 90]
    cv_noise = 10  # ~10px Gaussian (what we measured)
    llm_noise = 80  # conservative LLM estimate

    print("=" * 80)
    print("BASELINE vs FOV vs DETECTION METHOD — Triangulation Accuracy")
    print("=" * 80)

    for fov in fovs:
        print(f"\n{'─' * 80}")
        print(f"FOV = {fov}°")
        print(f"{'─' * 80}")
        print(f"{'Baseline':>10} {'Disparity':>10} │ {'CV (σ=10px)':^30} │ {'LLM (σ=80px)':^30}")
        print(f"{'(cm)':>10} {'(px)':>10} │ {'mean':>8} {'p50':>8} {'p90':>8} │ {'mean':>8} {'p50':>8} {'p90':>8}")
        print(f"{'':>10} {'':>10} │ {'(cm)':>8} {'(cm)':>8} {'(cm)':>8} │ {'(cm)':>8} {'(cm)':>8} {'(cm)':>8}")
        print("─" * 80)

        for bl in baselines:
            disp = get_disparity(bl, fov)
            cv_res = test_config(bl, fov, cv_noise)
            llm_res = test_config(bl, fov, llm_noise)

            cv_str = f"{cv_res['mean_cm']:>8} {cv_res['p50_cm']:>8} {cv_res['p90_cm']:>8}" if cv_res else "     N/A"
            llm_str = f"{llm_res['mean_cm']:>8} {llm_res['p50_cm']:>8} {llm_res['p90_cm']:>8}" if llm_res else "     N/A"

            marker = ""
            if cv_res and cv_res["p90_cm"] < 3:
                marker = " ✅"
            elif cv_res and cv_res["p90_cm"] < 5:
                marker = " ⚡"

            print(f"{bl:>8}cm {disp:>8}px │ {cv_str} │ {llm_str}{marker}")

    print(f"\n{'─' * 80}")
    print("Legend: ✅ = p90 < 3cm (excellent), ⚡ = p90 < 5cm (good enough for pick-and-place)")
    print("Gripper width ~6cm → max practical on-gripper baseline ≈ 6cm")
    print("Multi-view (arm repositioning) can achieve 15-30cm baseline")
    print()

    # Also: what if we increase resolution with wider baseline?
    print("\n" + "=" * 80)
    print("BONUS: Higher resolution (1024x768) with wider baseline + CV")
    print("=" * 80)

    for bl in [8, 10, 15, 20]:
        for fov in [60, 90]:
            # At higher res, CV noise stays ~same in px
            global W, H
            W, H = 1024, 768
            disp = get_disparity(bl, fov)
            res = test_config(bl, fov, cv_noise)
            W, H = 640, 480  # reset

            if res:
                print(f"  {bl}cm baseline, {fov}° FOV, 1024x768: "
                      f"disp={disp}px, mean={res['mean_cm']}cm, p90={res['p90_cm']}cm")


if __name__ == "__main__":
    main()
