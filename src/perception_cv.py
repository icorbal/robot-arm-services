"""Computer-vision-based perception pipeline.

Uses color segmentation to detect objects instead of LLM pixel estimation.
The LLM's pixel accuracy (~80-200px error) is the bottleneck for triangulation.
Classical CV provides sub-pixel accuracy at no API cost.

This module provides:
- Color-based object detection via HSV segmentation
- Hybrid mode: LLM identifies object types, CV refines pixel positions
"""

import io
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# HSV ranges for known object colors (tuned for MuJoCo rendered scenes)
# Format: (H_low, S_low, V_low), (H_high, S_high, V_high)
COLOR_RANGES = {
    "red": [
        ((0, 100, 100), (10, 255, 255)),      # low-H red
        ((170, 100, 100), (180, 255, 255)),    # high-H red (wraps)
    ],
    "blue": [
        ((100, 100, 50), (130, 255, 255)),
    ],
    "green": [
        ((35, 100, 50), (85, 255, 255)),
    ],
    "yellow": [
        ((20, 100, 100), (35, 255, 255)),
    ],
}

# Minimum contour area to consider (filters noise)
MIN_AREA = 50


def detect_objects_cv(
    image_bytes: bytes,
    known_colors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Detect colored objects in an image using HSV color segmentation.

    Args:
        image_bytes: PNG image as bytes
        known_colors: List of color names to look for. If None, searches all.

    Returns:
        List of detected objects with pixel coordinates and color info.
    """
    try:
        import cv2
    except ImportError:
        # Fallback using PIL + numpy only
        return _detect_objects_pil(image_bytes, known_colors)

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode image")
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    results = []

    colors_to_check = known_colors or list(COLOR_RANGES.keys())

    for color_name in colors_to_check:
        ranges = COLOR_RANGES.get(color_name, [])
        if not ranges:
            continue

        # Combine masks for all ranges of this color
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            # Centroid via moments
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Bounding box for size estimation
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)

            # Guess type from shape
            if 0.8 < aspect < 1.2:
                obj_type = "box"  # roughly square
            else:
                obj_type = "cylinder"

            suffix = f"_{i+1}" if i > 0 else ""
            results.append({
                "id": f"{color_name}_{obj_type}{suffix}",
                "color": color_name,
                "type": obj_type,
                "px": [round(float(cx), 1), round(float(cy), 1)],
                "area": int(area),
                "bbox": [x, y, w, h],
            })

    # Sort by area (largest first)
    results.sort(key=lambda r: r["area"], reverse=True)
    return results


def _detect_objects_pil(
    image_bytes: bytes,
    known_colors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fallback detection using PIL + numpy when OpenCV isn't available."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixels = np.array(img, dtype=np.float32)

    results = []
    colors_to_check = known_colors or list(COLOR_RANGES.keys())

    # Simple color detection based on RGB ratios
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    total = r + g + b + 1e-6  # avoid division by zero

    color_masks = {
        "red": (r > 150) & (g < 80) & (b < 80),
        "blue": (b > 150) & (r < 80) & (g < 80),
        "green": (g > 150) & (r < 80) & (b < 80),
        "yellow": (r > 150) & (g > 150) & (b < 80),
    }

    for color_name in colors_to_check:
        mask = color_masks.get(color_name)
        if mask is None:
            continue

        # Find centroid of masked pixels
        ys, xs = np.where(mask)
        if len(xs) < MIN_AREA:
            continue

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        results.append({
            "id": f"{color_name}_object",
            "color": color_name,
            "type": "box",
            "px": [round(cx, 1), round(cy, 1)],
            "area": int(len(xs)),
        })

    return results


def stereo_perceive_cv(
    img_left_bytes: bytes,
    img_right_bytes: bytes,
    cam_left: dict,
    cam_right: dict,
    width: int,
    height: int,
    known_colors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Full stereo perception pipeline using CV.

    Detects objects in both images, matches by color, triangulates.

    Args:
        img_left_bytes: Left camera image PNG bytes
        img_right_bytes: Right camera image PNG bytes
        cam_left: Left camera parameters
        cam_right: Right camera parameters
        width: Image width
        height: Image height
        known_colors: Colors to search for

    Returns:
        List of detected objects with 3D world positions.
    """
    from .perception import triangulate_point

    left_objects = detect_objects_cv(img_left_bytes, known_colors)
    right_objects = detect_objects_cv(img_right_bytes, known_colors)

    logger.info(f"CV detected {len(left_objects)} left, {len(right_objects)} right objects")

    # Match objects between views by color
    results = []
    used_right = set()

    for left_obj in left_objects:
        color = left_obj["color"]

        # Find matching right object (same color, not yet matched)
        best_match = None
        for j, right_obj in enumerate(right_objects):
            if j in used_right:
                continue
            if right_obj["color"] == color:
                best_match = (j, right_obj)
                break

        if best_match is None:
            logger.warning(f"No right match for {left_obj['id']}")
            continue

        j, right_obj = best_match
        used_right.add(j)

        left_px = left_obj["px"]
        right_px = right_obj["px"]

        try:
            world_pos = triangulate_point(
                left_px, right_px,
                cam_left, cam_right,
                width, height,
            )
            logger.info(
                f"{left_obj['id']}: left={left_px}, right={right_px} "
                f"→ [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]"
            )
        except Exception as e:
            logger.error(f"Triangulation failed for {left_obj['id']}: {e}")
            continue

        results.append({
            "id": left_obj["id"],
            "type": left_obj["type"],
            "color": color,
            "pos": [round(float(world_pos[0]), 4),
                    round(float(world_pos[1]), 4),
                    round(float(world_pos[2]), 4)],
            "size": [0.03, 0.03, 0.03],
            "confidence": "cv",
        })

    return results
