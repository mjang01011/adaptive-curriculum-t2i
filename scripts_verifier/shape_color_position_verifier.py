"""
Classical CV verifier for synthetic colored-shape spatial prompts.

Requires: opencv-python, numpy
  pip install opencv-python-headless numpy

Usage as module:
  from scripts_verifier.shape_color_position_verifier import verify_image
  result = verify_image(image_path, metadata_row)
"""
import math
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python required: pip install opencv-python-headless")

# ── HSV color ranges ────────────────────────────────────────────────────────
# Each entry is a list of (lo, hi) tuples in HSV (H:0-180, S:0-255, V:0-255)
COLOR_RANGES = {
    "red":    [((0,   70, 70), (12,  255, 255)),
               ((165, 70, 70), (180, 255, 255))],
    "orange": [((10,  80, 80), (22,  255, 255))],
    "yellow": [((22,  70, 70), (38,  255, 255))],
    "green":  [((38,  50, 50), (88,  255, 255))],
    "blue":   [((88,  50, 50), (132, 255, 255))],
    "purple": [((130, 35, 50), (165, 255, 255))],
}

MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


# ── color mask ───────────────────────────────────────────────────────────────

def _color_mask(hsv, color):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        lo_arr = np.array(lo, dtype=np.uint8)
        hi_arr = np.array(hi, dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo_arr, hi_arr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    return mask


# ── find largest blob ─────────────────────────────────────────────────────────

def _find_largest_blob(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a >= min_area and a > best_area:
            best_area = a
            best = cnt
    return best


# ── shape score ───────────────────────────────────────────────────────────────

def _shape_score(contour, expected_shape):
    area = cv2.contourArea(contour)
    if area < 1:
        return 0.0, "unknown"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n_verts = len(approx)
    circularity = 4 * math.pi * area / (peri * peri + 1e-6)

    if n_verts == 3:
        pred = "triangle"
    elif n_verts == 4:
        pred = "square"
    elif circularity > 0.65:
        pred = "circle"
    else:
        pred = "unknown"

    if expected_shape == "circle":
        if circularity > 0.65:
            return 1.0, pred
        if circularity > 0.45:
            return 0.5, pred
        return 0.0, pred

    if expected_shape == "square":
        if n_verts == 4:
            return 1.0, pred
        x, y, bw, bh = cv2.boundingRect(contour)
        ratio = bw / (bh + 1e-6)
        if 0.7 < ratio < 1.43:
            return 0.5, pred
        return 0.0, pred

    if expected_shape == "triangle":
        if n_verts == 3:
            return 1.0, pred
        if 3 <= n_verts <= 5:
            return 0.5, pred
        return 0.0, pred

    return 0.0, pred


# ── detection ────────────────────────────────────────────────────────────────

def _detect(hsv, color, shape, image_area):
    min_blob_area = max(80, 0.002 * image_area)
    mask = _color_mask(hsv, color)
    contour = _find_largest_blob(mask, min_blob_area)
    if contour is None:
        return {
            "color": color, "shape_expected": shape,
            "detected": False,
            "color_score": 0.0, "shape_score": 0.0,
            "center": None, "bbox": None, "area": 0,
            "circularity": 0.0, "shape_pred": "none",
        }
    area = float(cv2.contourArea(contour))
    M = cv2.moments(contour)
    cx = int(M["m10"] / (M["m00"] + 1e-6))
    cy = int(M["m01"] / (M["m00"] + 1e-6))
    x, y, bw, bh = cv2.boundingRect(contour)
    peri = cv2.arcLength(contour, True)
    circ = 4 * math.pi * area / (peri * peri + 1e-6)
    ss, pred = _shape_score(contour, shape)

    return {
        "color": color, "shape_expected": shape,
        "detected": True,
        "color_score": 1.0, "shape_score": ss,
        "center": [cx, cy], "bbox": [x, y, x + bw, y + bh],
        "area": area, "circularity": circ, "shape_pred": pred,
        "contour": contour,
    }


# ── IoU + separation ──────────────────────────────────────────────────────────

def _iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


def _separation_score(d1, d2):
    if not d1["detected"] or not d2["detected"]:
        return 0.0
    iou = _iou(d1["bbox"], d2["bbox"])
    if iou < 0.10:
        return 1.0
    if iou < 0.25:
        return 0.5
    return 0.0


# ── area ratio ────────────────────────────────────────────────────────────────

def _area_ratio_score(d1, d2):
    if not d1["detected"] or not d2["detected"]:
        return 0.0
    a1, a2 = d1["area"], d2["area"]
    if a1 == 0 or a2 == 0:
        return 0.0
    ratio = max(a1, a2) / (min(a1, a2) + 1e-6)
    if ratio < 4:
        return 1.0
    if ratio < 8:
        return 0.5
    return 0.0


# ── relation score ────────────────────────────────────────────────────────────

def _relation_score(d1, d2, relation, image_w, image_h):
    # both objects must be detected for relation to count
    if not d1["detected"] or not d2["detected"]:
        return 0.0
    cx1, cy1 = d1["center"]
    cx2, cy2 = d2["center"]
    margin = 0.05 * image_w

    if relation == "left_of":
        delta = cx2 - cx1
    elif relation == "right_of":
        delta = cx1 - cx2
    elif relation == "above":
        delta = cy2 - cy1
        margin = 0.05 * image_h
    elif relation == "below":
        delta = cy1 - cy2
        margin = 0.05 * image_h
    else:
        return 0.5

    if delta > margin:
        return 1.0
    if delta > -margin:
        return 0.5
    return 0.0


# ── quality score ─────────────────────────────────────────────────────────────

def _quality_score(detections, image_area):
    total_colored = sum(d["area"] for d in detections)
    frac = total_colored / (image_area + 1e-6)
    if frac < 0.005:
        return 0.0
    if frac < 0.015:
        return 0.5
    if frac > 0.80:
        return 0.3
    return 1.0


# ── object → shape family ─────────────────────────────────────────────────────

OBJECT_FAMILY = {
    # round
    "ball":  "circle",
    "coin":  "circle",
    "plate": "circle",
    # rectangular
    "book":  "rect",
    "box":   "rect",
    "card":  "rect",
    "block": "rect",
}

# map family names to what _shape_score expects
FAMILY_TO_SHAPE = {
    "circle":   "circle",
    "rect":     "square",   # square heuristics work for rectangles too
    "triangle": "triangle",
    # pass-through for raw shape mode
    "square":   "square",
}


def _resolve_shape(obj):
    """Return the shape string to pass to _shape_score for this object."""
    if "family" in obj:
        return FAMILY_TO_SHAPE.get(obj["family"], "circle")
    if "shape" in obj:
        return FAMILY_TO_SHAPE.get(obj["shape"], obj["shape"])
    if "object" in obj:
        family = OBJECT_FAMILY.get(obj["object"], "circle")
        return FAMILY_TO_SHAPE.get(family, "circle")
    return "circle"


# ── public API ────────────────────────────────────────────────────────────────

def verify_image(image_path, metadata):
    """
    Args:
        image_path: str or Path to a PNG/JPG
        metadata:   dict with keys "objects" (list of {color, shape, position})
                    and optional "relation"
    Returns:
        dict with keys: reward, components, detections
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {"reward": 0.0, "components": {}, "detections": [], "error": "cannot_read"}

    h, w = img_bgr.shape[:2]
    image_area = h * w
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    objects  = metadata.get("objects", [])
    relation = metadata.get("relation", None)

    dets = []
    for obj in objects:
        shape = _resolve_shape(obj)
        d = _detect(img_hsv, obj["color"], shape, image_area)
        dets.append(d)

    quality    = _quality_score(dets, image_area)

    if len(dets) == 2 and relation:
        rel_score  = _relation_score(dets[0], dets[1], relation, w, h)
        sep_score  = _separation_score(dets[0], dets[1])
        ar_score   = _area_ratio_score(dets[0], dets[1])
        components = {
            "obj1_color": dets[0]["color_score"],
            "obj2_color": dets[1]["color_score"],
            "obj1_shape": dets[0]["shape_score"],
            "obj2_shape": dets[1]["shape_score"],
            "relation":   rel_score,
            "separation": sep_score,
            "area_ratio": ar_score,
            "quality":    quality,
        }
        reward = (
            0.20  * components["obj1_color"]
            + 0.20  * components["obj2_color"]
            + 0.05  * components["obj1_shape"]
            + 0.05  * components["obj2_shape"]
            + 0.25  * components["relation"]
            + 0.20  * components["separation"]
            + 0.10  * components["area_ratio"]
            + 0.05  * components["quality"]
        )
    elif len(dets) == 2:
        sep_score  = _separation_score(dets[0], dets[1])
        ar_score   = _area_ratio_score(dets[0], dets[1])
        components = {
            "obj1_color": dets[0]["color_score"],
            "obj2_color": dets[1]["color_score"],
            "obj1_shape": dets[0]["shape_score"],
            "obj2_shape": dets[1]["shape_score"],
            "separation": sep_score,
            "area_ratio": ar_score,
            "quality":    quality,
        }
        reward = (
            0.25  * components["obj1_color"]
            + 0.25  * components["obj2_color"]
            + 0.10  * components["obj1_shape"]
            + 0.10  * components["obj2_shape"]
            + 0.15  * components["separation"]
            + 0.10  * components["area_ratio"]
            + 0.05  * components["quality"]
        )
    else:
        components = {"quality": quality}
        reward = quality * 0.1

    clean_dets = [{k: v for k, v in d.items() if k != "contour"} for d in dets]

    return {
        "reward":      round(float(reward), 4),
        "components":  {k: round(float(v), 4) for k, v in components.items()},
        "detections":  clean_dets,
        "_dets_with_contour": dets,
    }
