"""
Run the classical CV verifier on generated shape images and save results + debug overlays.

Usage:
  python scripts_verifier/eval_shape_verifier.py \
    --samples-jsonl outputs_verifier/base_shapes_val_g6/samples.jsonl \
    --out outputs_verifier/base_shapes_val_g6/verifier_results.jsonl \
    --summary-out outputs_verifier/base_shapes_val_g6/summary.json \
    --save-debug-overlays outputs_verifier/base_shapes_val_g6/debug_overlays \
    --images-root outputs_verifier/base_shapes_val_g6
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("pip install opencv-python-headless")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _draw_overlay(img_bgr, result, sample_id):
    """Draw color masks, bboxes, centers, relation line, and score text."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    dets = result.get("_dets_with_contour", result.get("detections", []))
    COLORS_BGR = {
        "red": (0, 0, 220), "blue": (220, 100, 0), "green": (0, 180, 0),
        "yellow": (0, 220, 220), "purple": (180, 0, 180), "orange": (0, 140, 255),
    }

    centers = []
    for d in dets:
        if not d.get("detected"):
            continue
        color_name = d["color"]
        draw_color = COLORS_BGR.get(color_name, (128, 128, 128))

        # draw contour / bbox
        if "contour" in d and d["contour"] is not None:
            cv2.drawContours(vis, [d["contour"]], -1, draw_color, 2)
        elif d.get("bbox"):
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), draw_color, 2)

        # draw center
        if d.get("center"):
            cx, cy = int(d["center"][0]), int(d["center"][1])
            cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(vis, (cx, cy), 5, draw_color, 2)
            centers.append((cx, cy))
            label = f"{color_name} {d['shape_expected']} (pred:{d['shape_pred']} s={d['shape_score']:.1f})"
            cv2.putText(vis, label, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 3)
            cv2.putText(vis, label, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, draw_color, 1)

    # relation line
    if len(centers) == 2:
        cv2.line(vis, centers[0], centers[1], (200, 200, 200), 1, cv2.LINE_AA)

    # reward summary
    comps = result.get("components", {})
    r = result.get("reward", 0.0)
    lines = [f"reward={r:.3f}"] + [f"  {k}={v:.2f}" for k, v in comps.items()]
    for li, txt in enumerate(lines):
        y = 14 + li * 14
        cv2.putText(vis, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 3)
        cv2.putText(vis, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 200), 1)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-jsonl",       required=True)
    parser.add_argument("--out",                 required=True)
    parser.add_argument("--summary-out",         required=True)
    parser.add_argument("--save-debug-overlays", default=None)
    parser.add_argument("--images-root",         default=None,
                        help="Root dir to resolve relative image paths in samples.jsonl")
    parser.add_argument("--max-overlays",        type=int, default=200,
                        help="Cap number of debug overlays saved (0=all)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts_verifier.shape_color_position_verifier import verify_image

    samples = load_jsonl(args.samples_jsonl)
    images_root = Path(args.images_root) if args.images_root else Path(args.samples_jsonl).parent

    overlay_dir = None
    if args.save_debug_overlays:
        overlay_dir = Path(args.save_debug_overlays)
        overlay_dir.mkdir(parents=True, exist_ok=True)

    out_path     = Path(args.out)
    summary_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rewards = []
    comp_accum  = {}
    n_missing   = 0
    overlays_saved = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, sample in enumerate(samples):
            img_path = Path(sample["image_path"])
            if not img_path.is_absolute():
                img_path = images_root / img_path

            if not img_path.exists():
                n_missing += 1
                print(f"[eval] missing image: {img_path}")
                continue

            result = verify_image(str(img_path), sample)

            # strip internal contour before writing
            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}

            row = {**{k: sample[k] for k in ("id", "prompt", "seed", "relation")
                      if k in sample},
                   **clean_result}
            f_out.write(json.dumps(row) + "\n")

            all_rewards.append(result["reward"])
            for k, v in result["components"].items():
                comp_accum.setdefault(k, []).append(v)

            # debug overlay
            if overlay_dir and (args.max_overlays == 0 or overlays_saved < args.max_overlays):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    vis = _draw_overlay(img_bgr, result, sample.get("id", str(i)))
                    fname = f"{sample.get('id', i)}_seed{sample.get('seed', 0)}.jpg"
                    cv2.imwrite(str(overlay_dir / fname), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    overlays_saved += 1

            if (i + 1) % 50 == 0:
                mean_so_far = sum(all_rewards) / len(all_rewards)
                print(f"  [{i+1}/{len(samples)}]  mean_reward={mean_so_far:.4f}")

    n = len(all_rewards)
    mean = sum(all_rewards) / n if n else 0.0
    se   = (sum((r - mean) ** 2 for r in all_rewards) / (n * (n - 1))) ** 0.5 if n > 1 else 0.0
    comp_means = {k: round(sum(v) / len(v), 4) for k, v in comp_accum.items()}

    summary = {
        "n":              n,
        "n_missing":      n_missing,
        "mean_reward":    round(mean, 4),
        "se_reward":      round(se, 4),
        "component_means": comp_means,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[eval] n={n}  mean_reward={mean:.4f}  se={se:.4f}")
    print(f"[eval] components: {comp_means}")
    if overlay_dir:
        print(f"[eval] {overlays_saved} overlays → {overlay_dir}")
    print(f"[eval] results → {out_path}")
    print(f"[eval] summary → {summary_path}")


if __name__ == "__main__":
    main()
