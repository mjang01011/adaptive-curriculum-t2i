"""
Test Janus on real-world object prompts scorable by the classical CV verifier.

Generates images for several object+color combinations, scores with the verifier,
and outputs a self-contained HTML report.

Usage (in januspro_venv):
  python /viscam/u/jj277/adaptive-curriculum-t2i/scripts_verifier/test_janus_objects.py \
    --out-dir /viscam/u/jj277/janus_project/outputs/objects_test
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── test cases ────────────────────────────────────────────────────────────────
# Objects chosen for: strong color association + compact blob shape
# family: circle → round verifier heuristic, rect → square heuristic

TEST_CASES = [
    {
        "prompt":   "a red apple on the left and a blue cup on the right, white background",
        "objects":  [{"color": "red",  "family": "circle", "position": "left"},
                     {"color": "blue", "family": "rect",   "position": "right"}],
        "relation": "left_of",
    },
    {
        "prompt":   "a green apple on the left and a red ball on the right, white background",
        "objects":  [{"color": "green", "family": "circle", "position": "left"},
                     {"color": "red",   "family": "circle", "position": "right"}],
        "relation": "left_of",
    },
    {
        "prompt":   "a yellow banana on the left and a red apple on the right, white background",
        "objects":  [{"color": "yellow", "family": "circle", "position": "left"},
                     {"color": "red",    "family": "circle", "position": "right"}],
        "relation": "left_of",
    },
    {
        "prompt":   "a red strawberry on the left and a green lime on the right, white background",
        "objects":  [{"color": "red",   "family": "circle", "position": "left"},
                     {"color": "green", "family": "circle", "position": "right"}],
        "relation": "left_of",
    },
    {
        "prompt":   "a blue mug on the left and a red apple on the right, white background",
        "objects":  [{"color": "blue", "family": "rect",   "position": "left"},
                     {"color": "red",  "family": "circle", "position": "right"}],
        "relation": "left_of",
    },
    {
        "prompt":   "a red apple above a blue cup, white background",
        "objects":  [{"color": "red",  "family": "circle", "position": "top"},
                     {"color": "blue", "family": "rect",   "position": "bottom"}],
        "relation": "above",
    },
    {
        "prompt":   "an orange on the left and a purple grape on the right, white background",
        "objects":  [{"color": "orange", "family": "circle", "position": "left"},
                     {"color": "purple", "family": "circle", "position": "right"}],
        "relation": "left_of",
    },
    {
        "prompt":   "a red tomato on the left and a yellow lemon on the right, white background",
        "objects":  [{"color": "red",    "family": "circle", "position": "left"},
                     {"color": "yellow", "family": "circle", "position": "right"}],
        "relation": "left_of",
    },
]

# ── verifier ──────────────────────────────────────────────────────────────────

def score(pil_img, metadata):
    import cv2
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts_verifier.shape_color_position_verifier import verify_image_bgr
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return verify_image_bgr(bgr, metadata)


# ── HTML ──────────────────────────────────────────────────────────────────────

def _b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_html(results, out_path):
    cards = ""
    for r in results:
        reward = r["result"]["reward"]
        comps  = r["result"].get("components", {})
        color  = '#6f6' if reward >= 0.5 else '#ff6' if reward >= 0.25 else '#f66'
        comp_rows = "".join(
            f"<tr><td>{k}</td>"
            f"<td style='color:{'#6f6' if v>=0.7 else '#ff6' if v>=0.4 else '#f66'}'>{v:.3f}</td></tr>"
            for k, v in comps.items()
        )
        cards += f"""
<div style="background:#222;border:1px solid #444;border-radius:6px;padding:10px;width:300px;flex-shrink:0">
  <img src="{_b64(r['pil'])}" style="width:100%;border:1px solid #555;margin-bottom:6px">
  <div style="color:#aef;font-size:10px;margin-bottom:4px">{r['prompt']}</div>
  <div style="font-size:14px;color:{color};font-weight:bold">reward = {reward:.4f}</div>
  <table style="font-size:11px;width:100%;margin-top:4px;border-collapse:collapse">{comp_rows}</table>
</div>"""

    # sort by reward descending for easy scanning
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body {{font-family:monospace;background:#111;color:#eee;margin:16px}}
  h1 {{color:#7cf}} p {{color:#aaa}}
  .grid {{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}}
</style></head><body>
<h1>Janus Real-World Object Test</h1>
<p>Sorted by reward. Green ≥ 0.5, yellow ≥ 0.25, red &lt; 0.25</p>
<div class="grid">{cards}</div>
</body></html>"""

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"[html] → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",    required=True)
    parser.add_argument("--model-path", default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--seeds",      type=int, nargs="+", default=None,
                        help="Generate multiple seeds per prompt")
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    args = parser.parse_args()

    seeds   = args.seeds or [args.seed]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts_janus.janus_wrapper import JanusProWrapper

    print(f"[test] Loading Janus... cfg_weight={args.cfg_weight}")
    wrapper = JanusProWrapper(model_path=args.model_path, cfg_weight=args.cfg_weight, temperature=1.0)
    _ = wrapper.model
    print(f"[test] Model loaded. Running {len(TEST_CASES)} prompts × {len(seeds)} seeds.\n")

    results = []
    for i, case in enumerate(TEST_CASES):
        metadata = {"objects": case["objects"], "relation": case["relation"]}
        for seed in seeds:
            print(f"[{i+1}/{len(TEST_CASES)}] seed={seed}  {case['prompt'][:70]}")
            out = wrapper.generate_images([case["prompt"]], seeds=[seed])
            pil = out["images"][0]

            fname = f"case{i:02d}_seed{seed}.png"
            pil.save(out_dir / fname)

            result = score(pil, metadata)
            print(f"         reward={result['reward']:.4f}  "
                  f"obj1_color={result['components'].get('obj1_color',0):.2f}  "
                  f"obj2_color={result['components'].get('obj2_color',0):.2f}  "
                  f"relation={result['components'].get('relation',0):.2f}\n")

            results.append({
                "prompt":   case["prompt"],
                "seed":     seed,
                "pil":      pil,
                "result":   result,
                "fname":    fname,
            })

    # sort by reward for HTML
    results_sorted = sorted(results, key=lambda r: r["result"]["reward"], reverse=True)
    make_html(results_sorted, out_dir / "objects_test.html")

    with open(out_dir / "results.json", "w") as f:
        json.dump([
            {"prompt": r["prompt"], "seed": r["seed"],
             "reward": r["result"]["reward"],
             "components": r["result"]["components"]}
            for r in results
        ], f, indent=2)

    print(f"\n[test] Best results:")
    for r in results_sorted[:3]:
        print(f"  {r['result']['reward']:.4f}  {r['prompt'][:60]}")


if __name__ == "__main__":
    main()
