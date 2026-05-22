"""
Test whether Janus can generate verifier-scorable shapes with different prompt prefixes.

Generates 4 images (one per prefix) for a fixed shape prompt, runs the CV verifier,
and saves images + an HTML summary so you can see both image and score side by side.

Usage (from janus_project dir, in januspro_venv):
  python /viscam/u/jj277/adaptive-curriculum-t2i/scripts_verifier/test_janus_shape_prefixes.py \
    --out-dir /viscam/u/jj277/janus_project/outputs/prefix_test \
    --model-path /viscam/u/jj277/janus_project/Janus
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── prompt setup ──────────────────────────────────────────────────────────────

BASE_PROMPT = "a red circle on the left and a blue square on the right"

PREFIXES = [
    "",
    "simple flat clipart on white background: ",
    "minimal flat 2D icon, solid white background, no shading: ",
    "flat colored geometric shapes on white background: ",
]

METADATA = {
    "objects": [
        {"color": "red",  "shape": "circle",  "family": "circle", "position": "left"},
        {"color": "blue", "shape": "square",  "family": "square", "position": "right"},
    ],
    "relation": "left_of",
}


# ── verifier ──────────────────────────────────────────────────────────────────

def score(pil_img):
    import cv2
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts_verifier.shape_color_position_verifier import verify_image_bgr
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return verify_image_bgr(bgr, METADATA)


# ── HTML report ───────────────────────────────────────────────────────────────

def _b64(pil_img):
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_html(results, out_path):
    cards = ""
    for r in results:
        comps = r["result"].get("components", {})
        reward = r["result"]["reward"]
        comp_rows = "".join(
            f"<tr><td>{k}</td><td style='color:{'#6f6' if v>=0.7 else '#ff6' if v>=0.4 else '#f66'}'>{v:.3f}</td></tr>"
            for k, v in comps.items()
        )
        prefix_display = r["prefix"] if r["prefix"] else "(no prefix)"
        full_prompt = r["full_prompt"]
        cards += f"""
<div style="background:#222;border:1px solid #444;border-radius:6px;padding:12px;width:340px;flex-shrink:0">
  <img src="{_b64(r['pil'])}" style="width:100%;border:1px solid #555"><br>
  <div style="color:#fa0;margin:6px 0;font-size:11px"><b>Prefix:</b> {prefix_display}</div>
  <div style="color:#aef;font-size:10px;margin-bottom:6px">{full_prompt}</div>
  <div style="font-size:13px;color:{'#6f6' if reward>=0.7 else '#ff6' if reward>=0.4 else '#f66'}">
    reward = {reward:.4f}
  </div>
  <table style="font-size:11px;width:100%;margin-top:4px">{comp_rows}</table>
</div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body {{font-family:monospace;background:#111;color:#eee;margin:16px}}
  h1 {{color:#7cf}} h2 {{color:#fa0}}
  .grid {{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}}
</style></head><body>
<h1>Janus Prefix Test</h1>
<h2>Base prompt: "{BASE_PROMPT}"</h2>
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load wrapper once (reused across all prefixes)
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from scripts_janus.janus_wrapper import JanusProWrapper

    print("[test] Loading Janus...")
    wrapper = JanusProWrapper(model_path=args.model_path, cfg_weight=5.0, temperature=1.0)
    _ = wrapper.model  # trigger load
    print("[test] Model loaded.\n")

    results = []
    for i, prefix in enumerate(PREFIXES):
        full_prompt = prefix + BASE_PROMPT
        print(f"[{i+1}/{len(PREFIXES)}] Generating: \"{full_prompt}\"")

        out = wrapper.generate_images([full_prompt], seeds=[args.seed])
        pil = out["images"][0]
        pil.save(out_dir / f"prefix_{i}.png")

        result = score(pil)
        print(f"         reward={result['reward']:.4f}  components={result['components']}\n")

        results.append({
            "prefix":      prefix,
            "full_prompt": full_prompt,
            "pil":         pil,
            "result":      result,
        })

    make_html(results, out_dir / "prefix_test.html")

    # also save raw results
    with open(out_dir / "results.json", "w") as f:
        json.dump([
            {"prefix": r["prefix"], "full_prompt": r["full_prompt"],
             "reward": r["result"]["reward"], "components": r["result"]["components"]}
            for r in results
        ], f, indent=2)

    print(f"\n[test] done. images + HTML → {out_dir}")
    best = max(results, key=lambda r: r["result"]["reward"])
    print(f"[test] best prefix: \"{best['prefix']}\"  reward={best['result']['reward']:.4f}")


if __name__ == "__main__":
    main()
