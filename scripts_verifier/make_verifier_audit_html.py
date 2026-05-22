"""
Build a visual audit HTML report for the shape verifier.

Usage:
  python scripts_verifier/make_verifier_audit_html.py \
    --verifier-results outputs_verifier/base_shapes_val_g6/verifier_results.jsonl \
    --images-root      outputs_verifier/base_shapes_val_g6 \
    --overlays-root    outputs_verifier/base_shapes_val_g6/debug_overlays \
    --out              outputs_verifier/base_shapes_val_g6/verifier_audit.html
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

# ── CSS + HTML helpers ────────────────────────────────────────────────────────

CSS = """
body { font-family: monospace; font-size: 12px; background: #111; color: #eee; margin: 16px; }
h1   { color: #7cf; }
h2   { color: #fa0; margin-top: 32px; border-bottom: 1px solid #444; padding-bottom: 4px; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.card {
  background: #222; border: 1px solid #444; border-radius: 6px;
  padding: 8px; width: 300px; flex-shrink: 0;
}
.card img { width: 140px; height: 140px; object-fit: contain; border: 1px solid #555; }
.images   { display: flex; gap: 4px; margin-bottom: 6px; }
.prompt   { color: #aef; margin-bottom: 4px; font-size: 11px; }
.scores   { font-size: 11px; line-height: 1.6; }
.hi  { color: #6f6; }
.mid { color: #ff6; }
.lo  { color: #f66; }
.reward-bar { height: 6px; background: #555; border-radius: 3px; margin: 4px 0; }
.reward-fill { height: 100%; border-radius: 3px; }
"""

def _color_class(v):
    if v >= 0.7:  return "hi"
    if v >= 0.4:  return "mid"
    return "lo"

def _bar(v, color="#5af"):
    pct = int(v * 100)
    return (f'<div class="reward-bar">'
            f'<div class="reward-fill" style="width:{pct}%;background:{color}"></div>'
            f'</div>')

def _rel_img(base_path, ref_path):
    """Return relative path from ref_path's parent to base_path."""
    try:
        return os.path.relpath(str(base_path), str(Path(ref_path).parent))
    except ValueError:
        return str(base_path)

def _card(row, images_root, overlays_root, html_path):
    sid   = row.get("id", "")
    seed  = row.get("seed", 0)
    prompt = row.get("prompt", "")
    reward = row.get("reward", 0.0)
    comps  = row.get("components", {})
    dets   = row.get("detections", [])

    # original image path
    img_rel = row.get("image_path", "")
    img_abs = Path(images_root) / img_rel if img_rel and not Path(img_rel).is_absolute() else Path(img_rel)
    img_src = _rel_img(img_abs, html_path) if img_abs.exists() else ""

    # overlay
    overlay_abs = Path(overlays_root) / f"{sid}_seed{seed}.jpg"
    ov_src = _rel_img(overlay_abs, html_path) if overlay_abs.exists() else ""

    img_tag = lambda src: f'<img src="{src}" loading="lazy">' if src else '<div style="width:140px;height:140px;background:#333;display:inline-block"></div>'

    det_lines = []
    for i, d in enumerate(dets[:2]):
        c  = d.get("color", "?")
        se = d.get("shape_expected", "?")
        sp = d.get("shape_pred", "?")
        det_lines.append(f"obj{i+1}: {c} {se} → pred {sp}  area={d.get('area', 0):.0f}")

    comp_html = ""
    for k, v in comps.items():
        cc = _color_class(v)
        comp_html += f'<span class="{cc}">{k}={v:.2f}</span>  '

    det_html = "<br>".join(det_lines)

    rc = _color_class(reward)
    return f"""
<div class="card">
  <div class="images">{img_tag(img_src)}{img_tag(ov_src)}</div>
  <div class="prompt">{prompt}</div>
  {_bar(reward)}
  <div class="scores">
    <span class="{rc}">reward={reward:.3f}</span><br>
    {comp_html}<br>
    <span style="color:#888">{det_html}</span>
  </div>
</div>"""


def _section(title, rows, images_root, overlays_root, html_path):
    cards = "".join(_card(r, images_root, overlays_root, html_path) for r in rows)
    return f'<h2>{title} ({len(rows)} samples)</h2><div class="grid">{cards}</div>\n'


# ── main ─────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-results", required=True)
    parser.add_argument("--images-root",      required=True)
    parser.add_argument("--overlays-root",    required=True)
    parser.add_argument("--out",              required=True)
    parser.add_argument("--n",                type=int, default=30, help="samples per section")
    parser.add_argument("--seed",             type=int, default=0)
    args = parser.parse_args()

    rows = load_jsonl(args.verifier_results)
    html_path = Path(args.out)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    sorted_by_reward   = sorted(rows, key=lambda r: r.get("reward", 0), reverse=True)
    sorted_by_relation = sorted(rows, key=lambda r: r.get("components", {}).get("relation", 0), reverse=True)

    top_reward    = sorted_by_reward[:args.n]
    bottom_reward = sorted_by_reward[-args.n:]
    rand_sample   = rng.sample(rows, min(args.n, len(rows)))
    top_relation  = sorted_by_relation[:args.n]

    # high reward but relation < 0.5 — verifier-suspicious cases
    high_wrong_rel = [r for r in sorted_by_reward
                      if r.get("components", {}).get("relation", 1) < 0.5][:args.n]

    # low reward despite both colors detected
    low_color_ok = [r for r in sorted(rows, key=lambda r: r.get("reward", 0))
                    if r.get("components", {}).get("obj1_color", 0) >= 0.8
                    and r.get("components", {}).get("obj2_color", 0) >= 0.5
                    and r.get("reward", 1) < 0.45][:args.n]

    mean_r = sum(r.get("reward", 0) for r in rows) / len(rows) if rows else 0
    comp_means = {}
    for k in (rows[0].get("components", {}) if rows else {}):
        vals = [r.get("components", {}).get(k, 0) for r in rows]
        comp_means[k] = sum(vals) / len(vals)

    summary_html = "<br>".join(f"<b>{k}</b>: {v:.3f}" for k, v in comp_means.items())

    body = f"""
<h1>Verifier Audit — {Path(args.verifier_results).parent.name}</h1>
<p>n={len(rows)}  mean_reward={mean_r:.4f}</p>
<p>{summary_html}</p>
"""
    body += _section(f"Top {args.n} by reward",    top_reward,      args.images_root, args.overlays_root, html_path)
    body += _section(f"Random {args.n}",           rand_sample,     args.images_root, args.overlays_root, html_path)
    body += _section(f"Bottom {args.n} by reward", bottom_reward,   args.images_root, args.overlays_root, html_path)
    body += _section(f"Top {args.n} by relation",  top_relation,    args.images_root, args.overlays_root, html_path)
    body += _section(f"Suspicious: high reward but relation < 0.5 ({len(high_wrong_rel)})",
                     high_wrong_rel, args.images_root, args.overlays_root, html_path)
    body += _section(f"Low reward despite colors detected ({len(low_color_ok)})",
                     low_color_ok,   args.images_root, args.overlays_root, html_path)

    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path.write_text(html, encoding="utf-8")
    print(f"[audit] HTML → {html_path}")

    # CSV
    csv_path = html_path.with_name("verifier_audit_candidates.csv")
    fieldnames = ["id", "seed", "image_path", "reward", "relation",
                  "obj1_color", "obj2_color", "obj1_shape", "obj2_shape", "quality"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            comps = r.get("components", {})
            writer.writerow({
                "id":         r.get("id", ""),
                "seed":       r.get("seed", 0),
                "image_path": r.get("image_path", ""),
                "reward":     round(r.get("reward", 0), 4),
                "relation":   round(comps.get("relation", 0), 4),
                "obj1_color": round(comps.get("obj1_color", 0), 4),
                "obj2_color": round(comps.get("obj2_color", 0), 4),
                "obj1_shape": round(comps.get("obj1_shape", 0), 4),
                "obj2_shape": round(comps.get("obj2_shape", 0), 4),
                "quality":    round(comps.get("quality", 0), 4),
            })
    print(f"[audit] CSV  → {csv_path}")


if __name__ == "__main__":
    main()
