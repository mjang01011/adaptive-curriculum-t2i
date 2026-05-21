"""
Generate a markdown comparison table from multiple eval_checkpoints_bucket.py outputs.

Usage:
  python scripts/make_final_comparison_table.py \
    --entries \
      "Base LlamaGen:none:outputs/<grpo_run>/fixed_eval_attribute.json:base" \
      "Direct GRPO stable:GRPO:outputs/<grpo_run>/fixed_eval_attribute.json:best" \
      "Best-of-G rerank:inference-time rerank:outputs/<best_of_g_run>/best_of_g_summary.json:best_grpo" \
      "Rejection-SFT:SFT:outputs/<sft_run>/fixed_eval_attribute.json:best" \
      "SFT + GRPO:SFT then GRPO:outputs/<sft_grpo_run>/fixed_eval_attribute.json:best" \
    --bucket attribute_binding \
    --out outputs/final_attribute_comparison.md

Each entry is: "display_name:training_type:eval_json_path:checkpoint_key"
The checkpoint_key selects which entry in the eval json's "results" dict to use.
For best-of-G summary json the format is different (see below).
"""
import argparse
import json
import math
from pathlib import Path


def _load_eval_result(json_path: str, checkpoint_key: str):
    """
    Load mean/stderr from an eval_checkpoints_bucket output JSON.
    Returns (mean, stderr, per_qtype) or raises on missing key.
    """
    with open(json_path) as f:
        data = json.load(f)

    # eval_checkpoints_bucket output
    if "results" in data:
        results = data["results"]
        if checkpoint_key not in results:
            available = list(results.keys())
            raise KeyError(
                f"Checkpoint '{checkpoint_key}' not found in {json_path}. "
                f"Available: {available}"
            )
        r = results[checkpoint_key]
        mean   = r.get("mean", float("nan"))
        stderr = r.get("stderr", float("nan"))
        per_q  = {}
        return mean, stderr, per_q

    # best_of_g_summary.json format (best_grpo / random keys)
    if checkpoint_key == "best_grpo" and "best_grpo" in data:
        mean   = data["best_grpo"].get("mean_hard_target", float("nan"))
        stderr = data["best_grpo"].get("se_hard_target", 0.0)
        return mean, stderr, {}
    if checkpoint_key == "random" and "random_sample" in data:
        mean   = data["random_sample"].get("mean_hard_target", float("nan"))
        stderr = data["random_sample"].get("se_hard_target", 0.0)
        return mean, stderr, {}

    # probe_eval_history.jsonl (find best epoch)
    if json_path.endswith(".jsonl"):
        best_mean = -1.0
        best_se   = 0.0
        with open(json_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                m = rec.get("mean_reward", -1.0)
                if m > best_mean:
                    best_mean = m
                    best_se   = rec.get("se_reward", 0.0)
        return best_mean, best_se, {}

    raise ValueError(f"Cannot parse eval json at {json_path} with key '{checkpoint_key}'")


def _fmt(val, digits=4):
    if math.isnan(val):
        return "—"
    return f"{val:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entries", nargs="+", required=True,
        help="Each: 'display_name:training_type:eval_json:checkpoint_key'"
    )
    parser.add_argument("--bucket", default="attribute_binding")
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()

    rows = []
    base_mean = None

    for entry in args.entries:
        parts = entry.split(":", 3)
        if len(parts) != 4:
            print(f"WARNING: skipping malformed entry: {entry}")
            continue
        display_name, training_type, json_path, ckpt_key = parts

        if not Path(json_path).exists():
            print(f"WARNING: file not found: {json_path} — skipping {display_name}")
            rows.append({
                "name": display_name, "type": training_type,
                "mean": float("nan"), "stderr": float("nan"),
                "delta": float("nan"), "notes": "file not found",
            })
            continue

        try:
            mean, stderr, _ = _load_eval_result(json_path, ckpt_key)
        except Exception as e:
            print(f"WARNING: failed to load {json_path}/{ckpt_key}: {e}")
            rows.append({
                "name": display_name, "type": training_type,
                "mean": float("nan"), "stderr": float("nan"),
                "delta": float("nan"), "notes": str(e)[:60],
            })
            continue

        if base_mean is None:
            base_mean = mean  # first entry is the base

        delta = mean - base_mean if not math.isnan(mean) else float("nan")
        rows.append({
            "name":   display_name,
            "type":   training_type,
            "mean":   mean,
            "stderr": stderr,
            "delta":  delta,
            "notes":  "",
        })
        print(f"  {display_name:30s}  mean={_fmt(mean)}  se={_fmt(stderr)}  "
              f"delta={_fmt(delta, 4) if not math.isnan(delta) else '—'}")

    # --- markdown table -------------------------------------------------
    header = (
        "| Method | Training type | Val hard target ↑ | SE | Δ vs base |\n"
        "| --- | --- | ---: | ---: | ---: |"
    )
    lines = [
        f"# Attribute binding — final comparison  (bucket: {args.bucket})\n",
        header,
    ]
    for r in rows:
        mean_str  = _fmt(r["mean"])
        se_str    = _fmt(r["stderr"])
        delta_str = (f"{r['delta']:+.4f}" if not math.isnan(r["delta"]) else "—")
        note      = r.get("notes", "")
        lines.append(
            f"| {r['name']} | {r['type']} | {mean_str} | {se_str} | {delta_str} |"
            + (f"  <!-- {note} -->" if note else "")
        )

    md = "\n".join(lines) + "\n"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n[table] Saved → {out_path}")
    print(md)


if __name__ == "__main__":
    main()
