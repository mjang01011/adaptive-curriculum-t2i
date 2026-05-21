"""
Select top-k candidates from rejection-SFT samples.jsonl for SFT training.

Selection rule (per prompt):
  1. Filter candidates where presence_component_score >= min_object_presence.
  2. Sort filtered candidates by grpo_total_score descending.
  3. Take top-k.
  4. If no candidates pass the filter, fall back to the best overall (fallback=true).

Usage:
  python scripts/select_rejection_sft_data.py \
    --input  outputs/rejection_sft_attribute_g6/samples.jsonl \
    --selection best_by_grpo_total \
    --top-k 1 \
    --min-object-presence 0.50 \
    --out outputs/rejection_sft_attribute_g6/selected_top1.jsonl
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",               required=True)
    parser.add_argument("--selection",           default="best_by_grpo_total",
                        choices=["best_by_grpo_total", "best_by_hard_target"])
    parser.add_argument("--top-k",               type=int, default=1)
    parser.add_argument("--min-object-presence", type=float, default=0.50)
    parser.add_argument("--out",                 required=True)
    args = parser.parse_args()

    score_key = "grpo_total_score" if args.selection == "best_by_grpo_total" else "hard_target_score"

    # --- load -----------------------------------------------------------
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[select] Loaded {len(rows)} candidate rows from {args.input}")

    # --- group by prompt_id --------------------------------------------
    groups: dict = defaultdict(list)
    for row in rows:
        groups[row["prompt_id"]].append(row)

    num_prompts  = len(groups)
    selected     = []
    fallback_count = 0

    for prompt_id, candidates in groups.items():
        passing = [
            c for c in candidates
            if _nan_safe_ge(c.get("presence_component_score", float("nan")), args.min_object_presence)
        ]

        if passing:
            pool     = sorted(passing, key=lambda x: x.get(score_key, 0.0), reverse=True)
            fallback = False
        else:
            pool     = sorted(candidates, key=lambda x: x.get(score_key, 0.0), reverse=True)
            fallback = True
            fallback_count += 1

        for rank, cand in enumerate(pool[:args.top_k], start=1):
            out_row = dict(cand)
            out_row["selected_rank"]    = rank
            out_row["selection_rule"]   = args.selection
            out_row["fallback_selected"] = fallback
            selected.append(out_row)

    # --- summary stats --------------------------------------------------
    sel_grpo  = [r.get("grpo_total_score", 0.0) for r in selected]
    sel_hard  = [r.get("hard_target_score", 0.0) for r in selected]

    # "random" baseline = mean over all candidate rows (one random sample per prompt = first seed)
    first_per_prompt = [min(g, key=lambda x: x.get("sample_index", 0)) for g in groups.values()]
    rand_hard  = _mean([r.get("hard_target_score", 0.0) for r in rows])
    first_hard = _mean([r.get("hard_target_score", 0.0) for r in first_per_prompt])

    summary = {
        "input":                       args.input,
        "selection_rule":              args.selection,
        "top_k":                       args.top_k,
        "min_object_presence":         args.min_object_presence,
        "num_prompts":                 num_prompts,
        "num_selected":                len(selected),
        "fallback_count":              fallback_count,
        "mean_selected_grpo_total":    _mean(sel_grpo),
        "mean_selected_hard_target":   _mean(sel_hard),
        "mean_random_hard_target":     rand_hard,
        "mean_first_hard_target":      first_hard,
        "delta_selected_vs_random":    _mean(sel_hard) - rand_hard,
        "delta_selected_vs_first":     _mean(sel_hard) - first_hard,
    }

    # --- write outputs ---------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    summary_path = out_path.parent / "selection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[select] Prompts: {num_prompts}  selected: {len(selected)}  fallbacks: {fallback_count}")
    print(f"[select] mean grpo (selected): {summary['mean_selected_grpo_total']:.4f}")
    print(f"[select] mean hard (selected): {summary['mean_selected_hard_target']:.4f}  "
          f"random: {rand_hard:.4f}  first: {first_hard:.4f}")
    print(f"[select] delta vs random: {summary['delta_selected_vs_random']:+.4f}")
    print(f"[select] Saved → {out_path}")
    print(f"[select] Summary → {summary_path}")


def _nan_safe_ge(val, threshold):
    """Returns True if val >= threshold, False if val is nan or below threshold."""
    if isinstance(val, float) and math.isnan(val):
        return False
    return val >= threshold


if __name__ == "__main__":
    main()
