"""
filter_sft_dataset.py — apply reward thresholds to all_samples.jsonl and
produce a filtered reward_sft_dataset.jsonl for train_reward_sft.py.

Run this whenever you want to change the filtering criteria without
regenerating any images.

Usage:
  python SFT/filter_sft_dataset.py \\
    --input   outputs/reward_sft_data/all_samples.jsonl \\
    --output  outputs/reward_sft_data/reward_sft_dataset.jsonl \\
    --min-reward   0.60 \\
    --max-neg      0.45 \\
    --reweight                # recompute weight from (reward - min_reward) / (1 - min_reward)

Optional per-category thresholds (override --min-reward):
  --min-reward-attribute_binding 0.55
  --min-reward-spatial_relation  0.60
  --min-reward-counting          0.65
  --min-reward-interaction_relation 0.55
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, help="all_samples.jsonl")
    p.add_argument("--output",   required=True, help="filtered output .jsonl")
    p.add_argument("--min-reward",   type=float, default=0.60)
    p.add_argument("--max-neg",      type=float, default=0.45,
                   help="Exclude if neg_caption_score > this")
    p.add_argument("--reweight",     action="store_true",
                   help="Recompute weight = (reward - min_reward) / (1 - min_reward)")
    # Per-category overrides
    p.add_argument("--min-reward-attribute_binding",    type=float, default=None)
    p.add_argument("--min-reward-spatial_relation",     type=float, default=None)
    p.add_argument("--min-reward-counting",             type=float, default=None)
    p.add_argument("--min-reward-interaction_relation", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    cat_thresholds = {
        "attribute_binding":    args.min_reward_attribute_binding    or args.min_reward,
        "spatial_relation":     args.min_reward_spatial_relation     or args.min_reward,
        "counting":             args.min_reward_counting             or args.min_reward,
        "interaction_relation": args.min_reward_interaction_relation or args.min_reward,
    }

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_kept = 0
    cat_counts: dict = {}
    cat_kept:   dict = {}

    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_total += 1
            cat = rec.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

            reward    = float(rec.get("reward", 0.0))
            neg_score = float(rec.get("scores", {}).get("neg_caption_score", 0.0))
            thresh    = cat_thresholds.get(cat, args.min_reward)

            if reward < thresh:
                continue
            if neg_score > args.max_neg:
                continue

            if args.reweight:
                span = max(1e-6, 1.0 - thresh)
                w = float(min(1.0, max(0.1, (reward - thresh) / span)))
                rec["weight"] = round(w, 4)

            f_out.write(json.dumps(rec) + "\n")
            n_kept += 1
            cat_kept[cat] = cat_kept.get(cat, 0) + 1

    print(f"\n[filter] {n_total} total  →  {n_kept} kept  ({100*n_kept/max(1,n_total):.1f}%)")
    print(f"  min_reward={args.min_reward}  max_neg={args.max_neg}")
    for cat in sorted(cat_counts):
        print(f"  {cat:30s}  {cat_kept.get(cat,0):4d} / {cat_counts[cat]:4d}")
    print(f"\n  → {out_path}")


if __name__ == "__main__":
    main()
