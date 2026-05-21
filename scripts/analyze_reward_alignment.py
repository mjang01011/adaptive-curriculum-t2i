"""
Analyze alignment between pseudo_soft training reward and hard_target validation reward.

Reads reward_details.jsonl written by the training loop and computes:
  - Pearson / Spearman correlation between soft_reward and hard_reward
  - Per-bucket breakdown
  - Confusion table (soft bins vs hard correctness)
  - Uncertain answer rate

Usage:
  python scripts/analyze_reward_alignment.py \
    --reward-log outputs/<run>/reward_details.jsonl \
    --out        outputs/<run>/reward_alignment.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _rank(values):
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j < len(sorted_idx) - 1 and values[sorted_idx[j + 1]] == values[sorted_idx[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-log", required=True)
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()

    log_path = Path(args.reward_log)
    if not log_path.exists():
        print(f"ERROR: {log_path} not found.")
        print("Make sure the training run has reward detail logging enabled.")
        sys.exit(1)

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in reward log.")
        return

    print(f"[align] Loaded {len(records)} reward records")

    # check fields
    sample = records[0]
    has_hard = "hard_reward" in sample
    has_soft = "soft_reward" in sample
    if not has_soft:
        print("ERROR: reward_details.jsonl missing 'soft_reward' field.")
        return
    if not has_hard:
        print("WARNING: 'hard_reward' not in records. "
              "Run with reward logging that captures both modes.")

    soft_vals = [r["soft_reward"] for r in records]
    uncertain_rate = sum(1 for r in records if r.get("has_uncertain", False)) / len(records)

    result = {
        "n_records": len(records),
        "uncertain_rate": uncertain_rate,
        "soft_reward_mean": sum(soft_vals) / len(soft_vals),
    }

    if has_hard:
        hard_vals = [r["hard_reward"] for r in records]
        pearson = _pearson(soft_vals, hard_vals)
        spearman = _spearman(soft_vals, hard_vals)
        result["pearson_soft_hard"] = pearson
        result["spearman_soft_hard"] = spearman
        result["hard_reward_mean"] = sum(hard_vals) / len(hard_vals)

        # per-bucket breakdown
        by_bucket = defaultdict(lambda: {"soft": [], "hard": []})
        for r in records:
            b = r.get("bucket", "unknown")
            by_bucket[b]["soft"].append(r["soft_reward"])
            by_bucket[b]["hard"].append(r["hard_reward"])

        bucket_stats = {}
        for b, vals in by_bucket.items():
            bucket_stats[b] = {
                "n": len(vals["soft"]),
                "pearson": _pearson(vals["soft"], vals["hard"]),
                "spearman": _spearman(vals["soft"], vals["hard"]),
                "soft_mean": sum(vals["soft"]) / len(vals["soft"]),
                "hard_mean": sum(vals["hard"]) / len(vals["hard"]),
            }
        result["per_bucket"] = bucket_stats

        # soft bin vs hard correct confusion
        bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
        bin_labels = ["0–0.25", "0.25–0.5", "0.5–0.75", "0.75–1.0"]
        confusion = {}
        for label, (lo, hi) in zip(bin_labels, bins):
            subset = [r["hard_reward"] for r in records if lo <= r["soft_reward"] < hi]
            if subset:
                confusion[label] = {
                    "n": len(subset),
                    "hard_mean": sum(subset) / len(subset),
                }
        result["soft_bin_vs_hard_mean"] = confusion

        # summary print
        print(f"\n{'='*50}")
        print(f"Reward alignment: soft (pseudo_soft_target) vs hard (hard_target)")
        print(f"{'='*50}")
        print(f"  n records      : {len(records)}")
        print(f"  soft mean      : {result['soft_reward_mean']:.4f}")
        print(f"  hard mean      : {result['hard_reward_mean']:.4f}")
        print(f"  pearson        : {pearson:.4f}  {'✓ good' if pearson > 0.5 else '⚠ low' if pearson > 0.2 else '✗ bad'}")
        print(f"  spearman       : {spearman:.4f}")
        print(f"  uncertain rate : {uncertain_rate:.2%}")
        print()
        print("  Per-bucket:")
        for b, s in bucket_stats.items():
            print(f"    {b:30s}  pearson={s['pearson']:+.3f}  soft={s['soft_mean']:.3f}  hard={s['hard_mean']:.3f}")
        print()
        print("  Soft bins → hard mean:")
        for label, v in confusion.items():
            print(f"    soft={label:10s}  hard_mean={v['hard_mean']:.3f}  n={v['n']}")

        interpretation = (
            "GOOD — soft reward aligns with hard" if pearson > 0.5 else
            "ACCEPTABLE — some alignment" if pearson > 0.2 else
            "BAD — soft reward misaligned with hard; consider using true_soft or hard_target for training"
        )
        print(f"\n  Interpretation: {interpretation}")
        result["interpretation"] = interpretation

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")


import sys
if __name__ == "__main__":
    main()
