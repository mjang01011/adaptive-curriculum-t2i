"""
Best-of-G diagnostic using the classical CV verifier as the reward signal.

For each prompt, compare:
  random   — score of seed 0
  best_of_g — best score across all G seeds (verifier IS the eval, so this equals oracle)

Success criterion: mean_best_of_g - mean_random >= +0.10

Usage:
  python scripts_verifier/best_of_g_from_verifier.py \
    --verifier-results outputs_verifier/base_shapes_val_g6/verifier_results.jsonl \
    --out outputs_verifier/base_shapes_val_g6/bog_summary.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy  = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-results", required=True)
    parser.add_argument("--out",              required=True)
    parser.add_argument("--success-threshold", type=float, default=0.10)
    args = parser.parse_args()

    rows = load_jsonl(args.verifier_results)

    # group by prompt id
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)

    # sort each group by seed to ensure reproducibility
    for pid in by_id:
        by_id[pid].sort(key=lambda r: r.get("seed", 0))

    mean_random_list  = []
    mean_best_list    = []
    component_random  = defaultdict(list)
    component_best    = defaultdict(list)
    per_prompt        = []

    for pid, group in by_id.items():
        rewards = [r["reward"] for r in group]
        random_reward = rewards[0]
        best_reward   = max(rewards)
        best_idx      = rewards.index(best_reward)

        mean_random_list.append(random_reward)
        mean_best_list.append(best_reward)

        random_comps = group[0].get("components", {})
        best_comps   = group[best_idx].get("components", {})
        for k, v in random_comps.items():
            component_random[k].append(v)
        for k, v in best_comps.items():
            component_best[k].append(v)

        per_prompt.append({
            "id":            pid,
            "prompt":        group[0].get("prompt", ""),
            "relation":      group[0].get("relation", ""),
            "rewards":       rewards,
            "random_reward": random_reward,
            "best_reward":   best_reward,
            "best_seed":     group[best_idx].get("seed", best_idx),
            "delta":         round(best_reward - random_reward, 4),
        })

    N = len(per_prompt)
    mean_random  = sum(mean_random_list)  / N
    mean_best    = sum(mean_best_list)    / N
    delta        = mean_best - mean_random

    comp_means_random = {k: round(sum(v) / len(v), 4) for k, v in component_random.items()}
    comp_means_best   = {k: round(sum(v) / len(v), 4) for k, v in component_best.items()}

    success = delta >= args.success_threshold

    summary = {
        "n_prompts":              N,
        "mean_random":            round(mean_random, 4),
        "mean_best_of_g":         round(mean_best, 4),
        "delta_best_vs_random":   round(delta, 4),
        "success_threshold":      args.success_threshold,
        "success":                success,
        "component_means_random": comp_means_random,
        "component_means_best":   comp_means_best,
        "per_prompt":             per_prompt,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[bog] ── Results ────────────────────────────────────────────")
    print(f"  prompts:         {N}")
    print(f"  mean_random:     {mean_random:.4f}")
    print(f"  mean_best_of_g:  {mean_best:.4f}  (delta={delta:+.4f})")
    print(f"  success (>={args.success_threshold}):  {success}")
    print(f"\n  component random: {comp_means_random}")
    print(f"  component best:   {comp_means_best}")

    if not success:
        print(f"\n  [warn] Best-of-G delta {delta:+.4f} < {args.success_threshold} threshold.")
        print("  Check debug overlays — verifier may have bad thresholds or model output is weak.")
    else:
        print(f"\n  [ok] Verifier reward signal is useful. Proceed to GRPO/GCPO-lite.")

    print(f"\n[bog] Saved → {out_path}")


if __name__ == "__main__":
    main()
