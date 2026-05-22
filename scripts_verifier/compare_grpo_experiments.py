"""
Print a comparison table of the three GRPO verifier experiments.

Usage:
  python scripts_verifier/compare_grpo_experiments.py \
    --runs-root /viscam/u/jj277/adaptive-curriculum-t2i/outputs_verifier/grpo_runs \
    --names synthetic_shapes_vanilla_grpo synthetic_shapes_winner_grpo synthetic_shapes_winner_gcpo_lite
"""
import argparse
import json
from pathlib import Path


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_metrics(run_dir: Path) -> list:
    p = run_dir / "train_metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--names",     nargs="+",
                        default=["synthetic_shapes_vanilla_grpo",
                                 "synthetic_shapes_winner_grpo",
                                 "synthetic_shapes_winner_gcpo_lite"])
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = []
    for name in args.names:
        run_dir = runs_root / name
        summary = load_summary(run_dir)
        metrics = load_metrics(run_dir)

        final_r  = metrics[-1]["mean_reward"] if metrics else float("nan")
        final_pg = metrics[-1]["pg_loss"]     if metrics else float("nan")

        rows.append({
            "name":              name,
            "objective":         summary.get("objective", "?"),
            "steps_run":         summary.get("total_steps", len(metrics)),
            "baseline_reward":   summary.get("baseline_reward", float("nan")),
            "baseline_relation": summary.get("baseline_relation", float("nan")),
            "best_relation":     summary.get("best_relation", float("nan")),
            "final_mean_reward": summary.get("final_mean_reward", final_r),
            "final_pg_loss":     final_pg,
        })

    # print table
    header = f"{'Run':<42}  {'obj':<22}  {'steps':>5}  {'base_r':>6}  {'base_rel':>8}  {'best_rel':>8}  {'final_r':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        short = r["name"].replace("synthetic_shapes_", "")
        print(f"{short:<42}  {r['objective']:<22}  {r['steps_run']:>5}  "
              f"{r['baseline_reward']:>6.4f}  {r['baseline_relation']:>8.4f}  "
              f"{r['best_relation']:>8.4f}  {r['final_mean_reward']:>7.4f}")

    # lift over baseline relation
    print("\nRelation lift over baseline:")
    for r in rows:
        lift = r["best_relation"] - r["baseline_relation"]
        short = r["name"].replace("synthetic_shapes_", "")
        print(f"  {short:<42}  Δrel={lift:+.4f}")


if __name__ == "__main__":
    main()
