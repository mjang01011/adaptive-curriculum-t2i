"""
Plotting utilities for curriculum training results.
Generates reward curves, UCB selection plots, and final comparison charts.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_jsonl(path: str) -> list:
    records = []
    if not Path(path).exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _save_csv(path: str, rows: list, columns: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(c, "")) for c in columns) + "\n")


def plot_reward_curves(run_dir: str, bucket_names: List[str], show: bool = False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not available, skipping plots")
        return

    run_path = Path(run_dir)
    records = _load_jsonl(str(run_path / "bucket_eval_history.jsonl"))
    if not records:
        return

    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # overall reward over time
    by_step: Dict[int, List[float]] = {}
    by_bucket: Dict[str, List] = {b: [] for b in bucket_names}

    for r in records:
        step = r["curriculum_step"]
        bucket = r.get("bucket", "")
        reward = r.get("mean_raw_reward", 0.0)
        by_step.setdefault(step, []).append(reward)
        if bucket in by_bucket:
            by_bucket[bucket].append((step, reward))

    # overall
    steps_sorted = sorted(by_step)
    avg_rewards = [sum(by_step[s]) / len(by_step[s]) for s in steps_sorted]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps_sorted, avg_rewards, label="avg reward")
    ax.set_xlabel("Curriculum Step")
    ax.set_ylabel("Mean Validation Reward")
    ax.set_title("Overall Validation Reward vs Curriculum Step")
    ax.legend()
    fig.savefig(str(plots_dir / "overall_reward.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _save_csv(
        str(plots_dir / "overall_reward.csv"),
        [{"step": s, "avg_reward": r} for s, r in zip(steps_sorted, avg_rewards)],
        ["step", "avg_reward"],
    )

    # per-bucket
    fig, ax = plt.subplots(figsize=(12, 5))
    for bucket, pairs in by_bucket.items():
        if not pairs:
            continue
        bsteps, brewards = zip(*sorted(pairs))
        ax.plot(bsteps, brewards, label=bucket)
    ax.set_xlabel("Curriculum Step")
    ax.set_ylabel("Mean Validation Reward")
    ax.set_title("Per-Bucket Validation Reward vs Curriculum Step")
    ax.legend()
    fig.savefig(str(plots_dir / "per_bucket_reward.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if show:
        plt.show()


def plot_ucb_selection_counts(run_dir: str, bucket_names: List[str]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    run_path = Path(run_dir)
    records = _load_jsonl(str(run_path / "curriculum_decisions.jsonl"))
    if not records:
        return

    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    counts: Dict[str, int] = {b: 0 for b in bucket_names}
    selection_history: List[str] = []
    for r in records:
        b = r.get("chosen_bucket", "")
        if b in counts:
            counts[b] += 1
        selection_history.append(b)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Times Selected")
    ax.set_title("UCB Bucket Selection Counts")
    fig.savefig(str(plots_dir / "selection_counts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _save_csv(
        str(plots_dir / "selection_counts.csv"),
        [{"bucket": b, "count": c} for b, c in counts.items()],
        ["bucket", "count"],
    )


def plot_ucb_score_components(run_dir: str, bucket_names: List[str]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    run_path = Path(run_dir)
    records = _load_jsonl(str(run_path / "curriculum_decisions.jsonl"))
    if not records:
        return

    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    steps = []
    imp_ma: Dict[str, List[float]] = {b: [] for b in bucket_names}
    ucb_scores: Dict[str, List[float]] = {b: [] for b in bucket_names}

    for r in records:
        step = r["curriculum_step"]
        steps.append(step)
        stats = r.get("bucket_stats", {})
        scores = r.get("ucb_scores", {})
        for b in bucket_names:
            imp_ma[b].append(stats.get(b, {}).get("improvement_ma", 0.0))
            ucb_scores[b].append(scores.get(b, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for b in bucket_names:
        axes[0].plot(steps, imp_ma[b], label=b)
        axes[1].plot(steps, ucb_scores[b], label=b)
    axes[0].set_title("Improvement MA per Bucket")
    axes[1].set_title("UCB Score per Bucket")
    for ax in axes:
        ax.set_xlabel("Curriculum Step")
        ax.legend(fontsize=7)
    fig.savefig(str(plots_dir / "ucb_components.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(summary_paths: Dict[str, str], out_path: str, bucket_names: List[str]):
    """Compare final bucket rewards across strategies."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    strategies = list(summary_paths.keys())
    data: Dict[str, Dict[str, float]] = {}
    for strat, path in summary_paths.items():
        if Path(path).exists():
            with open(path) as f:
                summary = json.load(f)
            data[strat] = summary.get("final_bucket_rewards", {})

    if not data:
        return

    x = np.arange(len(bucket_names))
    width = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, strat in enumerate(strategies):
        vals = [data.get(strat, {}).get(b, 0.0) for b in bucket_names]
        ax.bar(x + i * width, vals, width, label=strat)
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(bucket_names, rotation=20, ha="right")
    ax.set_ylabel("Final Mean Reward")
    ax.set_title("Final Reward Comparison: Uniform vs Static vs UCB")
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(run_dir: str, bucket_names: List[str]):
    plot_reward_curves(run_dir, bucket_names)
    plot_ucb_selection_counts(run_dir, bucket_names)
    plot_ucb_score_components(run_dir, bucket_names)
    print(f"[plots] Saved to {run_dir}/plots/")
