"""
Reward alignment dashboard.

Reads reward_details.jsonl written during training and computes:
  - Pearson/Spearman correlation: total_grpo_reward vs hard_target_score
  - Per-component correlations (target, presence, anti, quality, alignment)
  - Trend by training step
  - Failure cases: high grpo / low hard (reward hacking) and low grpo / high hard (missed)
  - Scatter plots and component trend plots

Thresholds:
  pearson > 0.5   → usable reward
  0.2–0.5         → noisy but possibly usable
  < 0.2           → misaligned — fix reward before continuing

Usage:
    python scripts/analyze_reward_alignment.py \
        --reward-log outputs/<run>/reward_details.jsonl \
        --out outputs/<run>/reward_alignment_summary.json
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Dict


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")

def _pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    if dx < 1e-9 or dy < 1e-9: return 0.0
    return num / (dx * dy)

def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0]*len(xs)
    for r, idx in enumerate(order): ranks[idx] = r
    return ranks

def _spearman(xs, ys): return _pearson(_rank(xs), _rank(ys))


# ---------------------------------------------------------------------------
# Component grouping
# ---------------------------------------------------------------------------

TARGET_QTYPES   = {"relation", "attribute", "count"}
ANTI_QTYPES     = {"anti_relation", "anti_swap", "anti_count"}
PRESENCE_QTYPES = {"object_presence"}
QUALITY_QTYPES  = {"image_quality"}
ALIGN_QTYPES    = {"prompt_alignment"}


def _extract_components(row: dict) -> dict:
    cs = row.get("component_scores", {})
    if cs:
        def _get(*keys):
            for k in keys:
                if k in cs: return cs[k]
            return float("nan")
        return {
            "target":    _get("relation", "attribute", "count"),
            "anti":      _get("anti_relation", "anti_swap", "anti_count"),
            "presence":  _get("object_presence"),
            "quality":   _get("image_quality"),
            "alignment": _get("prompt_alignment"),
        }
    # fallback: re-aggregate from question_scores
    qs = row.get("question_scores", [])
    buckets: Dict[str, List[float]] = defaultdict(list)
    for q in qs:
        qt = q.get("q_type", "")
        sc = q.get("score", float("nan"))
        if qt in TARGET_QTYPES:    buckets["target"].append(sc)
        elif qt in ANTI_QTYPES:    buckets["anti"].append(sc)
        elif qt in PRESENCE_QTYPES: buckets["presence"].append(sc)
        elif qt in QUALITY_QTYPES:  buckets["quality"].append(sc)
        elif qt in ALIGN_QTYPES:    buckets["alignment"].append(sc)
    return {k: _mean(v) for k, v in buckets.items()}


def _uncertain_rate(row: dict) -> float:
    qs = row.get("question_scores", [])
    if not qs: return float("nan")
    return sum(1 for q in qs if q.get("predicted","") == "uncertain") / len(qs)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _try_plots(rows: List[dict], out_dir: Path, by_step: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping plots")
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    grpo_scores = [r.get("soft_reward", float("nan")) for r in rows]
    hard_scores = [r.get("hard_reward", float("nan")) for r in rows]
    valid = [(g,h) for g,h in zip(grpo_scores,hard_scores) if not math.isnan(g) and not math.isnan(h)]
    if not valid: return
    gs, hs = zip(*valid)

    # 1. Scatter: grpo vs hard
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(gs, hs, alpha=0.3, s=10)
    ax.set_xlabel("GRPO total reward"); ax.set_ylabel("Hard target reward")
    ax.set_title("GRPO reward vs Hard target (all training images)")
    corr = _pearson(list(gs), list(hs))
    ax.text(0.05, 0.92, f"pearson r={corr:.3f}", transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(str(plots_dir/"reward_vs_hard_scatter.png"), dpi=120); plt.close(fig)

    # 2. Target component vs hard
    target_scores = [_extract_components(r).get("target", float("nan")) for r in rows]
    valid_t = [(t,h) for t,h in zip(target_scores,hard_scores) if not math.isnan(t) and not math.isnan(h)]
    if valid_t:
        ts, hs_t = zip(*valid_t)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(ts, hs_t, alpha=0.3, s=10, color="orange")
        ax.set_xlabel("Target component score"); ax.set_ylabel("Hard target reward")
        ax.set_title("Target component vs Hard target")
        corr_t = _pearson(list(ts), list(hs_t))
        ax.text(0.05, 0.92, f"pearson r={corr_t:.3f}", transform=ax.transAxes)
        fig.tight_layout(); fig.savefig(str(plots_dir/"target_component_vs_hard_scatter.png"), dpi=120); plt.close(fig)

    if not by_step: return

    steps = [s["step"] for s in by_step]

    # 3. Component trends by step
    fig, ax = plt.subplots(figsize=(9, 4))
    for comp in ("target", "presence", "anti", "quality", "alignment"):
        vals = [s.get(f"mean_{comp}_component", float("nan")) for s in by_step]
        if any(not math.isnan(v) for v in vals):
            ax.plot(steps, vals, label=comp, marker="o", markersize=3)
    ax.set_xlabel("Step"); ax.set_ylabel("Component score")
    ax.set_title("Reward components by step"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(str(plots_dir/"component_trends_by_step.png"), dpi=120); plt.close(fig)

    # 4. Hard target vs grpo by step
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, [s["mean_grpo_total"] for s in by_step], label="grpo_total", marker="o", markersize=3)
    ax.plot(steps, [s["mean_hard_target"] for s in by_step], label="hard_target", marker="s", markersize=3)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.set_title("GRPO vs Hard target by step"); ax.legend()
    fig.tight_layout(); fig.savefig(str(plots_dir/"hard_target_vs_grpo_by_step.png"), dpi=120); plt.close(fig)

    # 5. Uncertain rate by step
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(steps, [s.get("uncertain_rate", 0) for s in by_step], marker="o", markersize=3, color="red")
    ax.set_xlabel("Step"); ax.set_ylabel("Uncertain rate")
    ax.set_title("Uncertain rate by step")
    fig.tight_layout(); fig.savefig(str(plots_dir/"uncertain_rate_by_step.png"), dpi=120); plt.close(fig)

    print(f"[analyze] Plots saved to {plots_dir}")


def _save_failure_cases(rows: List[dict], out_dir: Path):
    fail_dir = out_dir / "failure_cases"
    fail_dir.mkdir(exist_ok=True)

    hacking = [r for r in rows if r.get("soft_reward",0) >= 0.8 and r.get("hard_reward",1) <= 0.25]
    missed  = [r for r in rows if r.get("soft_reward",1) <= 0.4 and r.get("hard_reward",0) >= 0.75]

    def _save(cases, fname):
        with open(fail_dir/fname, "w", encoding="utf-8") as f:
            for r in cases:
                f.write(json.dumps({
                    "prompt_id": r.get("prompt_id"), "prompt": r.get("prompt"),
                    "soft_reward": r.get("soft_reward"), "hard_reward": r.get("hard_reward"),
                    "step": r.get("global_step"), "image_path": r.get("image_path",""),
                    "component_scores": _extract_components(r),
                }) + "\n")
        print(f"[analyze] {len(cases)} {fname.replace('.jsonl','')} cases → {fail_dir/fname}")

    _save(hacking, "reward_hacking.jsonl")
    _save(missed,  "reward_missed_good.jsonl")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-log", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    log_path = Path(args.reward_log)
    if not log_path.exists():
        print(f"[ERROR] reward log not found: {log_path}"); return

    rows: List[dict] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: rows.append(json.loads(line))
    print(f"[analyze] Loaded {len(rows)} rows from {log_path}")
    if not rows: return

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global correlations
    grpo_scores = [r.get("soft_reward", float("nan")) for r in rows]
    hard_scores  = [r.get("hard_reward",  float("nan")) for r in rows]
    valid = [(g,h) for g,h in zip(grpo_scores,hard_scores) if not math.isnan(g) and not math.isnan(h)]
    gs, hs = (list(zip(*valid)) if valid else ([], []))
    pearson_total  = _pearson(list(gs), list(hs))
    spearman_total = _spearman(list(gs), list(hs))

    # Component correlations
    comp_data: Dict[str,List] = defaultdict(list)
    hard_for_comp: Dict[str,List] = defaultdict(list)
    for r, h in zip(rows, hard_scores):
        if math.isnan(h): continue
        for k,v in _extract_components(r).items():
            if not math.isnan(v):
                comp_data[k].append(v)
                hard_for_comp[k].append(h)
    component_corrs = {f"pearson_{k}_vs_hard": _pearson(comp_data[k], hard_for_comp[k]) for k in comp_data}

    # By-step breakdown
    by_step_raw: Dict[int,List] = defaultdict(list)
    for r in rows:
        by_step_raw[r.get("global_step", r.get("step", 0))].append(r)
    by_step = []
    for step in sorted(by_step_raw.keys()):
        sr = by_step_raw[step]
        sg = [v for v in [r.get("soft_reward",float("nan")) for r in sr] if not math.isnan(v)]
        sh = [v for v in [r.get("hard_reward", float("nan")) for r in sr] if not math.isnan(v)]
        sc: Dict[str,List] = defaultdict(list)
        uncertain = sum(int(r.get("has_uncertain",False)) for r in sr)
        for r in sr:
            for k,v in _extract_components(r).items():
                if not math.isnan(v): sc[k].append(v)
        entry = {
            "step": step,
            "mean_grpo_total": _mean(sg),
            "mean_hard_target": _mean(sh),
            "corr_total_vs_hard": _pearson(sg, sh),
            "uncertain_rate": uncertain / len(sr),
            **{f"mean_{k}_component": _mean(v) for k,v in sc.items()},
        }
        by_step.append(entry)

    # Interpretation
    if math.isnan(pearson_total):
        verdict = "insufficient data"
    elif pearson_total > 0.5:
        verdict = "USABLE — reward correlates with hard target"
    elif pearson_total >= 0.2:
        verdict = "NOISY — reward weakly correlates; monitor closely"
    else:
        verdict = "MISALIGNED — stop tuning; fix reward before continuing"

    target_corr = component_corrs.get("pearson_target_vs_hard", float("nan"))
    presence_corr = component_corrs.get("pearson_presence_vs_hard", float("nan"))
    quality_corr  = component_corrs.get("pearson_quality_vs_hard",  float("nan"))
    if not math.isnan(target_corr) and not math.isnan(pearson_total):
        if target_corr > pearson_total + 0.05:
            verdict += " | stabilizers diluting total — increase target weight"
        elif pearson_total > target_corr + 0.05:
            verdict += " | support questions help ranking — keep decomposed reward"
    if not math.isnan(presence_corr) and not math.isnan(target_corr):
        if presence_corr > 0.3 and target_corr < 0.2:
            verdict += " | model optimizing easy visual features, not composition"

    uncertain_rates = [_uncertain_rate(r) for r in rows if not math.isnan(_uncertain_rate(r))]
    summary = {
        "num_images": len(rows),
        "bucket": rows[0].get("bucket","unknown"),
        "pearson_total_vs_hard": round(pearson_total, 4),
        "spearman_total_vs_hard": round(spearman_total, 4),
        **{k: round(v,4) for k,v in component_corrs.items()},
        "mean_grpo_total":  round(_mean(list(gs)), 4),
        "mean_hard_target": round(_mean(list(hs)), 4),
        **{f"mean_{k}_component": round(_mean(comp_data[k]),4) for k in comp_data},
        "uncertain_rate_grpo": round(_mean(uncertain_rates), 4),
        "alignment_verdict": verdict,
        "by_step": by_step,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] Summary → {out_path}")

    _try_plots(rows, out_dir, by_step)
    _save_failure_cases(rows, out_dir)

    print("\n" + "="*60)
    print(f"  pearson(grpo, hard):  {pearson_total:.4f}   spearman={spearman_total:.4f}")
    for k,v in sorted(component_corrs.items()):
        print(f"  {k}: {v:.4f}")
    print(f"\n  VERDICT: {verdict}")
    print("="*60)


if __name__ == "__main__":
    main()
