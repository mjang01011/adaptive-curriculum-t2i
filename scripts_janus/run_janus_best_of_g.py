"""
Janus-Pro-1B Best-of-G diagnostic.
Checks whether the reward function ranks candidates correctly and whether Janus has
good candidate support, before committing to a full GRPO run.

Usage:
  python scripts_janus/run_janus_best_of_g.py \
    --data-root /viscam/u/jj277/adaptive-curriculum-t2i/data \
    --bucket attribute_binding \
    --num-prompts 20 \
    --num-generations 4 \
    --seeds 0 1 2 3 \
    --model-path deepseek-ai/Janus-Pro-1B \
    --cfg-weight 5.0 \
    --temperature 1.0 \
    --reward-mode pseudo_soft_grpo_target_heavy \
    --hard-reward-mode hard_target \
    --output-dir /viscam/u/jj277/janus_project/outputs_janus_best_of_g/janus_attribute_g4_$(date +%Y%m%d_%H%M%S)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs) ** 0.5)
    dy = (sum((y - my) ** 2 for y in ys) ** 0.5)
    return num / (dx * dy + 1e-12)


def _spearman(xs, ys):
    def rank(v):
        sorted_v = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank_i, idx in enumerate(sorted_v):
            r[idx] = rank_i + 1
        return r
    rx = rank(xs)
    ry = rank(ys)
    return _pearson(rx, ry)


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",         required=True)
    parser.add_argument("--bucket",            default="attribute_binding")
    parser.add_argument("--split",             default="val")
    parser.add_argument("--num-prompts",       type=int, default=20)
    parser.add_argument("--num-generations",   type=int, default=4)
    parser.add_argument("--seeds",             type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--model-path",        default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--cfg-weight",        type=float, default=5.0)
    parser.add_argument("--temperature",       type=float, default=1.0)
    parser.add_argument("--reward-mode",       default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--hard-reward-mode",  default="hard_target")
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--reward-model-path", default=None)
    args = parser.parse_args()

    G = args.num_generations
    assert G <= len(args.seeds), f"Need at least {G} seeds, got {len(args.seeds)}"

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise RuntimeError(f"Output dir already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    img_dir = out_dir / "images"
    img_dir.mkdir()

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from adaptive_curriculum.data.schemas import BucketItem
    from adaptive_curriculum.reward.vlm_reward import VLMRewardModel
    from scripts_janus.janus_wrapper import JanusProWrapper

    # ── load data ──────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    split_file = data_root / args.bucket / f"{args.bucket}_{args.split}_20.jsonl"
    if not split_file.exists():
        # fallback: look for any val file
        candidates = list((data_root / args.bucket).glob(f"*{args.split}*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No val file found under {data_root / args.bucket}")
        split_file = candidates[0]
    raw = load_jsonl(split_file)
    items = [BucketItem.from_dict(d) for d in raw[:args.num_prompts]]
    print(f"[bog] Loaded {len(items)} prompts from {split_file}")

    # ── load models ────────────────────────────────────────────────────
    reward_model = VLMRewardModel(model_path=args.reward_model_path)
    wrapper = JanusProWrapper(
        model_path=args.model_path,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    _ = wrapper.model
    print("[bog] Models loaded. Starting generation...")

    prompts = [item.text for item in items]
    N = len(items)

    # all_soft[i][g], all_hard[i][g], all_pil[i][g]
    all_soft = [[None] * G for _ in range(N)]
    all_hard = [[None] * G for _ in range(N)]
    all_pil  = [[None] * G for _ in range(N)]

    for g, seed in enumerate(args.seeds[:G]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        out = wrapper.generate_images(prompts, seeds=None)
        images = out["images"]

        for i, (item, pil_img) in enumerate(zip(items, images)):
            soft_r = reward_model.score_image(pil_img, item, mode=args.reward_mode)
            hard_r = reward_model.score_image(pil_img, item, mode=args.hard_reward_mode)
            all_soft[i][g] = float(soft_r["score"])
            all_hard[i][g] = float(hard_r["score"])
            all_pil[i][g]  = pil_img
            # save image
            fname = img_dir / f"prompt{i:03d}_seed{seed}.png"
            pil_img.save(fname)

        print(f"  [gen {g+1}/{G}] done (seed={seed})")

    # ── compute Best-of-G metrics ──────────────────────────────────────
    mean_hard_random            = []
    mean_hard_best_by_soft      = []
    mean_hard_best_by_hard      = []   # oracle
    pearson_vals                = []
    spearman_vals               = []

    samples_jsonl = []

    for i, item in enumerate(items):
        softs = all_soft[i]
        hards = all_hard[i]

        # random: pick sample 0 (seed-deterministic)
        hard_random = hards[0]
        # best by soft reward (grpo signal)
        best_soft_idx = softs.index(max(softs))
        hard_best_soft = hards[best_soft_idx]
        # oracle: best by hard reward
        hard_oracle = max(hards)

        mean_hard_random.append(hard_random)
        mean_hard_best_by_soft.append(hard_best_soft)
        mean_hard_best_by_hard.append(hard_oracle)

        pr = _pearson(softs, hards)
        sp = _spearman(softs, hards)
        pearson_vals.append(pr if not math.isnan(pr) else 0.0)
        spearman_vals.append(sp if not math.isnan(sp) else 0.0)

        samples_jsonl.append({
            "prompt_id": item.id,
            "prompt": item.text,
            "soft_scores": softs,
            "hard_scores": hards,
            "hard_random": hard_random,
            "hard_best_by_soft": hard_best_soft,
            "hard_oracle": hard_oracle,
            "pearson": pr,
            "spearman": sp,
        })

    mean_r    = sum(mean_hard_random) / N
    mean_soft = sum(mean_hard_best_by_soft) / N
    mean_ora  = sum(mean_hard_best_by_hard) / N
    pearson   = sum(pearson_vals) / len(pearson_vals)
    spearman  = sum(spearman_vals) / len(spearman_vals)

    reward_ranking_usable = (mean_soft - mean_r) >= 0.05
    oracle_headroom       = (mean_ora  - mean_r) >= 0.05
    success               = reward_ranking_usable and oracle_headroom

    summary = {
        "bucket": args.bucket,
        "model": args.model_path,
        "num_prompts": N,
        "num_generations": G,
        "reward_mode": args.reward_mode,
        "hard_reward_mode": args.hard_reward_mode,
        "mean_hard_random": round(mean_r, 4),
        "mean_hard_best_by_grpo_total": round(mean_soft, 4),
        "mean_hard_oracle": round(mean_ora, 4),
        "pearson_grpo_total_vs_hard": round(pearson, 4),
        "spearman_grpo_total_vs_hard": round(spearman, 4),
        "reward_ranking_usable": reward_ranking_usable,
        "oracle_headroom": oracle_headroom,
        "success": success,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "samples.jsonl", "w") as f:
        for rec in samples_jsonl:
            f.write(json.dumps(rec) + "\n")

    print("\n[bog] ── Results ──────────────────────────────────────")
    print(f"  random             : {mean_r:.4f}")
    print(f"  best_by_grpo_total : {mean_soft:.4f}  (delta={mean_soft-mean_r:+.4f})")
    print(f"  oracle             : {mean_ora:.4f}  (delta={mean_ora-mean_r:+.4f})")
    print(f"  pearson            : {pearson:.4f}")
    print(f"  spearman           : {spearman:.4f}")
    print(f"  reward_ranking_usable: {reward_ranking_usable}")
    print(f"  oracle_headroom    : {oracle_headroom}")
    print(f"  success            : {success}")
    if not reward_ranking_usable:
        print("  [warn] Reward ranking delta < 0.05 — reward signal may be too noisy for GRPO")
    if not oracle_headroom:
        print("  [warn] Oracle headroom < 0.05 — model candidates may be too uniform")
    print(f"\n[bog] Saved → {out_dir}")


if __name__ == "__main__":
    main()
