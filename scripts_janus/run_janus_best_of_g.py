"""
Janus-Pro-1B Best-of-G diagnostic — two-phase: generate then score.

Phase 1 (januspro_venv): generate images
  python3 scripts_janus/run_janus_best_of_g.py \
    --phase generate \
    --data-root $PROJ/data --bucket attribute_binding \
    --num-prompts 20 --num-generations 4 --seeds 0 1 2 3 \
    --output-dir $JANUS/outputs_janus_best_of_g/janus_attribute_g4_<ts>

Phase 2 (svl env): score images
  python3 scripts_janus/run_janus_best_of_g.py \
    --phase score \
    --data-root $PROJ/data --bucket attribute_binding \
    --num-prompts 20 --num-generations 4 --seeds 0 1 2 3 \
    --output-dir $JANUS/outputs_janus_best_of_g/janus_attribute_g4_<ts>
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
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy + 1e-12)


def _spearman(xs, ys):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank_i, idx in enumerate(s):
            r[idx] = rank_i + 1
        return r
    return _pearson(rank(xs), rank(ys))


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def img_fname(prompt_idx, seed):
    return f"prompt{prompt_idx:03d}_seed{seed}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",             choices=["generate", "score", "both"], default="both")
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
    parser.add_argument("--reward-model-id",   default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    G = args.num_generations
    assert G <= len(args.seeds), f"Need at least {G} seeds, got {len(args.seeds)}"

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from adaptive_curriculum.data.schemas import BucketItem

    # ── load data ──────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    split_file = data_root / args.bucket / f"{args.bucket}_{args.split}_20.jsonl"
    if not split_file.exists():
        candidates = list((data_root / args.bucket).glob(f"*{args.split}*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No val file found under {data_root / args.bucket}")
        split_file = candidates[0]
    raw = load_jsonl(split_file)
    items = [BucketItem.from_dict(d) for d in raw[:args.num_prompts]]
    N = len(items)
    print(f"[bog] {N} prompts, G={G}, seeds={args.seeds[:G]}, phase={args.phase}")

    # ── phase: generate ────────────────────────────────────────────────
    if args.phase in ("generate", "both"):
        if out_dir.exists() and args.phase == "both":
            raise RuntimeError(f"Output dir already exists: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)

        from scripts_janus.janus_wrapper import JanusProWrapper
        wrapper = JanusProWrapper(
            model_path=args.model_path,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
        _ = wrapper.model
        print("[bog] Janus loaded. Generating...")

        prompts = [item.text for item in items]
        for g, seed in enumerate(args.seeds[:G]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            out = wrapper.generate_images(prompts, seeds=None)
            for i, pil_img in enumerate(out["images"]):
                pil_img.save(img_dir / img_fname(i, seed))
            print(f"  gen {g+1}/{G} (seed={seed}) done")
        print(f"[bog] Images saved → {img_dir}")

    # ── phase: score ───────────────────────────────────────────────────
    if args.phase in ("score", "both"):
        import PIL.Image
        from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
        reward_model = Qwen3VLRewardModel(model_id=args.reward_model_id)
        print("[bog] Scoring images...")

        all_soft = [[None] * G for _ in range(N)]
        all_hard = [[None] * G for _ in range(N)]

        for g, seed in enumerate(args.seeds[:G]):
            for i, item in enumerate(items):
                path = img_dir / img_fname(i, seed)
                if not path.exists():
                    raise FileNotFoundError(f"Image not found: {path} — run generate phase first")
                pil_img = PIL.Image.open(path).convert("RGB")
                soft_r = reward_model.score_image(pil_img, item, mode=args.reward_mode)
                hard_r = reward_model.score_image(pil_img, item, mode=args.hard_reward_mode)
                all_soft[i][g] = float(soft_r["score"])
                all_hard[i][g] = float(hard_r["score"])
            print(f"  scored gen {g+1}/{G} (seed={seed})")

        mean_hard_random       = []
        mean_hard_best_by_soft = []
        mean_hard_oracle       = []
        pearson_vals           = []
        spearman_vals          = []
        samples_jsonl          = []

        for i, item in enumerate(items):
            softs = all_soft[i]
            hards = all_hard[i]
            hard_random    = hards[0]
            best_soft_idx  = softs.index(max(softs))
            hard_best_soft = hards[best_soft_idx]
            hard_oracle    = max(hards)
            mean_hard_random.append(hard_random)
            mean_hard_best_by_soft.append(hard_best_soft)
            mean_hard_oracle.append(hard_oracle)
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
        mean_ora  = sum(mean_hard_oracle) / N
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

        print("\n[bog] ── Results ────────────────────────────────────")
        print(f"  random             : {mean_r:.4f}")
        print(f"  best_by_grpo_total : {mean_soft:.4f}  (delta={mean_soft-mean_r:+.4f})")
        print(f"  oracle             : {mean_ora:.4f}  (delta={mean_ora-mean_r:+.4f})")
        print(f"  pearson            : {pearson:.4f}")
        print(f"  spearman           : {spearman:.4f}")
        print(f"  reward_ranking_usable: {reward_ranking_usable}")
        print(f"  oracle_headroom    : {oracle_headroom}")
        print(f"  success            : {success}")
        if not reward_ranking_usable:
            print("  [warn] Reward ranking delta < 0.05")
        if not oracle_headroom:
            print("  [warn] Oracle headroom < 0.05 — candidates may be too uniform")
        print(f"\n[bog] Saved → {out_dir}")


if __name__ == "__main__":
    main()
