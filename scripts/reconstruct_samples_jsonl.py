"""
Reconstruct samples.jsonl from existing image + token files without regenerating.

Usage:
  python scripts/reconstruct_samples_jsonl.py \
    --base-config adaptive_curriculum/configs/experiment.yaml \
    --data-root /viscam/.../data \
    --bucket attribute_binding \
    --output-dir outputs/rejection_sft_attribute_binding_g6 \
    --num-generations 6 \
    --reward-mode pseudo_soft_grpo_target_heavy
"""
import argparse
import json
import sys
from pathlib import Path


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config",    default="adaptive_curriculum/configs/experiment.yaml")
    parser.add_argument("--data-root",      required=True)
    parser.add_argument("--bucket",         default="attribute_binding")
    parser.add_argument("--split",          default="train")
    parser.add_argument("--num-prompts",    type=int, default=500)
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--reward-mode",    default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--overwrite",      action="store_true")
    args = parser.parse_args()

    out_dir    = Path(args.output_dir)
    tokens_dir = out_dir / "image_tokens"
    images_dir = out_dir / "images"
    samples_path = out_dir / "samples.jsonl"

    if samples_path.exists() and samples_path.stat().st_size > 0 and not args.overwrite:
        print(f"[reconstruct] {samples_path} already exists and is non-empty. Use --overwrite to redo.")
        return

    seeds = list(range(args.num_generations))

    from omegaconf import OmegaConf
    base_cfg = OmegaConf.load(args.base_config)
    base_cfg.paths.data_root = args.data_root

    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=args.data_root,
        bucket_names=[args.bucket],
        train_file=str(base_cfg.buckets.train_file),
        val_file=str(base_cfg.buckets.val_file),
    )
    all_items = datasets[args.bucket].train_items[:args.num_prompts]
    item_by_id = {item.id: item for item in all_items}
    print(f"[reconstruct] {len(all_items)} items, seeds={seeds}")

    from adaptive_curriculum.reward.vlm_reward import build_reward_model
    reward_model = build_reward_model(base_cfg)

    all_grpo_scores = []
    all_hard_scores = []
    total_samples   = 0
    total_uncertain = 0

    with open(samples_path, "w", encoding="utf-8") as f_out:
        for seed in seeds:
            seed_img_dir = images_dir / f"seed_{seed:02d}"
            print(f"\n[reconstruct] seed={seed}")
            found = 0
            for item in all_items:
                tok_path = tokens_dir / f"{item.id}_seed{seed}.pt"
                img_path = seed_img_dir / f"{item.id}.png"

                if not tok_path.exists():
                    print(f"  [warn] missing token: {tok_path}")
                    continue
                if not img_path.exists():
                    # try png with prompt-id as filename variants
                    candidates = list(seed_img_dir.glob(f"{item.id}*.png"))
                    if not candidates:
                        print(f"  [warn] missing image: {img_path}")
                        continue
                    img_path = candidates[0]

                grpo_result = reward_model.score_image(str(img_path), item, mode=args.reward_mode)
                hard_result = reward_model.score_image(str(img_path), item, mode="hard_target")

                cs_grpo = grpo_result.get("component_scores", {})
                qwen_grpo = [
                    {"q": q.get("question", ""), "predicted": q.get("predicted", ""),
                     "expected": q.get("expected", "")}
                    for q in grpo_result.get("question_scores", [])
                ]
                qwen_hard = [
                    {"q": q.get("question", ""), "predicted": q.get("predicted", ""),
                     "expected": q.get("expected", "")}
                    for q in hard_result.get("question_scores", [])
                ]
                unc_grpo = sum(1 for q in grpo_result.get("question_scores", [])
                               if q.get("predicted", "") == "uncertain")
                unc_hard = sum(1 for q in hard_result.get("question_scores", [])
                               if q.get("predicted", "") == "uncertain")

                row = {
                    "bucket": item.bucket,
                    "prompt_id": item.id,
                    "prompt": item.text,
                    "sample_index": seed,
                    "seed": seed,
                    "image_path": str(img_path),
                    "image_tokens_path": str(tok_path),
                    "grpo_total_score": float(grpo_result["score"]),
                    "hard_target_score": float(hard_result["score"]),
                    "target_component_score": float(
                        cs_grpo.get("attribute", cs_grpo.get("relation", float("nan")))
                    ),
                    "presence_component_score": float(
                        cs_grpo.get("object_presence", float("nan"))
                    ),
                    "anti_component_score": float(
                        cs_grpo.get("anti_swap", cs_grpo.get("anti_relation", float("nan")))
                    ),
                    "quality_component_score": float(
                        cs_grpo.get("image_quality", float("nan"))
                    ),
                    "alignment_component_score": float(
                        cs_grpo.get("prompt_alignment", float("nan"))
                    ),
                    "uncertain_count_grpo": unc_grpo,
                    "uncertain_count_target": unc_hard,
                    "qwen_answers_grpo": qwen_grpo,
                    "qwen_answers_target": qwen_hard,
                }
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()

                all_grpo_scores.append(float(grpo_result["score"]))
                all_hard_scores.append(float(hard_result["score"]))
                total_uncertain += unc_grpo
                total_samples   += 1
                found += 1

            print(f"  seed={seed} done  found={found}  cumulative={total_samples}")

    summary = {
        "bucket": args.bucket,
        "num_prompts": len(all_items),
        "G": len(seeds),
        "seeds": seeds,
        "total_samples": total_samples,
        "mean_grpo_score": _mean(all_grpo_scores),
        "mean_hard_target_score": _mean(all_hard_scores),
        "uncertain_rate": total_uncertain / total_samples if total_samples else 0.0,
        "reward_mode": args.reward_mode,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[reconstruct] Done.  total={total_samples}  "
          f"mean_grpo={summary['mean_grpo_score']:.4f}  "
          f"mean_hard={summary['mean_hard_target_score']:.4f}")
    print(f"[reconstruct] Saved → {samples_path}")


if __name__ == "__main__":
    main()
