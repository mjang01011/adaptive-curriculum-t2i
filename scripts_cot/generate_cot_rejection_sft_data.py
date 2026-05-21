"""
CoT-prompt rejection-SFT data generation.

Generates G images per training prompt using the **structured prompt** (better signal),
but records the **raw prompt** as the SFT conditioning key.  When train_rejection_sft.py
reads these rows it calls t5_cache[prompt_id] — which returns the raw-prompt embedding —
so the model learns to map raw prompts to reward-selected outputs generated via structured
prompts.  This is the distillation trick.

Output:
  <output-dir>/
    images/seed_N/<bucket>_<id>_sample0.png
    image_tokens/<id>_seed<N>.pt
    samples.jsonl
    summary.json

Usage:
  python scripts_cot/generate_cot_rejection_sft_data.py \
    --base-config  adaptive_curriculum/configs/experiment.yaml \
    --data-root    data \
    --structured-data-root data_cot_structured \
    --repo-root    /viscam/.../LlamaGen \
    --gpt-ckpt     .../t2i_XL_stage1_256.pt \
    --vq-ckpt      .../vq_ds16_t2i.pt \
    --t5-path      .../t5-ckpt \
    --t5-cache-dir .../data/t5_cache \
    --bucket       attribute_binding \
    --split        train \
    --num-prompts  500 \
    --num-generations 6 \
    --cfg-scale    2.0 \
    --reward-mode  pseudo_soft_grpo_target_heavy \
    --seeds        0 1 2 3 4 5 \
    --output-dir   outputs_cot_planning/cot_rejection_sft/attribute_g6_<timestamp>
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _find_structured_file(structured_root: Path, bucket: str, split: str) -> Path:
    candidates = list((structured_root / bucket).glob(f"{bucket}_{split}_*_structured.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No structured jsonl for {bucket}/{split} in {structured_root / bucket}"
        )
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config",           default="adaptive_curriculum/configs/experiment.yaml")
    parser.add_argument("--data-root",             required=True)
    parser.add_argument("--structured-data-root",  required=True)
    parser.add_argument("--repo-root",             required=True)
    parser.add_argument("--gpt-ckpt",              required=True)
    parser.add_argument("--vq-ckpt",               required=True)
    parser.add_argument("--t5-path",               required=True)
    parser.add_argument("--t5-cache-dir",          default=None)
    parser.add_argument("--bucket",                default="attribute_binding")
    parser.add_argument("--split",                 default="train")
    parser.add_argument("--num-prompts",           type=int, default=500)
    parser.add_argument("--num-generations",       type=int, default=6)
    parser.add_argument("--cfg-scale",             type=float, default=2.0)
    parser.add_argument("--reward-mode",           default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--seeds",                 nargs="+", type=int, default=None)
    parser.add_argument("--output-dir",            required=True)
    parser.add_argument("--batch-size",            type=int, default=4)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.num_generations))

    # refuse overwrite
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "images").mkdir()
    (out_dir / "image_tokens").mkdir()

    sys.path.insert(0, args.repo_root)

    from omegaconf import OmegaConf
    base_cfg = OmegaConf.load(args.base_config)
    base_cfg.paths.repo_root = args.repo_root
    base_cfg.paths.data_root = args.data_root
    base_cfg.model.gpt_ckpt  = args.gpt_ckpt
    base_cfg.model.vq_ckpt   = args.vq_ckpt
    base_cfg.model.t5_path   = args.t5_path
    base_cfg.model.cfg_scale = args.cfg_scale
    if args.t5_cache_dir:
        base_cfg.paths.t5_cache_dir = args.t5_cache_dir

    metadata = {
        "experiment": os.environ.get("EXPERIMENT", "cot_rejection_sft"),
        "bucket": args.bucket,
        "split": args.split,
        "num_prompts": args.num_prompts,
        "G": len(seeds),
        "seeds": seeds,
        "reward_mode": args.reward_mode,
        "cfg_scale": args.cfg_scale,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "output_dir": str(out_dir),
        "generation_condition": "structured_prompt",
        "sft_condition": "raw_prompt",
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- load original train items (for scoring questions + prompt_id) ---
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=args.data_root,
        bucket_names=[args.bucket],
        train_file=str(base_cfg.buckets.train_file),
        val_file=str(base_cfg.buckets.val_file),
    )
    all_items = datasets[args.bucket].train_items[:args.num_prompts]
    print(f"[cot_gen] Bucket={args.bucket}  prompts={len(all_items)}  G={len(seeds)}")

    # --- load structured prompts ----------------------------------------
    struct_path = _find_structured_file(
        Path(args.structured_data_root), args.bucket, args.split
    )
    print(f"[cot_gen] Structured prompts: {struct_path}")
    struct_map: dict = {}
    with open(struct_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            struct_map[row["id"]] = row.get("structured_prompt", row.get("prompt", ""))

    # items we have structured prompts for
    items = [it for it in all_items if it.id in struct_map]
    print(f"[cot_gen] Items with structured prompts: {len(items)}")
    if len(items) < len(all_items):
        print(f"[cot_gen] WARNING: {len(all_items) - len(items)} items missing structured prompts")

    # --- build models ----------------------------------------------------
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    model = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        gpt_model=str(base_cfg.model.gpt_model),
        image_size=int(base_cfg.model.image_size),
        t5_path=args.t5_path,
        t5_model_type=str(base_cfg.model.t5_model_type),
        t5_feature_max_len=int(base_cfg.model.t5_feature_max_len),
        cfg_scale=args.cfg_scale,
        precision=str(base_cfg.model.mixed_precision),
        use_lora=False,
    )

    from adaptive_curriculum.reward.vlm_reward import build_reward_model
    reward_model = build_reward_model(base_cfg)

    import torch

    def _iter_batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i + bs]

    total_samples   = 0
    all_grpo_scores = []
    all_hard_scores = []
    total_uncertain = 0

    samples_path = out_dir / "samples.jsonl"
    tokens_dir   = out_dir / "image_tokens"

    with open(samples_path, "w", encoding="utf-8") as f_out:
        for seed in seeds:
            seed_img_dir = out_dir / "images" / f"seed_{seed:02d}"
            seed_img_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[cot_gen] seed={seed}")

            for batch in _iter_batches(items, args.batch_size):
                prompt_ids    = [it.id for it in batch]
                struct_promps = [struct_map[it.id] for it in batch]
                bucket_names  = [it.bucket for it in batch]

                # generate with STRUCTURED prompt (no cache — different from raw)
                img_paths, tok_tensors = model.generate_with_tokens(
                    prompts=struct_promps,
                    out_dir=str(seed_img_dir),
                    prompt_ids=prompt_ids,
                    bucket_names=bucket_names,
                    num_samples_per_prompt=1,
                    seed=seed,
                    cached_embeddings=None,
                )

                for item, img_path, tokens in zip(batch, img_paths, tok_tensors):
                    # save token sequence
                    tok_fname = f"{item.id}_seed{seed}.pt"
                    tok_path  = str(tokens_dir / tok_fname)
                    torch.save(tokens, tok_path)

                    # score with GRPO reward (uses original item's grpo_reward_questions)
                    grpo_result = reward_model.score_image(img_path, item, mode=args.reward_mode)
                    hard_result = reward_model.score_image(img_path, item, mode="hard_target")

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

                    unc_grpo = sum(
                        1 for q in grpo_result.get("question_scores", [])
                        if q.get("predicted", "") == "uncertain"
                    )
                    unc_hard = sum(
                        1 for q in hard_result.get("question_scores", [])
                        if q.get("predicted", "") == "uncertain"
                    )

                    row = {
                        "bucket": item.bucket,
                        "prompt_id": item.id,
                        "raw_prompt": item.text,
                        "structured_prompt": struct_map[item.id],
                        "generation_condition": "structured_prompt",
                        "sft_condition": "raw_prompt",
                        "sample_index": seeds.index(seed),
                        "seed": seed,
                        "image_path": img_path,
                        "image_tokens_path": tok_path,
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

            print(f"  seed={seed} done  cumulative_samples={total_samples}")

    summary = {
        "bucket": args.bucket,
        "num_prompts": len(items),
        "G": len(seeds),
        "seeds": seeds,
        "total_samples": total_samples,
        "mean_grpo_score": _mean(all_grpo_scores),
        "mean_hard_target_score": _mean(all_hard_scores),
        "uncertain_rate": total_uncertain / total_samples if total_samples else 0.0,
        "reward_mode": args.reward_mode,
        "cfg_scale": args.cfg_scale,
        "generation_condition": "structured_prompt",
        "sft_condition": "raw_prompt",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[cot_gen] Done.  total={total_samples}  "
          f"mean_grpo={summary['mean_grpo_score']:.4f}  "
          f"mean_hard={summary['mean_hard_target_score']:.4f}")
    print(f"[cot_gen] Output → {out_dir}")
    print(f"[cot_gen] Next step: python scripts/select_rejection_sft_data.py "
          f"--input {out_dir}/samples.jsonl --top-k 1 "
          f"--out {out_dir}/selected_top1.jsonl")


if __name__ == "__main__":
    main()
