"""
Compare raw vs structured prompts: generate images with both, score with the same
target_questions, report the delta.

No training — pure inference-time experiment.

Usage:
  python scripts_cot/eval_structured_prompts.py \
    --base-config  adaptive_curriculum/configs/experiment.yaml \
    --data-root    data \
    --structured-data-root data_cot_structured \
    --repo-root    /viscam/.../LlamaGen \
    --gpt-ckpt     .../t2i_XL_stage1_256.pt \
    --vq-ckpt      .../vq_ds16_t2i.pt \
    --t5-path      .../t5-ckpt \
    --t5-cache-dir .../data/t5_cache \
    --bucket       attribute_binding \
    --split        val \
    --num-prompts  20 \
    --num-samples-per-prompt 8 \
    --cfg-scale    2.0 \
    --seeds        0 1 2 3 4 5 6 7 \
    --reward-mode  hard_target \
    --output-dir   outputs_cot_planning/structured_prompt_eval_attribute_<timestamp>
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _find_structured_file(structured_root: Path, bucket: str, split: str) -> Path:
    candidates = list((structured_root / bucket).glob(f"{bucket}_{split}_*_structured.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No structured jsonl for bucket={bucket} split={split} in {structured_root / bucket}"
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
    parser.add_argument("--split",                 default="val")
    parser.add_argument("--num-prompts",           type=int, default=20)
    parser.add_argument("--num-samples-per-prompt",type=int, default=8)
    parser.add_argument("--cfg-scale",             type=float, default=2.0)
    parser.add_argument("--seeds",                 nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--reward-mode",           default="hard_target")
    parser.add_argument("--output-dir",            required=True)
    args = parser.parse_args()

    # refuse overwrite
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    raw_dir    = out_dir / "raw_samples"
    struct_dir = out_dir / "structured_samples"
    raw_dir.mkdir()
    struct_dir.mkdir()

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

    # --- save metadata --------------------------------------------------
    metadata = {
        "experiment": os.environ.get("EXPERIMENT", "structured_prompt_eval"),
        "bucket": args.bucket,
        "split": args.split,
        "num_prompts": args.num_prompts,
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "seeds": args.seeds,
        "cfg_scale": args.cfg_scale,
        "reward_mode": args.reward_mode,
        "output_dir": str(out_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- load original val items (for target_questions) -----------------
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=args.data_root,
        bucket_names=[args.bucket],
        train_file=str(base_cfg.buckets.train_file),
        val_file=str(base_cfg.buckets.val_file),
        max_val_prompts=args.num_prompts,
    )
    orig_items = datasets[args.bucket].val_items
    print(f"[eval_sp] Original val items: {len(orig_items)}")

    # --- load structured prompts ----------------------------------------
    struct_path = _find_structured_file(
        Path(args.structured_data_root), args.bucket, args.split
    )
    print(f"[eval_sp] Structured prompts: {struct_path}")
    struct_map: dict = {}  # id -> structured_prompt string
    with open(struct_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            struct_map[row["id"]] = row.get("structured_prompt", row.get("prompt", ""))

    # filter to items we have structured prompts for
    items = [it for it in orig_items if it.id in struct_map][:args.num_prompts]
    print(f"[eval_sp] Items with structured prompts: {len(items)}")

    # --- T5 cache for raw prompts ---------------------------------------
    t5_cache = None
    if args.t5_cache_dir:
        from adaptive_curriculum.data.t5_cache import load_t5_cache
        t5_cache = load_t5_cache(args.t5_cache_dir, [args.bucket])
        if t5_cache:
            print(f"[eval_sp] T5 cache loaded (raw prompts)")

    # --- build model and reward -----------------------------------------
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

    # --- generate and score per seed -----------------------------------
    # raw_scores[i]    = list of scores across seeds for item i
    # struct_scores[i] = same for structured
    raw_per_item    = [[] for _ in items]
    struct_per_item = [[] for _ in items]
    qtype_raw:    dict = {}
    qtype_struct: dict = {}

    eval_records = []

    prompt_ids   = [it.id for it in items]
    raw_prompts  = [it.text for it in items]
    struct_prompts = [struct_map[it.id] for it in items]
    bucket_names = [it.bucket for it in items]

    for seed in args.seeds:
        print(f"\n[eval_sp] Seed {seed}")

        # --- raw generation ---------------------------------------------
        cached_embs = None
        if t5_cache is not None:
            cached_embs = t5_cache.bucket_embeddings(args.bucket)

        raw_paths = model.generate_images(
            prompts=raw_prompts,
            out_dir=str(raw_dir / f"seed_{seed:02d}"),
            prompt_ids=prompt_ids,
            bucket_names=bucket_names,
            num_samples_per_prompt=1,
            seed=seed,
            cached_embeddings=cached_embs,
        )

        # --- structured generation (no cache — different prompts) ------
        struct_paths = model.generate_images(
            prompts=struct_prompts,
            out_dir=str(struct_dir / f"seed_{seed:02d}"),
            prompt_ids=prompt_ids,
            bucket_names=bucket_names,
            num_samples_per_prompt=1,
            seed=seed,
            cached_embeddings=None,   # can't use raw-prompt cache here
        )

        for i, item in enumerate(items):
            r_path = raw_paths[i]
            s_path = struct_paths[i]

            r_result = reward_model.score_image(r_path, item, mode=args.reward_mode)
            s_result = reward_model.score_image(s_path, item, mode=args.reward_mode)

            raw_per_item[i].append(float(r_result["score"]))
            struct_per_item[i].append(float(s_result["score"]))

            # accumulate per-qtype
            for q in r_result.get("question_scores", []):
                qt = q.get("q_type", "unknown")
                qtype_raw.setdefault(qt, []).append(float(q.get("correct", 0)))
            for q in s_result.get("question_scores", []):
                qt = q.get("q_type", "unknown")
                qtype_struct.setdefault(qt, []).append(float(q.get("correct", 0)))

            record = {
                "prompt_id": item.id,
                "seed": seed,
                "raw_prompt": item.text,
                "structured_prompt": struct_map[item.id],
                "raw_image_path": r_path,
                "structured_image_path": s_path,
                "raw_score": float(r_result["score"]),
                "structured_score": float(s_result["score"]),
                "delta": float(s_result["score"]) - float(r_result["score"]),
            }
            eval_records.append(record)

    # --- aggregate -------------------------------------------------------
    raw_prompt_means    = [_mean(v) for v in raw_per_item]
    struct_prompt_means = [_mean(v) for v in struct_per_item]

    raw_mean    = _mean(raw_prompt_means)
    struct_mean = _mean(struct_prompt_means)
    delta       = struct_mean - raw_mean

    # per-qtype means
    raw_qt    = {qt: _mean(v) for qt, v in qtype_raw.items()}
    struct_qt = {qt: _mean(v) for qt, v in qtype_struct.items()}

    # per-prompt comparison
    per_prompt = [
        {
            "prompt_id": items[i].id,
            "prompt": items[i].text,
            "raw_mean":    raw_prompt_means[i],
            "struct_mean": struct_prompt_means[i],
            "delta": struct_prompt_means[i] - raw_prompt_means[i],
        }
        for i in range(len(items))
    ]

    summary = {
        "bucket": args.bucket,
        "num_prompts": len(items),
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "seeds": args.seeds,
        "raw_mean_hard_target":        raw_mean,
        "structured_mean_hard_target": struct_mean,
        "delta_structured_minus_raw":  delta,
    }
    for qt in sorted(set(list(qtype_raw.keys()) + list(qtype_struct.keys()))):
        summary[f"raw_{qt}_component"]    = raw_qt.get(qt, float("nan"))
        summary[f"structured_{qt}_component"] = struct_qt.get(qt, float("nan"))

    # success criterion
    success = delta >= 0.05
    summary["success"] = success
    summary["success_threshold"] = 0.05

    # --- write outputs ---------------------------------------------------
    with open(out_dir / "eval_results.jsonl", "w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec) + "\n")

    with open(out_dir / "per_prompt_comparison.json", "w") as f:
        json.dump(per_prompt, f, indent=2)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- print results ---------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Structured prompt eval — bucket: {args.bucket}")
    print(f"{'='*60}")
    print(f"raw    mean: {raw_mean:.4f}")
    print(f"struct mean: {struct_mean:.4f}")
    print(f"delta:       {delta:+.4f}  ({'PASS ✓' if success else 'FAIL — below +0.05 threshold'})")
    print()
    for qt in sorted(set(list(qtype_raw.keys()) + list(qtype_struct.keys()))):
        r = raw_qt.get(qt, float("nan"))
        s = struct_qt.get(qt, float("nan"))
        print(f"  {qt:25s}  raw={r:.4f}  struct={s:.4f}  delta={s-r:+.4f}")
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
