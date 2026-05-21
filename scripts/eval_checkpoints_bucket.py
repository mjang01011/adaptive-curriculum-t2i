"""
Fixed-seed checkpoint evaluation for one bucket.

Evaluates base model and saved LoRA checkpoints on the same prompts with the same
seeds, giving a low-variance comparison curve rather than the noisy online signal.

Usage:
  python scripts/eval_checkpoints_bucket.py \
    --run-dir   outputs/<run> \
    --bucket    attribute_binding \
    --checkpoints base step_000005 step_000010 step_000015 step_000020 \
    --num-val-prompts 20 \
    --num-samples-per-prompt 8 \
    --cfg-scale 2.0 \
    --reward-mode hard_target \
    --fixed-seeds 0 1 2 3 4 5 6 7 \
    --out outputs/<run>/fixed_eval_attribute.json
"""
import argparse
import json
import math
import sys
from pathlib import Path


def _stderr(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def _load_config(run_dir: Path):
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_resolved.yaml not found in {run_dir}")
    from omegaconf import OmegaConf
    return OmegaConf.load(str(cfg_path))


def _build_model(config, lora_checkpoint: str = None):
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    use_lora = lora_checkpoint is not None
    lora_cfg = None
    if use_lora:
        lora_cfg = {
            "rank": int(config.lora.rank),
            "alpha": int(config.lora.alpha),
            "dropout": float(config.lora.dropout),
            "target_modules": list(config.lora.get("target_modules", ["wqkv", "wo"])),
            "start_layer": int(getattr(config.lora, "start_layer", 0)),
        }

    model = LlamaGenWrapper(
        repo_root=config.paths.repo_root,
        vq_ckpt=config.model.vq_ckpt,
        gpt_ckpt=config.model.gpt_ckpt,
        gpt_model=config.model.gpt_model,
        image_size=config.model.image_size,
        t5_path=config.model.t5_path,
        t5_model_type=config.model.t5_model_type,
        t5_feature_max_len=config.model.t5_feature_max_len,
        cfg_scale=float(getattr(config.model, "cfg_scale", 2.0)),
        precision=config.model.mixed_precision,
        use_lora=use_lora,
        lora_config=lora_cfg,
    )

    if lora_checkpoint is not None:
        model.load_checkpoint(lora_checkpoint)

    return model


def _eval_checkpoint(
    model,
    reward_model,
    val_items,
    out_dir: Path,
    num_samples: int,
    seeds: list,
    reward_mode: str,
    t5_cache=None,
):
    """
    Evaluate model on val_items using multiple fixed seeds, return per-prompt rewards.
    Images are generated seeds × val_items and aggregated per prompt.
    """
    from adaptive_curriculum.train.evaluate_buckets import evaluate_bucket

    all_prompt_rewards = []  # list of per-prompt mean rewards across seeds

    for seed in seeds:
        summary = evaluate_bucket(
            model=model,
            reward_model=reward_model,
            val_items=val_items,
            out_dir=str(out_dir / f"seed_{seed:02d}"),
            num_samples_per_prompt=num_samples,
            seed=seed,
            t5_cache=t5_cache,
            reward_mode=reward_mode,
        )
        all_prompt_rewards.append(summary["reward_distribution"])

    # average per-prompt across seeds
    n_prompts = len(val_items)
    per_prompt_mean = []
    for p in range(n_prompts):
        vals = [all_prompt_rewards[s][p] for s in range(len(seeds)) if p < len(all_prompt_rewards[s])]
        per_prompt_mean.append(sum(vals) / len(vals) if vals else 0.0)

    mean = sum(per_prompt_mean) / len(per_prompt_mean)
    stderr = _stderr(per_prompt_mean)
    return {"mean": mean, "stderr": stderr, "per_prompt": per_prompt_mean, "n_seeds": len(seeds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",    required=True)
    parser.add_argument("--bucket",     required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="'base' or checkpoint names like 'step_000005'")
    parser.add_argument("--num-val-prompts",       type=int, default=20)
    parser.add_argument("--num-samples-per-prompt", type=int, default=8)
    parser.add_argument("--cfg-scale",  type=float, default=2.0)
    parser.add_argument("--reward-mode", default="hard_target")
    parser.add_argument("--fixed-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--data-root",  default=None, help="Override data root from config")
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    if args.data_root:
        config.paths.data_root = args.data_root

    # override cfg_scale in model config for eval
    config.model.cfg_scale = args.cfg_scale

    # load val items
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=config.paths.data_root,
        bucket_names=[args.bucket],
        train_file=config.buckets.train_file,
        val_file=config.buckets.val_file,
        max_val_prompts=args.num_val_prompts,
    )
    val_items = datasets[args.bucket].val_items
    print(f"[eval] Bucket: {args.bucket}  val prompts: {len(val_items)}")

    # load reward model
    from adaptive_curriculum.reward.vlm_reward import build_reward_model
    reward_model = build_reward_model(config)

    # load T5 cache if available
    t5_cache = None
    t5_cache_dir = getattr(config.paths, "t5_cache_dir", None)
    if t5_cache_dir and t5_cache_dir != "null":
        from adaptive_curriculum.data.t5_cache import load_t5_cache
        t5_cache = load_t5_cache(t5_cache_dir, [args.bucket])

    results = {}

    for ckpt_name in args.checkpoints:
        print(f"\n[eval] Checkpoint: {ckpt_name}")

        if ckpt_name == "base":
            lora_path = None
        else:
            # look for the checkpoint file
            ckpt_path = run_dir / "checkpoints" / f"{ckpt_name}.pt"
            if not ckpt_path.exists():
                print(f"  WARNING: {ckpt_path} not found, skipping")
                continue
            lora_path = str(ckpt_path)

        model = _build_model(config, lora_checkpoint=lora_path)
        out_dir = run_dir / "fixed_eval" / args.bucket / ckpt_name

        result = _eval_checkpoint(
            model=model,
            reward_model=reward_model,
            val_items=val_items,
            out_dir=out_dir,
            num_samples=args.num_samples_per_prompt,
            seeds=args.fixed_seeds,
            reward_mode=args.reward_mode,
            t5_cache=t5_cache,
        )
        results[ckpt_name] = result
        print(f"  mean={result['mean']:.4f}  stderr={result['stderr']:.4f}  "
              f"(seeds={result['n_seeds']}, prompts={len(val_items)}, "
              f"samples/prompt={args.num_samples_per_prompt})")

        # free GPU memory between checkpoints
        del model
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # print summary table
    print(f"\n{'='*50}")
    print(f"Fixed eval — bucket: {args.bucket}  reward_mode: {args.reward_mode}")
    print(f"{'='*50}")
    print(f"{'Checkpoint':20s}  {'Mean':>8s}  {'Stderr':>8s}")
    print("-" * 42)
    base_mean = results.get("base", {}).get("mean", None)
    for ckpt_name, r in results.items():
        delta = f"  ({r['mean'] - base_mean:+.4f})" if base_mean is not None and ckpt_name != "base" else ""
        print(f"{ckpt_name:20s}  {r['mean']:8.4f}  {r['stderr']:8.4f}{delta}")

    out_data = {
        "bucket": args.bucket,
        "reward_mode": args.reward_mode,
        "num_val_prompts": len(val_items),
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "fixed_seeds": args.fixed_seeds,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
