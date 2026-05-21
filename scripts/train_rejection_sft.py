"""
Epoch-based rejection-SFT training on reward-selected image token sequences.

For each epoch:
  - Shuffle selected rows
  - Mini-batch forward/backward with AR cross-entropy on pre-saved VQ tokens
  - Run fixed probe eval (hard_target) before training and after each epoch
  - Save checkpoint: epoch_N.pt, best.pt, final.pt

Saves config_resolved.yaml compatible with eval_checkpoints_bucket.py.

Usage (via train_rejection_sft.sh):
  python scripts/train_rejection_sft.py \
    --base-config adaptive_curriculum/configs/experiment.yaml \
    --sft-config  adaptive_curriculum/configs/experiments/attribute_rejection_sft_top1.yaml \
    --selected-jsonl outputs/rejection_sft_attribute_g6/selected_top1.jsonl \
    --repo-root   /viscam/.../LlamaGen \
    --data-root   /viscam/.../data \
    --gpt-ckpt    .../t2i_XL_stage1_256.pt \
    --vq-ckpt     .../vq_ds16_t2i.pt \
    --t5-path     .../t5-ckpt \
    --t5-cache-dir .../data/t5_cache \
    --output-dir  /viscam/.../outputs/<run_name> \
    --run-name    attribute_rejection_sft_top1_<job_id> \
    [--wandb]
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _probe_eval(model, reward_model, probe_items, probe_seeds, t5_cache, run_dir, label):
    """Run fixed probe across seeds. Returns (mean, stderr, per_qtype_means)."""
    from adaptive_curriculum.train.evaluate_buckets import evaluate_bucket

    rewards  = []
    qtype_accum: dict = {}

    for seed in probe_seeds:
        out = str(Path(run_dir) / "probe_evals" / label / f"seed_{seed}")
        summary = evaluate_bucket(
            model=model,
            reward_model=reward_model,
            val_items=probe_items,
            out_dir=out,
            num_samples_per_prompt=1,
            seed=seed,
            t5_cache=t5_cache,
            reward_mode="hard_target",
        )
        rewards.extend(summary.get("reward_distribution", []))
        for qt, acc in summary.get("per_qtype_accuracy", {}).items():
            qtype_accum.setdefault(qt, []).append(acc)

    mean   = _mean(rewards)
    stderr = (statistics.stdev(rewards) / len(rewards) ** 0.5) if len(rewards) > 1 else 0.0
    qt_means = {qt: _mean(v) for qt, v in qtype_accum.items()}
    return mean, stderr, qt_means, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config",    default="adaptive_curriculum/configs/experiment.yaml")
    parser.add_argument("--sft-config",     required=True)
    parser.add_argument("--selected-jsonl", required=True)
    parser.add_argument("--repo-root",      required=True)
    parser.add_argument("--data-root",      required=True)
    parser.add_argument("--gpt-ckpt",       required=True)
    parser.add_argument("--vq-ckpt",        required=True)
    parser.add_argument("--t5-path",        required=True)
    parser.add_argument("--t5-cache-dir",   default=None)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--run-name",       default=None)
    parser.add_argument("--wandb",          action="store_true")
    parser.add_argument("--wandb-project",  default="llamagen-adaptive-curriculum")
    parser.add_argument("--wandb-entity",   default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.repo_root)

    # --- config ----------------------------------------------------------
    from omegaconf import OmegaConf
    base_cfg = OmegaConf.load(args.base_config)
    sft_cfg  = OmegaConf.load(args.sft_config)
    config   = OmegaConf.merge(base_cfg, sft_cfg)

    # CLI path overrides
    config.paths.repo_root     = args.repo_root
    config.paths.data_root     = args.data_root
    config.model.gpt_ckpt      = args.gpt_ckpt
    config.model.vq_ckpt       = args.vq_ckpt
    config.model.t5_path       = args.t5_path
    if args.t5_cache_dir:
        config.paths.t5_cache_dir = args.t5_cache_dir

    sft_train  = config.training
    sft_eval   = config.evaluation
    sft_lora   = config.lora

    epochs         = int(getattr(sft_train, "epochs", 3))
    batch_size     = int(getattr(sft_train, "train_batch_size", 4))
    lr             = float(getattr(sft_train, "learning_rate", 2e-5))
    max_grad_norm  = float(getattr(sft_train, "max_grad_norm", 1.0))
    save_every_ep  = bool(getattr(sft_train, "save_every_epoch", True))
    bucket         = str(getattr(config, "bucket", "attribute_binding"))

    probe_num_prompts = int(getattr(sft_eval, "probe_num_prompts", 8))
    probe_seeds       = list(getattr(sft_eval, "probe_seeds", [0, 1, 2, 3]))
    probe_enabled     = bool(getattr(sft_eval, "eval_probe_fixed", True))

    # --- run dir ---------------------------------------------------------
    run_dir  = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "probe_evals", "logs"):
        (run_dir / sub).mkdir(exist_ok=True)

    run_name = args.run_name or run_dir.name
    print(f"[sft] Run dir: {run_dir}")
    print(f"[sft] Bucket: {bucket}  epochs: {epochs}  lr: {lr}  batch: {batch_size}")

    # save resolved config for eval_checkpoints_bucket.py compatibility
    OmegaConf.save(config, str(run_dir / "config_resolved.yaml"))

    import os, subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    metadata = {
        "mode": "rejection_sft",
        "bucket": bucket,
        "selected_jsonl": args.selected_jsonl,
        "epochs": epochs,
        "learning_rate": lr,
        "train_batch_size": batch_size,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "run_name": run_name,
        "git_commit": git_commit,
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- load selected data ---------------------------------------------
    selected_rows = []
    with open(args.selected_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                selected_rows.append(json.loads(line))
    print(f"[sft] Selected rows: {len(selected_rows)}")

    # --- T5 cache -------------------------------------------------------
    t5_cache = None
    if args.t5_cache_dir:
        from adaptive_curriculum.data.t5_cache import load_t5_cache
        t5_cache = load_t5_cache(args.t5_cache_dir, [bucket])
        if t5_cache:
            print(f"[sft] T5 cache loaded from {args.t5_cache_dir}")
        else:
            print("[sft] T5 cache not found — will fail at training time (cache required for SFT)")
            raise RuntimeError("T5 cache required for rejection-SFT but not found.")

    t5_cache_dict = t5_cache.bucket_embeddings(bucket)

    # --- val items for probe eval ---------------------------------------
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=args.data_root,
        bucket_names=[bucket],
        train_file=str(config.buckets.train_file),
        val_file=str(config.buckets.val_file),
        max_val_prompts=int(getattr(sft_eval, "num_val_prompts_per_bucket", 20)),
    )
    val_items   = datasets[bucket].val_items
    probe_items = val_items[:probe_num_prompts]
    print(f"[sft] Val items: {len(val_items)}  probe items: {len(probe_items)}")

    # --- build model ----------------------------------------------------
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    lora_cfg = {
        "rank":           int(sft_lora.rank),
        "alpha":          int(sft_lora.alpha),
        "dropout":        float(sft_lora.dropout),
        "target_modules": list(sft_lora.get("target_modules", ["wqkv", "wo"])),
        "start_layer":    int(getattr(sft_lora, "start_layer", 0)),
    }
    model = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        gpt_model=str(config.model.gpt_model),
        image_size=int(config.model.image_size),
        t5_path=args.t5_path,
        t5_model_type=str(config.model.t5_model_type),
        t5_feature_max_len=int(config.model.t5_feature_max_len),
        cfg_scale=float(getattr(config.model, "cfg_scale", 2.0)),
        precision=str(config.model.mixed_precision),
        use_lora=True,
        lora_config=lora_cfg,
        learning_rate=lr,
        max_grad_norm=max_grad_norm,
    )

    # --- reward model for probe eval ------------------------------------
    from adaptive_curriculum.reward.vlm_reward import build_reward_model
    reward_model = build_reward_model(config)

    # --- W&B ------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_cfg_dict = OmegaConf.to_container(config, resolve=True)
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config=wandb_cfg_dict,
            )
            print(f"[sft] W&B: {wandb_run.url}")
        except Exception as e:
            print(f"[sft] W&B init failed: {e}")

    def _wb_log(d, step):
        if wandb_run:
            wandb_run.log(d, step=step)

    # --- base probe (step -1) ------------------------------------------
    base_probe_mean = None
    if probe_enabled and probe_items:
        print("\n[sft] Running base probe (before training)...")
        pmean, pse, qt_means, _ = _probe_eval(
            model, reward_model, probe_items, probe_seeds, t5_cache, str(run_dir), "base"
        )
        base_probe_mean = pmean
        qt_str = "  ".join(f"{qt}={v:.3f}" for qt, v in sorted(qt_means.items()))
        print(f"  [probe base]  mean={pmean:.4f}  se={pse:.4f}  {qt_str}")
        wb_log = {f"probe_base/{bucket}/{k}": v for k, v in qt_means.items()}
        wb_log[f"probe_base/{bucket}/mean_reward"] = pmean
        wb_log[f"probe_base/{bucket}/se_reward"]   = pse
        _wb_log(wb_log, 0)

    # --- training metrics log ------------------------------------------
    train_log_path = run_dir / "train_metrics.jsonl"
    probe_log_path = run_dir / "probe_eval_history.jsonl"

    def _append_jsonl(path, record):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    import torch

    best_probe_mean  = base_probe_mean if base_probe_mean is not None else -1.0
    best_checkpoint  = None
    global_step      = 0

    # --- epoch loop -----------------------------------------------------
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        random.shuffle(selected_rows)

        epoch_losses    = []
        epoch_grad_norms = []
        n_batches       = 0

        print(f"\n[sft] === Epoch {epoch}/{epochs} ===  ({len(selected_rows)} rows)")

        for batch_start in range(0, len(selected_rows), batch_size):
            batch_rows = selected_rows[batch_start:batch_start + batch_size]
            if not batch_rows:
                continue

            prompt_ids  = [row["prompt_id"] for row in batch_rows]
            token_paths = [row["image_tokens_path"] for row in batch_rows]

            # load pre-saved VQ tokens
            try:
                tokens_list  = [torch.load(p, map_location="cpu") for p in token_paths]
                image_tokens = torch.stack(tokens_list)  # (B, seq_len)
            except Exception as e:
                print(f"  [sft] WARNING: failed to load tokens for batch: {e}")
                continue

            metrics = model.train_sft_step_from_tokens(
                image_tokens=image_tokens,
                prompt_ids=prompt_ids,
                t5_cache_dict=t5_cache_dict,
            )

            epoch_losses.append(metrics["loss"])
            epoch_grad_norms.append(metrics["grad_norm"])
            n_batches   += 1
            global_step += 1

            if n_batches % 20 == 0:
                print(f"  [epoch {epoch} batch {n_batches}]  "
                      f"loss={metrics['loss']:.4f}  grad_norm={metrics['grad_norm']:.3f}")

            _wb_log({
                "train/loss":      metrics["loss"],
                "train/grad_norm": metrics["grad_norm"],
                "train/lr":        metrics["lr"],
            }, global_step)
            _append_jsonl(train_log_path, {
                "epoch": epoch, "global_step": global_step,
                "loss": metrics["loss"], "grad_norm": metrics["grad_norm"],
                "lr": metrics["lr"],
            })

        epoch_time  = time.time() - epoch_start
        mean_loss   = _mean(epoch_losses)
        mean_gnorm  = _mean(epoch_grad_norms)
        print(f"  [epoch {epoch}] mean_loss={mean_loss:.4f}  "
              f"mean_grad_norm={mean_gnorm:.3f}  t={epoch_time:.1f}s  batches={n_batches}")

        # --- save checkpoint per epoch ----------------------------------
        if save_every_ep:
            ckpt_path = str(run_dir / "checkpoints" / f"epoch_{epoch}.pt")
            model.save_checkpoint(ckpt_path)

        # --- probe eval after epoch ------------------------------------
        if probe_enabled and probe_items:
            pmean, pse, qt_means, probe_rewards = _probe_eval(
                model, reward_model, probe_items, probe_seeds, t5_cache,
                str(run_dir), f"epoch_{epoch:02d}"
            )
            qt_str = "  ".join(f"{qt}={v:.3f}" for qt, v in sorted(qt_means.items()))
            delta_str = (f"  delta_from_base={pmean - base_probe_mean:+.4f}"
                         if base_probe_mean is not None else "")
            print(f"  [probe epoch={epoch}] {bucket}  mean={pmean:.4f}  se={pse:.4f}  "
                  f"n={len(probe_rewards)}{delta_str}  {qt_str}")

            probe_record = {
                "epoch": epoch, "bucket": bucket,
                "mean_reward": pmean, "se_reward": pse,
                "per_prompt_scores": probe_rewards,
                "per_qtype_accuracy": qt_means,
            }
            _append_jsonl(probe_log_path, probe_record)

            wb_log = {f"probe/{bucket}/mean_reward": pmean,
                      f"probe/{bucket}/se_reward":   pse}
            for qt, acc in qt_means.items():
                wb_log[f"probe/{bucket}/{qt}"] = acc
            _wb_log(wb_log, epoch)

            # track best checkpoint
            if pmean > best_probe_mean:
                best_probe_mean = pmean
                best_checkpoint = str(run_dir / "checkpoints" / "best.pt")
                model.save_checkpoint(best_checkpoint)
                print(f"  [sft] New best probe mean: {pmean:.4f} → saved best.pt")

    # --- final checkpoint -----------------------------------------------
    final_path = str(run_dir / "checkpoints" / "final.pt")
    model.save_checkpoint(final_path)
    print(f"\n[sft] Saved final.pt")

    summary = {
        "bucket": bucket,
        "epochs": epochs,
        "best_probe_mean": best_probe_mean,
        "base_probe_mean": base_probe_mean,
        "delta_best_vs_base": (best_probe_mean - base_probe_mean)
        if base_probe_mean is not None else None,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_path,
        "selected_jsonl": args.selected_jsonl,
        "num_selected": len(selected_rows),
    }
    with open(run_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[sft] Done.  best_probe={best_probe_mean:.4f}  "
          f"base_probe={base_probe_mean:.4f}  run_dir={run_dir}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
