"""
Core curriculum training loop shared by all strategies.
"""
import time
from pathlib import Path
from typing import Dict, Optional

from adaptive_curriculum.utils.seed import set_seed
from adaptive_curriculum.utils.paths import make_run_dir
from adaptive_curriculum.utils.logging import RunLogger
from adaptive_curriculum.utils.checkpointing import save_sampler_state
from adaptive_curriculum.utils.jsonl import write_json
from adaptive_curriculum.utils.plots import generate_all_plots
from adaptive_curriculum.train.evaluate_buckets import evaluate_bucket, evaluate_all_buckets


def build_sampler(strategy: str, bucket_names: list, config):
    if strategy == "uniform":
        from adaptive_curriculum.curriculum.uniform_sampler import UniformSampler
        return UniformSampler(bucket_names)
    elif strategy == "round_robin":
        from adaptive_curriculum.curriculum.round_robin_sampler import RoundRobinSampler
        return RoundRobinSampler(bucket_names)
    elif strategy == "fixed_bucket":
        from adaptive_curriculum.curriculum.fixed_bucket_sampler import FixedBucketSampler
        fixed = getattr(config, "fixed_bucket", None)
        if fixed is None:
            raise ValueError("strategy=fixed_bucket requires 'fixed_bucket: <name>' in config")
        return FixedBucketSampler(fixed)
    elif strategy == "pooled_random":
        from adaptive_curriculum.curriculum.pooled_random_sampler import PooledRandomSampler
        return PooledRandomSampler(bucket_names)
    elif strategy == "static":
        from adaptive_curriculum.curriculum.static_sampler import StaticSampler
        phases = [
            {"buckets": p["buckets"], "steps": p["steps"]}
            for p in config.static_curriculum.phases
        ]
        return StaticSampler(phases)
    elif strategy == "ucb":
        from adaptive_curriculum.curriculum.ucb_sampler import UCBSampler
        return UCBSampler(
            bucket_names=bucket_names,
            c=config.ucb.c,
            reward_ma_beta=config.ucb.reward_ma_beta,
            improvement_ma_beta=config.ucb.improvement_ma_beta,
            epsilon=float(getattr(config.ucb, "epsilon", 0.0)),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_curriculum_training(config, strategy: str, output_root: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    import os, subprocess
    set_seed(config.seed)

    if output_dir is not None:
        run_dir = Path(output_dir)
        if run_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing output_dir: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        for sub in ("checkpoints", "evals", "generations", "plots", "probe_evals"):
            (run_dir / sub).mkdir(exist_ok=True)
    else:
        output_root = output_root or config.paths.output_root
        experiment_name = getattr(getattr(config, "logging", None), "run_name", None)
        run_dir = make_run_dir(output_root, strategy, config.project_name, experiment_name=experiment_name)
    print(f"[train] Run dir: {run_dir}")

    # save run metadata
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = "unknown"
    run_name = getattr(getattr(config, "logging", None), "run_name", run_dir.name)
    fixed_bucket = getattr(config, "fixed_bucket", None)
    metadata = {
        "experiment": os.environ.get("EXPERIMENT", run_name),
        "strategy": strategy,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "run_name": run_name,
        "output_dir": str(run_dir),
        "git_commit": git_commit,
        "data_root": str(getattr(config.paths, "data_root", "data")),
        "bucket": fixed_bucket,
    }
    write_json(str(run_dir / "run_metadata.json"), metadata)
    print(f"[train] Metadata: experiment={metadata['experiment']}  job={metadata['slurm_job_id']}  bucket={fixed_bucket}")

    # save resolved config
    import yaml, dataclasses
    try:
        from omegaconf import OmegaConf
        OmegaConf.save(config, str(run_dir / "config_resolved.yaml"))
    except Exception:
        pass

    log_cfg = getattr(config, "logging", None)
    logger = RunLogger(
        run_dir=str(run_dir),
        strategy=strategy,
        use_wandb=getattr(log_cfg, "use_wandb", False),
        wandb_project=getattr(log_cfg, "wandb_project", None),
        wandb_entity=getattr(log_cfg, "wandb_entity", None),
        run_name=getattr(log_cfg, "run_name", None),
        config=config,
    )

    bucket_names = list(config.buckets.names)

    # build datasets
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    data_root = getattr(config.paths, "data_root", "/vol/data")
    datasets = load_bucket_datasets(
        data_root=data_root,
        bucket_names=bucket_names,
        train_file=config.buckets.train_file,
        val_file=config.buckets.val_file,
        max_val_prompts=getattr(config.evaluation, "num_val_prompts_per_bucket", None),
    )

    # load T5 embedding cache if available (eliminates T5 from eval hot path)
    t5_cache_dir = getattr(config.paths, "t5_cache_dir", None)
    t5_cache = None
    if t5_cache_dir and t5_cache_dir != "null":
        from adaptive_curriculum.data.t5_cache import load_t5_cache
        t5_cache = load_t5_cache(t5_cache_dir, bucket_names)
        if t5_cache:
            print(f"[train] T5 cache loaded from {t5_cache_dir}")
        else:
            print(f"[train] T5 cache not found at {t5_cache_dir}, will use live T5 inference")

    # build reward model
    from adaptive_curriculum.reward.vlm_reward import build_reward_model
    reward_model = build_reward_model(config)

    # build model (None for no-GPU dry runs)
    use_real_model = getattr(config, "_use_real_model", True)
    model = None
    if use_real_model:
        from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
        lora_cfg = {
            "rank": config.lora.rank,
            "alpha": config.lora.alpha,
            "dropout": config.lora.dropout,
            "target_modules": list(config.lora.get("target_modules", ["wqkv", "wo"])),
            "start_layer": int(getattr(config.lora, "start_layer", 0)),
        } if config.model.use_lora else None
        grpo_cfg_train = getattr(config, "grpo", None)
        _cfg_scale_train = (
            float(getattr(grpo_cfg_train, "cfg_scale_train", getattr(config.model, "cfg_scale", 2.0)))
            if grpo_cfg_train else float(getattr(config.model, "cfg_scale", 2.0))
        )
        _logprob_reduction = (
            str(getattr(grpo_cfg_train, "logprob_reduction", "sum_sqrt_len"))
            if grpo_cfg_train else "sum_sqrt_len"
        )
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
            cfg_scale_train=_cfg_scale_train,
            logprob_reduction=_logprob_reduction,
            precision=config.model.mixed_precision,
            use_lora=config.model.use_lora,
            lora_config=lora_cfg,
            learning_rate=config.training.learning_rate,
            max_grad_norm=config.training.max_grad_norm,
        )

    # for pooled_random: add a merged dataset entry so the loop can sample from it
    from adaptive_curriculum.curriculum.pooled_random_sampler import POOLED_BUCKET
    if strategy == "pooled_random":
        from adaptive_curriculum.data.bucket_dataset import PooledDataset
        datasets[POOLED_BUCKET] = PooledDataset(datasets)
        print(f"[train] Pooled dataset: {len(datasets[POOLED_BUCKET])} train items")

    # build curriculum sampler
    sampler = build_sampler(strategy, bucket_names, config)

    # initial evaluation
    eval_reward_mode = str(getattr(getattr(config, "evaluation", None), "reward_mode", "hard_target"))
    print("[train] Running initial bucket evaluation...")
    evals_dir = str(run_dir / "evals")
    initial_results = evaluate_all_buckets(
        model=model,
        reward_model=reward_model,
        datasets=datasets,
        out_dir=evals_dir,
        curriculum_step=-1,
        num_samples_per_prompt=config.evaluation.num_samples_per_prompt,
        seed=config.seed,
        t5_cache=t5_cache,
        reward_mode=eval_reward_mode,
    )
    initial_scores = {b: r["mean_raw_reward"] for b, r in initial_results.items()}
    sampler.initialize_rewards(initial_scores)
    print(f"[train] Initial scores: {initial_scores}")

    num_steps = config.training.num_curriculum_steps
    grad_steps_per = config.training.gradient_steps_per_curriculum_step
    train_batch_size = config.training.train_batch_size
    save_every = config.training.save_every
    save_checkpoints = bool(getattr(config.training, "save_checkpoints", True))
    full_eval_every = config.evaluation.full_eval_every_curriculum_step
    num_samples = config.evaluation.num_samples_per_prompt

    # fixed probe eval config
    eval_cfg = getattr(config, "evaluation", None)
    probe_enabled = bool(getattr(eval_cfg, "eval_probe_fixed", False))
    probe_num_prompts = int(getattr(eval_cfg, "probe_num_prompts", 8))
    probe_seeds = list(getattr(eval_cfg, "probe_seeds", [0, 1, 2, 3]))
    probe_every = int(getattr(eval_cfg, "eval_probe_every", 2))

    # build fixed probe sets: first probe_num_prompts val items per bucket
    probe_items = {
        b: ds.val_items[:probe_num_prompts]
        for b, ds in datasets.items()
        if b != "__pooled__"
    }

    total_generated = 0
    t_start = time.time()
    best_avg_reward = -float("inf")
    best_checkpoint = None

    # reward detail log — per-image soft+hard rewards for alignment analysis
    reward_detail_path = str(run_dir / "reward_details.jsonl")
    _reward_detail_file = open(reward_detail_path, "w", encoding="utf-8")

    for step in range(num_steps):
        t_step_start = time.time()

        # 1. Choose bucket
        bucket = sampler.choose_bucket(step)

        # 2. Train K GRPO steps
        grpo_cfg = getattr(config, "grpo", None)
        grpo_num_samples = getattr(grpo_cfg, "num_samples", 4) if grpo_cfg else 4
        grpo_beta = getattr(grpo_cfg, "beta", 0.01) if grpo_cfg else 0.01
        grpo_reward_mode = str(getattr(grpo_cfg, "reward_mode", "hard_target")) if grpo_cfg else "hard_target"
        grpo_advantage_eps = float(getattr(grpo_cfg, "advantage_eps", 1e-8)) if grpo_cfg else 1e-8

        train_metrics_list = []
        if model is not None:
            for g in range(grad_steps_per):
                batch = datasets[bucket].sample_train_batch(train_batch_size)
                if batch:
                    metrics = model.train_grpo_step(
                        batch=batch,
                        reward_model=reward_model,
                        num_samples=grpo_num_samples,
                        beta=grpo_beta,
                        t5_cache=t5_cache,
                        reward_mode=grpo_reward_mode,
                        advantage_eps=grpo_advantage_eps,
                    )
                    train_metrics_list.append(metrics)
                    # write per-image reward details for alignment analysis
                    if hasattr(model, "_last_sample_details"):
                        for detail in model._last_sample_details:
                            detail["global_step"] = step
                            detail["grad_step"] = g
                            _reward_detail_file.write(
                                __import__("json").dumps(detail) + "\n"
                            )
                        _reward_detail_file.flush()
                    if (g + 1) % 4 == 0:
                        print(f"  [grpo {g+1}/{grad_steps_per}] loss={metrics['loss']:.4f}  "
                              f"reward={metrics['mean_reward']:.3f}  grad_norm={metrics['grad_norm']:.3f}  "
                              f"zero_std={metrics.get('percent_groups_zero_std', 0):.0f}%  "
                              f"mean_adv={metrics.get('mean_abs_advantage', 0):.3f}")

        if train_metrics_list:
            n = len(train_metrics_list)
            def _avg(key, default=0.0):
                return sum(m.get(key, default) for m in train_metrics_list) / n
            logger.log_train_metrics(step, bucket, {
                "avg_loss": _avg("loss"),
                "pg_loss": _avg("pg_loss"),
                "kl_loss": _avg("kl_loss"),
                "train_mean_reward": _avg("mean_reward"),
                "train_reward_std": _avg("reward_std"),
                "reward_min": _avg("reward_min"),
                "reward_max": _avg("reward_max"),
                "lr": train_metrics_list[-1].get("lr", 0),
                "grad_norm": _avg("grad_norm"),
                "grad_norm_before_clip": _avg("grad_norm_before_clip"),
                "grad_norm_after_clip": _avg("grad_norm_after_clip"),
                "lora_weight_norm": train_metrics_list[-1].get("lora_weight_norm", 0),
                "lora_grad_norm": _avg("lora_grad_norm"),
                "percent_groups_zero_std": _avg("percent_groups_zero_std"),
                "mean_group_reward_std": _avg("mean_group_reward_std"),
                "median_group_reward_std": _avg("median_group_reward_std"),
                "mean_abs_advantage": _avg("mean_abs_advantage"),
                "fraction_nonzero_advantage": _avg("fraction_nonzero_advantage"),
                "seq_logprob_mean": _avg("seq_logprob_mean"),
                "seq_logprob_std": _avg("seq_logprob_std"),
                "cfg_scale_train": train_metrics_list[-1].get("cfg_scale_train", 2.0),
                "logprob_reduction": train_metrics_list[-1].get("logprob_reduction", "sum_sqrt_len"),
                "reward_mode": train_metrics_list[-1].get("reward_mode", grpo_reward_mode),
            })

            # reward component breakdown: aggregate across all grad steps
            if model is not None and hasattr(model, "_last_sample_details"):
                _import_json = __import__("json")
                comp_accum: dict = {}
                hard_accum: list = []
                uncertain_count = 0
                total_details = 0
                for detail in model._last_sample_details:
                    total_details += 1
                    hard_accum.append(detail.get("hard_reward", 0.0))
                    if detail.get("has_uncertain", False):
                        uncertain_count += 1
                    for qt, sc in detail.get("component_scores", {}).items():
                        comp_accum.setdefault(qt, []).append(sc)
                if total_details > 0:
                    component_log = {
                        f"grpo_component_{qt}": sum(v) / len(v)
                        for qt, v in comp_accum.items()
                    }
                    component_log["hard_target_on_train_images"] = sum(hard_accum) / len(hard_accum)
                    component_log["uncertain_rate_train"] = uncertain_count / total_details
                    logger.log_reward_components(step, bucket, component_log)
                    if step % 4 == 0:
                        comp_str = "  ".join(
                            f"{k.replace('grpo_component_','')}={v:.3f}"
                            for k, v in sorted(component_log.items())
                            if k.startswith("grpo_component_")
                        )
                        print(f"  [components] {comp_str}  hard_train={component_log['hard_target_on_train_images']:.3f}")

        # 3. Evaluate selected bucket (always hard_target for clean signal)
        # For pooled_random, training bucket is __pooled__ (no val items);
        # rotate through real buckets round-robin for per-step eval logging.
        eval_bucket = (
            sampler.get_eval_bucket() if bucket == POOLED_BUCKET else bucket
        )
        bucket_eval_dir = str(run_dir / "evals" / f"step_{step:06d}" / eval_bucket)
        bucket_summary = evaluate_bucket(
            model=model,
            reward_model=reward_model,
            val_items=datasets[eval_bucket].val_items,
            out_dir=bucket_eval_dir,
            num_samples_per_prompt=num_samples,
            seed=config.seed,
            t5_cache=t5_cache,
            reward_mode=eval_reward_mode,
        )
        total_generated += bucket_summary["num_images"]

        # 4. Fixed probe evaluation (hard_target, same prompts/seeds every time)
        if probe_enabled and step % probe_every == 0:
            _probe_bucket = eval_bucket if bucket == POOLED_BUCKET else bucket
            _probe_val_items = probe_items.get(_probe_bucket, [])
            if _probe_val_items and model is not None:
                import statistics as _stats
                probe_rewards = []
                uncertain_count = 0
                total_q = 0
                for seed in probe_seeds:
                    from adaptive_curriculum.train.evaluate_buckets import evaluate_bucket as _eval_bucket
                    _probe_out = str(run_dir / "probe_evals" / f"step_{step:06d}" / _probe_bucket / f"seed_{seed}")
                    _probe_summary = _eval_bucket(
                        model=model,
                        reward_model=reward_model,
                        val_items=_probe_val_items,
                        out_dir=_probe_out,
                        num_samples_per_prompt=1,
                        seed=seed,
                        t5_cache=t5_cache,
                        reward_mode="hard_target",
                    )
                    probe_rewards.extend(_probe_summary.get("reward_distribution", []))
                    for r in _probe_summary.get("per_qtype_accuracy", {}).values():
                        pass  # already aggregated
                if probe_rewards:
                    pmean = sum(probe_rewards) / len(probe_rewards)
                    pse = (_stats.stdev(probe_rewards) / (len(probe_rewards) ** 0.5)) if len(probe_rewards) > 1 else 0.0
                    probe_result = {
                        "mean_reward": pmean,
                        "se_reward": pse,
                        "num_images": len(probe_rewards),
                        "per_prompt_scores": probe_rewards,
                        "uncertain_rate": 0.0,
                    }
                    logger.log_probe_eval(step, _probe_bucket, probe_result)
                    print(f"  [probe] {_probe_bucket}  mean={pmean:.4f}  se={pse:.4f}  n={len(probe_rewards)}")

        # 5. Update sampler
        reward_info = {
            "raw_reward": bucket_summary["mean_raw_reward"],
            "eval_summary": bucket_summary,
        }
        sampler.update(bucket, reward_info)

        # 6. Log
        ucb_scores = sampler.get_scores() if hasattr(sampler, "get_scores") else {}
        bucket_stats = sampler.get_stats_dict() if hasattr(sampler, "get_stats_dict") else {}
        logger.log_curriculum_decision(step, bucket, ucb_scores, bucket_stats)
        logger.log_bucket_eval(step, bucket_summary)

        step_time = time.time() - t_step_start
        logger.log_step_time(step, step_time)
        logger.log_gpu_stats(step)

        log_bucket = eval_bucket if bucket == POOLED_BUCKET else bucket
        print(
            f"[step {step:4d}/{num_steps}] bucket={log_bucket:25s}  "
            f"reward={bucket_summary['mean_raw_reward']:.4f}  "
            f"t={step_time:.1f}s"
        )

        # 6. Periodic full evaluation
        if step % full_eval_every == 0:
            all_results = evaluate_all_buckets(
                model=model,
                reward_model=reward_model,
                datasets=datasets,
                out_dir=evals_dir,
                curriculum_step=step,
                num_samples_per_prompt=num_samples,
                seed=config.seed,
                t5_cache=t5_cache,
                reward_mode=eval_reward_mode,
            )
            logger.log_full_eval(step, all_results)
            avg_reward = sum(r["mean_raw_reward"] for r in all_results.values()) / len(all_results)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                if model is not None and save_checkpoints:
                    best_checkpoint = str(run_dir / "checkpoints" / f"best.pt")
                    model.save_checkpoint(best_checkpoint)

        # 7. Checkpoint
        if save_checkpoints and step % save_every == 0:
            if model is not None:
                ckpt_path = str(run_dir / "checkpoints" / f"step_{step:06d}.pt")
                model.save_checkpoint(ckpt_path)
                if best_checkpoint is None:
                    best_checkpoint = ckpt_path
            save_sampler_state(sampler, str(run_dir / "checkpoints" / f"sampler_step_{step:06d}.json"))

    # final evaluation
    final_results = evaluate_all_buckets(
        model=model,
        reward_model=reward_model,
        datasets=datasets,
        out_dir=evals_dir,
        curriculum_step=num_steps,
        num_samples_per_prompt=num_samples,
        seed=config.seed,
        t5_cache=t5_cache,
        reward_mode=eval_reward_mode,
    )
    final_bucket_rewards = {b: r["mean_raw_reward"] for b, r in final_results.items()}
    avg_final = sum(final_bucket_rewards.values()) / len(final_bucket_rewards)

    total_gpu_secs = time.time() - t_start
    summary = {
        "strategy": strategy,
        "final_bucket_rewards": final_bucket_rewards,
        "average_final_reward": avg_final,
        "best_checkpoint": best_checkpoint,
        "total_gpu_seconds": total_gpu_secs,
        "total_generated_images": total_generated,
    }
    write_json(str(run_dir / "final_summary.json"), summary)
    _reward_detail_file.close()
    print(f"\n[train] Done. avg_final_reward={avg_final:.4f}  run_dir={run_dir}")

    # plots
    generate_all_plots(str(run_dir), bucket_names)

    logger.finish()
    return str(run_dir)
