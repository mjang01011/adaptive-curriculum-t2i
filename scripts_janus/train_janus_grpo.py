"""
Janus-Pro-1B one-bucket GRPO training loop.

Usage:
  EXPERIMENT=janus_attribute_only_grpo_stable \
  CONFIG=configs_janus/janus_attribute_only_grpo_stable.yaml \
  python scripts_janus/train_janus_grpo.py --config $CONFIG

  # or via sbatch:
  EXPERIMENT=janus_attribute_only_grpo_stable \
  CONFIG=configs_janus/janus_attribute_only_grpo_stable.yaml \
  sbatch scripts_janus/train_janus_grpo.sh
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def write_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def make_run_dir(output_root: str, experiment: str, job_id: str) -> Path:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{experiment}_{job_id}_{ts}"
    run_dir = Path(output_root) / name
    if run_dir.exists():
        raise RuntimeError(f"Run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    for sub in ("checkpoints", "evals", "generations", "probe_evals", "reward_details"):
        (run_dir / sub).mkdir()
    return run_dir


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ── probe eval ────────────────────────────────────────────────────────────────

def run_probe_eval(wrapper, reward_model, val_items, out_path, num_prompts=8, seeds=(0, 1, 2, 3)):
    import random
    prompts_items = val_items[:num_prompts]
    prompts = [item.text for item in prompts_items]
    hard_scores = []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        wrapper.model.eval()
        out = wrapper.generate_images(prompts, seeds=None)
        for item, pil_img in zip(prompts_items, out["images"]):
            result = reward_model.score_image(pil_img, item, mode="hard_target")
            hard_scores.append(float(result["score"]))
    n = len(hard_scores)
    mean = sum(hard_scores) / n
    se   = (sum((s - mean) ** 2 for s in hard_scores) / (n * (n - 1))) ** 0.5 if n > 1 else 0.0
    probe_result = {"mean": round(mean, 4), "se": round(se, 4), "n": n, "scores": hard_scores}
    write_json(out_path, probe_result)
    return probe_result


# ── early stopping ────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, base: float, drop_threshold: float = 0.05, patience: int = 2):
        self.base = base
        self.drop_threshold = drop_threshold
        self.patience = patience
        self._consecutive_drops = 0

    def update(self, current_mean: float) -> bool:
        if current_mean <= self.base - self.drop_threshold:
            self._consecutive_drops += 1
        else:
            self._consecutive_drops = 0
        return self._consecutive_drops >= self.patience


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from adaptive_curriculum.data.schemas import BucketItem
    from adaptive_curriculum.reward.vlm_reward import VLMRewardModel
    from scripts_janus.janus_wrapper import JanusProWrapper

    experiment  = os.environ.get("EXPERIMENT", cfg.experiment_name)
    job_id      = os.environ.get("SLURM_JOB_ID", "local")
    output_root = cfg.training.output_root

    run_dir = make_run_dir(output_root, experiment, job_id)
    print(f"[train] Run dir: {run_dir}")

    # save metadata
    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = "unknown"
    write_json(run_dir / "run_metadata.json", {
        "experiment": experiment,
        "slurm_job_id": job_id,
        "run_dir": str(run_dir),
        "git_commit": git_commit,
        "config": OmegaConf.to_container(cfg, resolve=True),
    })

    # ── load data ──────────────────────────────────────────────────────
    data_root = Path(cfg.data.root)
    bucket = cfg.fixed_bucket

    train_raw = load_jsonl(cfg.data.train_file)
    val_raw   = load_jsonl(cfg.data.val_file)
    train_items = [BucketItem.from_dict(d) for d in train_raw]
    val_items   = [BucketItem.from_dict(d) for d in val_raw]
    print(f"[train] bucket={bucket}  train={len(train_items)}  val={len(val_items)}")

    # ── load reward model ──────────────────────────────────────────────
    reward_model = VLMRewardModel(model_path=getattr(cfg, "reward_model_path", None))
    print("[train] Reward model loaded.")

    # ── load Janus with LoRA ───────────────────────────────────────────
    lora_cfg_dict = {
        "r":              cfg.lora.r,
        "alpha":          cfg.lora.alpha,
        "dropout":        cfg.lora.dropout,
        "target_scope":   cfg.lora.target_scope,
        "target_modules": list(cfg.lora.target_modules),
    }
    grpo_cfg = cfg.grpo
    gen_cfg  = cfg.generation

    wrapper = JanusProWrapper(
        model_path=cfg.model if isinstance(cfg.model, str) else "deepseek-ai/Janus-Pro-1B",
        lora_config=lora_cfg_dict,
        cfg_weight=gen_cfg.cfg_weight,
        temperature=gen_cfg.temperature,
        logprob_reduction=grpo_cfg.logprob_reduction,
        learning_rate=cfg.training.learning_rate,
        max_grad_norm=cfg.training.max_grad_norm,
    )
    _ = wrapper.model
    print("[train] Model loaded with LoRA.")

    # ── W&B setup ──────────────────────────────────────────────────────
    use_wandb = getattr(getattr(cfg, "logging", None), "use_wandb", False)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=getattr(cfg.logging, "wandb_project", "janus-grpo"),
                name=experiment,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        except Exception as e:
            print(f"[train] W&B init failed: {e} — continuing without W&B")
            use_wandb = False

    # ── probe eval before training ─────────────────────────────────────
    eval_cfg = cfg.evaluation
    base_probe = None
    if eval_cfg.probe_eval_before_training:
        print("[train] Running base probe eval...")
        probe_result = run_probe_eval(
            wrapper, reward_model, val_items,
            out_path=str(run_dir / "probe_evals" / "probe_base.json"),
            num_prompts=eval_cfg.probe_num_prompts,
            seeds=list(eval_cfg.probe_seeds),
        )
        base_probe = probe_result["mean"]
        print(f"[train] Base probe: mean={base_probe:.4f}  se={probe_result['se']:.4f}")
        if use_wandb and wandb_run:
            wandb_run.log({"probe/hard_mean": base_probe, "probe/hard_se": probe_result["se"], "step": 0})

    early_stopper = EarlyStopper(base=base_probe or 0.0, drop_threshold=0.05, patience=2)

    # ── training loop ──────────────────────────────────────────────────
    num_steps      = cfg.training.num_curriculum_steps
    grad_steps     = cfg.training.gradient_steps_per_curriculum_step
    batch_size     = cfg.training.train_batch_size
    num_samples    = grpo_cfg.num_samples
    beta           = grpo_cfg.beta
    reward_mode    = grpo_cfg.reward_mode
    advantage_eps  = grpo_cfg.advantage_eps
    save_every     = cfg.training.save_every
    eval_every     = eval_cfg.eval_probe_every

    # gcpo-lite config
    token_weighting = getattr(grpo_cfg, "token_weighting", None)
    gcpo_config = None
    if token_weighting == "gcpo_lite":
        gcpo_config = {
            "grid_size":             getattr(grpo_cfg, "grid_size", 24),
            "initial_ratio":         getattr(grpo_cfg, "initial_ratio", 0.10),
            "entropy_gradient_ratio": getattr(grpo_cfg, "entropy_gradient_ratio", 0.20),
            "background_weight":     getattr(grpo_cfg, "background_weight", 0.2),
        }

    import random
    print(f"[train] Starting {num_steps} curriculum steps × {grad_steps} grad steps...")

    for step in range(1, num_steps + 1):
        step_metrics = {"step": step}

        for grad_step in range(grad_steps):
            batch = random.choices(train_items, k=batch_size)
            result = wrapper.train_grpo_step(
                batch=batch,
                reward_model=reward_model,
                num_samples=num_samples,
                beta=beta,
                reward_mode=reward_mode,
                advantage_eps=advantage_eps,
                token_weighting=token_weighting,
                gcpo_config=gcpo_config,
            )
            # log reward details
            if hasattr(wrapper, "_last_sample_details"):
                write_jsonl(
                    str(run_dir / "reward_details" / f"step{step:04d}_grad{grad_step}.jsonl"),
                    wrapper._last_sample_details,
                )
            step_metrics.update({f"grad{grad_step}_{k}": v for k, v in result.items()})

        # aggregate mean_reward over grad steps for logging
        agg_reward = sum(step_metrics.get(f"grad{g}_mean_reward", 0.0) for g in range(grad_steps)) / grad_steps
        agg_loss   = sum(step_metrics.get(f"grad{g}_loss", 0.0) for g in range(grad_steps)) / grad_steps
        log_msg = (f"  step={step}/{num_steps}  mean_reward={agg_reward:.4f}  "
                   f"loss={agg_loss:.4f}  lr={result['lr']:.2e}")
        print(log_msg)
        write_jsonl(str(run_dir / "train_log.jsonl"), [step_metrics])

        if use_wandb and wandb_run:
            wandb_run.log({"train/mean_reward": agg_reward, "train/loss": agg_loss, "step": step})

        # checkpoint
        if step % save_every == 0 or step == num_steps:
            ckpt_path = str(run_dir / "checkpoints" / f"step_{step:04d}.pt")
            wrapper.save_checkpoint(ckpt_path)

        # probe eval
        if step % eval_every == 0 or step == num_steps:
            print(f"  [eval] probe at step {step}...")
            probe_result = run_probe_eval(
                wrapper, reward_model, val_items,
                out_path=str(run_dir / "probe_evals" / f"probe_step{step:04d}.json"),
                num_prompts=eval_cfg.probe_num_prompts,
                seeds=list(eval_cfg.probe_seeds),
            )
            print(f"  [eval] mean={probe_result['mean']:.4f}  se={probe_result['se']:.4f}")
            write_jsonl(str(run_dir / "probe_history.jsonl"), [{
                "step": step,
                "mean": probe_result["mean"],
                "se": probe_result["se"],
                "n": probe_result["n"],
            }])
            if use_wandb and wandb_run:
                wandb_run.log({
                    "probe/hard_mean": probe_result["mean"],
                    "probe/hard_se": probe_result["se"],
                    "step": step,
                })

            # early stopping
            if base_probe is not None and early_stopper.update(probe_result["mean"]):
                print(f"[train] Early stop: probe dropped {base_probe:.4f} → {probe_result['mean']:.4f} "
                      f"for {early_stopper.patience} consecutive evals")
                break

    # ── final summary ──────────────────────────────────────────────────
    # collect probe history
    probe_history = []
    for line in open(run_dir / "probe_history.jsonl"):
        probe_history.append(json.loads(line))

    final_mean = probe_history[-1]["mean"] if probe_history else None
    improvement = round(final_mean - base_probe, 4) if (final_mean and base_probe) else None
    success     = improvement is not None and improvement >= 0.05

    summary = {
        "experiment": experiment,
        "bucket": bucket,
        "base_probe": base_probe,
        "final_probe": final_mean,
        "improvement": improvement,
        "success": success,
        "strong_success": improvement is not None and improvement >= 0.08,
        "num_steps_run": step,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "final_summary.json", summary)
    print("\n[train] ── Final Summary ──────────────────────────────")
    print(f"  base_probe  : {base_probe}")
    print(f"  final_probe : {final_mean}")
    print(f"  improvement : {improvement}")
    print(f"  success     : {success}")
    print(f"  run_dir     : {run_dir}")

    if use_wandb and wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
