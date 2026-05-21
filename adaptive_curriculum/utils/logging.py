import json
from pathlib import Path
from typing import Optional

from adaptive_curriculum.utils.jsonl import append_jsonl


class RunLogger:
    def __init__(
        self,
        run_dir: str,
        strategy: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config=None,
    ):
        self.run_dir = Path(run_dir)
        self.strategy = strategy
        self.use_wandb = use_wandb
        self._wandb_run = None
        self._bucket_selection_counts: dict = {}

        if use_wandb:
            try:
                import wandb
                wandb_cfg = {}
                if config is not None:
                    try:
                        from omegaconf import OmegaConf
                        wandb_cfg = OmegaConf.to_container(config, resolve=True)
                    except Exception:
                        pass
                self._wandb_run = wandb.init(
                    project=wandb_project or "llamagen-adaptive-curriculum",
                    entity=wandb_entity or None,
                    name=run_name or self.run_dir.name,
                    config=wandb_cfg,
                )
                print(f"[Logger] W&B run: {self._wandb_run.url}")
            except Exception as e:
                print(f"[Logger] wandb init failed: {e}")

    # ── training ──────────────────────────────────────────────────────────────

    def log_train_metrics(self, step: int, bucket: str, metrics: dict):
        record = {"curriculum_step": step, "bucket": bucket, "strategy": self.strategy, **metrics}
        append_jsonl(str(self.run_dir / "train_metrics.jsonl"), record)
        if self._wandb_run:
            self._wandb_run.log(
                {
                    "train/loss": metrics.get("avg_loss", 0),
                    "train/pg_loss": metrics.get("pg_loss", 0),
                    "train/kl_loss": metrics.get("kl_loss", 0),
                    "train/mean_reward": metrics.get("train_mean_reward", 0),
                    "train/reward_std": metrics.get("train_reward_std", 0),
                    "train/lr": metrics.get("lr", 0),
                    "train/grad_norm": metrics.get("grad_norm", 0),
                },
                step=step,
            )

    # ── curriculum decisions ──────────────────────────────────────────────────

    def log_curriculum_decision(
        self,
        step: int,
        chosen_bucket: str,
        ucb_scores: Optional[dict] = None,
        bucket_stats: Optional[dict] = None,
    ):
        record = {
            "curriculum_step": step,
            "strategy": self.strategy,
            "chosen_bucket": chosen_bucket,
            "ucb_scores": ucb_scores or {},
            "bucket_stats": bucket_stats or {},
        }
        append_jsonl(str(self.run_dir / "curriculum_decisions.jsonl"), record)

        # update selection counts
        self._bucket_selection_counts[chosen_bucket] = (
            self._bucket_selection_counts.get(chosen_bucket, 0) + 1
        )
        total_selections = sum(self._bucket_selection_counts.values())

        if self._wandb_run:
            wandb_log = {"curriculum/chosen_bucket": chosen_bucket}

            # UCB component breakdown per bucket
            for bucket, stats in (bucket_stats or {}).items():
                if isinstance(stats, dict):
                    for key in ("improvement_ma", "exploration_bonus", "ucb_score",
                                "raw_reward_ma", "last_raw_reward", "n_selected"):
                        if key in stats and isinstance(stats[key], (int, float)):
                            wandb_log[f"ucb/{bucket}/{key}"] = stats[key]

            # selection frequency
            for bucket, count in self._bucket_selection_counts.items():
                wandb_log[f"curriculum/selection_count/{bucket}"] = count
                wandb_log[f"curriculum/selection_frac/{bucket}"] = count / total_selections

            self._wandb_run.log(wandb_log, step=step)

    # ── per-step bucket eval ──────────────────────────────────────────────────

    def log_bucket_eval(self, step: int, eval_summary: dict):
        record = {"curriculum_step": step, **{
            k: v for k, v in eval_summary.items()
            if k not in ("sample_image_paths", "reward_distribution")
        }}
        append_jsonl(str(self.run_dir / "bucket_eval_history.jsonl"), record)

        if self._wandb_run:
            bucket = eval_summary.get("bucket", "unknown")
            wandb_log = {
                f"eval/{bucket}/mean_reward": eval_summary.get("mean_raw_reward", 0),
                f"eval/{bucket}/std_reward": eval_summary.get("std_raw_reward", 0),
            }
            for q_type, acc in eval_summary.get("per_qtype_accuracy", {}).items():
                wandb_log[f"eval/{bucket}/qtype/{q_type}"] = acc
            self._wandb_run.log(wandb_log, step=step)

    # ── full evaluation across all buckets ───────────────────────────────────

    def log_full_eval(self, step: int, all_results: dict):
        record = {"curriculum_step": step, "results": {
            b: {k: v for k, v in s.items() if k not in ("sample_image_paths", "reward_distribution")}
            for b, s in all_results.items()
        }}
        append_jsonl(str(self.run_dir / "full_eval_history.jsonl"), record)

        if not self._wandb_run:
            return

        import wandb
        wandb_log = {}
        rewards = []

        for bucket, summary in all_results.items():
            r = summary.get("mean_raw_reward", 0)
            rewards.append(r)
            wandb_log[f"full_eval/{bucket}/mean_reward"] = r
            wandb_log[f"full_eval/{bucket}/std_reward"] = summary.get("std_raw_reward", 0)

            # per-question-type accuracy (target vs anti)
            for q_type, acc in summary.get("per_qtype_accuracy", {}).items():
                wandb_log[f"full_eval/{bucket}/qtype/{q_type}"] = acc

            # reward distribution histogram
            dist = summary.get("reward_distribution", [])
            if dist:
                wandb_log[f"full_eval/{bucket}/reward_hist"] = wandb.Histogram(dist)

            # sample generated images (up to 4 per bucket)
            img_paths = summary.get("sample_image_paths", [])
            valid_imgs = [p for p in img_paths if Path(p).exists()]
            if valid_imgs:
                wandb_log[f"full_eval/{bucket}/samples"] = [
                    wandb.Image(p, caption=f"{bucket} step={step}")
                    for p in valid_imgs[:4]
                ]

        if rewards:
            wandb_log["full_eval/avg_reward"] = sum(rewards) / len(rewards)

        self._wandb_run.log(wandb_log, step=step)

    # ── fixed probe eval ─────────────────────────────────────────────────────

    def log_probe_eval(self, step: int, bucket: str, probe_result: dict):
        record = {"curriculum_step": step, "bucket": bucket, **{
            k: v for k, v in probe_result.items()
            if k not in ("per_prompt_scores",)
        }, "per_prompt_scores": probe_result.get("per_prompt_scores", [])}
        append_jsonl(str(self.run_dir / "probe_eval_history.jsonl"), record)
        if self._wandb_run:
            wandb_log = {
                f"probe/{bucket}/mean_reward": probe_result.get("mean_reward", 0),
                f"probe/{bucket}/se_reward": probe_result.get("se_reward", 0),
                f"probe/{bucket}/uncertain_rate": probe_result.get("uncertain_rate", 0),
            }
            for qt, acc in probe_result.get("per_qtype_accuracy", {}).items():
                wandb_log[f"probe/{bucket}/{qt}"] = acc
            self._wandb_run.log(wandb_log, step=step)

    # ── reward component logging ───────────────────────────────────────────────

    def log_reward_components(self, step: int, bucket: str, components: dict):
        record = {"curriculum_step": step, "bucket": bucket, **components}
        append_jsonl(str(self.run_dir / "reward_components.jsonl"), record)
        if self._wandb_run:
            wandb_log = {}
            for key, val in components.items():
                if isinstance(val, (int, float)):
                    wandb_log[f"reward_components/{bucket}/{key}"] = val
            self._wandb_run.log(wandb_log, step=step)

    # ── GPU stats ─────────────────────────────────────────────────────────────

    def log_gpu_stats(self, step: int):
        from adaptive_curriculum.utils.gpu_stats import get_gpu_stats
        stats = get_gpu_stats()
        if not stats:
            return
        append_jsonl(str(self.run_dir / "gpu_stats.jsonl"), {"step": step, **stats})
        if self._wandb_run:
            self._wandb_run.log(stats, step=step)

    # ── step timing ───────────────────────────────────────────────────────────

    def log_step_time(self, step: int, elapsed_sec: float):
        if self._wandb_run:
            self._wandb_run.log({"perf/step_time_sec": elapsed_sec}, step=step)

    def finish(self):
        if self._wandb_run:
            self._wandb_run.finish()
