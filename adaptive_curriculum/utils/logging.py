import json
from pathlib import Path
from typing import Optional

from adaptive_curriculum.utils.jsonl import append_jsonl


class RunLogger:
    def __init__(self, run_dir: str, strategy: str, use_wandb: bool = False):
        self.run_dir = Path(run_dir)
        self.strategy = strategy
        self.use_wandb = use_wandb
        self._wandb_run = None

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(project="llamagen_ucb_curriculum", name=str(self.run_dir.name))
            except Exception as e:
                print(f"[Logger] wandb init failed: {e}")

    def log_train_metrics(self, step: int, bucket: str, metrics: dict):
        record = {"curriculum_step": step, "bucket": bucket, "strategy": self.strategy, **metrics}
        append_jsonl(str(self.run_dir / "train_metrics.jsonl"), record)
        if self._wandb_run:
            self._wandb_run.log({"train/" + k: v for k, v in metrics.items()}, step=step)

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

    def log_bucket_eval(self, step: int, eval_summary: dict):
        record = {"curriculum_step": step, **eval_summary}
        append_jsonl(str(self.run_dir / "bucket_eval_history.jsonl"), record)
        if self._wandb_run:
            bucket = eval_summary.get("bucket", "unknown")
            self._wandb_run.log({
                f"eval/{bucket}/mean_reward": eval_summary.get("mean_raw_reward", 0),
            }, step=step)

    def finish(self):
        if self._wandb_run:
            self._wandb_run.finish()
