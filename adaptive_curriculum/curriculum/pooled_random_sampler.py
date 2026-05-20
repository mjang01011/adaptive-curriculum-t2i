from typing import List

from adaptive_curriculum.curriculum.base_sampler import CurriculumSampler

POOLED_BUCKET = "__pooled__"


class PooledRandomSampler(CurriculumSampler):
    """
    No curriculum — every training batch is sampled from the combined pool of
    all bucket train items.  The training loop detects POOLED_BUCKET and draws
    from the merged dataset instead of a single bucket dataset.

    Per-step eval still rotates through real buckets (round-robin) so training
    logs remain comparable; full evals every N steps cover all buckets anyway.
    """

    def __init__(self, bucket_names: List[str]):
        self.bucket_names = bucket_names
        self._eval_idx = 0  # round-robin index for per-step eval bucket

    def choose_bucket(self, step: int) -> str:
        return POOLED_BUCKET

    def get_eval_bucket(self) -> str:
        """Return the real bucket to use for this step's per-step eval."""
        bucket = self.bucket_names[self._eval_idx % len(self.bucket_names)]
        self._eval_idx += 1
        return bucket

    def state_dict(self) -> dict:
        return {"bucket_names": self.bucket_names, "eval_idx": self._eval_idx}

    def load_state_dict(self, state: dict):
        self.bucket_names = state["bucket_names"]
        self._eval_idx = state.get("eval_idx", 0)
