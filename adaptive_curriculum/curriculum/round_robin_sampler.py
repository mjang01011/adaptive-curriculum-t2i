from typing import List

from adaptive_curriculum.curriculum.base_sampler import CurriculumSampler


class RoundRobinSampler(CurriculumSampler):
    """Cycles through buckets in a fixed order, ignoring reward signal."""

    def __init__(self, bucket_names: List[str]):
        self.bucket_names = bucket_names
        self._step = 0

    def choose_bucket(self, step: int) -> str:
        return self.bucket_names[step % len(self.bucket_names)]

    def state_dict(self) -> dict:
        return {"bucket_names": self.bucket_names}

    def load_state_dict(self, state: dict):
        self.bucket_names = state["bucket_names"]
