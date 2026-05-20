import random
from typing import List

from adaptive_curriculum.curriculum.base_sampler import CurriculumSampler


class UniformSampler(CurriculumSampler):
    def __init__(self, bucket_names: List[str]):
        self.bucket_names = bucket_names

    def choose_bucket(self, step: int) -> str:
        return random.choice(self.bucket_names)

    def state_dict(self) -> dict:
        return {"bucket_names": self.bucket_names}

    def load_state_dict(self, state: dict):
        self.bucket_names = state["bucket_names"]
