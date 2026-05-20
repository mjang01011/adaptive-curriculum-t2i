from adaptive_curriculum.curriculum.base_sampler import CurriculumSampler


class FixedBucketSampler(CurriculumSampler):
    """Always trains on the same bucket. Used for single-skill overfit experiments."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    def choose_bucket(self, step: int) -> str:
        return self.bucket_name

    def state_dict(self) -> dict:
        return {"bucket_name": self.bucket_name}

    def load_state_dict(self, state: dict):
        self.bucket_name = state["bucket_name"]
