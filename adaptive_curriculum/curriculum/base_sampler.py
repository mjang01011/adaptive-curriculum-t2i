from typing import List


class CurriculumSampler:
    def choose_bucket(self, step: int) -> str:
        raise NotImplementedError

    def initialize_rewards(self, initial_scores: dict):
        """Called once before training with initial per-bucket val rewards."""
        pass

    def update(self, bucket: str, reward_info: dict):
        """Called after each curriculum step with the evaluated reward."""
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict):
        pass
