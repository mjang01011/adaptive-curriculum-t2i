import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from adaptive_curriculum.curriculum.base_sampler import CurriculumSampler


@dataclass
class BucketStats:
    name: str
    n_selected: int = 0
    raw_reward_ma: float = 0.0
    prev_raw_reward_ma: float = 0.0
    improvement_ma: float = 0.0
    last_raw_reward: float = 0.0
    last_improvement: float = 0.0
    ucb_score: float = 0.0


def compute_ucb_score(stats: BucketStats, t: int, c: float) -> float:
    if stats.n_selected == 0:
        return float("inf")
    exploration = c * math.sqrt(math.log(t + 1) / stats.n_selected)
    return stats.improvement_ma + exploration


class UCBSampler(CurriculumSampler):
    def __init__(
        self,
        bucket_names: List[str],
        c: float = 0.25,
        reward_ma_beta: float = 0.8,
        improvement_ma_beta: float = 0.8,
    ):
        self.bucket_names = bucket_names
        self.c = c
        self.reward_ma_beta = reward_ma_beta
        self.improvement_ma_beta = improvement_ma_beta
        self.t = 0
        self.buckets: Dict[str, BucketStats] = {
            name: BucketStats(name=name) for name in bucket_names
        }
        self._last_chosen: Optional[str] = None

    def initialize_rewards(self, initial_scores: dict):
        """Seed raw_reward_ma from pre-training baseline evaluation."""
        for name, score in initial_scores.items():
            if name in self.buckets:
                self.buckets[name].raw_reward_ma = score
                self.buckets[name].last_raw_reward = score

    def choose_bucket(self, step: int) -> str:
        scores = {
            name: compute_ucb_score(stats, self.t, self.c)
            for name, stats in self.buckets.items()
        }
        chosen = max(scores, key=lambda k: scores[k])
        # store ucb scores on stats for logging
        for name, score in scores.items():
            self.buckets[name].ucb_score = score
        self._last_chosen = chosen
        return chosen

    def update(self, bucket: str, reward_info: dict):
        raw_reward = reward_info["raw_reward"]
        stats = self.buckets[bucket]

        old_m = stats.raw_reward_ma
        new_m = self.reward_ma_beta * old_m + (1 - self.reward_ma_beta) * raw_reward
        delta = new_m - old_m
        new_d = self.improvement_ma_beta * stats.improvement_ma + (1 - self.improvement_ma_beta) * delta

        stats.prev_raw_reward_ma = old_m
        stats.raw_reward_ma = new_m
        stats.last_raw_reward = raw_reward
        stats.last_improvement = delta
        stats.improvement_ma = new_d
        stats.n_selected += 1
        self.t += 1

    def get_scores(self) -> Dict[str, float]:
        return {
            name: compute_ucb_score(stats, self.t, self.c)
            for name, stats in self.buckets.items()
        }

    def get_stats_dict(self) -> Dict[str, dict]:
        return {
            name: {
                "n_selected": s.n_selected,
                "raw_reward_ma": s.raw_reward_ma,
                "improvement_ma": s.improvement_ma,
                "exploration_bonus": (
                    self.c * math.sqrt(math.log(self.t + 1) / s.n_selected)
                    if s.n_selected > 0 else float("inf")
                ),
                "last_raw_reward": s.last_raw_reward,
                "last_improvement": s.last_improvement,
                "ucb_score": s.ucb_score,
            }
            for name, s in self.buckets.items()
        }

    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "c": self.c,
            "reward_ma_beta": self.reward_ma_beta,
            "improvement_ma_beta": self.improvement_ma_beta,
            "buckets": {
                name: {
                    "n_selected": s.n_selected,
                    "raw_reward_ma": s.raw_reward_ma,
                    "prev_raw_reward_ma": s.prev_raw_reward_ma,
                    "improvement_ma": s.improvement_ma,
                    "last_raw_reward": s.last_raw_reward,
                    "last_improvement": s.last_improvement,
                    "ucb_score": s.ucb_score,
                }
                for name, s in self.buckets.items()
            },
        }

    def load_state_dict(self, state: dict):
        self.t = state["t"]
        self.c = state["c"]
        self.reward_ma_beta = state["reward_ma_beta"]
        self.improvement_ma_beta = state["improvement_ma_beta"]
        for name, sd in state["buckets"].items():
            if name in self.buckets:
                s = self.buckets[name]
                s.n_selected = sd["n_selected"]
                s.raw_reward_ma = sd["raw_reward_ma"]
                s.prev_raw_reward_ma = sd["prev_raw_reward_ma"]
                s.improvement_ma = sd["improvement_ma"]
                s.last_raw_reward = sd["last_raw_reward"]
                s.last_improvement = sd["last_improvement"]
                s.ucb_score = sd["ucb_score"]
