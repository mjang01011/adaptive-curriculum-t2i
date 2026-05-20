import random
from typing import List, Dict

from adaptive_curriculum.curriculum.base_sampler import CurriculumSampler


class StaticSampler(CurriculumSampler):
    """Easy-to-hard fixed curriculum defined by phases."""

    def __init__(self, phases: List[Dict]):
        # phases: list of {"buckets": [...], "steps": N}
        self.phases = phases
        self._build_schedule()

    def _build_schedule(self):
        # precompute cumulative step counts per phase
        self._boundaries = []
        cumulative = 0
        for phase in self.phases:
            cumulative += phase["steps"]
            self._boundaries.append(cumulative)

    def choose_bucket(self, step: int) -> str:
        active_buckets = self.phases[-1]["buckets"]
        for i, boundary in enumerate(self._boundaries):
            if step < boundary:
                active_buckets = self.phases[i]["buckets"]
                break
        return random.choice(active_buckets)

    def state_dict(self) -> dict:
        return {"phases": [{"buckets": list(p["buckets"]), "steps": int(p["steps"])} for p in self.phases]}

    def load_state_dict(self, state: dict):
        self.phases = state["phases"]
        self._build_schedule()
