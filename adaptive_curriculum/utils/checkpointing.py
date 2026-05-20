import json
from pathlib import Path


def save_sampler_state(sampler, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = sampler.state_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_sampler_state(sampler, path: str):
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    sampler.load_state_dict(state)
    return sampler
