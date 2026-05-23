"""
CARGO: Component-Aware Reward-Grounded token advantage GRPO for LlamaGen T2I.

Key exports:
    CARGOTrainer      — GRPO trainer with per-token importance-weighted advantages
                        + elite-replay SFT auxiliary loss
    CARGORewardModel  — Qwen3-VL reward model with v2 reward modes
                        (distinct_object_score for spatial, triple-gate formula)
    CARGO_MODES       — frozenset of supported v2 reward mode names
    META_KEYS         — component-score keys excluded from mask computation
    compute_cargo_mask — winner-aligned token importance mask (16×16 smoothed)

Scripts (run from project root):
    CARGO/train.py        — training entry point
    CARGO/diagnostics.py  — per-prompt mask overlays + comp_rewards.json
    CARGO/viz.py          — training curves, CARGO panel, progression grid
"""
from CARGO.trainer import CARGOTrainer
from CARGO.scoring import CARGORewardModel
from CARGO.rewards import CARGO_MODES, apply_cargo_reward, META_KEYS
from CARGO.masks   import compute_cargo_mask

__all__ = [
    "CARGOTrainer",
    "CARGORewardModel",
    "CARGO_MODES",
    "META_KEYS",
    "apply_cargo_reward",
    "compute_cargo_mask",
]
