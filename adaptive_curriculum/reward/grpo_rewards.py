"""
Compositional gated GRPO reward formulas.

Each function takes a component-score dict (keyed by q_type, values in [0,1])
and returns (reward: float, debug: dict).

Component scores are pseudo-soft averages per q_type, computed by vlm_reward.py
before calling these functions. Missing components fall back to neutral defaults.
"""
import math


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float, tau: float = 0.20) -> float:
    return 1.0 / (1.0 + math.exp(-x / tau))


def _smooth_gate(x: float, low: float, high: float) -> float:
    """0 if x <= low, 1 if x >= high, linear ramp between."""
    return _clamp((x - low) / (high - low), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Attribute binding  (bucket: attribute_binding)
# ---------------------------------------------------------------------------

def grpo_attr_presence_gated_v2(comp: dict) -> tuple:
    """
    Gate on object_presence: optimize attribute only after objects are visible.
    Anti-swap is a constraint penalty, not a positive driver.
    """
    presence   = comp.get("object_presence", 0.5)
    attribute  = comp.get("attribute", 0.5)
    anti_swap  = comp.get("anti_swap", 1.0)
    alignment  = comp.get("prompt_alignment", 0.5)
    quality    = comp.get("image_quality", 0.5)
    uncertain  = comp.get("uncertain_frac", 0.0)

    gate = _smooth_gate(presence, low=0.35, high=0.80)

    pre_reward = (
        0.80 * presence
      + 0.10 * alignment
      + 0.10 * quality
    )

    post_reward = (
        0.55 * attribute
      + 0.30 * presence
      + 0.10 * alignment
      + 0.05 * quality
    )

    reward = (1.0 - gate) * pre_reward + gate * post_reward

    penalty = 0.0
    if presence >= 0.55 and anti_swap < 0.50:
        penalty = 0.10 * (0.50 - anti_swap) / 0.50
        reward -= penalty

    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "gate":             gate,
        "pre_reward":       pre_reward,
        "post_reward":      post_reward,
        "penalty":          penalty,
        "uncertain_penalty": uncertain_penalty,
    }


# ---------------------------------------------------------------------------
# Spatial relation  (bucket: spatial_relations, spatial_relations_anchored)
# ---------------------------------------------------------------------------

def grpo_spatial_presence_gated_v2(comp: dict, anchored: bool = False) -> tuple:
    """
    Gate on object_presence (stricter than attribute because spatial is noisier).
    Anti-relation is a constraint penalty only once objects are clearly present.
    Set anchored=True for spatial_relations_anchored (upweights relation slightly).
    """
    presence     = comp.get("object_presence", 0.5)
    relation     = comp.get("relation", 0.5)
    anti_relation = comp.get("anti_relation", 1.0)
    alignment    = comp.get("prompt_alignment", 0.5)
    quality      = comp.get("image_quality", 0.5)
    uncertain    = comp.get("uncertain_frac", 0.0)

    gate = _smooth_gate(presence, low=0.45, high=0.85)

    pre_reward = (
        0.85 * presence
      + 0.10 * alignment
      + 0.05 * quality
    )

    if anchored:
        post_reward = (
            0.60 * relation
          + 0.30 * presence
          + 0.07 * alignment
          + 0.03 * quality
        )
    else:
        post_reward = (
            0.55 * relation
          + 0.35 * presence
          + 0.07 * alignment
          + 0.03 * quality
        )

    reward = (1.0 - gate) * pre_reward + gate * post_reward

    penalty = 0.0
    if presence >= 0.60 and anti_relation < 0.50:
        penalty = 0.15 * (0.50 - anti_relation) / 0.50
        reward -= penalty

    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "gate":             gate,
        "pre_reward":       pre_reward,
        "post_reward":      post_reward,
        "penalty":          penalty,
        "uncertain_penalty": uncertain_penalty,
    }


# ---------------------------------------------------------------------------
# Counting  (bucket: counting)
# ---------------------------------------------------------------------------

def grpo_counting_presence_gated_v2(comp: dict) -> tuple:
    """
    Gate on object_presence. Exact count is primary; count_close gives partial credit.
    Overcount penalty only fires once the object category is clearly visible.
    """
    presence     = comp.get("object_presence", 0.5)
    count_correct = comp.get("count_correct", 0.5)
    count_close  = comp.get("count_close", count_correct)
    overcount_ok = comp.get("overcount_ok", 1.0)
    alignment    = comp.get("prompt_alignment", 0.5)
    quality      = comp.get("image_quality", 0.5)
    uncertain    = comp.get("uncertain_frac", 0.0)

    gate = _smooth_gate(presence, low=0.35, high=0.80)

    pre_reward = (
        0.85 * presence
      + 0.10 * alignment
      + 0.05 * quality
    )

    post_reward = (
        0.50 * count_correct
      + 0.20 * count_close
      + 0.20 * presence
      + 0.07 * alignment
      + 0.03 * quality
    )

    reward = (1.0 - gate) * pre_reward + gate * post_reward

    penalty = 0.0
    if presence >= 0.55 and overcount_ok < 0.50:
        penalty = 0.10 * (0.50 - overcount_ok) / 0.50
        reward -= penalty

    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "gate":             gate,
        "pre_reward":       pre_reward,
        "post_reward":      post_reward,
        "penalty":          penalty,
        "uncertain_penalty": uncertain_penalty,
    }


# ---------------------------------------------------------------------------
# Contrastive rubric  (attribute + spatial)
# ---------------------------------------------------------------------------

def grpo_attr_contrastive_rubric_v1(comp: dict) -> tuple:
    """
    Rubric-prompted contrastive attribute reward.
    contrastive_attr = sigmoid((target_soft - swap_soft) / tau), computed in vlm_reward.py.
    """
    presence         = comp.get("object_presence", 0.5)
    contrastive_attr = comp.get("contrastive_attr", 0.5)
    attribute        = comp.get("attribute", 0.5)
    alignment        = comp.get("prompt_alignment", 0.5)
    quality          = comp.get("image_quality", 0.5)
    uncertain        = comp.get("uncertain_frac", 0.0)

    reward = (
        0.30 * presence
      + 0.40 * contrastive_attr
      + 0.15 * attribute
      + 0.08 * alignment
      + 0.07 * quality
    )
    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "contrastive_attr":  contrastive_attr,
        "uncertain_penalty": uncertain_penalty,
    }


def grpo_spatial_contrastive_rubric_v1(comp: dict, anchored: bool = False) -> tuple:
    """
    Rubric-prompted contrastive spatial reward.
    contrastive_relation = sigmoid((target_soft - opposite_soft) / tau), computed in vlm_reward.py.
    """
    presence             = comp.get("object_presence", 0.5)
    contrastive_relation = comp.get("contrastive_relation", 0.5)
    relation             = comp.get("relation", 0.5)
    separation           = comp.get("separation_clarity", 0.5)
    alignment            = comp.get("prompt_alignment", 0.5)
    quality              = comp.get("image_quality", 0.5)
    uncertain            = comp.get("uncertain_frac", 0.0)

    reward = (
        0.35 * presence
      + 0.35 * contrastive_relation
      + 0.10 * relation
      + 0.10 * separation
      + 0.05 * alignment
      + 0.05 * quality
    )
    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "contrastive_relation": contrastive_relation,
        "uncertain_penalty":    uncertain_penalty,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FORMULA_MAP = {
    "grpo_attr_presence_gated_v2":        grpo_attr_presence_gated_v2,
    "grpo_spatial_presence_gated_v2":     grpo_spatial_presence_gated_v2,
    "grpo_counting_presence_gated_v2":    grpo_counting_presence_gated_v2,
}

_CONTRASTIVE_FORMULA_MAP = {
    "grpo_attr_contrastive_rubric_v1":    grpo_attr_contrastive_rubric_v1,
    "grpo_spatial_contrastive_rubric_v1": grpo_spatial_contrastive_rubric_v1,
}

GATED_V2_MODES         = frozenset(_FORMULA_MAP.keys())
CONTRASTIVE_RUBRIC_MODES = frozenset(_CONTRASTIVE_FORMULA_MAP.keys())


def apply_gated_v2(mode: str, comp: dict, bucket: str = "") -> tuple:
    fn = _FORMULA_MAP[mode]
    if mode == "grpo_spatial_presence_gated_v2":
        anchored = "anchored" in bucket
        return fn(comp, anchored=anchored)
    return fn(comp)


def apply_contrastive_rubric(mode: str, comp: dict, bucket: str = "") -> tuple:
    fn = _CONTRASTIVE_FORMULA_MAP[mode]
    if mode == "grpo_spatial_contrastive_rubric_v1":
        anchored = "anchored" in bucket
        return fn(comp, anchored=anchored)
    return fn(comp)
