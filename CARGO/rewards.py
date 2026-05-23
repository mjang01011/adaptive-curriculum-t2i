"""
CARGO v2 reward formulas (self-contained, no adaptive_curriculum imports).

grpo_attr_contrastive_rubric_v2
    Stronger attribute_subscore weight + explicit target_binding term vs v1.

grpo_spatial_contrastive_rubric_v2
    Triple gate: relation_effective = contrastive_relation * object_presence * distinct_object_score.
    Prevents reward hacking where shadows/patches are scored as correct spatial relations.
"""
import math


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# v2 Attribute binding
# ---------------------------------------------------------------------------

def grpo_attr_contrastive_rubric_v2(comp: dict) -> tuple:
    """
    V2 attribute reward:
      0.25 * presence + 0.30 * contrastive_attr + 0.25 * attribute_subscore
      + 0.10 * target_binding + 0.05 * alignment + 0.05 * quality

    Changes from v1 (0.30/0.40/0.15/0/0.08/0.07):
      - attribute_sub bumped 0.15 → 0.25 (more object-level signal)
      - target_binding added explicitly (0.10)
      - presence reduced 0.30 → 0.25
    """
    presence       = comp.get("object_presence", 0.5)
    contrastive    = comp.get("contrastive_attr", 0.5)
    attribute_sub  = comp.get("attribute_subscore", comp.get("attribute", 0.5))
    target_binding = comp.get("target_binding", 0.5)
    alignment      = comp.get("prompt_alignment", 0.5)
    quality        = comp.get("image_quality", 0.5)
    uncertain      = comp.get("uncertain_frac", 0.0)

    reward = (
        0.25 * presence
      + 0.30 * contrastive
      + 0.25 * attribute_sub
      + 0.10 * target_binding
      + 0.05 * alignment
      + 0.05 * quality
    )
    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "contrastive_attr":  contrastive,
        "attribute_sub":     attribute_sub,
        "target_binding":    target_binding,
        "uncertain_penalty": uncertain_penalty,
    }


# ---------------------------------------------------------------------------
# v2 Spatial relations
# ---------------------------------------------------------------------------

def grpo_spatial_contrastive_rubric_v2(comp: dict, anchored: bool = False) -> tuple:
    """
    V2 spatial reward with triple gate:
      relation_effective = contrastive_relation * object_presence * distinct_object_score

    Distinct_object_score gates out images where the model produces a shadow, colour
    patch, or fused shape that superficially satisfies the spatial statement.

    Weights: 0.25*presence + 0.25*distinct + 0.30*relation_effective
             + 0.10*separation + 0.05*alignment + 0.05*quality
    """
    presence    = comp.get("object_presence", 0.5)
    contrastive = comp.get("contrastive_relation", 0.5)
    distinct    = comp.get("distinct_object_score", 0.5)
    separation  = comp.get("separation_clarity", 0.5)
    alignment   = comp.get("prompt_alignment", 0.5)
    quality     = comp.get("image_quality", 0.5)
    uncertain   = comp.get("uncertain_frac", 0.0)

    relation_effective = contrastive * presence * distinct

    reward = (
        0.25 * presence
      + 0.25 * distinct
      + 0.30 * relation_effective
      + 0.10 * separation
      + 0.05 * alignment
      + 0.05 * quality
    )
    uncertain_penalty = 0.05 * uncertain
    reward -= uncertain_penalty

    return _clamp(reward), {
        "contrastive_relation":  contrastive,
        "distinct_object_score": distinct,
        "relation_effective":    relation_effective,
        "uncertain_penalty":     uncertain_penalty,
    }


# ---------------------------------------------------------------------------
# Metadata / derived component keys
# These appear in component_scores but are not primary reward signals and
# should be excluded from per-component mask computation.
# ---------------------------------------------------------------------------

META_KEYS = frozenset({
    "uncertain_frac",
    "mean_logit_margin",
    # Attribute: raw negative evidence (superseded by contrastive_attr)
    "swapped_binding",
    # Attribute: per-object breakdown (already in attribute_subscore)
    "attr1_score",
    "attr2_score",
    # Spatial: raw directional scores (superseded by relation_effective triple gate)
    "target_relation",
    "opposite_relation",
    "contrastive_relation",      # raw sigmoid; fooled by patches/fusions — use relation_effective
})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

CARGO_MODES = frozenset([
    "grpo_attr_contrastive_rubric_v2",
    "grpo_spatial_contrastive_rubric_v2",
])

_CARGO_FORMULA_MAP = {
    "grpo_attr_contrastive_rubric_v2":    grpo_attr_contrastive_rubric_v2,
    "grpo_spatial_contrastive_rubric_v2": grpo_spatial_contrastive_rubric_v2,
}


def apply_cargo_reward(mode: str, comp: dict, bucket: str = "") -> tuple:
    fn = _CARGO_FORMULA_MAP[mode]
    if mode == "grpo_spatial_contrastive_rubric_v2":
        anchored = "anchored" in bucket
        return fn(comp, anchored=anchored)
    return fn(comp)
