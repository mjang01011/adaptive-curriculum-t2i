"""
Build an expanded validation JSONL that mixes easy, medium, and hard
compositional binding examples.

Difficulty levels
-----------------
easy   — single common attribute, objects rarely confused
medium — two attributes, mild swap risk
hard   — color/attribute swap, spatial binding, size+color combos

Usage
-----
  python scripts_data/build_val_set.py \
    --existing data/attribute_binding/attribute_binding_val_20.jsonl \
    --output   data/attribute_binding/val_mixed_60.jsonl
"""
import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Example definitions
# ---------------------------------------------------------------------------

def _make_attr(id_, bucket, prompt, obj1, attr1, obj2, attr2, difficulty):
    """Two-object attribute binding example (material, color, or pattern)."""
    tq = [
        {"type": "attribute",  "question": f"Is the {obj1} {attr1}?",  "answer": "yes"},
        {"type": "attribute",  "question": f"Is the {obj2} {attr2}?",  "answer": "yes"},
        {"type": "anti_swap",  "question": f"Is the {obj1} {attr2}?",  "answer": "no"},
        {"type": "anti_swap",  "question": f"Is the {obj2} {attr1}?",  "answer": "no"},
    ]
    gq = [
        {"type": "object_presence", "question": f"Is the {obj1} clearly visible?", "answer": "yes", "weight": 0.1},
        {"type": "object_presence", "question": f"Is the {obj2} clearly visible?", "answer": "yes", "weight": 0.1},
        {"type": "attribute",  "question": f"Is the {obj1} {attr1}?",  "answer": "yes", "weight": 0.2},
        {"type": "attribute",  "question": f"Is the {obj2} {attr2}?",  "answer": "yes", "weight": 0.2},
        {"type": "anti_swap",  "question": f"Is the {obj1} {attr2}?",  "answer": "no",  "weight": 0.1},
        {"type": "anti_swap",  "question": f"Is the {obj2} {attr1}?",  "answer": "no",  "weight": 0.1},
        {"type": "image_quality",   "question": "Is the image clear and mostly free of severe visual artifacts?", "answer": "yes", "weight": 0.1},
        {"type": "prompt_alignment","question": "Does the image generally match the prompt?", "answer": "yes", "weight": 0.1},
    ]
    return _wrap(id_, bucket, prompt, tq, gq, difficulty)


def _make_spatial_color(id_, prompt, obj1, color1, obj2, color2, relation, difficulty):
    """Spatial + color: obj1(color1) RELATION obj2(color2)."""
    tq = [
        {"type": "attribute",  "question": f"Is the {obj1} {color1}?",  "answer": "yes"},
        {"type": "attribute",  "question": f"Is the {obj2} {color2}?",  "answer": "yes"},
        {"type": "anti_swap",  "question": f"Is the {obj1} {color2}?",  "answer": "no"},
        {"type": "anti_swap",  "question": f"Is the {obj2} {color1}?",  "answer": "no"},
        {"type": "relation",   "question": f"Is the {obj1} {relation} the {obj2}?", "answer": "yes"},
    ]
    gq = [
        {"type": "object_presence", "question": f"Is the {obj1} clearly visible?", "answer": "yes", "weight": 0.1},
        {"type": "object_presence", "question": f"Is the {obj2} clearly visible?", "answer": "yes", "weight": 0.1},
        {"type": "attribute",  "question": f"Is the {obj1} {color1}?", "answer": "yes", "weight": 0.15},
        {"type": "attribute",  "question": f"Is the {obj2} {color2}?", "answer": "yes", "weight": 0.15},
        {"type": "anti_swap",  "question": f"Is the {obj1} {color2}?", "answer": "no",  "weight": 0.1},
        {"type": "anti_swap",  "question": f"Is the {obj2} {color1}?", "answer": "no",  "weight": 0.1},
        {"type": "relation",   "question": f"Is the {obj1} {relation} the {obj2}?", "answer": "yes", "weight": 0.15},
        {"type": "image_quality",   "question": "Is the image clear and mostly free of severe visual artifacts?", "answer": "yes", "weight": 0.075},
        {"type": "prompt_alignment","question": "Does the image generally match the prompt?", "answer": "yes", "weight": 0.075},
    ]
    n_tq = len(tq)
    rule  = _rule(n_tq)
    return _wrap(id_, "spatial_color", prompt, tq, gq, difficulty, rule=rule)


def _make_size_color(id_, prompt, small_obj, small_color, large_obj, large_color, difficulty):
    """Size + color: a small COLOR1 OBJ1 and a large COLOR2 OBJ2."""
    tq = [
        {"type": "attribute",  "question": f"Is the {small_obj} {small_color}?", "answer": "yes"},
        {"type": "attribute",  "question": f"Is the {large_obj} {large_color}?", "answer": "yes"},
        {"type": "anti_swap",  "question": f"Is the {small_obj} {large_color}?", "answer": "no"},
        {"type": "anti_swap",  "question": f"Is the {large_obj} {small_color}?", "answer": "no"},
        {"type": "size",       "question": f"Does the {large_obj} appear larger than the {small_obj}?", "answer": "yes"},
    ]
    gq = [
        {"type": "object_presence", "question": f"Is the {small_obj} clearly visible?", "answer": "yes", "weight": 0.1},
        {"type": "object_presence", "question": f"Is the {large_obj} clearly visible?", "answer": "yes", "weight": 0.1},
        {"type": "attribute",  "question": f"Is the {small_obj} {small_color}?", "answer": "yes", "weight": 0.15},
        {"type": "attribute",  "question": f"Is the {large_obj} {large_color}?", "answer": "yes", "weight": 0.15},
        {"type": "anti_swap",  "question": f"Is the {small_obj} {large_color}?", "answer": "no",  "weight": 0.1},
        {"type": "anti_swap",  "question": f"Is the {large_obj} {small_color}?", "answer": "no",  "weight": 0.1},
        {"type": "size",       "question": f"Does the {large_obj} appear larger than the {small_obj}?", "answer": "yes", "weight": 0.15},
        {"type": "image_quality",   "question": "Is the image clear and mostly free of severe visual artifacts?", "answer": "yes", "weight": 0.075},
        {"type": "prompt_alignment","question": "Does the image generally match the prompt?", "answer": "yes", "weight": 0.075},
    ]
    n_tq = len(tq)
    rule  = _rule(n_tq)
    return _wrap(id_, "size_color", prompt, tq, gq, difficulty, rule=rule)


def _rule(n_tq):
    return {
        "type": "normalized_hard_target_accuracy",
        "formula": f"correct_target_questions / {n_tq}",
        "max_reward": 1.0, "min_reward": 0.0,
        "vlm_answer_space": ["yes", "no", "uncertain"],
        "uncertain_score": 0.0,
        "used_for": ["validation", "ucb", "final_eval"],
    }


_GRPO_RULE = {
    "type": "weighted_pseudo_soft_accuracy",
    "formula": "sum(weight_i * pseudo_soft_correctness_i) / sum(weight_i)",
    "max_reward": 1.0, "min_reward": 0.0,
    "vlm_answer_space": ["yes", "no", "uncertain"],
    "pseudo_soft_scoring": {
        "expected_yes": {"yes": 1.0, "uncertain": 0.5, "no": 0.0},
        "expected_no":  {"no": 1.0, "uncertain": 0.5, "yes": 0.0},
    },
    "used_for": ["grpo_training"],
}


def _wrap(id_, bucket, prompt, tq, gq, difficulty, rule=None):
    if rule is None:
        rule = _rule(len(tq))
    return {
        "id": id_,
        "bucket": bucket,
        "prompt": prompt,
        "difficulty": difficulty,
        "target_questions": tq,
        "grpo_reward_questions": gq,
        "target_reward_rule": rule,
        "grpo_reward_rule": _GRPO_RULE,
    }


# ---------------------------------------------------------------------------
# New examples
# ---------------------------------------------------------------------------

NEW_EXAMPLES = [
    # ── Easy: single common color/material, unambiguous objects ───────────────
    _make_attr("val_easy_001", "attribute_binding",
               "A red apple and a blue vase.",
               "apple", "red", "vase", "blue", "easy"),

    _make_attr("val_easy_002", "attribute_binding",
               "A yellow banana and a purple grape.",
               "banana", "yellow", "grape", "purple", "easy"),

    _make_attr("val_easy_003", "attribute_binding",
               "A wooden table and a metal chair.",
               "table", "wooden", "chair", "metal", "easy"),

    _make_attr("val_easy_004", "attribute_binding",
               "A green frog and an orange fish.",
               "frog", "green", "fish", "orange", "easy"),

    _make_attr("val_easy_005", "attribute_binding",
               "A white rabbit and a black cat.",
               "rabbit", "white", "cat", "black", "easy"),

    # ── Medium: two attributes, mild swap risk ────────────────────────────────
    _make_attr("val_med_001", "attribute_binding",
               "A red car and a blue truck.",
               "car", "red", "truck", "blue", "medium"),

    _make_attr("val_med_002", "attribute_binding",
               "A striped towel and a spotted pillow.",
               "towel", "striped", "pillow", "spotted", "medium"),

    _make_attr("val_med_003", "attribute_binding",
               "A glass bottle and a ceramic mug.",
               "bottle", "glass", "mug", "ceramic", "medium"),

    _make_attr("val_med_004", "attribute_binding",
               "A blue hat and a red scarf.",
               "hat", "blue", "scarf", "red", "medium"),

    _make_attr("val_med_005", "attribute_binding",
               "A green backpack and a yellow umbrella.",
               "backpack", "green", "umbrella", "yellow", "medium"),

    _make_attr("val_med_006", "attribute_binding",
               "A leather wallet and a plastic comb.",
               "wallet", "leather", "comb", "plastic", "medium"),

    _make_attr("val_med_007", "attribute_binding",
               "A pink flamingo and a grey elephant.",
               "flamingo", "pink", "elephant", "grey", "medium"),

    # ── Hard: color swap confusable pairs ─────────────────────────────────────
    _make_attr("val_hard_color_001", "attribute_binding",
               "A red cube and a blue sphere.",
               "cube", "red", "sphere", "blue", "hard"),

    _make_attr("val_hard_color_002", "attribute_binding",
               "A blue car and a red bus.",
               "car", "blue", "bus", "red", "hard"),

    _make_attr("val_hard_color_003", "attribute_binding",
               "A red rose and a blue lily.",
               "rose", "red", "lily", "blue", "hard"),

    _make_attr("val_hard_color_004", "attribute_binding",
               "A green apple and a red pear.",
               "apple", "green", "pear", "red", "hard"),

    _make_attr("val_hard_color_005", "attribute_binding",
               "A black horse and a white sheep.",
               "horse", "black", "sheep", "white", "hard"),

    _make_attr("val_hard_color_006", "attribute_binding",
               "A yellow cup and a brown bowl.",
               "cup", "yellow", "bowl", "brown", "hard"),

    # ── Hard: spatial + color ─────────────────────────────────────────────────
    _make_spatial_color("val_hard_spat_001",
                        "A red cube on top of a blue sphere.",
                        "cube", "red", "sphere", "blue", "on top of", "hard"),

    _make_spatial_color("val_hard_spat_002",
                        "A green apple to the left of a red orange on a wooden table.",
                        "apple", "green", "orange", "red", "to the left of", "hard"),

    _make_spatial_color("val_hard_spat_003",
                        "A blue vase in front of a yellow painting.",
                        "vase", "blue", "painting", "yellow", "in front of", "hard"),

    _make_spatial_color("val_hard_spat_004",
                        "A red ball behind a blue box.",
                        "ball", "red", "box", "blue", "behind", "hard"),

    _make_spatial_color("val_hard_spat_005",
                        "A white candle next to a black lantern.",
                        "candle", "white", "lantern", "black", "next to", "hard"),

    # ── Hard: size + color ────────────────────────────────────────────────────
    _make_size_color("val_hard_size_001",
                     "A small white cat sitting next to a large black dog.",
                     "cat", "white", "dog", "black", "hard"),

    _make_size_color("val_hard_size_002",
                     "A small red ball next to a large blue cube.",
                     "ball", "red", "cube", "blue", "hard"),

    _make_size_color("val_hard_size_003",
                     "A small yellow flower beside a large purple vase.",
                     "flower", "yellow", "vase", "purple", "hard"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--existing", default=None, help="Existing val JSONL to prepend")
    p.add_argument("--output",   required=True)
    args = p.parse_args()

    rows = []
    if args.existing and Path(args.existing).exists():
        with open(args.existing) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line.strip()))
        print(f"Loaded {len(rows)} existing examples from {args.existing}")

    rows.extend(NEW_EXAMPLES)
    print(f"Added {len(NEW_EXAMPLES)} new examples")

    # Deduplicate by id
    seen, deduped = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in deduped:
            f.write(json.dumps(r) + "\n")

    by_diff = {}
    for r in deduped:
        d = r.get("difficulty", "existing")
        by_diff[d] = by_diff.get(d, 0) + 1

    print(f"Wrote {len(deduped)} total examples → {out}")
    for d, n in sorted(by_diff.items()):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()
