"""
Generate synthetic prompts for CV verifier feasibility test.

Two modes:
  --mode shapes  : raw geometric shapes (circle, square, triangle)
  --mode objects : simple iconic objects grounded to shape families (ball, book, kite)

Usage:
  python scripts_verifier/generate_synthetic_shape_prompts.py \
    --mode objects \
    --out-dir data_synthetic_shapes \
    --num-train 200 --num-val 50 --seed 42
"""
import argparse
import json
import random
from pathlib import Path

# ── shape mode ────────────────────────────────────────────────────────────────

COLORS  = ["red", "blue", "green", "yellow", "purple", "orange"]
SHAPES  = ["circle", "square", "triangle"]
SHAPE_RELATIONS  = ["left_of", "right_of", "above", "below"]

SHAPE_TEMPLATES = {
    "left_of":  "a {c1} {s1} on the left and a {c2} {s2} on the right",
    "right_of": "a {c1} {s1} on the right and a {c2} {s2} on the left",
    "above":    "a {c1} {s1} above a {c2} {s2}",
    "below":    "a {c1} {s1} below a {c2} {s2}",
}

# ── objects mode ──────────────────────────────────────────────────────────────

# Each object maps to a shape family for the verifier
OBJECT_FAMILY = {
    "ball":  "circle",
    "book":  "rect",
    "kite":  "triangle",
}
OBJECTS  = list(OBJECT_FAMILY.keys())
OBJECT_RELATIONS = ["left_of", "right_of"]

OBJECT_TEMPLATES = {
    "left_of":  "a {c1} {o1} on the left and a {c2} {o2} on the right",
    "right_of": "a {c1} {o1} on the right and a {c2} {o2} on the left",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _positions(rel):
    return {
        "left_of":  ("left",   "right"),
        "right_of": ("right",  "left"),
        "above":    ("top",    "bottom"),
        "below":    ("bottom", "top"),
    }[rel]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows)} rows → {path}")


# ── shapes mode ───────────────────────────────────────────────────────────────

def _shape_combos():
    combos = []
    for rel in SHAPE_RELATIONS:
        for c1 in COLORS:
            for c2 in COLORS:
                if c1 == c2:
                    continue
                for s1 in SHAPES:
                    for s2 in SHAPES:
                        combos.append((rel, c1, s1, c2, s2))
    return combos


def _shape_row(idx, rel, c1, s1, c2, s2):
    prompt = SHAPE_TEMPLATES[rel].format(c1=c1, s1=s1, c2=c2, s2=s2)
    pos1, pos2 = _positions(rel)
    return {
        "id":       f"shape_spatial_{idx:06d}",
        "prompt":   prompt,
        "objects":  [
            {"color": c1, "shape": s1, "family": s1, "position": pos1},
            {"color": c2, "shape": s2, "family": s2, "position": pos2},
        ],
        "relation": rel,
    }


# ── objects mode ──────────────────────────────────────────────────────────────

def _object_combos():
    combos = []
    for rel in OBJECT_RELATIONS:
        for c1 in COLORS:
            for c2 in COLORS:
                if c1 == c2:
                    continue
                for o1 in OBJECTS:
                    for o2 in OBJECTS:
                        combos.append((rel, c1, o1, c2, o2))
    return combos


def _object_row(idx, rel, c1, o1, c2, o2):
    prompt = OBJECT_TEMPLATES[rel].format(c1=c1, o1=o1, c2=c2, o2=o2)
    pos1, pos2 = _positions(rel)
    return {
        "id":       f"object_spatial_{idx:06d}",
        "prompt":   prompt,
        "objects":  [
            {"color": c1, "object": o1, "family": OBJECT_FAMILY[o1], "position": pos1},
            {"color": c2, "object": o2, "family": OBJECT_FAMILY[o2], "position": pos2},
        ],
        "relation": rel,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      default="objects", choices=["shapes", "objects"])
    parser.add_argument("--out-dir",   default="data_synthetic_shapes")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val",   type=int, default=50)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.mode == "objects":
        combos   = _object_combos()
        make_row = _object_row
    else:
        combos   = _shape_combos()
        make_row = _shape_row

    rng.shuffle(combos)
    total = args.num_train + args.num_val
    if total > len(combos):
        raise ValueError(f"Requested {total} prompts but only {len(combos)} unique combos exist")

    train_combos = combos[:args.num_train]
    val_combos   = combos[args.num_train:total]

    train_rows = [make_row(i + 1,                  *c) for i, c in enumerate(train_combos)]
    val_rows   = [make_row(args.num_train + i + 1, *c) for i, c in enumerate(val_combos)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl",   val_rows)
    print(f"[gen_prompts] mode={args.mode}  train={len(train_rows)}  val={len(val_rows)}")


if __name__ == "__main__":
    main()
