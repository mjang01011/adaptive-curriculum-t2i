"""
Generate synthetic colored-shape spatial prompts for CV verifier feasibility test.

Usage:
  python scripts_verifier/generate_synthetic_shape_prompts.py \
    --out-dir data_synthetic_shapes \
    --num-train 200 \
    --num-val 50 \
    --seed 42
"""
import argparse
import json
import random
from pathlib import Path

COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
SHAPES = ["circle", "square", "triangle"]
RELATIONS = ["left_of", "right_of", "above", "below"]

TEMPLATES = {
    "left_of":  "a {c1} {s1} on the left and a {c2} {s2} on the right",
    "right_of": "a {c1} {s1} on the right and a {c2} {s2} on the left",
    "above":    "a {c1} {s1} above a {c2} {s2}",
    "below":    "a {c1} {s1} below a {c2} {s2}",
}

PLAIN_SUFFIX = ", plain white background"


def all_combos():
    combos = []
    for rel in RELATIONS:
        for c1 in COLORS:
            for c2 in COLORS:
                if c1 == c2:
                    continue
                for s1 in SHAPES:
                    for s2 in SHAPES:
                        combos.append((rel, c1, s1, c2, s2))
    return combos


def make_row(idx, rel, c1, s1, c2, s2):
    prompt = TEMPLATES[rel].format(c1=c1, s1=s1, c2=c2, s2=s2)
    if rel == "left_of":
        pos1, pos2 = "left", "right"
    elif rel == "right_of":
        pos1, pos2 = "right", "left"
    elif rel == "above":
        pos1, pos2 = "top", "bottom"
    else:
        pos1, pos2 = "bottom", "top"
    return {
        "id":       f"shape_spatial_{idx:06d}",
        "prompt":   prompt,
        "objects":  [
            {"color": c1, "shape": s1, "position": pos1},
            {"color": c2, "shape": s2, "position": pos2},
        ],
        "relation": rel,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows)} rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",   default="data_synthetic_shapes")
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-val",   type=int, default=50)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    combos = all_combos()
    rng.shuffle(combos)
    total = args.num_train + args.num_val
    if total > len(combos):
        raise ValueError(f"Requested {total} prompts but only {len(combos)} unique combos exist")

    selected = combos[:total]
    train_combos = selected[:args.num_train]
    val_combos   = selected[args.num_train:]

    train_rows = [make_row(i + 1,          *c) for i, c in enumerate(train_combos)]
    val_rows   = [make_row(args.num_train + i + 1, *c) for i, c in enumerate(val_combos)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl",   val_rows)
    print(f"[gen_prompts] Done. train={len(train_rows)}  val={len(val_rows)}")


if __name__ == "__main__":
    main()
