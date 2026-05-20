"""
Utility to generate toy bucket JSONL files for smoke tests and early experiments.
Run: python -m adaptive_curriculum.data.build_buckets --out-dir /path/to/data
"""
import json
import argparse
from pathlib import Path


TOY_DATA = {
    "attribute_binding": {
        "train": [
            {
                "id": "attr_train_{i:06d}",
                "bucket": "attribute_binding",
                "caption": "A {color} {obj}.",
                "eval_questions": [
                    {"question": "Is there a {color} {obj}?", "answer": "yes", "weight": 1.0},
                ],
            }
        ],
        "templates": [
            ("red", "cube"), ("blue", "sphere"), ("green", "cylinder"),
            ("yellow", "pyramid"), ("purple", "cone"), ("orange", "box"),
            ("pink", "ball"), ("white", "ring"), ("black", "disk"), ("brown", "block"),
        ],
    },
    "counting": {
        "templates": [
            (1, "dog"), (2, "cats"), (3, "birds"), (4, "fish"), (5, "apples"),
            (2, "oranges"), (3, "cubes"), (1, "horse"), (4, "stars"), (2, "trees"),
        ],
    },
    "spatial_relations": {
        "templates": [
            ("red cube", "left of", "blue sphere"),
            ("green cylinder", "above", "yellow box"),
            ("cat", "to the right of", "dog"),
            ("apple", "below", "orange"),
            ("book", "next to", "lamp"),
            ("car", "in front of", "building"),
            ("bird", "above", "tree"),
            ("chair", "beside", "table"),
            ("ball", "behind", "wall"),
            ("cup", "on top of", "saucer"),
        ],
    },
    "complex_composition": {
        "templates": [
            ("red cube", "blue sphere", "left of"),
            ("green cylinder", "yellow pyramid", "above"),
            ("black dog", "white cat", "right of"),
            ("big apple", "small orange", "next to"),
            ("tall tree", "short bush", "behind"),
            ("open book", "closed laptop", "beside"),
            ("hot coffee", "cold water", "in front of"),
            ("round ball", "square box", "above"),
            ("old car", "new bike", "to the left of"),
            ("flying bird", "swimming fish", "near"),
        ],
    },
}


def _make_attribute_binding_items(n: int = 10) -> list:
    templates = TOY_DATA["attribute_binding"]["templates"]
    items = []
    for i in range(n):
        color, obj = templates[i % len(templates)]
        items.append({
            "id": f"attr_{i:06d}",
            "bucket": "attribute_binding",
            "prompt": f"A {color} {obj}.",
            "eval_questions": [
                {"question": f"Is there a {color} {obj}?", "answer": "yes", "weight": 1.5},
                {"question": f"Is there a {obj}?", "answer": "yes", "weight": 1.0},
                {"question": f"What color is the {obj}?", "answer": color, "weight": 1.0},
            ],
        })
    return items


def _make_counting_items(n: int = 10) -> list:
    templates = TOY_DATA["counting"]["templates"]
    num_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    items = []
    for i in range(n):
        count, obj = templates[i % len(templates)]
        nw = num_words.get(count, str(count))
        items.append({
            "id": f"count_{i:06d}",
            "bucket": "counting",
            "prompt": f"{nw.capitalize()} {obj}.",
            "eval_questions": [
                {"question": f"Are there exactly {count} {obj}?", "answer": "yes", "weight": 2.0},
                {"question": f"Is there at least one {obj}?", "answer": "yes", "weight": 1.0},
            ],
        })
    return items


def _make_spatial_items(n: int = 10) -> list:
    templates = TOY_DATA["spatial_relations"]["templates"]
    items = []
    for i in range(n):
        subj, rel, obj = templates[i % len(templates)]
        items.append({
            "id": f"spatial_{i:06d}",
            "bucket": "spatial_relations",
            "prompt": f"A {subj} is {rel} a {obj}.",
            "eval_questions": [
                {"question": f"Is there a {subj}?", "answer": "yes", "weight": 1.0},
                {"question": f"Is there a {obj}?", "answer": "yes", "weight": 1.0},
                {"question": f"Is the {subj} {rel} the {obj}?", "answer": "yes", "weight": 2.0},
            ],
        })
    return items


def _make_complex_items(n: int = 10) -> list:
    templates = TOY_DATA["complex_composition"]["templates"]
    items = []
    for i in range(n):
        subj, obj, rel = templates[i % len(templates)]
        items.append({
            "id": f"complex_{i:06d}",
            "bucket": "complex_composition",
            "prompt": f"A {subj} {rel} a {obj}.",
            "eval_questions": [
                {"question": f"Is there a {subj}?", "answer": "yes", "weight": 1.0},
                {"question": f"Is there a {obj}?", "answer": "yes", "weight": 1.0},
                {"question": f"Is the {subj} {rel} the {obj}?", "answer": "yes", "weight": 2.0},
            ],
        })
    return items


BUCKET_BUILDERS = {
    "attribute_binding": _make_attribute_binding_items,
    "counting": _make_counting_items,
    "spatial_relations": _make_spatial_items,
    "complex_composition": _make_complex_items,
}


def build_toy_data(out_dir: str, n_train: int = 20, n_val: int = 10):
    root = Path(out_dir) / "buckets"
    for bucket, builder in BUCKET_BUILDERS.items():
        bucket_dir = root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        all_items = builder(n_train + n_val)
        train_items = all_items[:n_train]
        val_items = all_items[n_train : n_train + n_val]

        for split, items in [("train", train_items), ("val", val_items)]:
            path = bucket_dir / f"{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            print(f"  wrote {len(items)} items -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=10)
    args = parser.parse_args()

    print(f"Building toy bucket data in {args.out_dir}")
    build_toy_data(args.out_dir, args.n_train, args.n_val)
    print("Done.")
