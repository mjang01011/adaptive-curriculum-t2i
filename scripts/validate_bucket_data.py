"""
Validate the new grounded v2 bucket data structure.

Checks per bucket:
  - train split has exactly 500 items
  - val split has exactly 20 items
  - every item has non-empty target_questions
  - every item has non-empty grpo_reward_questions (for anchored buckets)
  - grpo_reward_questions weights sum to ~1.0 (±0.01)
  - no duplicate IDs within a split

Usage:
    python scripts/validate_bucket_data.py --data-root data/
    python scripts/validate_bucket_data.py --data-root data/ --buckets spatial_relations_anchored
"""
import argparse
import json
import sys
from pathlib import Path


BUCKETS_WITH_GRPO_QS = {"spatial_relations_anchored"}

EXPECTED_TRAIN = 500
EXPECTED_VAL = 20


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append((lineno, json.loads(line)))
            except json.JSONDecodeError as e:
                print(f"  [ERROR] {path.name}:{lineno} JSON decode error: {e}")
    return items


def validate_split(path: Path, split: str, expected_count: int, bucket: str) -> int:
    errors = 0
    if not path.exists():
        print(f"  [ERROR] {split} file not found: {path}")
        return 1

    rows = load_jsonl(path)
    n = len(rows)
    if n != expected_count:
        print(f"  [ERROR] {split}: expected {expected_count} items, got {n}")
        errors += 1
    else:
        print(f"  [OK]    {split}: {n} items")

    ids_seen = set()
    for lineno, item in rows:
        item_id = item.get("id", f"<missing_id_line_{lineno}>")

        # Duplicate ID check
        if item_id in ids_seen:
            print(f"  [ERROR] {split}:{lineno} duplicate id={item_id!r}")
            errors += 1
        ids_seen.add(item_id)

        # target_questions
        tqs = item.get("target_questions") or []
        if not tqs:
            print(f"  [ERROR] {split}:{lineno} id={item_id} has no target_questions")
            errors += 1

        # grpo_reward_questions (only required for anchored buckets)
        if bucket in BUCKETS_WITH_GRPO_QS:
            grpo_qs = item.get("grpo_reward_questions") or []
            if not grpo_qs:
                print(f"  [ERROR] {split}:{lineno} id={item_id} has no grpo_reward_questions")
                errors += 1
            else:
                weight_sum = sum(q.get("weight", 0.0) for q in grpo_qs)
                if abs(weight_sum - 1.0) > 0.01:
                    print(
                        f"  [ERROR] {split}:{lineno} id={item_id} "
                        f"grpo_reward_questions weights sum to {weight_sum:.4f} (expected 1.0)"
                    )
                    errors += 1

    return errors


def validate_bucket(data_root: Path, bucket: str) -> int:
    print(f"\n=== {bucket} ===")
    errors = 0

    train_path = data_root / bucket / f"{bucket}_train_500.jsonl"
    val_path = data_root / bucket / f"{bucket}_val_20.jsonl"

    errors += validate_split(train_path, "train", EXPECTED_TRAIN, bucket)
    errors += validate_split(val_path, "val", EXPECTED_VAL, bucket)
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate bucket data for new grounded v2 schema")
    parser.add_argument("--data-root", type=str, default="data", help="Path to data/ directory")
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["attribute_binding", "counting", "spatial_relations_anchored", "complex_composition"],
        help="Buckets to validate",
    )
    parser.add_argument("--expected-train-size", type=int, default=EXPECTED_TRAIN)
    parser.add_argument("--expected-val-size", type=int, default=EXPECTED_VAL)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] data-root not found: {data_root}")
        sys.exit(1)

    global EXPECTED_TRAIN, EXPECTED_VAL
    EXPECTED_TRAIN = args.expected_train_size
    EXPECTED_VAL = args.expected_val_size

    total_errors = 0
    for bucket in args.buckets:
        total_errors += validate_bucket(data_root, bucket)

    print()
    if total_errors == 0:
        print(f"All {len(args.buckets)} buckets passed validation.")
        sys.exit(0)
    else:
        print(f"Validation FAILED with {total_errors} error(s) across {len(args.buckets)} bucket(s).")
        sys.exit(1)


if __name__ == "__main__":
    main()
