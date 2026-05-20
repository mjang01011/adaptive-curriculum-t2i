"""
Validate the new 500/20 target-only bucket dataset.

Usage:
    python -m adaptive_curriculum.data.validate_buckets \
        --data-root /vol/data/buckets \
        --expected-train-size 500 \
        --expected-val-size 20 \
        --question-field target_questions
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

BUCKET_NAMES = ["attribute_binding", "counting", "spatial_relations", "complex_composition"]

# Expected question counts per bucket (target_questions only)
EXPECTED_Q_COUNT = {
    "attribute_binding": 4,
    "counting": 2,
    "spatial_relations": 2,
    "complex_composition": 4,
}


def _load_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def validate_dataset(
    data_root: str,
    expected_train_size: int = 500,
    expected_val_size: int = 20,
    question_field: str = "target_questions",
    train_file_pattern: str = "{bucket}_train_{n}.jsonl",
    val_file_pattern: str = "{bucket}_val_{n}.jsonl",
) -> bool:
    root = Path(data_root)
    errors: List[str] = []
    warnings: List[str] = []

    train_pattern = train_file_pattern.format(bucket="{bucket}", n=expected_train_size)
    val_pattern = val_file_pattern.format(bucket="{bucket}", n=expected_val_size)

    for bucket in BUCKET_NAMES:
        bucket_dir = root / bucket
        if not bucket_dir.exists():
            errors.append(f"[{bucket}] directory not found: {bucket_dir}")
            continue

        train_path = bucket_dir / train_pattern.format(bucket=bucket)
        val_path = bucket_dir / val_pattern.format(bucket=bucket)

        # --- train file ---
        if not train_path.exists():
            errors.append(f"[{bucket}] train file not found: {train_path}")
        else:
            train_items = _load_jsonl(train_path)
            if len(train_items) != expected_train_size:
                errors.append(
                    f"[{bucket}] train size {len(train_items)} != expected {expected_train_size}"
                )
            else:
                print(f"  [OK] {bucket} train: {len(train_items)} items")

            missing_q = [i for i, d in enumerate(train_items) if not d.get(question_field)]
            if missing_q:
                errors.append(
                    f"[{bucket}] train: {len(missing_q)} items missing '{question_field}'"
                )

            wrong_q_count = [
                (i, len(d[question_field]))
                for i, d in enumerate(train_items)
                if d.get(question_field) and len(d[question_field]) != EXPECTED_Q_COUNT[bucket]
            ]
            if wrong_q_count:
                sample = wrong_q_count[:3]
                errors.append(
                    f"[{bucket}] train: {len(wrong_q_count)} items with wrong question count "
                    f"(expected {EXPECTED_Q_COUNT[bucket]}), e.g. indices {sample}"
                )

            # check answer values
            bad_answers = []
            for i, d in enumerate(train_items):
                for q in d.get(question_field, []):
                    if q.get("answer") not in ("yes", "no"):
                        bad_answers.append((i, q.get("answer")))
            if bad_answers:
                errors.append(
                    f"[{bucket}] train: {len(bad_answers)} questions with invalid answer "
                    f"(not yes/no), e.g. {bad_answers[:3]}"
                )

        # --- val file ---
        if not val_path.exists():
            errors.append(f"[{bucket}] val file not found: {val_path}")
        else:
            val_items = _load_jsonl(val_path)
            if len(val_items) != expected_val_size:
                errors.append(
                    f"[{bucket}] val size {len(val_items)} != expected {expected_val_size}"
                )
            else:
                print(f"  [OK] {bucket} val:   {len(val_items)} items")

            missing_q = [i for i, d in enumerate(val_items) if not d.get(question_field)]
            if missing_q:
                errors.append(
                    f"[{bucket}] val: {len(missing_q)} items missing '{question_field}'"
                )

            wrong_q_count = [
                (i, len(d[question_field]))
                for i, d in enumerate(val_items)
                if d.get(question_field) and len(d[question_field]) != EXPECTED_Q_COUNT[bucket]
            ]
            if wrong_q_count:
                errors.append(
                    f"[{bucket}] val: {len(wrong_q_count)} items with wrong question count "
                    f"(expected {EXPECTED_Q_COUNT[bucket]})"
                )

        # warn if legacy eval_questions field present
        if train_path.exists():
            first = _load_jsonl(train_path)[:1]
            if first and "eval_questions" in first[0]:
                warnings.append(
                    f"[{bucket}] legacy 'eval_questions' key found — "
                    "schema uses 'target_questions' now"
                )

    print()
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print(f"\nValidation FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  [FAIL] {e}")
        return False

    print("Validation PASSED. All checks OK.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate adaptive curriculum bucket dataset")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of bucket data")
    parser.add_argument("--expected-train-size", type=int, default=500)
    parser.add_argument("--expected-val-size", type=int, default=20)
    parser.add_argument("--question-field", type=str, default="target_questions")
    args = parser.parse_args()

    print(f"Validating dataset at: {args.data_root}")
    print(f"Expected: {args.expected_train_size} train / {args.expected_val_size} val per bucket")
    print(f"Question field: '{args.question_field}'")
    print()

    ok = validate_dataset(
        data_root=args.data_root,
        expected_train_size=args.expected_train_size,
        expected_val_size=args.expected_val_size,
        question_field=args.question_field,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
