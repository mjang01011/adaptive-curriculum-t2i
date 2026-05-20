"""
BucketDataset: loads train/val JSONL per bucket, samples training batches.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from adaptive_curriculum.data.schemas import BucketItem


class BucketDataset:
    def __init__(self, bucket_name: str, train_items: List[BucketItem], val_items: List[BucketItem]):
        self.name = bucket_name
        self.train_items = train_items
        self.val_items = val_items

    def sample_train_batch(self, batch_size: int) -> List[BucketItem]:
        if not self.train_items:
            return []
        return random.choices(self.train_items, k=batch_size)

    def __len__(self):
        return len(self.train_items)


def _load_jsonl(path: Path) -> List[BucketItem]:
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))
    return items


def load_bucket_datasets(
    data_root: str,
    bucket_names: List[str],
    train_file: str = "{bucket}_train_500.jsonl",
    val_file: str = "{bucket}_val_20.jsonl",
    max_val_prompts: Optional[int] = None,
) -> Dict[str, BucketDataset]:
    root = Path(data_root)
    datasets: Dict[str, BucketDataset] = {}
    for name in bucket_names:
        train_path = root / name / train_file.format(bucket=name)
        val_path = root / name / val_file.format(bucket=name)
        train_items = _load_jsonl(train_path)
        val_items = _load_jsonl(val_path)
        if max_val_prompts is not None:
            val_items = val_items[:max_val_prompts]
        datasets[name] = BucketDataset(name, train_items, val_items)
    return datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["attribute_binding", "counting", "spatial_relations", "complex_composition"],
    )
    args = parser.parse_args()

    datasets = load_bucket_datasets(args.data_root, args.buckets)
    for name, ds in datasets.items():
        print(f"[{name}] train={len(ds.train_items)}  val={len(ds.val_items)}")
        if ds.val_items:
            sample = ds.val_items[0]
            print(f"  sample id={sample.id}  text={sample.text!r}")
        if ds.train_items:
            batch = ds.sample_train_batch(2)
            print(f"  train batch[0] id={batch[0].id}")
