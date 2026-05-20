"""
Load and access pre-extracted T5 embedding caches.

Cache format (one .pt per bucket):
  {
    "embeddings": {item_id: tensor(1, seq_len, 2048)},
    "meta":       {item_id: {"split": "train"|"val", "prompt": str}},
  }
"""
from pathlib import Path
from typing import Dict, List, Optional

import torch


class T5EmbeddingCache:
    def __init__(self, cache_dir: str, bucket_names: List[str], device: str = "cpu"):
        self.cache_dir = Path(cache_dir)
        self.device = device
        # {bucket: {item_id: tensor(1, seq_len, 2048)}}
        self._embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        self._meta: Dict[str, Dict[str, dict]] = {}
        self._load(bucket_names)

    def _load(self, bucket_names: List[str]):
        missing = []
        for bucket in bucket_names:
            path = self.cache_dir / f"{bucket}.pt"
            if not path.exists():
                missing.append(str(path))
                continue
            data = torch.load(str(path), map_location=self.device, weights_only=False)
            self._embeddings[bucket] = data["embeddings"]
            self._meta[bucket] = data.get("meta", {})
            n = len(data["embeddings"])
            print(f"[T5Cache] loaded {bucket}: {n} embeddings from {path.name}")

        if missing:
            raise FileNotFoundError(
                f"T5 cache files not found:\n  " + "\n  ".join(missing) +
                f"\nRun: python -m adaptive_curriculum.data.extract_t5_embeddings "
                f"--data-root <data> --out-dir {self.cache_dir} --t5-path <t5_ckpt>"
            )

    def get(self, bucket: str, item_id: str) -> torch.Tensor:
        """Returns tensor(1, seq_len, 2048)."""
        return self._embeddings[bucket][item_id]

    def bucket_embeddings(self, bucket: str) -> Dict[str, torch.Tensor]:
        """Returns the full {item_id: tensor} dict for a bucket."""
        return self._embeddings[bucket]

    def split_embeddings(self, bucket: str, split: str) -> Dict[str, torch.Tensor]:
        """Returns only train or val embeddings for a bucket."""
        meta = self._meta.get(bucket, {})
        return {
            k: v for k, v in self._embeddings[bucket].items()
            if meta.get(k, {}).get("split") == split
        }

    def available_buckets(self) -> List[str]:
        return list(self._embeddings.keys())


def load_t5_cache(
    cache_dir: Optional[str],
    bucket_names: List[str],
    device: str = "cpu",
) -> Optional[T5EmbeddingCache]:
    """Returns None if cache_dir is not set, so callers can fall back to live T5."""
    if not cache_dir:
        return None
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None
    # check at least one cache file exists
    if not any(cache_path.glob("*.pt")):
        return None
    return T5EmbeddingCache(cache_dir, bucket_names, device=device)
