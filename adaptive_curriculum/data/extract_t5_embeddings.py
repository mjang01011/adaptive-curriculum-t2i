"""
Extract T5 embeddings for all bucket prompts and save as one .pt cache file per bucket.

Format: {bucket}.pt  ->  dict mapping item_id -> tensor(1, seq_len, 2048)
        seq_len is the actual token length (not padded), matching LlamaGen's native .npy shape.

Usage (local, CPU):
  python -m adaptive_curriculum.data.extract_t5_embeddings \
      --data-root C:/Users/desti/Desktop/llamagen-cl/data \
      --out-dir   C:/Users/desti/Desktop/llamagen-cl/data/t5_cache \
      --repo-root C:/Users/desti/Desktop/llamagen-cl/LlamaGen \
      --t5-path   /path/to/pretrained_models/t5-ckpt \
      --device cpu

Usage (Modal / GPU):
  python -m adaptive_curriculum.data.extract_t5_embeddings \
      --data-root /vol/data \
      --out-dir   /vol/data/t5_cache \
      --repo-root /vol/repo/LlamaGen \
      --t5-path   /vol/pretrained_models/t5-ckpt \
      --batch-size 64
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch


BUCKET_SPLITS = {
    "attribute_binding":       {"train": "attribute_binding_train_500.jsonl",            "val": "attribute_binding_val_20.jsonl"},
    "counting":                {"train": "counting_train_500.jsonl",                     "val": "counting_val_20.jsonl"},
    "spatial_relations_anchored": {"train": "spatial_relations_anchored_train_500.jsonl","val": "spatial_relations_anchored_val_20.jsonl"},
    "complex_composition":     {"train": "complex_composition_train_500.jsonl",          "val": "complex_composition_val_20.jsonl"},
}


def _load_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _encode_batch(
    t5_model,
    prompts: List[str],
    ids: List[str],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Returns {id: tensor(1, seq_len, 2048)} with actual token lengths (unpadded)."""
    with torch.no_grad():
        caption_embs, emb_masks = t5_model.get_text_embeddings(prompts)
    # caption_embs: (B, max_len, 2048)  emb_masks: (B, max_len)
    result = {}
    for item_id, emb, mask in zip(ids, caption_embs, emb_masks):
        valid_len = int(mask.sum().item())
        # store only the real tokens — matches LlamaGen .npy convention
        result[item_id] = emb[:valid_len].unsqueeze(0).cpu()   # (1, valid_len, 2048)
    return result


def extract_all(
    data_root: str,
    out_dir: str,
    repo_root: str,
    t5_path: str,
    t5_model_type: str = "flan-t5-xl",
    t5_feature_max_len: int = 120,
    batch_size: int = 32,
    device: str = None,
    precision: str = "bf16",
    buckets: List[str] = None,
):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from language.t5 import T5Embedder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    print(f"Loading T5 ({t5_model_type}) on {device}...")
    t5 = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=t5_path,
        dir_or_name=t5_model_type,
        torch_dtype=dtype,
        model_max_length=t5_feature_max_len,
    )
    print("T5 loaded.\n")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_root)

    bucket_list = buckets or list(BUCKET_SPLITS.keys())

    for bucket in bucket_list:
        splits = BUCKET_SPLITS[bucket]
        bucket_dir = data_path / bucket

        # collect all items across both splits — one cache covers train+val
        all_items: List[dict] = []
        for split_name, filename in splits.items():
            fpath = bucket_dir / filename
            if not fpath.exists():
                print(f"  WARNING: {fpath} not found, skipping")
                continue
            items = _load_jsonl(fpath)
            for item in items:
                item["_split"] = split_name
            all_items.extend(items)

        if not all_items:
            print(f"[{bucket}] No items found, skipping.")
            continue

        print(f"[{bucket}] {len(all_items)} prompts (encoding in batches of {batch_size})...")

        embeddings: Dict[str, torch.Tensor] = {}
        for start in range(0, len(all_items), batch_size):
            batch = all_items[start : start + batch_size]
            prompts = [item["prompt"] for item in batch]
            ids = [item["id"] for item in batch]
            batch_embs = _encode_batch(t5, prompts, ids, device)
            embeddings.update(batch_embs)
            print(f"  {min(start + batch_size, len(all_items))}/{len(all_items)}", end="\r")

        # save: one .pt per bucket
        # structure: {"embeddings": {id: tensor(1, seq_len, 2048)}, "meta": {id: {split, prompt}}}
        meta = {item["id"]: {"split": item["_split"], "prompt": item["prompt"]} for item in all_items}
        cache = {"embeddings": embeddings, "meta": meta}

        out_file = out_path / f"{bucket}.pt"
        torch.save(cache, str(out_file))

        # sanity check
        sample_id = list(embeddings.keys())[0]
        sample_shape = embeddings[sample_id].shape
        print(f"\n  saved {len(embeddings)} embeddings -> {out_file}")
        print(f"  sample shape: {sample_shape}  (1, seq_len, 2048)")

    print(f"\nDone. Cache files written to: {out_dir}")
    _print_summary(out_dir)


def _print_summary(out_dir: str):
    out_path = Path(out_dir)
    print("\n--- Cache Summary ---")
    for f in sorted(out_path.glob("*.pt")):
        cache = torch.load(str(f), map_location="cpu", weights_only=False)
        embs = cache["embeddings"]
        meta = cache["meta"]
        n_train = sum(1 for v in meta.values() if v["split"] == "train")
        n_val   = sum(1 for v in meta.values() if v["split"] == "val")
        shapes  = set(tuple(v.shape) for v in embs.values())
        print(f"  {f.name:40s}  total={len(embs)}  train={n_train}  val={n_val}  shapes={shapes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",   type=str, required=True)
    parser.add_argument("--out-dir",     type=str, required=True)
    parser.add_argument("--repo-root",   type=str, default=None)
    parser.add_argument("--t5-path",     type=str, required=True)
    parser.add_argument("--t5-model",    type=str, default="flan-t5-xl",
                        choices=["flan-t5-xl", "t5-v1_1-xl", "t5-v1_1-xxl"])
    parser.add_argument("--t5-max-len",  type=int, default=120)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--precision",   type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--buckets",     nargs="+", default=None,
                        help="Subset of buckets to process (default: all four)")
    args = parser.parse_args()

    # auto-detect repo root
    repo_root = args.repo_root
    if repo_root is None:
        for candidate in ["./LlamaGen", "../LlamaGen", "/vol/repo/LlamaGen"]:
            if Path(candidate).exists():
                repo_root = str(Path(candidate).resolve())
                break
    if repo_root is None:
        raise RuntimeError("Cannot find LlamaGen repo root. Pass --repo-root explicitly.")

    extract_all(
        data_root=args.data_root,
        out_dir=args.out_dir,
        repo_root=repo_root,
        t5_path=args.t5_path,
        t5_model_type=args.t5_model,
        t5_feature_max_len=args.t5_max_len,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        buckets=args.buckets,
    )
