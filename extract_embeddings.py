"""
Run on cluster:
  cd /viscam/u/jj277/adaptive-curriculum-t2i
  python extract_embeddings.py
"""
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
LLAMAGEN_ROOT = "/viscam/u/jj277/svl/B3S/baselines/LlamaGen"
T5_CKPT       = "/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/t5-ckpt"
DATA_ROOT     = "/viscam/u/jj277/adaptive-curriculum-t2i/data"
OUT_DIR       = "/viscam/u/jj277/adaptive-curriculum-t2i/data/t5_cache"
T5_MODEL      = "flan-t5-xl"
BATCH_SIZE    = 32
# ─────────────────────────────────────────────────────────────────────────────

BUCKETS = {
    "attribute_binding":   ("attribute_binding_train_500.jsonl",    "attribute_binding_val_20.jsonl"),
    "counting":            ("counting_train_500.jsonl",              "counting_val_20.jsonl"),
    "spatial_relations":   ("spatial_relations_train_500.jsonl",     "spatial_relations_val_20.jsonl"),
    "complex_composition": ("complex_composition_train_500.jsonl",   "complex_composition_val_20.jsonl"),
}

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "t5_embedder",
    f"{LLAMAGEN_ROOT}/language/t5.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
T5Embedder = _mod.T5Embedder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

print(f"loading T5 ({T5_MODEL})...")
t5 = T5Embedder(
    device=device,
    local_cache=True,
    cache_dir=T5_CKPT,
    dir_or_name=T5_MODEL,
    torch_dtype=torch.bfloat16,
    model_max_length=120,
)
print("T5 loaded\n")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

for bucket, (train_file, val_file) in BUCKETS.items():
    all_items = []
    for split, fname in [("train", train_file), ("val", val_file)]:
        fpath = Path(DATA_ROOT) / bucket / fname
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                item["_split"] = split
                all_items.append(item)

    print(f"[{bucket}] {len(all_items)} prompts")

    embeddings = {}
    meta = {}
    batches = range(0, len(all_items), BATCH_SIZE)
    for start in tqdm(batches, desc=bucket, unit="batch"):
        batch = all_items[start : start + BATCH_SIZE]
        prompts = [x["prompt"] for x in batch]
        with torch.no_grad():
            embs, masks = t5.get_text_embeddings(prompts)
        for item, emb, mask in zip(batch, embs, masks):
            valid_len = int(mask.sum().item())
            embeddings[item["id"]] = emb[:valid_len].unsqueeze(0).cpu()  # (1, seq_len, 2048)
            meta[item["id"]] = {"split": item["_split"], "prompt": item["prompt"]}

    out_file = Path(OUT_DIR) / f"{bucket}.pt"
    torch.save({"embeddings": embeddings, "meta": meta}, str(out_file))

    sample_id = list(embeddings.keys())[0]
    print(f"\n  saved -> {out_file}  sample shape: {embeddings[sample_id].shape}")

print("\nDone.")

# ── upload to HuggingFace ─────────────────────────────────────────────────────
HF_REPO = "mjang01011/adaptive-curriculum-t2i-embeddings"  # change to your HF repo

from huggingface_hub import HfApi
api = HfApi()
for pt_file in Path(OUT_DIR).glob("*.pt"):
    print(f"uploading {pt_file.name} ...")
    api.upload_file(
        path_or_fileobj=str(pt_file),
        path_in_repo=pt_file.name,
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print(f"  done: {pt_file.name}")
print("All uploaded.")
