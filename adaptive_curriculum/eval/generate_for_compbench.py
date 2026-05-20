"""
Generate images for T2I-CompBench++ evaluation.

Supports three prompt input modes:
  1. T2I-CompBench++ JSON  (--prompts path/to/compbench.json)
       expected format: list of {id/index, prompt, category} dicts
  2. Flat JSONL            (--prompts path/to/prompts.jsonl)
       expected format: one {"id": ..., "prompt": ..., "category": ...} per line
  3. Our internal val sets (--use-val-sets --data-root ...)
       runs on all four curriculum buckets

Output layout (compatible with T2I-CompBench++ eval pipeline):
  <out_dir>/<model_name>/
    <category>/
      <prompt_id>_s<sample>.png   # e.g. 00042_s0.png
    manifest.json                 # {prompt_id: {prompt, category, image_paths}}

Usage examples
--------------
# Raw LlamaGen baseline (no LoRA):
python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root      $LLAMAGEN \
    --gpt-ckpt       $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt        $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path        $PRETRAINED/t5-ckpt \
    --prompts        /path/to/t2icompbench_prompts.json \
    --out-dir        /path/to/outputs/compbench \
    --model-name     base

# Trained LoRA checkpoint:
python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root      $LLAMAGEN \
    --gpt-ckpt       $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt        $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path        $PRETRAINED/t5-ckpt \
    --lora-checkpoint /path/to/run/checkpoints/best.pt \
    --prompts        /path/to/t2icompbench_prompts.json \
    --out-dir        /path/to/outputs/compbench \
    --model-name     ucb_lora

# Multiple checkpoints in one call:
python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root $LLAMAGEN --gpt-ckpt ... --vq-ckpt ... --t5-path ... \
    --lora-checkpoint ckpt1.pt ckpt2.pt ckpt3.pt \
    --model-name ucb_s20 ucb_s40 pooled_s40 \
    --prompts /path/to/prompts.json --out-dir /path/to/out

# Use our internal val sets instead of compbench prompts:
python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root $LLAMAGEN --gpt-ckpt ... --vq-ckpt ... --t5-path ... \
    --use-val-sets --data-root /path/to/data \
    --out-dir /path/to/out --model-name base
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── prompt loading helpers ─────────────────────────────────────────────────────

def _load_compbench_json(path: Path) -> List[dict]:
    """
    Handles all T2I-CompBench++ prompt formats:
      1. Plain string list:  ["prompt0", "prompt1", ...]          ← most common in their repo
      2. Flat dict list:     [{"prompt": ..., "category": ...}]
      3. Nested by category: {"color_attr": ["p0", ...], ...}     ← when loading a merged file
    Category is inferred from the filename stem if not present in the data.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_hint = path.stem  # e.g. "color_attr", "spatial"
    items = []

    if isinstance(data, list):
        for i, entry in enumerate(data):
            if isinstance(entry, str):
                # plain string list
                items.append({"id": str(i).zfill(5), "prompt": entry, "category": category_hint})
            else:
                items.append({
                    "id": str(entry.get("id", entry.get("index", i))).zfill(5),
                    "prompt": entry["prompt"],
                    "category": entry.get("category", category_hint),
                })
    elif isinstance(data, dict):
        # nested: {category: [str | dict, ...]}
        for category, prompts in data.items():
            for i, entry in enumerate(prompts):
                if isinstance(entry, str):
                    items.append({"id": str(i).zfill(5), "prompt": entry, "category": category})
                else:
                    items.append({
                        "id": str(entry.get("id", entry.get("index", i))).zfill(5),
                        "prompt": entry["prompt"],
                        "category": category,
                    })
    else:
        raise ValueError(f"Unrecognised JSON structure in {path}")

    return items


def _load_compbench_txt(path: Path) -> List[dict]:
    """T2I-CompBench++ .txt format: one prompt per line, category = filename stem."""
    category = path.stem
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                items.append({"id": str(i).zfill(5), "prompt": line, "category": category})
    return items


# Test split files only (_val.txt = 30% held-out test set, ~300 prompts each)
COMPBENCH_MAIN_FILES = [
    "color_val.txt", "shape_val.txt", "texture_val.txt",
    "spatial_val.txt", "3d_spatial_val.txt",
    "non_spatial_val.txt", "complex_val.txt", "numeracy_val.txt",
]


def _load_compbench_dir(prompts_dir: Path) -> List[dict]:
    """
    Load T2I-CompBench++ prompts from examples/dataset/ directory.
    Uses the main evaluation files only (not _train/_val splits).
    """
    items = []
    found = []
    for fname in COMPBENCH_MAIN_FILES:
        fpath = prompts_dir / fname
        if fpath.exists():
            batch = _load_compbench_txt(fpath)
            items.extend(batch)
            found.append(f"{fname}({len(batch)})")
        else:
            print(f"  [warn] {fname} not found in {prompts_dir}, skipping")
    print(f"[prompts] Loaded {len(items)} prompts: {', '.join(found)}")
    return items


def _load_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                items.append({
                    "id": str(entry.get("id", len(items))).zfill(5),
                    "prompt": entry["prompt"],
                    "category": entry.get("category", entry.get("bucket", "unknown")),
                })
    return items


def _load_prompts(prompts_path: str) -> List[dict]:
    p = Path(prompts_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts path not found: {p}")
    if p.is_dir():
        return _load_compbench_dir(p)
    if p.suffix == ".txt":
        return _load_compbench_txt(p)
    if p.suffix == ".jsonl":
        return _load_jsonl(p)
    return _load_compbench_json(p)


def _load_val_prompts(data_root: str) -> List[dict]:
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    BUCKETS = ["attribute_binding", "counting", "spatial_relations", "complex_composition"]
    datasets = load_bucket_datasets(data_root=data_root, bucket_names=BUCKETS)
    items = []
    for bucket, ds in datasets.items():
        for item in ds.val_items:
            items.append({
                "id": item.id,
                "prompt": item.text,
                "category": bucket,
            })
    print(f"[prompts] Loaded {len(items)} val items from {data_root}")
    return items


# ── model loading ──────────────────────────────────────────────────────────────

def _build_model(
    repo_root: str,
    gpt_ckpt: str,
    vq_ckpt: str,
    t5_path: str,
    t5_model_type: str,
    t5_feature_max_len: int,
    cfg_scale: float,
    gpt_model: str,
    image_size: int,
    precision: str,
    lora_checkpoint: Optional[str],
    lora_rank: int,
    lora_alpha: int,
    lora_start_layer: int,
):
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    use_lora = lora_checkpoint is not None
    lora_cfg = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "dropout": 0.0,
        "target_modules": ["wqkv", "wo"],
        "start_layer": lora_start_layer,
    } if use_lora else None

    model = LlamaGenWrapper(
        repo_root=repo_root,
        vq_ckpt=vq_ckpt,
        gpt_ckpt=gpt_ckpt,
        gpt_model=gpt_model,
        image_size=image_size,
        t5_path=t5_path,
        t5_model_type=t5_model_type,
        t5_feature_max_len=t5_feature_max_len,
        cfg_scale=cfg_scale,
        precision=precision,
        use_lora=use_lora,
        lora_config=lora_cfg,
    )

    if lora_checkpoint is not None:
        model.load_checkpoint(lora_checkpoint)

    return model


# ── generation ─────────────────────────────────────────────────────────────────

def _generate_for_model(
    model,
    items: List[dict],
    out_dir: Path,
    model_name: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    t5_cache_dir: Optional[str],
) -> dict:
    """
    Generate images for all items and write to out_dir/<model_name>/<category>/.
    Returns manifest dict.
    """
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # load T5 cache if available
    t5_cache = None
    if t5_cache_dir:
        try:
            from adaptive_curriculum.data.t5_cache import load_t5_cache
            categories = list({it["category"] for it in items})
            t5_cache = load_t5_cache(t5_cache_dir, categories)
            if t5_cache:
                print(f"[t5_cache] loaded from {t5_cache_dir}")
        except Exception as e:
            print(f"[t5_cache] could not load: {e}")

    manifest: dict = {}
    total = len(items)
    t0 = time.time()

    # process in batches per category so images land in the right subfolder
    categories = {}
    for item in items:
        categories.setdefault(item["category"], []).append(item)

    for category, cat_items in categories.items():
        cat_dir = model_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # process in mini-batches
        for start in range(0, len(cat_items), batch_size):
            batch = cat_items[start : start + batch_size]
            prompts = [it["prompt"] for it in batch]
            ids = [it["id"] for it in batch]

            # resolve cached embeddings for this batch if available
            cached_embs = None
            if t5_cache is not None:
                try:
                    cached_embs = t5_cache.bucket_embeddings(category)
                except Exception:
                    cached_embs = None

            image_paths = model.generate_images(
                prompts=prompts,
                out_dir=str(cat_dir),
                prompt_ids=ids,
                bucket_names=[category] * len(batch),
                num_samples_per_prompt=num_samples,
                seed=seed,
                cached_embeddings=cached_embs,
            )

            # group returned paths by item
            n = len(batch)
            for item_idx, item in enumerate(batch):
                item_paths = [
                    image_paths[k * n + item_idx]
                    for k in range(num_samples)
                    if k * n + item_idx < len(image_paths)
                ]
                manifest[item["id"]] = {
                    "prompt": item["prompt"],
                    "category": item["category"],
                    "image_paths": item_paths,
                }

            done = sum(len(v) for v in categories.values() if v is not cat_items) + start + len(batch)
            elapsed = time.time() - t0
            print(f"  [{model_name}] {min(done, total)}/{total}  {elapsed:.0f}s", end="\r")

    print(f"\n[{model_name}] Done. {total} prompts × {num_samples} samples = "
          f"{total * num_samples} images  ({time.time() - t0:.0f}s)")

    manifest_path = model_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[{model_name}] Manifest → {manifest_path}")

    return manifest


# ── summary table ──────────────────────────────────────────────────────────────

def _print_summary(out_dir: Path, model_names: List[str]):
    print("\n" + "=" * 60)
    print("Output summary")
    print("=" * 60)
    for name in model_names:
        model_dir = out_dir / name
        if not model_dir.exists():
            continue
        n_images = sum(1 for _ in model_dir.rglob("*.png"))
        manifest_path = model_dir / "manifest.json"
        n_prompts = 0
        if manifest_path.exists():
            with open(manifest_path) as f:
                n_prompts = len(json.load(f))
        print(f"  {name:30s}  {n_prompts} prompts  {n_images} images → {model_dir}")
    print()
    print("Next step: run T2I-CompBench++ evaluation scripts on the generated images.")
    print("  git clone https://github.com/Karine-Huang/T2I-CompBench")
    print("  Then point their eval scripts at each <out_dir>/<model_name>/ directory.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate images for T2I-CompBench++ evaluation"
    )

    # model paths
    parser.add_argument("--repo-root",   required=True)
    parser.add_argument("--gpt-ckpt",    required=True)
    parser.add_argument("--vq-ckpt",     required=True)
    parser.add_argument("--t5-path",     required=True)
    parser.add_argument("--t5-model",    default="flan-t5-xl")
    parser.add_argument("--t5-max-len",  type=int, default=120)
    parser.add_argument("--gpt-model",   default="GPT-XL")
    parser.add_argument("--image-size",  type=int, default=256)
    parser.add_argument("--precision",   default="bf16", choices=["bf16", "fp16", "fp32"])

    # LoRA checkpoints (pass multiple for batch comparison)
    parser.add_argument(
        "--lora-checkpoint", nargs="*", default=[],
        help="Path(s) to LoRA checkpoint(s). Omit for base model only."
    )
    parser.add_argument(
        "--model-name", nargs="*", default=[],
        help="Name(s) for output subdirectory. 'base' is always added. "
             "Must match --lora-checkpoint count if provided."
    )

    # prompts
    parser.add_argument("--prompts", default=None,
                        help="Path to T2I-CompBench++ JSON or flat JSONL prompts file")
    parser.add_argument("--use-val-sets", action="store_true",
                        help="Use our internal val sets instead of an external prompts file")
    parser.add_argument("--data-root", default=None,
                        help="Required when --use-val-sets is set")

    # generation params
    parser.add_argument("--cfg-scale",   type=float, default=2.0)
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Images per prompt (T2I-CompBench++ standard is 4)")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="Prompts per forward pass")
    parser.add_argument("--seed",        type=int, default=42)

    # output
    parser.add_argument("--out-dir",     required=True)
    parser.add_argument("--t5-cache-dir", default=None,
                        help="Optional pre-extracted T5 cache directory")

    # LoRA architecture (must match training config)
    parser.add_argument("--lora-rank",         type=int, default=16)
    parser.add_argument("--lora-alpha",        type=int, default=32)
    parser.add_argument("--lora-start-layer",  type=int, default=0)

    args = parser.parse_args()

    # ── validate args ──
    if args.use_val_sets and args.data_root is None:
        parser.error("--data-root is required when --use-val-sets is set")
    if not args.use_val_sets and args.prompts is None:
        parser.error("provide --prompts or --use-val-sets")

    lora_ckpts: List[Optional[str]] = args.lora_checkpoint or []
    # build name list: always include "base" as first entry
    names: List[str] = ["base"] + (args.model_name or [f"lora_{i}" for i in range(len(lora_ckpts))])
    model_list: List[Optional[str]] = [None] + lora_ckpts  # None → base model

    if len(names) != len(model_list):
        parser.error(
            f"--model-name count ({len(args.model_name)}) must equal "
            f"--lora-checkpoint count ({len(lora_ckpts)})"
        )

    # ── load prompts ──
    if args.use_val_sets:
        items = _load_val_prompts(args.data_root)
    else:
        items = _load_prompts(args.prompts)
    print(f"[prompts] {len(items)} prompts across "
          f"{len(set(it['category'] for it in items))} categories")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── generate for each model ──
    for model_name, lora_ckpt in zip(names, model_list):
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (lora_ckpt={lora_ckpt or 'none (base)'})")
        print(f"{'='*60}")

        model = _build_model(
            repo_root=args.repo_root,
            gpt_ckpt=args.gpt_ckpt,
            vq_ckpt=args.vq_ckpt,
            t5_path=args.t5_path,
            t5_model_type=args.t5_model,
            t5_feature_max_len=args.t5_max_len,
            cfg_scale=args.cfg_scale,
            gpt_model=args.gpt_model,
            image_size=args.image_size,
            precision=args.precision,
            lora_checkpoint=lora_ckpt,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_start_layer=args.lora_start_layer,
        )

        _generate_for_model(
            model=model,
            items=items,
            out_dir=out_dir,
            model_name=model_name,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            t5_cache_dir=args.t5_cache_dir,
        )

        # release GPU memory between models
        del model
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_summary(out_dir, names)


if __name__ == "__main__":
    main()
