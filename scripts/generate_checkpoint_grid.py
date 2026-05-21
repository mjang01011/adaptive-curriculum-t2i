"""
Generate qualitative image grids across checkpoints for visual inspection.

For each prompt, generates a row: [base | step_5 | step_10 | step_20 | ...]
Saves individual images and a combined grid PNG per prompt.

Usage:
  python scripts/generate_checkpoint_grid.py \
    --run-dir   outputs/<run> \
    --bucket    attribute_binding \
    --checkpoints base step_000005 step_000010 step_000020 \
    --num-prompts 16 \
    --num-samples 2 \
    --cfg-scale 2.0 \
    --out outputs/<run>/qual_grid_attribute
"""
import argparse
import json
import sys
from pathlib import Path


def _load_config(run_dir: Path):
    from omegaconf import OmegaConf
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_resolved.yaml not found in {run_dir}")
    return OmegaConf.load(str(cfg_path))


def _build_model(config, lora_checkpoint=None):
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    use_lora = lora_checkpoint is not None
    lora_cfg = None
    if use_lora:
        lora_cfg = {
            "rank": int(config.lora.rank),
            "alpha": int(config.lora.alpha),
            "dropout": float(config.lora.dropout),
            "target_modules": list(config.lora.get("target_modules", ["wqkv", "wo"])),
            "start_layer": int(getattr(config.lora, "start_layer", 0)),
        }
    model = LlamaGenWrapper(
        repo_root=config.paths.repo_root,
        vq_ckpt=config.model.vq_ckpt,
        gpt_ckpt=config.model.gpt_ckpt,
        gpt_model=config.model.gpt_model,
        image_size=config.model.image_size,
        t5_path=config.model.t5_path,
        t5_model_type=config.model.t5_model_type,
        t5_feature_max_len=config.model.t5_feature_max_len,
        cfg_scale=float(getattr(config.model, "cfg_scale", 2.0)),
        precision=config.model.mixed_precision,
        use_lora=use_lora,
        lora_config=lora_cfg,
    )
    if lora_checkpoint:
        model.load_checkpoint(lora_checkpoint)
    return model


def _make_grid(image_rows, out_path: Path, labels, prompt_texts):
    """
    image_rows: list of lists of PIL images, shape [n_prompts][n_checkpoints * n_samples]
    labels: checkpoint names for column headers
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[grid] Pillow not available, skipping grid generation. pip install Pillow")
        return

    if not image_rows or not image_rows[0]:
        return

    img_w = image_rows[0][0].width
    img_h = image_rows[0][0].height
    n_cols = len(image_rows[0])
    n_rows = len(image_rows)

    label_h = 24
    prompt_h = 20
    pad = 4

    grid_w = n_cols * (img_w + pad) + pad
    grid_h = n_rows * (img_h + pad + prompt_h) + label_h + pad

    grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid)

    # column labels
    col_w = img_w + pad
    for c, label in enumerate(labels):
        x = pad + c * col_w
        draw.text((x + 2, 2), label[:16], fill=(50, 50, 50))

    for r, (row_imgs, prompt_text) in enumerate(zip(image_rows, prompt_texts)):
        y_base = label_h + r * (img_h + pad + prompt_h)
        # prompt text
        short_prompt = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
        draw.text((pad, y_base), short_prompt, fill=(80, 80, 80))
        for c, img in enumerate(row_imgs):
            x = pad + c * col_w
            y = y_base + prompt_h
            grid.paste(img, (x, y))

    grid.save(str(out_path))
    print(f"  [grid] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",     required=True)
    parser.add_argument("--bucket",      required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="'base' or names like 'step_000005'")
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--cfg-scale",   type=float, default=2.0)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--data-root",   default=None)
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    if args.data_root:
        config.paths.data_root = args.data_root
    config.model.cfg_scale = args.cfg_scale

    # load val prompts
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=config.paths.data_root,
        bucket_names=[args.bucket],
        train_file=config.buckets.train_file,
        val_file=config.buckets.val_file,
        max_val_prompts=args.num_prompts,
    )
    val_items = datasets[args.bucket].val_items[:args.num_prompts]
    print(f"[grid] {len(val_items)} prompts, {len(args.checkpoints)} checkpoints, "
          f"{args.num_samples} samples each")

    t5_cache = None
    t5_cache_dir = getattr(config.paths, "t5_cache_dir", None)
    if t5_cache_dir and t5_cache_dir != "null":
        from adaptive_curriculum.data.t5_cache import load_t5_cache
        t5_cache = load_t5_cache(t5_cache_dir, [args.bucket])

    # {ckpt_name: [img_path_per_prompt_per_sample]}
    all_image_paths = {}

    for ckpt_name in args.checkpoints:
        print(f"\n[grid] Generating for checkpoint: {ckpt_name}")

        if ckpt_name == "base":
            lora_path = None
        else:
            ckpt_path = run_dir / "checkpoints" / f"{ckpt_name}.pt"
            if not ckpt_path.exists():
                print(f"  WARNING: {ckpt_path} not found, skipping")
                continue
            lora_path = str(ckpt_path)

        model = _build_model(config, lora_checkpoint=lora_path)
        ckpt_out = out_dir / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)

        cached_embs = None
        if t5_cache is not None:
            try:
                cached_embs = t5_cache.bucket_embeddings(args.bucket)
            except Exception:
                pass

        image_paths = model.generate_images(
            prompts=[it.text for it in val_items],
            out_dir=str(ckpt_out),
            prompt_ids=[it.id for it in val_items],
            bucket_names=[args.bucket] * len(val_items),
            num_samples_per_prompt=args.num_samples,
            seed=args.seed,
            cached_embeddings=cached_embs,
        )
        all_image_paths[ckpt_name] = image_paths

        del model
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # build grids: one grid per prompt
    try:
        from PIL import Image
        n_items = len(val_items)
        n_ckpts = len(all_image_paths)
        ckpt_names = list(all_image_paths.keys())

        print("\n[grid] Building per-prompt grids ...")
        for item_idx, item in enumerate(val_items):
            row_imgs = []
            col_labels = []
            for ckpt_name in ckpt_names:
                paths = all_image_paths[ckpt_name]
                for s in range(args.num_samples):
                    path_idx = s * n_items + item_idx
                    if path_idx < len(paths):
                        img = Image.open(paths[path_idx]).convert("RGB")
                        row_imgs.append(img)
                        col_labels.append(f"{ckpt_name}_s{s}")

            grid_path = out_dir / f"grid_{item_idx:04d}_{item.id}.png"
            _make_grid([row_imgs], grid_path, col_labels, [item.text])

        # also build a mega-grid of first sample per checkpoint for each prompt
        print("[grid] Building summary grid (1 sample/ckpt, all prompts) ...")
        summary_rows = []
        for item_idx, item in enumerate(val_items):
            row = []
            for ckpt_name in ckpt_names:
                paths = all_image_paths[ckpt_name]
                path_idx = 0 * n_items + item_idx  # sample 0
                if path_idx < len(paths):
                    row.append(Image.open(paths[path_idx]).convert("RGB"))
            summary_rows.append(row)

        _make_grid(
            summary_rows,
            out_dir / "summary_grid.png",
            labels=ckpt_names,
            prompt_texts=[it.text for it in val_items],
        )

    except ImportError:
        print("[grid] Pillow not installed — images saved, grids skipped. pip install Pillow")

    # save manifest
    manifest = {
        "bucket": args.bucket,
        "prompts": [{"id": it.id, "text": it.text} for it in val_items],
        "checkpoints": ckpt_names,
        "image_paths": {k: v for k, v in all_image_paths.items()},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[grid] Done. Output → {out_dir}")


if __name__ == "__main__":
    main()
