"""
Generate images for synthetic shape prompts using LlamaGen.

Usage:
  python scripts_verifier/generate_shape_images.py \
    --input-jsonl data_synthetic_shapes/val.jsonl \
    --model llamagen \
    --num-generations 6 \
    --cfg-scale 2.0 \
    --seeds 0 1 2 3 4 5 \
    --output-dir outputs_verifier/base_shapes_val_g6 \
    --base-config adaptive_curriculum/configs/experiment.yaml \
    --repo-root /viscam/.../LlamaGen \
    --gpt-ckpt .../t2i_XL_stage1_256.pt \
    --vq-ckpt .../vq_ds16_t2i.pt \
    --t5-path .../t5-ckpt \
    --batch-size 4
"""
import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl",   required=True)
    parser.add_argument("--model",         default="llamagen", choices=["llamagen"])
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--cfg-scale",     type=float, default=2.0)
    parser.add_argument("--seeds",         type=int, nargs="+", default=None)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--batch-size",    type=int, default=4)
    # LlamaGen model paths
    parser.add_argument("--base-config",   default="adaptive_curriculum/configs/experiment.yaml")
    parser.add_argument("--repo-root",     required=True)
    parser.add_argument("--gpt-ckpt",      required=True)
    parser.add_argument("--vq-ckpt",       required=True)
    parser.add_argument("--t5-path",       required=True)
    parser.add_argument("--t5-cache-dir",  default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.num_generations))
    assert len(seeds) >= args.num_generations, "Need at least num_generations seeds"
    seeds = seeds[:args.num_generations]

    out_dir   = Path(args.output_dir)
    img_dir   = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    rows = load_jsonl(args.input_jsonl)
    print(f"[gen_shapes] {len(rows)} prompts  seeds={seeds}  cfg={args.cfg_scale}")

    sys.path.insert(0, args.repo_root)
    from omegaconf import OmegaConf
    base_cfg = OmegaConf.load(args.base_config)
    base_cfg.paths.repo_root = args.repo_root
    base_cfg.model.gpt_ckpt  = args.gpt_ckpt
    base_cfg.model.vq_ckpt   = args.vq_ckpt
    base_cfg.model.t5_path   = args.t5_path
    base_cfg.model.cfg_scale  = args.cfg_scale
    if args.t5_cache_dir:
        base_cfg.paths.t5_cache_dir = args.t5_cache_dir

    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    model = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        gpt_model=str(base_cfg.model.gpt_model),
        image_size=int(base_cfg.model.image_size),
        t5_path=args.t5_path,
        t5_model_type=str(base_cfg.model.t5_model_type),
        t5_feature_max_len=int(base_cfg.model.t5_feature_max_len),
        cfg_scale=args.cfg_scale,
        precision=str(base_cfg.model.mixed_precision),
        use_lora=False,
    )

    def _batches(items, bs):
        for i in range(0, len(items), bs):
            yield items[i:i + bs]

    samples = []

    for seed in seeds:
        seed_dir = img_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(exist_ok=True)
        print(f"\n[gen_shapes] seed={seed}")

        for batch in _batches(rows, args.batch_size):
            prompts     = [r["prompt"] for r in batch]
            prompt_ids  = [r["id"]     for r in batch]
            bucket_names = ["synthetic_shapes"] * len(batch)

            img_paths, _ = model.generate_with_tokens(
                prompts=prompts,
                out_dir=str(seed_dir),
                prompt_ids=prompt_ids,
                bucket_names=bucket_names,
                num_samples_per_prompt=1,
                seed=seed,
                cached_embeddings=None,
            )

            for row, img_path in zip(batch, img_paths):
                # normalise to relative path if possible
                try:
                    rel = str(Path(img_path).relative_to(out_dir))
                except ValueError:
                    rel = img_path
                samples.append({
                    "id":           row["id"],
                    "prompt":       row["prompt"],
                    "sample_index": seeds.index(seed),
                    "seed":         seed,
                    "image_path":   rel,
                    "objects":      row["objects"],
                    "relation":     row.get("relation", ""),
                })

        print(f"  seed={seed} done")

    samples_path = out_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n[gen_shapes] {len(samples)} samples saved → {samples_path}")


if __name__ == "__main__":
    main()
