"""
Generate N LlamaGen images per prompt for T2I-CompBench evaluation.

For each batch of B prompts, repeats conditioning N times and calls generate()
once for all B*N images — much faster than N separate passes.

Output filename format:
    <sanitized_prompt>_<qid * N + sample_idx :06d>.png

Usage:
  python scripts_compbench/generate_llamagen_compbench_Nsample.py \
    --prompt-file  /path/to/T2I-CompBench/examples/dataset/color_val.txt \
    --category     color \
    --repo-root    LlamaGen \
    --gpt-ckpt     pretrained/t2i_XL_stage1_256.pt \
    --vq-ckpt      pretrained/vq_ds16_t2i.pt \
    --t5-path      pretrained/t5-ckpt \
    --output-dir   outputs_compbench/run1/color/samples \
    --num-samples  10 --batch-size 4 --cfg-scale 2.0 --seed 0

Optional LoRA:
    --lora-checkpoint outputs/<run>/best.pt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it


_MAX_PROMPT_LEN = 150


def _sanitize(prompt: str) -> str:
    safe = re.sub(r'[/\\:*?"<>|\x00]', "", prompt)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:_MAX_PROMPT_LEN]


def _read_prompts(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-file",       required=True)
    p.add_argument("--category",          required=True)
    p.add_argument("--repo-root",         required=True)
    p.add_argument("--gpt-ckpt",          required=True)
    p.add_argument("--vq-ckpt",           required=True)
    p.add_argument("--t5-path",           required=True)
    p.add_argument("--output-dir",        required=True)
    p.add_argument("--lora-checkpoint",   default=None)
    p.add_argument("--num-samples",       type=int,   default=10)
    p.add_argument("--batch-size",        type=int,   default=4,
                   help="Prompts per batch; total GPU batch = batch_size * num_samples")
    p.add_argument("--seed",              type=int,   default=0)
    p.add_argument("--cfg-scale",         type=float, default=2.0)
    p.add_argument("--image-size",        type=int,   default=256)
    p.add_argument("--temperature",       type=float, default=1.0)
    p.add_argument("--top-k",             type=int,   default=2000)
    p.add_argument("--precision",         default="bf16")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(args.prompt_file)
    N = args.num_samples
    print(f"[compbench_gen] category={args.category}  prompts={len(prompts)}  "
          f"N={N}  total_images={len(prompts)*N}", flush=True)

    sys.path.insert(0, args.repo_root)

    use_lora = args.lora_checkpoint is not None
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    model = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        gpt_model="GPT-XL",
        image_size=args.image_size,
        t5_path=args.t5_path,
        t5_model_type="flan-t5-xl",
        t5_feature_max_len=120,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=1.0,
        precision=args.precision,
        use_lora=use_lora,
        lora_config={"rank": 16, "alpha": 32, "dropout": 0.0,
                     "start_layer": 0, "target_modules": ["wqkv", "wo", "w1", "w2", "w3"]}
                    if use_lora else None,
    )

    if use_lora:
        model.load_checkpoint(args.lora_checkpoint)
        print(f"[compbench_gen] Loaded LoRA: {args.lora_checkpoint}", flush=True)

    torch.manual_seed(args.seed)

    from autoregressive.models.generate import generate
    from torchvision.utils import save_image

    print("[compbench_gen] Loading GPT...", flush=True)
    gpt      = model.gpt.eval()
    print("[compbench_gen] Loading VQ...", flush=True)
    vq       = model.vq_model
    print("[compbench_gen] Loading T5...", flush=True)
    t5       = model.t5
    print("[compbench_gen] All models loaded.", flush=True)
    ls       = model.latent_size
    cb       = model.codebook_embed_dim
    dtype    = model.dtype
    device   = model.device

    metadata = []
    saved    = 0

    batches = range(0, len(prompts), args.batch_size)
    for batch_start in tqdm(batches, desc=args.category, unit="batch"):
        batch_prompts = prompts[batch_start: batch_start + args.batch_size]
        B = len(batch_prompts)

        try:
            with torch.no_grad():
                # T5 encode batch of B prompts
                embs, masks = t5.get_text_embeddings(batch_prompts)
                new_embs = []
                for emb, mask in zip(embs, masks):
                    valid = int(mask.sum().item())
                    new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
                c = torch.stack(new_embs) * torch.flip(masks, dims=[-1])[:, :, None]
                c      = c.to(device=device, dtype=dtype)       # [B, T, D]
                c_mask = torch.flip(masks, dims=[-1]).to(device=device, dtype=dtype)  # [B, T]

                # Repeat each prompt N times → [B*N, T, D]
                c_rep      = c.repeat_interleave(N, dim=0)
                c_mask_rep = c_mask.repeat_interleave(N, dim=0)

                index_sample = generate(
                    gpt, c_rep, ls ** 2, c_mask_rep,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    top_k=args.top_k, top_p=1.0,
                    sample_logits=True,
                )  # [B*N, ls**2]

                samples = vq.decode_code(index_sample, [B * N, cb, ls, ls])  # [B*N, 3, H, W]

        except Exception as e:
            print(f"  [warn] batch starting at {batch_start} failed: {e}", flush=True)
            continue

        for i, prompt in enumerate(batch_prompts):
            qid       = batch_start + i
            sanitized = _sanitize(prompt)
            for si in range(N):
                flat_id = qid * N + si
                fname   = f"{sanitized}_{flat_id:06d}.png"
                fpath   = out_dir / fname
                img_idx = i * N + si
                save_image(samples[img_idx:img_idx+1], str(fpath),
                           normalize=True, value_range=(-1, 1))
                metadata.append({
                    "question_id": qid,
                    "sample_index": si,
                    "flat_id": flat_id,
                    "prompt": prompt,
                    "filename": fname,
                    "category": args.category,
                })
                saved += 1


    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec) + "\n")

    print(f"[compbench_gen] Done. {saved} images → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
