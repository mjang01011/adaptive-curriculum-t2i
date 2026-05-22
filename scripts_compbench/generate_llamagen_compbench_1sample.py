"""
Generate LlamaGen images for T2I-CompBench evaluation.

Reads a category prompt file (one prompt per line), generates 1 image per prompt,
and saves files in CompBench's expected format:

    <sanitized_prompt>_<six_digit_id>.png

Also writes metadata.jsonl mapping question_id → full prompt for downstream use.

Usage:
  python scripts_compbench/generate_llamagen_compbench_1sample.py \
    --prompt-file /viscam/.../T2I-CompBench/examples/dataset/color_val.txt \
    --category color \
    --repo-root  /viscam/.../LlamaGen \
    --gpt-ckpt   /viscam/.../t2i_XL_stage1_256.pt \
    --vq-ckpt    /viscam/.../vq_ds16_t2i.pt \
    --t5-path    /viscam/.../t5-ckpt \
    --output-dir /viscam/.../outputs_compbench_vanilla/<RUN>/color/samples \
    --seed 0 --cfg-scale 2.0 --image-size 256 --batch-size 8

Optional LoRA:
    --lora-checkpoint /viscam/.../outputs/<run>/checkpoints/best.pt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch


_MAX_PROMPT_LEN = 150  # chars kept in filename; rest captured in metadata.jsonl


def _sanitize(prompt: str) -> str:
    safe = re.sub(r'[/\\:*?"<>|\x00]', "", prompt)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:_MAX_PROMPT_LEN]


def _read_prompts(path: str) -> list:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file",        required=True)
    parser.add_argument("--category",           required=True)
    parser.add_argument("--repo-root",          required=True)
    parser.add_argument("--gpt-ckpt",           required=True)
    parser.add_argument("--vq-ckpt",            required=True)
    parser.add_argument("--t5-path",            required=True)
    parser.add_argument("--output-dir",         required=True)
    parser.add_argument("--lora-checkpoint",    default=None)
    parser.add_argument("--num-samples-per-prompt", type=int, default=1)
    parser.add_argument("--seed",               type=int, default=0)
    parser.add_argument("--cfg-scale",          type=float, default=2.0)
    parser.add_argument("--image-size",         type=int, default=256)
    parser.add_argument("--batch-size",         type=int, default=8)
    parser.add_argument("--precision",          default="bf16")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(args.prompt_file)
    print(f"[compbench_gen] category={args.category}  prompts={len(prompts)}")

    sys.path.insert(0, args.repo_root)

    use_lora = args.lora_checkpoint is not None
    lora_config = {"rank": 16, "alpha": 32, "dropout": 0.0,
                   "start_layer": 0, "target_modules": ["wqkv", "wo"]}

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
        precision=args.precision,
        use_lora=use_lora,
        lora_config=lora_config if use_lora else None,
    )

    if use_lora:
        model.load_checkpoint(args.lora_checkpoint)
        print(f"[compbench_gen] Loaded LoRA checkpoint: {args.lora_checkpoint}")

    torch.manual_seed(args.seed)

    from torchvision.utils import save_image
    from autoregressive.models.generate import generate

    metadata = []
    saved = 0

    def _iter_batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield i, lst[i:i + bs]

    model.gpt.eval()

    for start_idx, batch_prompts in _iter_batches(prompts, args.batch_size):
        batch_ids = list(range(start_idx, start_idx + len(batch_prompts)))

        with torch.no_grad():
            caption_embs, emb_masks = model.t5.get_text_embeddings(batch_prompts)
            new_embs = []
            for emb, mask in zip(caption_embs, emb_masks):
                valid = int(mask.sum().item())
                new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
            c_indices = torch.stack(new_embs)
            c_emb_masks = torch.flip(emb_masks, dims=[-1])
            c_indices = c_indices * c_emb_masks[:, :, None]

            latent_size = args.image_size // 16
            qzshape = [len(batch_prompts), model.codebook_embed_dim, latent_size, latent_size]

            index_sample = generate(
                model.gpt, c_indices, latent_size ** 2,
                c_emb_masks,
                cfg_scale=args.cfg_scale,
                temperature=model.temperature,
                top_k=model.top_k,
                top_p=model.top_p,
                sample_logits=True,
            )
            samples = model.vq_model.decode_code(index_sample, qzshape)

        for i, (prompt, qid) in enumerate(zip(batch_prompts, batch_ids)):
            sanitized = _sanitize(prompt)
            fname = f"{sanitized}_{qid:06d}.png"
            fpath = out_dir / fname
            save_image(samples[i:i+1], str(fpath), normalize=True, value_range=(-1, 1))
            metadata.append({
                "question_id": qid,
                "prompt": prompt,
                "filename": fname,
                "category": args.category,
            })
            saved += 1

        if (start_idx // args.batch_size) % 10 == 0:
            print(f"  [{saved}/{len(prompts)}] saved")

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for rec in metadata:
            f.write(json.dumps(rec) + "\n")

    print(f"[compbench_gen] Done. {saved} images → {out_dir}")


if __name__ == "__main__":
    main()
