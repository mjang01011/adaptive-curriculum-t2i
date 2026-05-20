"""
Thin wrapper around LlamaGen's sampling logic.
Accepts a list of prompts, generates images, saves them with deterministic filenames.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def generate_images(
    prompts: List[str],
    prompt_ids: List[str],
    bucket_names: List[str],
    out_dir: str,
    repo_root: str,
    vq_ckpt: str,
    gpt_ckpt: str,
    gpt_model: str = "GPT-XL",
    image_size: int = 256,
    downsample_size: int = 16,
    codebook_size: int = 16384,
    codebook_embed_dim: int = 8,
    cls_token_num: int = 120,
    t5_path: str = "pretrained_models/t5-ckpt",
    t5_model_type: str = "flan-t5-xl",
    t5_feature_max_len: int = 120,
    cfg_scale: float = 7.5,
    temperature: float = 1.0,
    top_k: int = 1000,
    top_p: float = 1.0,
    precision: str = "bf16",
    seed: Optional[int] = None,
    num_samples_per_prompt: int = 1,
    model_checkpoint: Optional[str] = None,
    gpt_model_obj=None,
    vq_model_obj=None,
    t5_model_obj=None,
    device: str = "cuda",
) -> List[dict]:
    """
    Generate images for each prompt. Optionally reuse pre-loaded model objects.
    Returns list of metadata dicts aligned with expanded (prompt x sample) list.
    """
    import torch
    from torchvision.utils import save_image

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from tokenizer.tokenizer_image.vq_model import VQ_models
    from language.t5 import T5Embedder
    from autoregressive.models.gpt import GPT_models
    from autoregressive.models.generate import generate

    precision_map = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = precision_map.get(precision, torch.bfloat16)

    if seed is not None:
        torch.manual_seed(seed)

    # load VQ model if not provided
    if vq_model_obj is None:
        vq_model_obj = VQ_models["VQ-16"](
            codebook_size=codebook_size,
            codebook_embed_dim=codebook_embed_dim,
        )
        vq_model_obj.to(device)
        vq_model_obj.eval()
        ckpt = torch.load(vq_ckpt, map_location="cpu")
        vq_model_obj.load_state_dict(ckpt["model"])
        del ckpt

    # load GPT model if not provided
    if gpt_model_obj is None:
        latent_size = image_size // downsample_size
        gpt_model_obj = GPT_models[gpt_model](
            block_size=latent_size ** 2,
            cls_token_num=cls_token_num,
            model_type="t2i",
        ).to(device=device, dtype=dtype)
        ckpt_path = model_checkpoint or gpt_ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu")
        weight_key = next((k for k in ("model", "module", "state_dict") if k in ckpt), None)
        model_weight = ckpt[weight_key] if weight_key else ckpt
        gpt_model_obj.load_state_dict(model_weight, strict=False)
        gpt_model_obj.eval()
        del ckpt

    # load T5 if not provided
    if t5_model_obj is None:
        t5_model_obj = T5Embedder(
            device=device,
            local_cache=True,
            cache_dir=t5_path,
            dir_or_name=t5_model_type,
            torch_dtype=dtype,
            model_max_length=t5_feature_max_len,
        )

    latent_size = image_size // downsample_size
    metadata = []

    torch.set_grad_enabled(False)
    with torch.no_grad():
        caption_embs, emb_masks = t5_model_obj.get_text_embeddings(prompts)

        # left-padding
        new_caption_embs = []
        for caption_emb, emb_mask in zip(caption_embs, emb_masks):
            valid_num = int(emb_mask.sum().item())
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
        new_emb_masks = torch.flip(emb_masks, dims=[-1])

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks

        for k in range(num_samples_per_prompt):
            qzshape = [len(c_indices), codebook_embed_dim, latent_size, latent_size]
            index_sample = generate(
                gpt_model_obj, c_indices, latent_size ** 2,
                c_emb_masks,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_logits=True,
            )
            samples = vq_model_obj.decode_code(index_sample, qzshape)

            for i, (prompt_id, bucket, prompt) in enumerate(zip(prompt_ids, bucket_names, prompts)):
                img_tensor = samples[i:i+1]
                filename = f"{bucket}_{prompt_id}_sample{k}.png"
                img_path = str(out_path / filename)
                save_image(img_tensor, img_path, normalize=True, value_range=(-1, 1))
                metadata.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "bucket": bucket,
                    "image_path": img_path,
                    "model_checkpoint": model_checkpoint or gpt_ckpt,
                    "seed": seed,
                    "sample_idx": k,
                })

    meta_path = out_path / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    return metadata
