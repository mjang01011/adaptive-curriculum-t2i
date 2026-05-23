"""
CARGO visual diagnostics: validate that mask hot-spots align with
compositionally discriminative image regions.

Usage:
  python CARGO/diagnostics.py \\
    --val-jsonl   $PROJECT/data/attribute_binding/attribute_binding_val_20.jsonl \\
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \\
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \\
    --t5-path     $PRETRAINED/t5-ckpt \\
    --repo-root   $PROJECT/LlamaGen \\
    --qwen-model  $PRETRAINED/Qwen3-VL-4B-Instruct \\
    --reward-mode grpo_attr_contrastive_rubric_v2 \\
    --num-prompts 4 \\
    --num-generations 8 \\
    --output-dir  $PROJECT/outputs/cargo_diagnostics

Outputs per prompt:
  {output_dir}/{prompt_id}/
    best.png             — best-scoring generated image
    worst.png            — worst-scoring generated image
    cargo_attr_mask.png  — CARGO mask for contrastive_attr component, overlaid on best
    cargo_presence_mask.png — mask for object_presence component, overlaid on best
    grid.png             — all G images annotated with component scores
    comp_rewards.json    — raw component reward matrix (B=1, G)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="CARGO mask visual diagnostics")
    p.add_argument("--val-jsonl",       required=True)
    p.add_argument("--repo-root",       required=True)
    p.add_argument("--gpt-ckpt",        required=True)
    p.add_argument("--vq-ckpt",         required=True)
    p.add_argument("--t5-path",         required=True)
    p.add_argument("--qwen-model",      default=None)
    p.add_argument("--reward-mode",     default="grpo_attr_contrastive_rubric_v2")
    p.add_argument("--num-prompts",     type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--cargo-mask-floor", type=float, default=0.30)
    p.add_argument("--cfg-scale-train", type=float, default=4.0)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--top-k",           type=int,   default=1000)
    p.add_argument("--top-p",           type=float, default=1.0)
    p.add_argument("--precision",       default="bf16")
    return p.parse_args()


def load_jsonl(path):
    from adaptive_curriculum.data.schemas import BucketItem
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))
    return items


def _make_image_grid(images, labels, cols=4):
    """Arrange PIL images in a grid with text labels."""
    from PIL import Image, ImageDraw, ImageFont
    rows = (len(images) + cols - 1) // cols
    W, H = images[0].size
    label_h = 20
    grid = Image.new("RGB", (W * cols, (H + label_h) * rows), (40, 40, 40))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(i, cols)
        y = row * (H + label_h)
        grid.paste(img, (col * W, y + label_h))
        draw = ImageDraw.Draw(grid)
        draw.text((col * W + 2, y + 2), label[:38], fill=(220, 220, 220), font=font)
    return grid


def run_diagnostics_for_item(
    item, wrapper, reward_model, args, out_dir: Path
):
    import contextlib
    from autoregressive.models.generate import generate
    from CARGO.masks   import compute_cargo_mask, overlay_mask_on_image
    from CARGO.rewards import META_KEYS
    import torchvision.transforms.functional as TF
    from torchvision.utils import save_image

    out_dir.mkdir(parents=True, exist_ok=True)
    G      = args.num_generations
    device = wrapper.device

    # ── Generate G images ─────────────────────────────────────────────────────
    wrapper.gpt.eval()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _sdpa_ctx = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                          SDPBackend.EFFICIENT_ATTENTION,
                                          SDPBackend.MATH])
    except Exception:
        _sdpa_ctx = contextlib.nullcontext

    with torch.no_grad():
        c_indices, c_emb_masks = wrapper._get_conditioning([item])

    qzshape = [1, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
    all_tokens = []
    all_pils   = []

    with torch.no_grad(), _sdpa_ctx():
        for _ in range(G):
            idx_sample = generate(
                wrapper.gpt, c_indices, wrapper.latent_size ** 2,
                c_emb_masks,
                cfg_scale=wrapper.cfg_scale_train,
                temperature=wrapper.temperature,
                top_k=wrapper.top_k,
                top_p=wrapper.top_p,
                sample_logits=True,
            )  # (1, seq_len)
            all_tokens.append(idx_sample)
            decoded = wrapper.vq_model.decode_code(idx_sample, qzshape)
            img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
            all_pils.append(TF.to_pil_image(img_t.cpu()))

    wrapper._disable_kv_cache()

    # ── Score all G images ────────────────────────────────────────────────────
    pairs   = [(pil, item) for pil in all_pils]
    results = reward_model.score_images_batch(pairs, mode=args.reward_mode)

    scores = [float(r["score"]) for r in results]
    comp_matrices = {}   # key → (G,) list of floats

    for g, r in enumerate(results):
        for k, v in (r.get("component_scores") or {}).items():
            comp_matrices.setdefault(k, []).append(float(v))

    # ── Save images sorted by score ───────────────────────────────────────────
    order = sorted(range(G), key=lambda i: scores[i], reverse=True)
    all_pils[order[0]].save(str(out_dir / "best.png"))
    all_pils[order[-1]].save(str(out_dir / "worst.png"))

    # ── Compute and save CARGO masks for primary components ───────────────────
    stacked_tokens = torch.stack(all_tokens, dim=0).squeeze(1)  # (G, seq_len)

    cargo_keys = [k for k in comp_matrices if k not in META_KEYS]
    for key in cargo_keys:
        R_c = torch.tensor(comp_matrices[key], dtype=torch.float32).to(device)
        mask = compute_cargo_mask(
            stacked_tokens.to(device), R_c,
            latent_size=wrapper.latent_size,
            mask_floor=args.cargo_mask_floor,
        )
        overlaid = overlay_mask_on_image(all_pils[order[0]], mask,
                                          latent_size=wrapper.latent_size)
        safe_key = key.replace("/", "_")
        overlaid.save(str(out_dir / f"cargo_{safe_key}_mask.png"))

    # ── Build annotated grid of all G images ──────────────────────────────────
    grid_labels = []
    for g in range(G):
        comp_str = "  ".join(
            f"{k[:6]}={comp_matrices[k][g]:.2f}"
            for k in sorted(comp_matrices)
            if k not in META_KEYS
        )
        grid_labels.append(f"r={scores[g]:.3f}  {comp_str[:30]}")

    grid = _make_image_grid(all_pils, grid_labels, cols=min(4, G))
    grid.save(str(out_dir / "grid.png"))

    # ── Save raw component rewards ────────────────────────────────────────────
    with open(out_dir / "comp_rewards.json", "w") as f:
        json.dump({
            "prompt": item.text,
            "scores": scores,
            "comp_matrices": comp_matrices,
        }, f, indent=2)

    print(f"  [{item.id}] best={max(scores):.3f}  worst={min(scores):.3f}  "
          f"spread={max(scores)-min(scores):.3f}")


def main():
    args = parse_args()

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    val_items = load_jsonl(args.val_jsonl)
    items = val_items[:args.num_prompts]
    print(f"[diag] Running diagnostics on {len(items)} prompts, G={args.num_generations}")

    print("[diag] Loading LlamaGenWrapper ...")
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=2.0,
        cfg_scale_train=args.cfg_scale_train,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        precision=args.precision,
        use_lora=False,   # diagnostics run on base model unless checkpoint specified
    )
    _ = wrapper.gpt
    _ = wrapper.vq_model
    _ = wrapper.t5

    print("[diag] Loading CARGORewardModel ...")
    from CARGO.scoring import CARGORewardModel
    reward_model_id = args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
    reward_model = CARGORewardModel(model_id=reward_model_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        item_dir = out_dir / item.id
        print(f"[diag] {item.id}: {item.text[:60]}")
        run_diagnostics_for_item(item, wrapper, reward_model, args, item_dir)

    print(f"\n[diag] Done. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
