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
    p.add_argument("--cargo-mask-source", default="pixel", choices=["pixel", "vq"],
                   help="pixel (default): L1 RGB patch distances; vq: VQ token identity (ablation)")
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


def _annotate_pil(pil_img, lines, bg=(20, 20, 20), fg=(220, 220, 100)):
    """Attach a text strip below a PIL image."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = pil_img.size
    lh   = 14
    strip = Image.new("RGB", (W, lh * len(lines) + 4), bg)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for i, ln in enumerate(lines):
        draw.text((3, 2 + i * lh), ln[:55], fill=fg, font=font)
    canvas = Image.new("RGB", (W, H + strip.height))
    canvas.paste(pil_img, (0, 0))
    canvas.paste(strip, (0, H))
    return canvas


def _diff_heatmap(winner_tokens, all_tokens_list, R_c, latent_size, out_size=256):
    """
    Pixel heatmap of token positions that differ between winner and losers,
    margin-weighted. Returns a PIL image at out_size × out_size.
    """
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F

    winner_idx = int(R_c.argmax().item())
    seq_len    = winner_tokens.shape[0]
    I = torch.zeros(seq_len, dtype=torch.float32)
    n = 0
    for g, tok in enumerate(all_tokens_list):
        if g == winner_idx:
            continue
        margin = max(float(R_c[winner_idx]) - float(R_c[g]), 0.0)
        if margin < 1e-6:
            continue
        I += (winner_tokens != tok).float().cpu() * margin
        n += 1
    if n > 0:
        I = I / n

    I_2d = I.reshape(1, 1, latent_size, latent_size)
    I_sm = F.avg_pool2d(I_2d, 3, 1, 1).reshape(latent_size, latent_size).numpy()
    I_sm = (I_sm - I_sm.min()) / max(I_sm.max() - I_sm.min(), 1e-6)

    try:
        import matplotlib.cm as mcm
        rgb = (mcm.get_cmap("hot")(I_sm)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        rgb = (np.stack([I_sm, np.zeros_like(I_sm), 1 - I_sm], -1) * 255).astype(np.uint8)

    return Image.fromarray(rgb).resize((out_size, out_size), Image.NEAREST)


def run_diagnostics_for_item(
    item, wrapper, reward_model, args, out_dir: Path
):
    import contextlib
    from autoregressive.models.generate import generate
    from CARGO.masks import (
        compute_cargo_mask_with_stats,
        compute_cargo_mask_pixel_with_stats,
        make_diversity_mask,
        make_early_token_mask,
        make_random_mask,
        overlay_mask_on_image,
        make_raw_heatmap,
    )
    from CARGO.rewards import META_KEYS
    import torchvision.transforms.functional as TF
    from PIL import Image

    use_pixel = (args.cargo_mask_source == "pixel")

    out_dir.mkdir(parents=True, exist_ok=True)
    G      = args.num_generations
    device = wrapper.device
    ls     = wrapper.latent_size

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

    qzshape    = [1, wrapper.codebook_embed_dim, ls, ls]
    all_tokens = []
    all_pils   = []

    with torch.no_grad(), _sdpa_ctx():
        for _ in range(G):
            idx = generate(
                wrapper.gpt, c_indices, ls ** 2,
                c_emb_masks,
                cfg_scale=wrapper.cfg_scale_train,
                temperature=wrapper.temperature,
                top_k=wrapper.top_k,
                top_p=wrapper.top_p,
                sample_logits=True,
            )  # (1, seq_len)
            all_tokens.append(idx)
            decoded = wrapper.vq_model.decode_code(idx, qzshape)
            img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
            all_pils.append(TF.to_pil_image(img_t.cpu()))

    wrapper._disable_kv_cache()

    stacked_tokens = torch.stack(all_tokens, dim=0).squeeze(1).to(device)  # (G, seq_len)
    seq_len = stacked_tokens.shape[1]

    # ── Score all G images ────────────────────────────────────────────────────
    pairs   = [(pil, item) for pil in all_pils]
    results = reward_model.score_images_batch(pairs, mode=args.reward_mode)

    scores = [float(r["score"]) for r in results]
    comp_matrices = {}
    for g, r in enumerate(results):
        for k, v in (r.get("component_scores") or {}).items():
            comp_matrices.setdefault(k, []).append(float(v))

    order      = sorted(range(G), key=lambda i: scores[i], reverse=True)
    best_idx   = order[0]
    worst_idx  = order[-1]
    best_pil   = all_pils[best_idx]
    worst_pil  = all_pils[worst_idx]
    best_toks  = stacked_tokens[best_idx]

    # ── Save best / worst ─────────────────────────────────────────────────────
    best_pil.save(str(out_dir / "best.png"))
    worst_pil.save(str(out_dir / "worst.png"))

    # ── Baselines (computed once; shared across components) ───────────────────
    div_mask   = make_diversity_mask(stacked_tokens, latent_size=ls,
                                      mask_floor=args.cargo_mask_floor)
    early_mask = make_early_token_mask(seq_len, mask_floor=args.cargo_mask_floor,
                                        device=device)
    rand_mask  = make_random_mask(seq_len, mask_floor=args.cargo_mask_floor,
                                   device=device, seed=0)

    # ── Per-component panels ──────────────────────────────────────────────────
    cargo_keys   = [k for k in comp_matrices if k not in META_KEYS]
    all_stats    = {}
    W256         = best_pil.size[0]

    for key in cargo_keys:
        R_c  = torch.tensor(comp_matrices[key], dtype=torch.float32, device=device)
        safe = key.replace("/", "_")

        # CARGO mask + stats — dispatch on mask source
        if use_pixel:
            mask, raw_I, stats = compute_cargo_mask_pixel_with_stats(
                all_pils, R_c, latent_size=ls, mask_floor=args.cargo_mask_floor
            )
        else:
            mask, raw_I, stats = compute_cargo_mask_with_stats(
                stacked_tokens, R_c, latent_size=ls, mask_floor=args.cargo_mask_floor
            )
        all_stats[key] = stats

        # ── Overlay with stats annotation ─────────────────────────────────────
        overlay = overlay_mask_on_image(best_pil, mask, latent_size=ls,
                                         stats=stats, title=f"CARGO ({args.cargo_mask_source}): {key}")
        overlay.save(str(out_dir / f"cargo_{safe}_mask.png"))

        # ── Raw heatmap (no image, shows true structure) ──────────────────────
        make_raw_heatmap(mask, latent_size=ls, out_size=W256).save(
            str(out_dir / f"cargo_{safe}_raw.png")
        )
        make_raw_heatmap(raw_I if raw_I.max() > 1e-8 else torch.ones(seq_len, device=device) * 0.5,
                         latent_size=ls, out_size=W256).save(
            str(out_dir / f"cargo_{safe}_raw_I.png")
        )

        # ── Comparison panel: best | worst | diff | CARGO | diversity | early ──
        reward_spread = float(R_c.max() - R_c.min())
        diff_img = _diff_heatmap(best_toks, [stacked_tokens[g] for g in range(G)],
                                  R_c, ls, out_size=W256)

        panels = [
            _annotate_pil(best_pil,   [f"BEST  r={scores[best_idx]:.3f}",  key]),
            _annotate_pil(worst_pil,  [f"WORST r={scores[worst_idx]:.3f}", f"spread={reward_spread:.3f}"]),
            _annotate_pil(diff_img,   ["winner-loser diff (raw I)", f"n_valid={stats['n_valid_losers']}"]),
            overlay_mask_on_image(best_pil, mask,      latent_size=ls,
                                   stats=stats, title="CARGO"),
            overlay_mask_on_image(best_pil, div_mask,  latent_size=ls,
                                   title="diversity (no reward weighting)"),
            overlay_mask_on_image(best_pil, early_mask, latent_size=ls,
                                   title="early-token baseline"),
        ]

        # Pad all to same height then concatenate horizontally
        max_h = max(p.size[1] for p in panels)
        padded = []
        for p in panels:
            if p.size[1] < max_h:
                canvas = Image.new("RGB", (p.size[0], max_h), (40, 40, 40))
                canvas.paste(p, (0, 0))
                padded.append(canvas)
            else:
                padded.append(p)

        row_w = sum(p.size[0] for p in padded)
        panel_img = Image.new("RGB", (row_w, max_h))
        x = 0
        for p in padded:
            panel_img.paste(p, (x, 0))
            x += p.size[0]

        panel_img.save(str(out_dir / f"panel_{safe}.png"))

    # ── Annotated grid of all G images ────────────────────────────────────────
    grid_labels = []
    for g in range(G):
        comp_str = "  ".join(
            f"{k[:6]}={comp_matrices[k][g]:.2f}"
            for k in sorted(comp_matrices)
            if k not in META_KEYS
        )
        grid_labels.append(f"r={scores[g]:.3f}  {comp_str[:28]}")

    grid = _make_image_grid(all_pils, grid_labels, cols=min(4, G))
    grid.save(str(out_dir / "grid.png"))

    # ── Save raw data ─────────────────────────────────────────────────────────
    with open(out_dir / "comp_rewards.json", "w") as f:
        json.dump({
            "prompt":        item.text,
            "scores":        scores,
            "comp_matrices": comp_matrices,
            "mask_stats":    all_stats,
        }, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    spread = max(scores) - min(scores)
    stat_lines = [
        f"  [{item.id}] best={max(scores):.3f}  worst={min(scores):.3f}  "
        f"spread={spread:.3f}  mask_source={args.cargo_mask_source}"
    ]
    for k, s in all_stats.items():
        # Interpret signal quality:
        #   raw_I_cv > 1.0  → concentrated peaks (good)
        #   raw_I_gini > 0.4 → unequal distribution (good)
        #   top10_frac > 0.30 → top tokens capture most importance (good)
        signal = "STRONG" if s["raw_I_cv"] > 1.5 and s["raw_I_gini"] > 0.4 else \
                 "MODERATE" if s["raw_I_cv"] > 0.5 and s["raw_I_max"] > 0.5 else \
                 "WEAK"
        stat_lines.append(
            f"    {k:<26} raw_I_max={s['raw_I_max']:.3f}  cv={s['raw_I_cv']:.2f}"
            f"  gini={s['raw_I_gini']:.2f}  top10%={s['raw_I_top10_frac']:.2f}"
            f"  valid={s['n_valid_losers']}  [{signal}]"
        )
    print("\n".join(stat_lines))


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
