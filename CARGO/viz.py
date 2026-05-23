"""
CARGO visualization grid.

Three entry points:

1. plot_training_curves(train_log_jsonl, output_path)
   Reads train_log.jsonl and plots:
     - Hard reward and smooth reward over val steps
     - Per-component EMA over training steps
     - Elite SFT loss and advantage magnitudes

2. make_cargo_panel(images, masks_by_comp, comp_scores, rewards, prompt, output_path)
   Per-step inline panel: generated images sorted by reward, CARGO mask overlays
   per component, and a component score bar chart.

3. make_progression_panel(val_images_dir, prompt_ids, steps, output_path)
   Shows how specific images evolve across training steps side-by-side.

Usage (post-training analysis):

  # Training curves
  python CARGO/viz.py curves \\
    --log $PROJECT/outputs/cargo_attr_v2/train_log.jsonl \\
    --out $PROJECT/outputs/cargo_attr_v2/training_curves.png

  # Mask panel for one checkpoint + set of prompts
  python CARGO/viz.py masks \\
    --val-jsonl   $PROJECT/data/attribute_binding/attribute_binding_val_20.jsonl \\
    --gpt-ckpt    $PROJECT/outputs/cargo_attr_v2/best_checkpoint.pt \\
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \\
    --t5-path     $PRETRAINED/t5-ckpt \\
    --qwen-model  $PRETRAINED/Qwen3-VL-4B-Instruct \\
    --repo-root   $PROJECT/LlamaGen \\
    --reward-mode grpo_attr_contrastive_rubric_v2 \\
    --num-prompts 4 --num-generations 8 \\
    --out         $PROJECT/outputs/cargo_attr_v2/mask_panel.png

  # Training progression panel
  python CARGO/viz.py progression \\
    --val-images-dir $PROJECT/outputs/cargo_attr_v2/val_images \\
    --steps 0 15 30 60 120 \\
    --num-prompts 4 \\
    --out $PROJECT/outputs/cargo_attr_v2/progression.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(log_path: str, output_path: str):
    """
    Parse train_log.jsonl and write a multi-panel matplotlib figure.

    Panels:
      A) Val hard reward and val smooth reward vs validation step
      B) Train EMA reward and per-component EMAs vs training step
      C) Advantage magnitude (scalar vs CARGO) and SFT loss vs step
      D) KL divergence and clip fraction vs step
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    val_steps       = []
    val_hard        = []
    val_smooth      = []

    train_steps     = []
    train_rewards   = []
    comp_emas: dict = {}       # key → list of (step, val)
    sft_losses      = []
    abs_adv_scalar  = []
    abs_adv_cargo   = []
    kl_vals         = []
    clip_fracs      = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            step = d.get("step", 0)

            if "val_reward" in d:
                val_steps.append(step)
                val_hard.append(d["val_reward"])
                val_smooth.append(d.get("val_smooth_reward", d["val_reward"]))

            if "mean_reward" in d:
                train_steps.append(step)
                train_rewards.append(d["mean_reward"])

                for k, v in d.items():
                    if k.startswith("train/ema/"):
                        comp_key = k[len("train/ema/"):]
                        comp_emas.setdefault(comp_key, []).append((step, v))

                if "sft_loss" in d and d["sft_loss"] > 0:
                    sft_losses.append((step, d["sft_loss"]))

                if "mean_abs_advantage" in d:
                    abs_adv_scalar.append((step, d["mean_abs_advantage"]))
                if "mean_abs_cargo_advantage" in d:
                    abs_adv_cargo.append((step, d["mean_abs_cargo_advantage"]))
                if "mean_kl" in d:
                    kl_vals.append((step, d["mean_kl"]))
                if "clip_frac" in d:
                    clip_fracs.append((step, d["clip_frac"]))

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # ── A: val rewards ────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    if val_steps:
        ax_a.plot(val_steps, val_hard,   "o-", color="steelblue", label="hard", linewidth=1.8)
        ax_a.plot(val_steps, val_smooth, "s--", color="darkorange", label="smooth (CARGO v2)", linewidth=1.8)
    ax_a.set_title("Val Reward vs Step", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("step"); ax_a.set_ylabel("reward")
    ax_a.legend(fontsize=9); ax_a.grid(alpha=0.3)

    # ── B: per-component EMAs ─────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]
    for ci, (comp_key, vals) in enumerate(sorted(comp_emas.items())):
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        color = _PALETTE[ci % len(_PALETTE)]
        ax_b.plot(xs, ys, color=color, linewidth=1.4,
                  label=comp_key[:20], alpha=0.85)
    ax_b.set_title("Component EMA Scores vs Step", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("step"); ax_b.set_ylabel("EMA score")
    if comp_emas:
        ax_b.legend(fontsize=7, ncol=2, loc="lower right")
    ax_b.grid(alpha=0.3)

    # ── C: advantage magnitudes + SFT loss ───────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    if abs_adv_scalar:
        xs, ys = zip(*abs_adv_scalar)
        ax_c.plot(xs, ys, "-", color="royalblue", label="|adv| scalar", linewidth=1.4)
    if abs_adv_cargo:
        xs, ys = zip(*abs_adv_cargo)
        ax_c.plot(xs, ys, "-", color="darkorange", label="|adv| CARGO", linewidth=1.4)
    ax_c2 = ax_c.twinx()
    if sft_losses:
        xs, ys = zip(*sft_losses)
        ax_c2.plot(xs, ys, "--", color="green", label="SFT loss", linewidth=1.2, alpha=0.7)
        ax_c2.set_ylabel("SFT loss", color="green", fontsize=9)
        ax_c2.tick_params(axis="y", labelcolor="green")
    ax_c.set_title("Advantage Magnitudes & Elite SFT Loss", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("step"); ax_c.set_ylabel("|advantage|")
    lines_c, labels_c = ax_c.get_legend_handles_labels()
    lines_c2, labels_c2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines_c + lines_c2, labels_c + labels_c2, fontsize=8)
    ax_c.grid(alpha=0.3)

    # ── D: KL + clip fraction ────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    if kl_vals:
        xs, ys = zip(*kl_vals)
        ax_d.plot(xs, ys, "-", color="purple", label="mean KL", linewidth=1.4)
    ax_d2 = ax_d.twinx()
    if clip_fracs:
        xs, ys = zip(*clip_fracs)
        ax_d2.plot(xs, ys, "--", color="crimson", label="clip frac", linewidth=1.2, alpha=0.8)
        ax_d2.set_ylabel("clip fraction", color="crimson", fontsize=9)
        ax_d2.tick_params(axis="y", labelcolor="crimson")
    ax_d.set_title("KL Divergence & PPO Clip Fraction", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("step"); ax_d.set_ylabel("mean KL")
    lines_d, labels_d = ax_d.get_legend_handles_labels()
    lines_d2, labels_d2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines_d + lines_d2, labels_d + labels_d2, fontsize=8)
    ax_d.grid(alpha=0.3)

    fig.suptitle(f"CARGO Training: {Path(log_path).parent.name}", fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved training curves → {output_path}")


# ---------------------------------------------------------------------------
# 2. CARGO mask panel
# ---------------------------------------------------------------------------

def make_cargo_panel(
    images:        list,         # G PIL images
    masks_by_comp: dict,         # {comp_key: (seq_len,) tensor}
    comp_scores:   list,         # list of G dicts {comp_key: float}
    rewards:       list,         # G floats
    prompt:        str,
    output_path:   str,
    latent_size:   int = 16,
    n_cols_images: int = 4,
):
    """
    Creates a panel with three sections:
      Top: generated images sorted by reward (best first), hard/smooth score annotations
      Mid: CARGO mask overlaid on the best image, one sub-panel per component
      Bot: bar chart of component scores for best vs worst sample
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from CARGO.masks import overlay_mask_on_image

    G = len(images)
    order = sorted(range(G), key=lambda i: rewards[i], reverse=True)

    n_comp = len(masks_by_comp)
    # Layout: row 0 = images, row 1 = masks, row 2 = bar chart
    n_img_rows = (G + n_cols_images - 1) // n_cols_images
    fig_h      = 3.0 * n_img_rows + 3.0 + 3.0  # images + masks + bar
    fig_w      = max(n_cols_images, n_comp, 4) * 3.0
    fig, axes  = plt.subplots(
        n_img_rows + 2,
        max(n_cols_images, n_comp, 1),
        figsize=(fig_w, fig_h),
    )
    # Make axes always 2D
    if axes.ndim == 1:
        axes = axes[None, :]

    # ── Row(s) 0..n_img_rows-1: generated images ─────────────────────────────
    for slot, g_idx in enumerate(order):
        row, col = divmod(slot, n_cols_images)
        ax = axes[row, col]
        ax.imshow(images[g_idx])
        r = rewards[g_idx]
        comp_str = "  ".join(
            f"{k[:5]}={v:.2f}" for k, v in sorted(comp_scores[g_idx].items())
            if k not in {"uncertain_frac", "mean_logit_margin"}
        )[:38]
        rank_label = "BEST" if slot == 0 else ("WORST" if slot == G - 1 else f"#{slot+1}")
        ax.set_title(f"{rank_label}  r={r:.3f}\n{comp_str}", fontsize=6.5, pad=2)
        ax.axis("off")

    # Hide unused image slots
    for slot in range(G, n_img_rows * n_cols_images):
        row, col = divmod(slot, n_cols_images)
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis("off")

    # ── Row n_img_rows: mask overlays on best image ───────────────────────────
    mask_row = n_img_rows
    best_img  = images[order[0]]
    for ci, (comp_key, mask) in enumerate(sorted(masks_by_comp.items())):
        if ci >= axes.shape[1]:
            break
        ax = axes[mask_row, ci]
        overlaid = overlay_mask_on_image(best_img, mask, latent_size=latent_size)
        ax.imshow(overlaid)
        ax.set_title(f"CARGO mask\n{comp_key[:18]}", fontsize=7, pad=2)
        ax.axis("off")
    for ci in range(len(masks_by_comp), axes.shape[1]):
        axes[mask_row, ci].axis("off")

    # ── Row n_img_rows+1: component score bars (best vs worst) ───────────────
    bar_row   = n_img_rows + 1
    comp_keys = sorted(
        k for k in comp_scores[0].keys()
        if k not in {"uncertain_frac", "mean_logit_margin"}
    )
    x       = np.arange(len(comp_keys))
    best_v  = [comp_scores[order[0]].get(k, 0.0)  for k in comp_keys]
    worst_v = [comp_scores[order[-1]].get(k, 0.0) for k in comp_keys]

    ax_bar = fig.add_subplot(fig.axes[-1])   # reuse last cell
    # Span the full bottom row
    for ci in range(axes.shape[1]):
        axes[bar_row, ci].axis("off")
    # Create a new axes that spans the full bottom row
    ax_bar = fig.add_axes([
        axes[bar_row, 0].get_position().x0,
        axes[bar_row, 0].get_position().y0,
        axes[bar_row, -1].get_position().x1 - axes[bar_row, 0].get_position().x0,
        axes[bar_row, 0].get_position().height,
    ])
    w = 0.35
    ax_bar.bar(x - w/2, best_v,  w, color="steelblue", alpha=0.85, label=f"best  r={rewards[order[0]]:.3f}")
    ax_bar.bar(x + w/2, worst_v, w, color="salmon",    alpha=0.85, label=f"worst r={rewards[order[-1]]:.3f}")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([k[:12] for k in comp_keys], rotation=30, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.set_ylabel("score")
    ax_bar.set_title("Component Scores: Best vs Worst Sample", fontsize=9)
    ax_bar.legend(fontsize=8)

    prompt_short = prompt[:90] + ("…" if len(prompt) > 90 else "")
    fig.suptitle(f'"{prompt_short}"', fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved CARGO panel → {output_path}")


# ---------------------------------------------------------------------------
# 3. Training progression panel
# ---------------------------------------------------------------------------

def make_progression_panel(
    val_images_dir: str,
    prompt_ids:     list,    # list of item.id strings to show
    steps:          list,    # list of int steps (must match saved val_images dirs)
    output_path:    str,
    img_size:       int = 128,
):
    """
    Grid: rows = prompts, cols = training steps.
    Shows how each prompt's generated image evolves across checkpoints.
    Requires val_images saved by run_val() (step_{step:06d}/{item_id}.png).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_rows = len(prompt_ids)
    n_cols = len(steps)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.0, n_rows * 2.2),
                              squeeze=False)

    val_dir = Path(val_images_dir)
    for ci, step in enumerate(steps):
        step_dir = val_dir / f"step_{step:06d}"
        for ri, pid in enumerate(prompt_ids):
            ax = axes[ri, ci]
            img_path = step_dir / f"{pid}.png"
            if img_path.exists():
                img = Image.open(img_path).resize((img_size, img_size))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
            ax.axis("off")
            if ri == 0:
                ax.set_title(f"step {step}", fontsize=8, pad=2)
            if ci == 0:
                ax.set_ylabel(pid[:16], fontsize=7, rotation=0,
                               ha="right", va="center", labelpad=4)

    fig.suptitle("CARGO Training Progression", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved progression panel → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _cmd_curves(args):
    plot_training_curves(args.log, args.out)


def _cmd_masks(args):
    import torch
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    from CARGO.scoring import CARGORewardModel
    from CARGO.masks   import compute_cargo_mask
    from CARGO.rewards import META_KEYS

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    from adaptive_curriculum.data.schemas import BucketItem

    val_items = []
    with open(args.val_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                val_items.append(BucketItem.from_dict(json.loads(line)))
    items = val_items[:args.num_prompts]

    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=2.0, cfg_scale_train=4.0,
        temperature=1.0, top_k=1000, top_p=1.0,
        precision="bf16",
        use_lora=bool(args.lora_ckpt),
    )
    if args.lora_ckpt:
        wrapper.load_checkpoint(args.lora_ckpt)
    _ = wrapper.gpt; _ = wrapper.vq_model; _ = wrapper.t5

    reward_model = CARGORewardModel(model_id=args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct")

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    import contextlib
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    for item in items:
        G = args.num_generations
        device = wrapper.device

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
                idx_s = generate(wrapper.gpt, c_indices, wrapper.latent_size ** 2,
                                  c_emb_masks, cfg_scale=wrapper.cfg_scale_train,
                                  temperature=1.0, top_k=1000, top_p=1.0, sample_logits=True)
                all_tokens.append(idx_s)
                dec = wrapper.vq_model.decode_code(idx_s, qzshape)
                img_t = (dec[0].float().clamp(-1, 1) + 1) / 2
                all_pils.append(TF.to_pil_image(img_t.cpu()))
        wrapper._disable_kv_cache()

        pairs    = [(pil, item) for pil in all_pils]
        results  = reward_model.score_images_batch(pairs, mode=args.reward_mode)
        rewards  = [r["score"] for r in results]
        comp_scores = [r.get("component_scores", {}) for r in results]

        # Build CARGO masks per component
        stacked = torch.stack(all_tokens, dim=0).squeeze(1).to(device)  # (G, seq_len)
        comp_matrices = {}
        for g, r in enumerate(results):
            for k, v in r.get("component_scores", {}).items():
                comp_matrices.setdefault(k, []).append(float(v))

        masks_by_comp = {}
        for key, vals in comp_matrices.items():
            if key in META_KEYS:
                continue
            R_c = torch.tensor(vals, dtype=torch.float32).to(device)
            masks_by_comp[key] = compute_cargo_mask(stacked, R_c,
                                                     latent_size=wrapper.latent_size)

        suffix = args.out.replace(".png", f"_{item.id}.png")
        make_cargo_panel(
            all_pils, masks_by_comp, comp_scores, rewards,
            prompt=item.text,
            output_path=suffix,
            latent_size=wrapper.latent_size,
        )


def _cmd_progression(args):
    # Discover prompt IDs from first step directory
    val_dir = Path(args.val_images_dir)
    steps   = [int(s) for s in args.steps]
    # Use first available step dir to enumerate prompt IDs
    first_step_dir = val_dir / f"step_{steps[0]:06d}"
    if args.prompt_ids:
        prompt_ids = args.prompt_ids
    elif first_step_dir.exists():
        prompt_ids = [p.stem for p in sorted(first_step_dir.glob("*.png"))][:args.num_prompts]
    else:
        print(f"[viz] No images found in {first_step_dir}")
        return
    make_progression_panel(args.val_images_dir, prompt_ids, steps, args.out)


def main():
    p = argparse.ArgumentParser(description="CARGO visualization tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    # curves
    pc = sub.add_parser("curves", help="Plot training curves from train_log.jsonl")
    pc.add_argument("--log", required=True)
    pc.add_argument("--out", required=True)

    # masks
    pm = sub.add_parser("masks", help="Generate CARGO mask panel for a set of prompts")
    pm.add_argument("--val-jsonl",       required=True)
    pm.add_argument("--repo-root",       required=True)
    pm.add_argument("--gpt-ckpt",        required=True)
    pm.add_argument("--vq-ckpt",         required=True)
    pm.add_argument("--t5-path",         required=True)
    pm.add_argument("--qwen-model",      default=None)
    pm.add_argument("--lora-ckpt",       default=None, help="LoRA checkpoint (omit for base model)")
    pm.add_argument("--reward-mode",     default="grpo_attr_contrastive_rubric_v2")
    pm.add_argument("--num-prompts",     type=int, default=4)
    pm.add_argument("--num-generations", type=int, default=8)
    pm.add_argument("--out",             required=True, help="Output PNG path (will be suffixed with prompt ID)")

    # progression
    pp = sub.add_parser("progression", help="Show image progression across training steps")
    pp.add_argument("--val-images-dir", required=True)
    pp.add_argument("--steps",          nargs="+", required=True,
                    help="Training steps to include (e.g. 0 15 30 60 120)")
    pp.add_argument("--prompt-ids",     nargs="*", default=None,
                    help="Specific item IDs to show (default: first --num-prompts from step dir)")
    pp.add_argument("--num-prompts",    type=int, default=4)
    pp.add_argument("--out",            required=True)

    args = p.parse_args()
    {"curves": _cmd_curves, "masks": _cmd_masks, "progression": _cmd_progression}[args.cmd](args)


if __name__ == "__main__":
    main()
