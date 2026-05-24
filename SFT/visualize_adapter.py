"""
Visual sanity-check for ImplicitCompositionAdapter.

Generates images for a fixed set of compositional prompts with the adapter
DISABLED and ENABLED, then saves a side-by-side grid.

Usage
-----
  python SFT/visualize_adapter.py \
    --adapter-ckpt outputs/implicit_adapter_v3/ckpt_step200.pt \
    --repo-root    LlamaGen \
    --gpt-ckpt     pretrained_models/t2i_XL_stage1_256.pt \
    --vq-ckpt      pretrained_models/vq_ds16_t2i.pt \
    --t5-path      pretrained_models/t5-ckpt \
    --out          eval_grid.png

  # Compare multiple checkpoints
  python SFT/visualize_adapter.py \
    --adapter-ckpt outputs/implicit_adapter_v3/ckpt_step200.pt \
                   outputs/implicit_adapter_v3/ckpt_step400.pt \
    --repo-root    LlamaGen \
    --gpt-ckpt     pretrained_models/t2i_XL_stage1_256.pt \
    --vq-ckpt      pretrained_models/vq_ds16_t2i.pt \
    --t5-path      pretrained_models/t5-ckpt \
    --out          eval_grid_compare.png
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

# ── Test prompts — designed to stress compositional binding ──────────────────

DEFAULT_PROMPTS = [
    "A red cube on top of a blue sphere.",
    "A small white cat sitting next to a large black dog.",
    "A green apple to the left of a red orange on a wooden table.",
    "A striped shirt hanging above a polka dot skirt.",
    "A metal chair in front of a wooden desk.",
    "A yellow rubber duck inside a glass bowl.",
    "A tall blue vase next to a short red candle.",
    "A silver fork to the right of a golden spoon.",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_models(args, device):
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        precision="bf16",
    )
    gpt = wrapper.gpt
    t5  = wrapper.t5
    vq  = wrapper.vq_model

    for p in gpt.parameters():
        p.requires_grad = False
    for p in t5.model.parameters():
        p.requires_grad = False

    return wrapper, gpt, t5, vq


def _attach_adapter(gpt, adapter_ckpt_path, device):
    from adaptive_curriculum.model.implicit_comp_adapter import (
        ImplicitCompositionAdapter, attach_implicit_adapter,
    )
    adapter = ImplicitCompositionAdapter(d_model=1280, n_comp_q=8, n_heads=8).to(device)
    ckpt = torch.load(adapter_ckpt_path, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()
    adapted_cls = attach_implicit_adapter(gpt, adapter)
    return adapter, adapted_cls


def _t5_encode(prompts, t5, device):
    caption_embs, emb_masks = t5.model.get_text_embeddings(prompts)
    # left-pad: move padding to front so last token is always real
    new_embs = []
    for emb, mask in zip(caption_embs, emb_masks):
        valid = int(mask.sum().item())
        new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
    embs = torch.stack(new_embs)
    masks = torch.flip(emb_masks, dims=[-1])
    c_indices = embs * masks[:, :, None]
    return c_indices.to(device), masks.to(device)


@torch.no_grad()
def _generate(gpt, vq, c_indices, c_masks, device, cfg_scale=7.5, temperature=1.0, top_k=2000, seed=42):
    from autoregressive.models.generate import generate
    torch.manual_seed(seed)
    latent_size = 16  # 256 // 16
    qzshape = [len(c_indices), 8, latent_size, latent_size]
    index_sample = generate(
        gpt, c_indices, latent_size ** 2,
        c_masks,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_k=top_k,
        top_p=1.0,
        sample_logits=True,
    )
    samples = vq.decode_code(index_sample, qzshape)  # [-1, 1]
    return samples  # [B, 3, 256, 256]


def _tensor_to_pil(t):
    """[-1,1] CHW → PIL"""
    t = (t.float().clamp(-1, 1) + 1) / 2  # [0, 1]
    return TF.to_pil_image(t.cpu())


def _add_label(img: Image.Image, text: str, font_size: int = 14) -> Image.Image:
    label_h = font_size + 6
    out = Image.new("RGB", (img.width, img.height + label_h), (30, 30, 30))
    out.paste(img, (0, label_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 2), text, fill=(220, 220, 220), font=font)
    return out


def _build_grid(col_images: list, col_labels: list, prompt_labels: list) -> Image.Image:
    """
    col_images: list of lists — col_images[col][row] = PIL image
    col_labels: header for each column
    prompt_labels: short prompt text per row
    """
    n_cols = len(col_images)
    n_rows = len(col_images[0])
    W, H   = col_images[0][0].size

    label_h  = 20
    header_h = 28
    cell_w   = W
    cell_h   = H + label_h
    grid_w   = n_cols * cell_w
    grid_h   = header_h + n_rows * cell_h

    canvas = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))
    draw   = ImageDraw.Draw(canvas)
    try:
        font_hdr  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_hdr  = ImageFont.load_default()
        font_body = font_hdr

    # Column headers
    for c, lbl in enumerate(col_labels):
        draw.text((c * cell_w + 4, 4), lbl, fill=(255, 215, 0), font=font_hdr)

    # Images + row prompt labels
    for r in range(n_rows):
        for c in range(n_cols):
            x = c * cell_w
            y = header_h + r * cell_h
            canvas.paste(col_images[c][r], (x, y + label_h))
            if c == 0:
                short = prompt_labels[r][:60]
                draw.text((x + 2, y + 2), short, fill=(180, 180, 255), font=font_body)

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = args.prompts or DEFAULT_PROMPTS
    print(f"[viz] {len(prompts)} prompts, {len(args.adapter_ckpt) + 1} columns (baseline + checkpoints)")

    wrapper, gpt, t5, vq = _load_models(args, device)
    gpt.eval()

    c_indices, c_masks = _t5_encode(prompts, t5, device)
    c_indices = c_indices.to(dtype=wrapper.dtype)
    c_masks   = c_masks.to(dtype=wrapper.dtype)

    col_images = []
    col_labels = []

    # ── Baseline (no adapter) ─────────────────────────────────────────────────
    print("[viz] Generating baseline (no adapter) ...")
    samples_base = _generate(gpt, vq, c_indices, c_masks, device, seed=args.seed)
    col_images.append([_tensor_to_pil(s) for s in samples_base])
    col_labels.append("Baseline (no adapter)")

    # ── Each adapter checkpoint ───────────────────────────────────────────────
    for ckpt_path in args.adapter_ckpt:
        step_tag = Path(ckpt_path).stem  # e.g. "ckpt_step200"
        print(f"[viz] Loading adapter from {ckpt_path} ...")
        adapter, adapted_cls = _attach_adapter(gpt, ckpt_path, device)
        adapted_cls._enabled = True

        # Log effective ratio for each prompt
        with torch.no_grad():
            C_base = gpt.cls_embedding.orig(c_indices, train=False)
            C_out, info = adapter(C_base.float())
            ratio = (C_out - C_base.float()).norm() / (C_base.float().norm() + 1e-8)
        print(f"[viz]   {step_tag}: γ={info['gamma']:.5f}  γΔ/base={ratio.item():.4f}")

        print(f"[viz] Generating images for {step_tag} ...")
        samples = _generate(gpt, vq, c_indices, c_masks, device, seed=args.seed)
        col_images.append([_tensor_to_pil(s) for s in samples])
        col_labels.append(step_tag)

        # Detach adapter for next checkpoint
        gpt.cls_embedding = gpt.cls_embedding.orig

    # ── Build grid ────────────────────────────────────────────────────────────
    short_prompts = [p[:55] + ("…" if len(p) > 55 else "") for p in prompts]
    grid = _build_grid(col_images, col_labels, short_prompts)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"[viz] Saved grid → {out_path}  ({grid.width}×{grid.height})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-ckpt", nargs="+", required=True,
                   help="One or more adapter checkpoint .pt files")
    p.add_argument("--repo-root",    required=True)
    p.add_argument("--gpt-ckpt",     required=True)
    p.add_argument("--vq-ckpt",      required=True)
    p.add_argument("--t5-path",      required=True)
    p.add_argument("--out",          default="eval_grid.png")
    p.add_argument("--prompts",      nargs="+", default=None,
                   help="Override default prompts (one string per prompt)")
    p.add_argument("--cfg-scale",    type=float, default=7.5)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top-k",        type=int,   default=2000)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()
    main(args)
