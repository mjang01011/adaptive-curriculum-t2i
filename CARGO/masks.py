"""
CARGO mask utilities: Component-Aware Reward-Grounded token importance maps.

compute_cargo_mask()            — winner-aligned importance, 16×16 smoothing, soft floor
compute_cargo_mask_with_stats() — same + returns raw I and diagnostic stats dict
make_diversity_mask()           — pure token diversity (no reward weighting) baseline
make_random_mask()              — random baseline (same floor/scale)
make_early_token_mask()         — early-token sharpening baseline
overlay_mask_on_image()         — heatmap blend with optional stats annotation
make_raw_heatmap()              — colormap-only image (no image overlay)
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core mask computation
# ---------------------------------------------------------------------------

def compute_cargo_mask(
    tokens_b: torch.Tensor,   # (G, seq_len) int64 VQ tokens for one batch item
    R_c_b:    torch.Tensor,   # (G,) float component rewards for one batch item
    latent_size: int  = 16,
    mask_floor: float = 0.30,
) -> torch.Tensor:
    """
    Winner-aligned CARGO importance mask for a single batch item.

    Algorithm:
      winner = argmax R_c_b
      I[t] = mean_over_losers( 1[z_winner[t] != z_loser[t]] * max(R_winner - R_loser, 0) )
      Reshape to (latent_size, latent_size), apply 3×3 avg_pool (spatial smoothing),
      normalize to [0,1], apply soft floor: m = floor + (1-floor) * I_norm.

    Returns mask of shape (seq_len,) in [mask_floor, 1.0].
    Falls back to uniform 0.5-filled mask if all rewards are identical or G=1.
    """
    G, seq_len = tokens_b.shape
    device = tokens_b.device

    winner_idx    = int(R_c_b.argmax().item())
    winner_tokens = tokens_b[winner_idx]  # (seq_len,)

    I = torch.zeros(seq_len, device=device, dtype=torch.float32)
    n_valid = 0
    for g in range(G):
        if g == winner_idx:
            continue
        margin = max(float(R_c_b[winner_idx].item()) - float(R_c_b[g].item()), 0.0)
        if margin < 1e-6:
            continue
        diff = (winner_tokens != tokens_b[g]).float()
        I += diff * margin
        n_valid += 1

    if n_valid > 0:
        I = I / n_valid

    # Smooth in 16×16 image space (NOT in flat token order — adjacent tokens
    # are not necessarily horizontally adjacent in the latent grid).
    I_2d     = I.reshape(1, 1, latent_size, latent_size)
    I_smooth = F.avg_pool2d(I_2d, kernel_size=3, stride=1, padding=1).reshape(seq_len)

    # Normalize to [0, 1]
    I_min, I_max = I_smooth.min(), I_smooth.max()
    if (I_max - I_min).item() > 1e-6:
        I_norm = (I_smooth - I_min) / (I_max - I_min)
    else:
        I_norm = torch.full_like(I_smooth, 0.5)

    return mask_floor + (1.0 - mask_floor) * I_norm


def compute_cargo_mask_with_stats(
    tokens_b:    torch.Tensor,
    R_c_b:       torch.Tensor,
    latent_size: int   = 16,
    mask_floor:  float = 0.30,
) -> tuple:
    """
    Same as compute_cargo_mask but also returns (raw_I, stats_dict).

    raw_I    : (seq_len,) float32 — importance before smoothing/normalization/floor.
               All-zero if fallback triggered (reward tie or G=1).
    stats    : dict with min, max, mean, std, entropy of the FINAL mask,
               plus reward_spread and n_valid_losers.
    """
    G, seq_len = tokens_b.shape
    device = tokens_b.device

    winner_idx    = int(R_c_b.argmax().item())
    winner_tokens = tokens_b[winner_idx]

    I = torch.zeros(seq_len, device=device, dtype=torch.float32)
    n_valid = 0
    for g in range(G):
        if g == winner_idx:
            continue
        margin = max(float(R_c_b[winner_idx].item()) - float(R_c_b[g].item()), 0.0)
        if margin < 1e-6:
            continue
        I += (winner_tokens != tokens_b[g]).float() * margin
        n_valid += 1

    raw_I = I.clone()
    if n_valid > 0:
        I = I / n_valid

    I_2d     = I.reshape(1, 1, latent_size, latent_size)
    I_smooth = F.avg_pool2d(I_2d, kernel_size=3, stride=1, padding=1).reshape(seq_len)

    I_min, I_max = I_smooth.min(), I_smooth.max()
    if (I_max - I_min).item() > 1e-6:
        I_norm = (I_smooth - I_min) / (I_max - I_min)
    else:
        I_norm = torch.full_like(I_smooth, 0.5)

    mask = mask_floor + (1.0 - mask_floor) * I_norm

    reward_spread = float(R_c_b.max().item()) - float(R_c_b.min().item())

    # --- Stats on raw_I (before smoothing / normalization / floor) ---
    # These are the informative metrics.  The final mask is always in
    # [mask_floor, 1.0] so its std/entropy are dominated by normalization
    # artifacts and say nothing about spatial structure.
    ri_mean = float(raw_I.mean().item())
    ri_std  = float(raw_I.std().item())
    ri_max  = float(raw_I.max().item())
    ri_cv   = ri_std / (ri_mean + 1e-8)     # coeff of variation; high = concentrated

    # Gini coefficient of raw_I — 0 = perfectly uniform, 1 = all mass in one token
    ri_sorted = raw_I.sort().values.cpu()
    n_f   = float(seq_len)
    cumsum = ri_sorted.cumsum(0)
    ri_gini = float(
        1.0 - 2.0 * cumsum.sum().item() / (n_f * ri_sorted.sum().item() + 1e-9) + 1.0 / n_f
    )

    # Fraction of total raw_I in the top-10% of tokens (top 26 of 256)
    top_k  = max(1, seq_len // 10)
    ri_top10_frac = float(
        ri_sorted[-top_k:].sum().item() / (ri_sorted.sum().item() + 1e-9)
    )

    # --- Stats on raw_I_smooth (after spatial smoothing, before floor/norm) ---
    I_2d_s   = raw_I.reshape(1, 1, latent_size, latent_size)
    I_sm     = F.avg_pool2d(I_2d_s, 3, 1, 1).reshape(seq_len).cpu()
    sm_cv    = float(I_sm.std().item()) / (float(I_sm.mean().item()) + 1e-8)

    stats = {
        # Raw importance before any processing — primary signal quality metrics
        "raw_I_max":       ri_max,
        "raw_I_mean":      ri_mean,
        "raw_I_cv":        ri_cv,       # high (>1) = spatially concentrated
        "raw_I_gini":      ri_gini,     # high (→1) = concentrated, low (→0) = diffuse
        "raw_I_top10_frac": ri_top10_frac,  # >0.3 means top-10% tokens hold most signal
        "smooth_cv":       sm_cv,       # CV after spatial smoothing
        "reward_spread":   reward_spread,
        "n_valid_losers":  n_valid,
    }
    return mask, raw_I, stats


# ---------------------------------------------------------------------------
# Comparison baselines
# ---------------------------------------------------------------------------


def make_diversity_mask(
    tokens_b:    torch.Tensor,   # (G, seq_len) int64
    latent_size: int   = 16,
    mask_floor:  float = 0.30,
) -> torch.Tensor:
    """
    Pure token diversity baseline: fraction of generations that differ from
    the mode token at each position (no reward weighting).
    Useful for checking whether CARGO adds signal beyond raw diversity.
    """
    G, seq_len = tokens_b.shape
    device = tokens_b.device

    # mode token per position
    mode, _ = tokens_b.mode(dim=0)   # (seq_len,)
    diversity = (tokens_b != mode.unsqueeze(0)).float().mean(dim=0)  # (seq_len,)

    I_2d     = diversity.reshape(1, 1, latent_size, latent_size)
    I_smooth = F.avg_pool2d(I_2d, kernel_size=3, stride=1, padding=1).reshape(seq_len)

    I_min, I_max = I_smooth.min(), I_smooth.max()
    if (I_max - I_min).item() > 1e-6:
        I_norm = (I_smooth - I_min) / (I_max - I_min)
    else:
        I_norm = torch.full_like(I_smooth, 0.5)

    return mask_floor + (1.0 - mask_floor) * I_norm

def make_random_mask(
    seq_len:    int,
    mask_floor: float = 0.30,
    device:     torch.device = None,
    seed:       int  = None,
) -> torch.Tensor:
    """Random baseline mask with same floor/scale as CARGO."""
    gen = torch.Generator(device=device or torch.device("cpu"))
    if seed is not None:
        gen.manual_seed(seed)
    raw = torch.rand(seq_len, generator=gen,
                     device=device or torch.device("cpu"))
    return mask_floor + (1.0 - mask_floor) * raw


def make_early_token_mask(
    seq_len:    int,
    early_frac: float = 0.25,
    mask_floor: float = 0.30,
    device:     torch.device = None,
) -> torch.Tensor:
    """
    Early-token sharpening baseline: high weight on first (early_frac) tokens,
    floor weight on the rest. Models the hypothesis that early tokens determine
    coarse structure.
    """
    m = torch.full((seq_len,), mask_floor,
                   dtype=torch.float32, device=device or torch.device("cpu"))
    n_early = max(1, int(seq_len * early_frac))
    m[:n_early] = 1.0
    return m


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def overlay_mask_on_image(
    pil_img,
    mask_256:    torch.Tensor,   # (256,) or (16,16) float in [mask_floor,1]
    latent_size: int   = 16,
    alpha:       float = 0.55,
    colormap:    str   = "hot",
    stats:       dict  = None,   # if provided, annotate stats on image
    title:       str   = "",
) -> "PIL.Image.Image":
    """
    Resize 16×16 mask to image resolution, blend as heatmap overlay.
    Optionally annotate min/max/mean/std/norm_entropy from a stats dict.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    mask = mask_256.detach().cpu().float()
    if mask.ndim == 1:
        mask = mask.reshape(latent_size, latent_size)

    m_np = mask.numpy()
    m_np = (m_np - m_np.min()) / max(m_np.max() - m_np.min(), 1e-6)

    try:
        import matplotlib.cm as mcm
        cmap      = mcm.get_cmap(colormap)
        heat_rgba = (cmap(m_np) * 255).astype(np.uint8)
        heat_rgb  = heat_rgba[:, :, :3]
    except Exception:
        heat_rgb = (np.stack([m_np, np.zeros_like(m_np), 1.0 - m_np], axis=-1) * 255).astype(np.uint8)

    heat_pil = Image.fromarray(heat_rgb, "RGB").resize(pil_img.size, Image.NEAREST)
    orig_np  = np.array(pil_img.convert("RGB")).astype(np.float32)
    heat_np  = np.array(heat_pil).astype(np.float32)
    blended  = np.clip((1 - alpha) * orig_np + alpha * heat_np, 0, 255).astype(np.uint8)
    out = Image.fromarray(blended, "RGB")

    if stats or title:
        W, H = out.size
        lines = []
        if title:
            lines.append(title)
        if stats:
            lines.append(
                f"cv={stats.get('raw_I_cv',0):.2f}  gini={stats.get('raw_I_gini',0):.2f}"
                f"  top10%={stats.get('raw_I_top10_frac',0):.2f}"
                f"  spread={stats.get('reward_spread',0):.3f}"
            )
            lines.append(
                f"raw_I_max={stats.get('raw_I_max',0):.3f}"
                f"  valid={stats.get('n_valid_losers','?')}"
            )
        lh = 14
        strip = Image.new("RGB", (W, lh * len(lines) + 4), (20, 20, 20))
        draw  = ImageDraw.Draw(strip)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        for i, ln in enumerate(lines):
            draw.text((3, 2 + i * lh), ln, fill=(220, 220, 100), font=font)
        canvas = Image.new("RGB", (W, H + strip.height))
        canvas.paste(out, (0, 0))
        canvas.paste(strip, (0, H))
        return canvas

    return out


def make_raw_heatmap(
    mask_256:    torch.Tensor,   # (seq_len,) or (latent,latent)
    latent_size: int  = 16,
    out_size:    int  = 256,
    colormap:    str  = "hot",
) -> "PIL.Image.Image":
    """
    Pure colormap image of the mask values (no image blend).
    Useful when the overlay hides weak structure.
    """
    import numpy as np
    from PIL import Image

    mask = mask_256.detach().cpu().float()
    if mask.ndim == 1:
        mask = mask.reshape(latent_size, latent_size)
    m_np = mask.numpy()
    m_np = (m_np - m_np.min()) / max(m_np.max() - m_np.min(), 1e-6)

    try:
        import matplotlib.cm as mcm
        cmap  = mcm.get_cmap(colormap)
        rgb   = (cmap(m_np)[:, :, :3] * 255).astype(np.uint8)
    except Exception:
        rgb = (np.stack([m_np, np.zeros_like(m_np), 1.0 - m_np], axis=-1) * 255).astype(np.uint8)

    return Image.fromarray(rgb, "RGB").resize((out_size, out_size), Image.NEAREST)
