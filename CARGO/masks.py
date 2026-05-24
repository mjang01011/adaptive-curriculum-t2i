"""
CARGO mask utilities: Component-Aware Reward-Grounded token importance maps.

Pixel-space masks (default, --cargo-mask-source pixel):
  compute_cargo_mask_pixel()            — winner-aligned L1 patch distances
  compute_cargo_mask_pixel_with_stats() — same + raw I map and diagnostic stats

VQ-token masks (ablation, --cargo-mask-source vq):
  compute_cargo_mask()            — winner-aligned token-identity differences
  compute_cargo_mask_with_stats() — same + raw I and stats

Baselines:
  make_diversity_mask()    — pure token diversity (no reward weighting)
  make_random_mask()       — random baseline
  make_early_token_mask()  — early-token sharpening baseline

Visualization:
  overlay_mask_on_image()  — heatmap blend with optional stats annotation
  make_raw_heatmap()       — colormap-only image (no image overlay)
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mask_stats(raw_I: torch.Tensor, mask: torch.Tensor,
                reward_spread: float, n_valid: int, seq_len: int) -> dict:
    """Compute diagnostic stats on raw_I (pre-normalization). Mask stats are
    intentionally excluded — they're always near-uniform after normalization."""
    ri_mean = float(raw_I.mean().item())
    ri_std  = float(raw_I.std().item())
    ri_max  = float(raw_I.max().item())
    ri_cv   = ri_std / (ri_mean + 1e-8)

    ri_sorted = raw_I.sort().values.cpu()
    n_f = float(seq_len)
    cumsum = ri_sorted.cumsum(0)
    ri_gini = float(
        1.0 - 2.0 * cumsum.sum().item() / (n_f * ri_sorted.sum().item() + 1e-9) + 1.0 / n_f
    )
    top_k = max(1, seq_len // 10)
    ri_top10_frac = float(
        ri_sorted[-top_k:].sum().item() / (ri_sorted.sum().item() + 1e-9)
    )
    return {
        "raw_I_max":        ri_max,
        "raw_I_mean":       ri_mean,
        "raw_I_cv":         ri_cv,
        "raw_I_gini":       ri_gini,
        "raw_I_top10_frac": ri_top10_frac,
        "reward_spread":    reward_spread,
        "n_valid_losers":   n_valid,
    }


def _pil_to_patch_grid(
    pil_img,
    latent_size: int = 16,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert a PIL image → (3, latent_size, latent_size) float32 in [0, 1]
    by average-pooling the 256×256 RGB image into a latent_size×latent_size grid.
    Each cell is the mean RGB of one 16×16 pixel patch.
    """
    import torchvision.transforms.functional as TF
    img_t = TF.to_tensor(pil_img.convert("RGB"))   # (3, H, W) [0,1]
    if img_t.shape[-2] != 256 or img_t.shape[-1] != 256:
        img_t = TF.resize(img_t, [256, 256], antialias=True)
    patch_size = 256 // latent_size                # 16 for latent_size=16
    grid = F.avg_pool2d(
        img_t.unsqueeze(0), kernel_size=patch_size, stride=patch_size
    ).squeeze(0)                                   # (3, latent_size, latent_size)
    return grid.to(device) if device is not None else grid


def _apply_smooth_normalize_floor(
    I_2d: torch.Tensor,   # (latent_size, latent_size)
    latent_size: int,
    mask_floor: float,
) -> torch.Tensor:
    """3×3 smooth → normalize [0,1] → soft floor. Returns (seq_len,) mask."""
    seq_len  = latent_size * latent_size
    I_sm     = F.avg_pool2d(
        I_2d.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze()                                    # (latent_size, latent_size)
    I_min, I_max = I_sm.min(), I_sm.max()
    if (I_max - I_min).item() > 1e-6:
        I_norm = (I_sm - I_min) / (I_max - I_min)
    else:
        I_norm = torch.full_like(I_sm, 0.5)
    return (mask_floor + (1.0 - mask_floor) * I_norm).reshape(seq_len)


# ---------------------------------------------------------------------------
# Pixel-space mask (default)
# ---------------------------------------------------------------------------

def compute_cargo_mask_pixel(
    images_b,             # list of G PIL images for one batch item
    R_c_b: torch.Tensor,  # (G,) float component rewards
    latent_size: int  = 16,
    mask_floor:  float = 0.30,
) -> torch.Tensor:
    """
    Winner-aligned CARGO mask from decoded pixel space.

    For each 16×16 patch:
      I[r,c] = mean_over_valid_losers(
          L1_RGB(winner_patch[r,c], loser_patch[r,c]) * max(R_winner - R_loser, 0)
      )
    Then: 3×3 spatial smooth → normalize [0,1] → soft floor.

    Returns (seq_len,) float32 in [mask_floor, 1.0].
    Pixel L1 differences are semantically consistent across generations, unlike
    VQ token identities which can vary arbitrarily even for similar image content.
    """
    G      = len(images_b)
    device = R_c_b.device
    ls     = latent_size

    winner_idx  = int(R_c_b.argmax().item())
    winner_grid = _pil_to_patch_grid(images_b[winner_idx], ls, device)  # (3, L, L)

    I       = torch.zeros(ls, ls, device=device, dtype=torch.float32)
    n_valid = 0
    for g in range(G):
        if g == winner_idx:
            continue
        margin = max(float(R_c_b[winner_idx].item()) - float(R_c_b[g].item()), 0.0)
        if margin < 1e-6:
            continue
        loser_grid = _pil_to_patch_grid(images_b[g], ls, device)   # (3, L, L)
        l1 = (winner_grid - loser_grid).abs().mean(dim=0)           # (L, L)
        I += l1 * margin
        n_valid += 1

    if n_valid > 0:
        I = I / n_valid

    return _apply_smooth_normalize_floor(I, ls, mask_floor)


def compute_cargo_mask_pixel_with_stats(
    images_b,
    R_c_b:       torch.Tensor,
    latent_size: int   = 16,
    mask_floor:  float = 0.30,
) -> tuple:
    """
    Same as compute_cargo_mask_pixel but also returns (raw_I_flat, stats_dict).
    raw_I_flat: (seq_len,) float32 — per-patch L1 importance before
                smoothing/normalization/floor.
    """
    G      = len(images_b)
    device = R_c_b.device
    ls     = latent_size
    seq_len = ls * ls

    winner_idx  = int(R_c_b.argmax().item())
    winner_grid = _pil_to_patch_grid(images_b[winner_idx], ls, device)

    I       = torch.zeros(ls, ls, device=device, dtype=torch.float32)
    n_valid = 0
    for g in range(G):
        if g == winner_idx:
            continue
        margin = max(float(R_c_b[winner_idx].item()) - float(R_c_b[g].item()), 0.0)
        if margin < 1e-6:
            continue
        loser_grid = _pil_to_patch_grid(images_b[g], ls, device)
        I += (winner_grid - loser_grid).abs().mean(dim=0) * margin
        n_valid += 1

    if n_valid > 0:
        I = I / n_valid

    raw_I_flat = I.reshape(seq_len)
    mask       = _apply_smooth_normalize_floor(I, ls, mask_floor)
    reward_spread = float(R_c_b.max().item()) - float(R_c_b.min().item())
    stats      = _mask_stats(raw_I_flat, mask, reward_spread, n_valid, seq_len)
    return mask, raw_I_flat, stats


# ---------------------------------------------------------------------------
# VQ-token mask (ablation, --cargo-mask-source vq)
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

    # Smooth in 16×16 image space (NOT in flat token order).
    return _apply_smooth_normalize_floor(
        I.reshape(latent_size, latent_size), latent_size, mask_floor
    )


def compute_cargo_mask_with_stats(
    tokens_b:    torch.Tensor,
    R_c_b:       torch.Tensor,
    latent_size: int   = 16,
    mask_floor:  float = 0.30,
) -> tuple:
    """
    Same as compute_cargo_mask but also returns (raw_I, stats_dict).
    raw_I: (seq_len,) float32 before smoothing/normalization/floor.
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

    mask          = _apply_smooth_normalize_floor(
        I.reshape(latent_size, latent_size), latent_size, mask_floor
    )
    reward_spread = float(R_c_b.max().item()) - float(R_c_b.min().item())
    stats         = _mask_stats(raw_I, mask, reward_spread, n_valid, seq_len)
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
