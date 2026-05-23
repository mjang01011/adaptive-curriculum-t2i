"""
CARGO mask utilities: Component-Aware Reward-Grounded token importance maps.

compute_cargo_mask()   — winner-aligned importance, 16×16 spatial smoothing, soft floor
make_random_mask()     — random baseline (same floor/scale)
make_early_token_mask()— early-token sharpening baseline
overlay_mask_on_image()— heatmap blend for visual diagnostics
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
    Falls back to uniform mask if all rewards are identical or G=1.
    """
    G, seq_len = tokens_b.shape
    device = tokens_b.device

    winner_idx = int(R_c_b.argmax().item())
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

    # Spatial smoothing in latent image space (do NOT smooth in flat token order)
    I_2d    = I.reshape(1, 1, latent_size, latent_size)
    I_smooth = F.avg_pool2d(I_2d, kernel_size=3, stride=1, padding=1).reshape(seq_len)

    # Normalize to [0, 1]
    I_min, I_max = I_smooth.min(), I_smooth.max()
    if (I_max - I_min).item() > 1e-6:
        I_norm = (I_smooth - I_min) / (I_max - I_min)
    else:
        I_norm = torch.full_like(I_smooth, 0.5)

    return mask_floor + (1.0 - mask_floor) * I_norm


# ---------------------------------------------------------------------------
# Comparison baselines
# ---------------------------------------------------------------------------

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
    mask_256:    torch.Tensor,   # (256,) or (16,16) float in [0,1]
    latent_size: int   = 16,
    alpha:       float = 0.55,
    colormap:    str   = "hot",
) -> "PIL.Image.Image":
    """
    Resize 16×16 mask to image resolution, blend as heatmap overlay.
    mask_256 values: higher = more important token (brighter in heatmap).
    alpha: heatmap opacity (0 = original, 1 = full heatmap).
    colormap: any matplotlib colormap name.
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
        cmap = mcm.get_cmap(colormap)
        heat_rgba = (cmap(m_np) * 255).astype(np.uint8)
        heat_rgb  = heat_rgba[:, :, :3]
    except Exception:
        heat_rgb = (np.stack([m_np, np.zeros_like(m_np), 1.0 - m_np], axis=-1) * 255).astype(np.uint8)

    heat_pil = Image.fromarray(heat_rgb, "RGB").resize(pil_img.size, Image.NEAREST)
    orig_np  = np.array(pil_img.convert("RGB")).astype(np.float32)
    heat_np  = np.array(heat_pil).astype(np.float32)
    blended  = np.clip((1 - alpha) * orig_np + alpha * heat_np, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, "RGB")
