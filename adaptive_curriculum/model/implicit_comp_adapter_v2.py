"""
ImplicitCompositionAdapter v2 — adds hard residual cap.

Identical architecture to v1 but AdaptedCaptionEmbedder enforces a hard
clamp on the effective delta ratio (||γΔ|| / ||C_base||) instead of relying
solely on a soft regularization penalty.

New class: HardCapAdaptedCaptionEmbedder
  - Replaces the soft-only residual control in v1.
  - After computing delta, scales it so ratio <= target_ratio always.
  - Still differentiable: scale is a clamp, gradient flows through delta.

Nothing in v1 is changed. Import from here when using likelihood-contrastive
training; keep importing from v1 for the embedding-only experiments.
"""
import torch
import torch.nn as nn
from typing import Optional

from adaptive_curriculum.model.implicit_comp_adapter import (
    ImplicitCompositionAdapter,
    AdaptedCaptionEmbedder,
    attach_implicit_adapter,          # re-export for convenience
    count_adapter_params,             # re-export
    _attn_entropy,
)


class HardCapAdaptedCaptionEmbedder(AdaptedCaptionEmbedder):
    """
    AdaptedCaptionEmbedder with a hard residual ratio cap.

    After computing C_out = C_base + gamma * delta, rescales the residual so
    that ||C_out - C_base|| / ||C_base|| <= target_ratio at every forward pass.

    Still differentiable: the scale factor is clamped but the gradient
    flows through delta * scale.

    Parameters
    ----------
    target_ratio : float
        Maximum allowed ||γΔ|| / ||C_base||. Default 0.05 (conservative).
        Set to a larger value (0.10) to match v1 behavior.
    """

    def __init__(
        self,
        orig_cls_embedding: nn.Module,
        adapter: ImplicitCompositionAdapter,
        target_ratio: float = 0.05,
    ):
        super().__init__(orig_cls_embedding, adapter)
        self.target_ratio = target_ratio

    def forward(self, caption, train, force_drop_ids=None):
        C_base = self.orig(caption, train, force_drop_ids)   # [B, 120, d]
        if not self._enabled:
            self._last_info = None
            return C_base

        C_out_f32, info = self.adapter(C_base.float())

        if torch.isnan(C_out_f32).any() or torch.isinf(C_out_f32).any():
            print("[adapter_v2] WARNING: NaN/inf, falling back to C_base", flush=True)
            self._last_info = None
            return C_base

        # ── Hard residual cap ────────────────────────────────────────────────
        delta      = C_out_f32 - C_base.float()              # [B, 120, d]
        base_norm  = C_base.float().norm().detach() + 1e-8
        delta_norm = delta.norm()
        ratio      = delta_norm / base_norm
        # scale ∈ (0, 1]: shrinks delta when ratio > target, else no-op
        scale      = torch.clamp(
            torch.tensor(self.target_ratio, device=delta.device) / (ratio + 1e-8),
            max=1.0,
        )
        C_out_f32  = C_base.float() + delta * scale

        # Augment info with cap diagnostics
        info["hard_cap_scale"]   = float(scale.item())
        info["pre_cap_ratio"]    = float(ratio.item())
        info["post_cap_ratio"]   = float((delta * scale).norm().item() / (base_norm.item()))

        C_out = C_out_f32.to(dtype=C_base.dtype)
        self._last_info = info
        return C_out


def attach_hard_cap_adapter(
    gpt_model: nn.Module,
    adapter: ImplicitCompositionAdapter,
    target_ratio: float = 0.05,
) -> HardCapAdaptedCaptionEmbedder:
    """
    Replace gpt_model.cls_embedding with HardCapAdaptedCaptionEmbedder.
    Returns the new module.
    """
    adapted = HardCapAdaptedCaptionEmbedder(
        gpt_model.cls_embedding, adapter, target_ratio=target_ratio,
    )
    gpt_model.cls_embedding = adapted
    return adapted
