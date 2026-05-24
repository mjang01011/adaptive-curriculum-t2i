"""
ImplicitCompositionAdapter: learns to extract and re-inject compositional
structure from T5 prompt embeddings.

At inference, the only input is a normal text prompt — no parser, no slots.

Architecture (two-stage attention bottleneck)
--------------------------------------------
C_base:  [B, 120, d_model]   — output of CaptionEmbedder.cap_proj

Stage 1 (extract):
  Q_comp = learned composition queries  [n_comp_q, d_model]
  Z = CrossAttn(Q=Q_comp, KV=C_base)   → [B, n_comp_q, d_model]
  "What compositional structure is in this prompt?"

Stage 2 (inject):
  ΔC = CrossAttn(Q=C_base, KV=Z)       → [B, 120, d_model]
  "How should each conditioning token be adjusted?"

Output:
  C_out = C_base + γ * out_proj(LayerNorm(ΔC))

Zero-init guarantee
-------------------
  out_proj.weight = 0,  out_proj.bias = 0,  γ = 0
  → C_out == C_base exactly at initialization
  → base LlamaGen is perfectly reproduced before any gradient step

Composition queries
-------------------
Default n_comp_q=8. Each query specialises via gradient descent to attend
to different semantic aspects (objects, attributes, relations, etc).
The query role is not hand-designed; it emerges from the training signal.

Usage
-----
  adapter     = ImplicitCompositionAdapter(d_model=1280)
  adapted_cls = attach_implicit_adapter(gpt_model, adapter)
  # gpt_model.cls_embedding is now AdaptedCaptionEmbedder

  # No context-setting needed at inference — adapter runs automatically.
  logits, _ = gpt_model(idx=tokens[:, :-1], cond_idx=c_indices, ...)
"""
import torch
import torch.nn as nn
from typing import Optional


class ImplicitCompositionAdapter(nn.Module):
    """
    Composition adapter that works on any raw T5 conditioning.
    No external slot input at inference.
    """

    def __init__(
        self,
        d_model:    int   = 1280,
        n_comp_q:   int   = 8,
        n_heads:    int   = 8,
        dropout:    float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_comp_q   = n_comp_q
        self.d_model    = d_model

        # Learned composition query tokens
        self.comp_queries = nn.Parameter(torch.randn(n_comp_q, d_model) * 0.02)

        # Stage 1: extract compositional structure from C_base
        self.extract_attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_extract_q    = nn.LayerNorm(d_model)
        self.ln_extract_kv   = nn.LayerNorm(d_model)

        # Stage 2: inject back into C_base
        self.inject_attn     = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_inject_q     = nn.LayerNorm(d_model)
        self.ln_inject_kv    = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.gamma    = nn.Parameter(torch.tensor(1e-3))

        # Zero-init out_proj so adapter contributes ~nothing at initialization.
        # gamma=1e-3 (not 0) so gradients flow reliably from step 0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, C_base: torch.Tensor) -> tuple:
        """
        C_base: [B, 120, d_model]
        Returns (C_out, info_dict).
        """
        B = C_base.shape[0]

        # ── Stage 1: extract compositional summary Z ─────────────────────────
        Q  = self.comp_queries.unsqueeze(0).expand(B, -1, -1)   # [B, n_q, d]
        Z, extract_attn_w = self.extract_attn(
            self.ln_extract_q(Q),
            self.ln_extract_kv(C_base),
            self.ln_extract_kv(C_base),
        )   # Z: [B, n_q, d]

        # ── Stage 2: inject Z back into conditioning tokens ───────────────────
        delta, inject_attn_w = self.inject_attn(
            self.ln_inject_q(C_base),
            self.ln_inject_kv(Z),
            self.ln_inject_kv(Z),
        )   # delta: [B, 120, d]

        delta = self.out_proj(delta)
        C_out = C_base + self.gamma * delta

        g           = float(self.gamma.item())
        d_norm      = float(delta.norm().item())
        b_norm      = float(C_base.norm().item())
        return C_out, {
            "delta":                   delta.detach(),
            "comp_summary":            Z.detach(),
            "extract_attn":            extract_attn_w.detach() if extract_attn_w is not None else None,
            "inject_attn":             inject_attn_w.detach()  if inject_attn_w  is not None else None,
            "gamma":                   g,
            "delta_norm":              d_norm,
            "base_norm":               b_norm,
            "delta_to_base":           d_norm / (b_norm + 1e-8),
            "effective_delta_to_base": abs(g) * d_norm / (b_norm + 1e-8),
            "slot_attn_entropy":       _attn_entropy(extract_attn_w),
        }


def _attn_entropy(attn_weights: Optional[torch.Tensor]) -> float:
    """Mean entropy of attention distribution — higher = more diffuse."""
    if attn_weights is None:
        return 0.0
    try:
        p    = attn_weights.float().clamp(min=1e-9)
        ent  = -(p * p.log()).sum(-1).mean().item()
        return float(ent)
    except Exception:
        return 0.0


class AdaptedCaptionEmbedder(nn.Module):
    """
    Wraps LlamaGen's CaptionEmbedder to run ImplicitCompositionAdapter
    after cap_proj on every forward pass.

    No slot context needed — adapter runs automatically on any prompt.
    """

    def __init__(
        self,
        orig_cls_embedding: nn.Module,
        adapter:            ImplicitCompositionAdapter,
    ):
        super().__init__()
        self.orig       = orig_cls_embedding
        self.adapter    = adapter
        self._last_info: Optional[dict] = None
        self._enabled   = True  # set False to bypass adapter (for identity checks)

    @property
    def last_adapter_info(self) -> Optional[dict]:
        return self._last_info

    # ── Proxy attributes needed by LlamaGen Transformer.forward() ────────────

    @property
    def uncond_prob(self):
        return self.orig.uncond_prob

    @uncond_prob.setter
    def uncond_prob(self, v):
        self.orig.uncond_prob = v

    @property
    def uncond_embedding(self):
        return self.orig.uncond_embedding

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, caption, train, force_drop_ids=None):
        C_base = self.orig(caption, train, force_drop_ids)   # [B, 120, d_model]
        if self._enabled:
            # Run adapter in float32: bf16 attention scores can exceed ln(65504)≈11
            # causing softmax overflow; 0*NaN=NaN even through zero-init out_proj.
            C_out_f32, info = self.adapter(C_base.float())
            if torch.isnan(C_out_f32).any() or torch.isinf(C_out_f32).any():
                print("[adapter] WARNING: NaN/inf in adapter output, falling back to C_base", flush=True)
                self._last_info = None
                return C_base
            C_out = C_out_f32.to(dtype=C_base.dtype)
            # Sanitize gradients at the bf16↔float32 boundary: the bf16 GPT backward
            # can produce NaN/inf grads for C_out, which would corrupt float32 adapter
            # params on the very first optimizer step.
            if C_out.requires_grad:
                C_out.register_hook(
                    lambda g: torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=-1.0)
                )
            self._last_info = info
            return C_out
        self._last_info = None
        return C_base


def attach_implicit_adapter(
    gpt_model: nn.Module,
    adapter:   ImplicitCompositionAdapter,
) -> AdaptedCaptionEmbedder:
    """
    Replace gpt_model.cls_embedding with AdaptedCaptionEmbedder in-place.
    Returns the new module (also accessible as gpt_model.cls_embedding).
    """
    adapted = AdaptedCaptionEmbedder(gpt_model.cls_embedding, adapter)
    gpt_model.cls_embedding = adapted
    return adapted


def count_adapter_params(adapter: ImplicitCompositionAdapter) -> int:
    return sum(p.numel() for p in adapter.parameters())
