"""
SlotResidualAdapter: cross-attention conditioning delta for LlamaGen.

Injects after CaptionEmbedder.cap_proj (T5→d_model projection), operating in
the GPT model's d_model space. Zero-initialized so base LlamaGen behaviour is
unchanged at startup (C_out == C_base when gamma==0).

Usage
-----
    adapter = SlotResidualAdapter(d_model=1280, t5_dim=2048)
    adapted_cls = attach_slot_adapter(gpt_model, adapter)
    # gpt_model.cls_embedding is now the AdaptedCaptionEmbedder

    # Before each forward pass that should use slot conditioning:
    adapted_cls.set_slot_context(slot_embs, slot_mask)
    logits, _ = gpt_model(idx=tokens[:, :-1], cond_idx=c_indices, ...)
    adapted_cls.clear_slot_context()

    # With no slot context, forward is identical to base LlamaGen.

Dimensions (GPT-XL)
-------------------
    d_model  = 1280   (GPT-XL hidden dim)
    t5_dim   = 2048   (flan-t5-xl output dim)
    cls_len  = 120    (conditioning token count, unchanged)
"""
import torch
import torch.nn as nn
from typing import Optional


class SlotResidualAdapter(nn.Module):
    """
    C_base:    [B, 120, d_model]  — output of CaptionEmbedder.cap_proj
    slot_embs: [B, K, t5_dim]    — T5 embeddings of slot_texts (K <= MAX_SLOTS)
    slot_mask: [B, K] bool        — True = padding position (passed to MHA as key_padding_mask)

    C_out = C_base + gamma * out_proj(CrossAttn(ln_q(C_base), ln_s(slot_proj(slot_embs))))

    Zero-init guarantee: out_proj.weight and out_proj.bias are zero at init,
    so the adapter contributes exactly zero until first gradient step.
    """

    def __init__(
        self,
        d_model:  int   = 1280,
        t5_dim:   int   = 2048,
        n_heads:  int   = 8,
        dropout:  float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.slot_proj  = nn.Linear(t5_dim, d_model, bias=False)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.out_proj   = nn.Linear(d_model, d_model, bias=True)
        self.ln_q       = nn.LayerNorm(d_model)
        self.ln_s       = nn.LayerNorm(d_model)
        self.gamma      = nn.Parameter(torch.tensor(0.0))

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        C_base:    torch.Tensor,
        slot_embs: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        S      = self.slot_proj(slot_embs)              # [B, K, d_model]
        q      = self.ln_q(C_base)                      # [B, 120, d_model]
        kv     = self.ln_s(S)                           # [B, K, d_model]
        delta, attn = self.cross_attn(q, kv, kv, key_padding_mask=slot_mask)
        delta  = self.out_proj(delta)
        C_out  = C_base + self.gamma * delta
        return C_out, {
            "delta":        delta.detach(),
            "attn":         attn.detach() if attn is not None else None,
            "gamma":        float(self.gamma.item()),
            "delta_norm":   float(delta.norm().item()),
            "base_norm":    float(C_base.norm().item()),
        }


class AdaptedCaptionEmbedder(nn.Module):
    """
    Wraps LlamaGen's CaptionEmbedder to inject a SlotResidualAdapter after
    cap_proj, without modifying LlamaGen source code.

    When no slot context is set, forward() is a drop-in pass-through to the
    original CaptionEmbedder — zero overhead, identical output.
    """

    def __init__(
        self,
        orig_cls_embedding: nn.Module,
        adapter:            SlotResidualAdapter,
    ):
        super().__init__()
        self.orig    = orig_cls_embedding
        self.adapter = adapter
        self._slot_embs:  Optional[torch.Tensor] = None
        self._slot_mask:  Optional[torch.Tensor] = None
        self._last_info:  Optional[dict]          = None

    # ── Slot context management ───────────────────────────────────────────────

    def set_slot_context(
        self,
        slot_embs: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
    ):
        self._slot_embs = slot_embs
        self._slot_mask = slot_mask

    def clear_slot_context(self):
        self._slot_embs  = None
        self._slot_mask  = None
        self._last_info  = None

    @property
    def last_adapter_info(self) -> Optional[dict]:
        return self._last_info

    # ── Proxy attributes needed by Transformer.forward() ─────────────────────

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
        if self._slot_embs is not None:
            C_out, info  = self.adapter(C_base, self._slot_embs, self._slot_mask)
            self._last_info = info
            return C_out
        return C_base


def attach_slot_adapter(
    gpt_model: nn.Module,
    adapter:   SlotResidualAdapter,
) -> AdaptedCaptionEmbedder:
    """
    Replace gpt_model.cls_embedding with AdaptedCaptionEmbedder in-place.
    Returns the new module (also accessible as gpt_model.cls_embedding).
    """
    adapted = AdaptedCaptionEmbedder(gpt_model.cls_embedding, adapter)
    gpt_model.cls_embedding = adapted
    return adapted


def count_adapter_params(adapter: SlotResidualAdapter) -> int:
    return sum(p.numel() for p in adapter.parameters())
