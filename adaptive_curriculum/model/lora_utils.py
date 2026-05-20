"""
LoRA injection for LlamaGen GPT model.
LlamaGen attention uses wqkv (fused QKV) and wo (output projection).
FeedForward uses w1, w2, w3.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base.weight.shape[1]
        out_features = base.weight.shape[0]

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def print_trainable_linear_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {tuple(module.weight.shape)}")


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Replace every nn.Linear whose name ends with one of target_modules with LoRALinear.
    Returns the model with LoRA injected (in-place on module tree).
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # get parent and attribute name
        parts = name.split(".")
        if not parts:
            continue
        attr = parts[-1]
        if attr not in target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # navigate to parent
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        lora_mod = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_mod)
        replaced += 1

    print(f"[LoRA] Replaced {replaced} linear layers with LoRA (rank={rank}, alpha={alpha})")
    return model


def freeze_base_model(model: nn.Module):
    """Freeze all parameters not inside a LoRALinear."""
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_lora_weights(model: nn.Module, path: str):
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)


def load_lora_weights(model: nn.Module, path: str, strict: bool = False):
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA load mismatch: missing={missing}, unexpected={unexpected}")
    return model
