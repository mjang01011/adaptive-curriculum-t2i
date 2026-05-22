"""
GRPO training on synthetic colored-shape prompts using the classical CV verifier as reward.

Supports three objectives:
  standard_grpo        — full group-relative policy gradient
  winner_only          — only the group winner gets a positive advantage
  winner_only_gcpo_lite — winner-only + per-token entropy-gradient weighting

Usage:
  python scripts_verifier/train_verifier_grpo.py \
    --config configs_verifier/synthetic_shapes_vanilla_grpo.yaml \
    [--run-name my_run]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

try:
    import cv2
except ImportError:
    raise ImportError("pip install opencv-python-headless")

# ── add project root to path ──────────────────────────────────────────────────

_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from scripts_verifier.shape_color_position_verifier import verify_image_bgr


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShapeDataItem:
    id:       str
    text:     str
    bucket:   str
    objects:  list   # [{color, shape/family, position}, ...]
    relation: str

    def to_metadata(self):
        return {"objects": self.objects, "relation": self.relation}


def load_shape_dataset(jsonl_path: str, bucket: str = "synthetic_shapes") -> List[ShapeDataItem]:
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items.append(ShapeDataItem(
                id=r["id"],
                text=r["prompt"],
                bucket=bucket,
                objects=r["objects"],
                relation=r.get("relation", ""),
            ))
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Reward model
# ══════════════════════════════════════════════════════════════════════════════

class ShapeVerifierRewardModel:
    """Wraps the classical CV verifier; accepts PIL images (no disk I/O)."""

    def score_pil(self, pil_img, item: ShapeDataItem) -> dict:
        import numpy as np
        rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return verify_image_bgr(bgr, item.to_metadata())

    def score_batch(self, pil_imgs: list, items: List[ShapeDataItem]) -> torch.Tensor:
        rewards = []
        for img, item in zip(pil_imgs, items):
            result = self.score_pil(img, item)
            rewards.append(result["reward"])
        return torch.tensor(rewards, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# GCPO-lite helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_per_token_lp_and_entropy(wrapper, tokens, c_indices, c_emb_masks):
    """
    Single forward pass → per-token log probs and per-token Shannon entropy.
    Returns (token_lp, entropy) both shape (B, seq_len), with grad on token_lp.
    """
    import torch.nn as _nn
    wrapper.gpt.train()
    saved_uncond_prob = wrapper.gpt.cls_embedding.uncond_prob
    wrapper.gpt.cls_embedding.uncond_prob = 0.0
    dropout_mods = [m for m in wrapper.gpt.modules() if isinstance(m, _nn.Dropout)]
    for m in dropout_mods:
        m.eval()
    try:
        tokens_long = tokens.long()
        c_cast = c_indices.float()
        logits, _ = wrapper.gpt(
            idx=tokens_long[:, :-1],
            cond_idx=c_cast,
            input_pos=None,
            targets=None,
            mask=None,
            valid=None,
        )
        logits = logits.float().nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
        log_p = F.log_softmax(logits, dim=-1)                      # (B, L, V)
        token_lp = log_p.gather(-1, tokens_long.unsqueeze(-1)).squeeze(-1)  # (B, L)
        token_lp = token_lp.clamp(min=-20.0)

        with torch.no_grad():
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1)                     # (B, L)

        return token_lp, entropy
    finally:
        wrapper.gpt.cls_embedding.uncond_prob = saved_uncond_prob
        for m in dropout_mods:
            m.train()


def build_gcpo_lite_weights(entropy: torch.Tensor, latent_size: int, alpha: float = 2.0) -> torch.Tensor:
    """
    Compute per-token importance weights from per-token entropy.

    Tokens with high spatial entropy-gradient (object boundaries) are upweighted.
    Also apply a mild positional decay to emphasize early (coarse layout) tokens.

    entropy: (B, seq_len)  seq_len = latent_size^2
    Returns: (B, seq_len) weights ≥ 1
    """
    B, L = entropy.shape
    assert L == latent_size * latent_size, f"expected {latent_size**2} tokens, got {L}"

    grid = entropy.reshape(B, latent_size, latent_size)

    # finite-difference gradient magnitude in 2D spatial grid
    dy = grid[:, 1:, :] - grid[:, :-1, :]   # (B, ls-1, ls)
    dx = grid[:, :, 1:] - grid[:, :, :-1]   # (B, ls, ls-1)
    # zero-pad to restore shape
    dy = F.pad(dy, (0, 0, 0, 1))            # (B, ls, ls)
    dx = F.pad(dx, (0, 1, 0, 0))            # (B, ls, ls)
    grad_mag = (dx ** 2 + dy ** 2).sqrt()   # (B, ls, ls)
    grad_flat = grad_mag.reshape(B, L)

    # normalize per sequence to [0, 1]
    gmin = grad_flat.min(dim=1, keepdim=True).values
    gmax = grad_flat.max(dim=1, keepdim=True).values
    grad_norm = (grad_flat - gmin) / (gmax - gmin + 1e-6)

    # boundary weight
    weights = 1.0 + alpha * grad_norm       # (B, L)

    # mild early-token boost: positional factor decays with sqrt(pos)
    pos = torch.arange(L, device=entropy.device, dtype=entropy.dtype)
    pos_w = (L / (pos + 1)).sqrt()          # large for early tokens
    pos_w = pos_w / pos_w.mean()            # mean = 1
    weights = weights * pos_w.unsqueeze(0)

    # renormalize so mean = 1 per sequence
    weights = weights / weights.mean(dim=1, keepdim=True)
    return weights.detach()


# ══════════════════════════════════════════════════════════════════════════════
# Training step
# ══════════════════════════════════════════════════════════════════════════════

def grpo_step(wrapper, reward_model, batch, cfg, global_step, wandb_run=None):
    """
    One GRPO update step.  Returns metrics dict.
    objective: standard_grpo | winner_only | winner_only_gcpo_lite
    """
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    objective   = cfg.training.objective
    G           = cfg.training.num_samples          # samples per prompt
    beta        = cfg.training.beta
    margin_thr  = cfg.training.get("margin_threshold", 0.05)
    gcpo_alpha  = cfg.training.get("gcpo_alpha", 2.0)
    grad_accum  = cfg.training.get("grad_accum_steps", 1)
    adv_eps     = cfg.training.get("advantage_eps", 1e-8)

    B = len(batch)

    # ── 1. conditioning ───────────────────────────────────────────────────────
    with torch.no_grad():
        c_indices, c_emb_masks = wrapper._get_conditioning(batch, t5_cache=None)

    # ── 2. generate G samples per prompt (no grad) ────────────────────────────
    wrapper.gpt.eval()
    all_tokens = []   # list of G tensors each (B, seq_len)
    all_pil    = []   # list of G lists each [B PIL images]
    qzshape    = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]

    with torch.no_grad():
        for _ in range(G):
            idx_sample = generate(
                wrapper.gpt, c_indices, wrapper.latent_size ** 2,
                c_emb_masks,
                cfg_scale=wrapper.cfg_scale_train,
                temperature=wrapper.temperature,
                top_k=wrapper.top_k,
                top_p=wrapper.top_p,
                sample_logits=True,
            )  # (B, seq_len)
            all_tokens.append(idx_sample)

            decoded = wrapper.vq_model.decode_code(idx_sample, qzshape)
            pil_batch = []
            for i in range(B):
                img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
                pil_batch.append(TF.to_pil_image(img_t.cpu()))
            all_pil.append(pil_batch)

    wrapper._disable_kv_cache()

    # ── 3. score all samples ─────────────────────────────────────────────────
    # rewards[b, g] = reward for prompt b, sample g
    rewards = torch.zeros(B, G)
    reward_details = []
    for g in range(G):
        for b, item in enumerate(batch):
            result = reward_model.score_pil(all_pil[g][b], item)
            rewards[b, g] = result["reward"]
            reward_details.append({
                "id": item.id, "g": g, "reward": result["reward"],
                "components": result.get("components", {}),
            })

    # ── 4. compute advantages ─────────────────────────────────────────────────
    mean_r = rewards.mean(dim=1, keepdim=True)    # (B, 1)
    std_r  = rewards.std(dim=1, keepdim=True) + adv_eps
    norm_adv = ((rewards - mean_r) / std_r).nan_to_num(nan=0.0)  # (B, G)

    if "winner_only" in objective:
        # only the winner sample gets a non-zero advantage
        winner_mask = torch.zeros_like(rewards)
        best_g = rewards.argmax(dim=1)  # (B,)
        for b in range(B):
            max_r = rewards[b, best_g[b]].item()
            # skip if winner barely beats the group mean (noise-dominated)
            if max_r - mean_r[b, 0].item() >= margin_thr:
                winner_mask[b, best_g[b]] = 1.0
        advantages = winner_mask  # use raw 1.0 for winner (not normalized)
    else:
        advantages = norm_adv  # (B, G)

    # ── 5. stack tokens: (B*G, seq_len) ──────────────────────────────────────
    stacked_tokens = torch.stack(all_tokens, dim=1).reshape(B * G, -1)  # (B*G, seq_len)
    rep_c    = c_indices.repeat_interleave(G, dim=0)
    rep_mask = c_emb_masks.repeat_interleave(G, dim=0)
    flat_adv = advantages.reshape(-1).to(wrapper.device)  # (B*G,)

    # ── 6. reference log probs (KL regularisation) ───────────────────────────
    wrapper.gpt.float()
    ref_lp = None
    if beta > 0.0:
        with torch.no_grad():
            ref_lp = wrapper._compute_log_probs_ref(stacked_tokens, rep_c, rep_mask, cfg_scale=1.0)

    # ── 7. policy gradient loss ───────────────────────────────────────────────
    wrapper._optimizer.zero_grad()
    total_pg   = 0.0
    total_kl   = 0.0
    chunk_size = B  # one prompt worth at a time to avoid OOM

    for start in range(0, B * G, chunk_size):
        end      = min(start + chunk_size, B * G)
        c_tok    = stacked_tokens[start:end]
        c_cond   = rep_c[start:end]
        c_msk    = rep_mask[start:end]
        c_adv    = flat_adv[start:end]
        weight   = (end - start) / (B * G)

        if objective == "winner_only_gcpo_lite":
            # per-token log probs + entropy in a single forward pass
            tok_lp, entropy = _compute_per_token_lp_and_entropy(
                wrapper, c_tok, c_cond, c_msk
            )
            gcpo_w = build_gcpo_lite_weights(
                entropy, wrapper.latent_size, alpha=gcpo_alpha
            )  # (chunk, seq_len)
            seq_lp = (tok_lp * gcpo_w).sum(dim=-1) / math.sqrt(tok_lp.shape[-1])
        else:
            seq_lp = wrapper._compute_log_probs(c_tok, c_cond, c_msk, cfg_scale=1.0)

        pg = -(c_adv * seq_lp).mean() * weight

        if beta > 0.0 and ref_lp is not None:
            kl_chunk  = (seq_lp - ref_lp[start:end]).mean() * weight
            chunk_loss = pg + beta * kl_chunk
            total_kl  += kl_chunk.item()
        else:
            chunk_loss = pg

        chunk_loss.backward()
        total_pg += pg.item()

        # sanitise NaN grads immediately (cumulative NaN propagation)
        for p in wrapper.gpt.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in wrapper.gpt.parameters() if p.requires_grad],
        wrapper.max_grad_norm,
    ).item()
    wrapper._optimizer.step()
    wrapper._step += 1

    metrics = {
        "step":          global_step,
        "pg_loss":       total_pg,
        "kl_loss":       total_kl,
        "total_loss":    total_pg + beta * total_kl,
        "grad_norm":     grad_norm,
        "mean_reward":   float(rewards.mean()),
        "max_reward":    float(rewards.max()),
        "min_reward":    float(rewards.min()),
        "mean_adv":      float(advantages.mean()),
        "lr":            wrapper._optimizer.param_groups[0]["lr"],
    }
    return metrics, reward_details


# ══════════════════════════════════════════════════════════════════════════════
# Probe evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_probe(wrapper, reward_model, val_items, cfg, step, out_dir, wandb_run=None):
    """
    Generate 1 image per val prompt, score, log summary.
    Returns mean_reward and mean_relation.
    """
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    probe_bs = cfg.training.get("probe_batch_size", 4)
    wrapper.gpt.eval()

    all_rewards   = []
    all_relations = []

    with torch.no_grad():
        for start in range(0, len(val_items), probe_bs):
            batch = val_items[start:start + probe_bs]
            c_indices, c_emb_masks = wrapper._get_conditioning(batch, t5_cache=None)
            B = len(batch)
            qzshape = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
            idx = generate(
                wrapper.gpt, c_indices, wrapper.latent_size ** 2, c_emb_masks,
                cfg_scale=wrapper.cfg_scale,
                temperature=wrapper.temperature,
                top_k=wrapper.top_k, top_p=wrapper.top_p,
                sample_logits=True,
            )
            wrapper._disable_kv_cache()
            decoded = wrapper.vq_model.decode_code(idx, qzshape)
            for i, item in enumerate(batch):
                img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
                pil = TF.to_pil_image(img_t.cpu())
                result = reward_model.score_pil(pil, item)
                all_rewards.append(result["reward"])
                all_relations.append(result.get("components", {}).get("relation", 0.0))

    mean_r   = float(np.mean(all_rewards))
    mean_rel = float(np.mean(all_relations))

    print(f"  [probe step={step}] mean_reward={mean_r:.4f}  mean_relation={mean_rel:.4f}  n={len(all_rewards)}")

    if wandb_run:
        wandb_run.log({"probe/mean_reward": mean_r, "probe/mean_relation": mean_rel, "step": step})

    return mean_r, mean_rel


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run_name = args.run_name or cfg.get("run_name", Path(args.config).stem)

    # ── seed ─────────────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── output dir ────────────────────────────────────────────────────────────
    out_dir = Path(cfg.paths.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"[train] run_name={run_name}  out_dir={out_dir}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.get("wandb", {}).get("enabled", False):
        import wandb
        os.environ.setdefault("WANDB_API_KEY",
            "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")
        wandb_run = wandb.init(
            project=cfg.wandb.get("project", "llamagen-verifier-grpo"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── data ──────────────────────────────────────────────────────────────────
    train_items = load_shape_dataset(cfg.data.train_jsonl)
    val_items   = load_shape_dataset(cfg.data.val_jsonl)
    print(f"[train] {len(train_items)} train  {len(val_items)} val")

    # ── model ─────────────────────────────────────────────────────────────────
    sys.path.insert(0, cfg.paths.repo_root)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    wrapper = LlamaGenWrapper(
        repo_root=cfg.paths.repo_root,
        vq_ckpt=cfg.model.vq_ckpt,
        gpt_ckpt=cfg.model.gpt_ckpt,
        gpt_model=cfg.model.gpt_model,
        image_size=cfg.model.image_size,
        t5_path=cfg.model.t5_path,
        t5_model_type=cfg.model.t5_model_type,
        t5_feature_max_len=cfg.model.t5_feature_max_len,
        cfg_scale=float(cfg.model.cfg_scale),
        cfg_scale_train=float(cfg.model.get("cfg_scale_train", cfg.model.cfg_scale)),
        temperature=float(cfg.model.get("temperature", 1.0)),
        top_k=int(cfg.model.get("top_k", 1000)),
        top_p=float(cfg.model.get("top_p", 1.0)),
        precision=cfg.model.mixed_precision,
        use_lora=cfg.model.use_lora,
        lora_config=OmegaConf.to_container(cfg.lora, resolve=True),
        learning_rate=float(cfg.training.lr),
        max_grad_norm=float(cfg.training.get("max_grad_norm", 1.0)),
    )
    # trigger lazy load
    _ = wrapper.gpt
    _ = wrapper.vq_model

    reward_model = ShapeVerifierRewardModel()

    # ── baseline probe ────────────────────────────────────────────────────────
    probe_every = cfg.training.get("probe_every_steps", 2)
    early_stop_rel = cfg.training.get("early_stop_relation_drop", 0.10)

    baseline_r, baseline_rel = run_probe(wrapper, reward_model, val_items, cfg, step=0, out_dir=out_dir, wandb_run=wandb_run)
    best_rel = baseline_rel

    # ── training loop ─────────────────────────────────────────────────────────
    total_steps    = cfg.training.total_steps
    batch_size     = cfg.training.batch_size
    objective      = cfg.training.objective

    print(f"[train] objective={objective}  total_steps={total_steps}  G={cfg.training.num_samples}  bs={batch_size}")

    train_idx   = list(range(len(train_items)))
    metrics_log = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        random.shuffle(train_idx)
        batch = [train_items[i] for i in train_idx[:batch_size]]

        metrics, details = grpo_step(wrapper, reward_model, batch, cfg, global_step=step, wandb_run=wandb_run)

        elapsed = time.time() - t0
        print(f"  step={step}/{total_steps}  "
              f"loss={metrics['total_loss']:.4f}  "
              f"pg={metrics['pg_loss']:.4f}  "
              f"mean_r={metrics['mean_reward']:.4f}  "
              f"grad_norm={metrics['grad_norm']:.3f}  "
              f"t={elapsed:.0f}s")

        metrics_log.append(metrics)
        if wandb_run:
            wandb_run.log({f"train/{k}": v for k, v in metrics.items()})

        # append reward details
        with open(out_dir / "reward_details.jsonl", "a", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d) + "\n")

        # probe + early stopping
        if step % probe_every == 0:
            probe_r, probe_rel = run_probe(
                wrapper, reward_model, val_items, cfg, step=step, out_dir=out_dir, wandb_run=wandb_run
            )
            if probe_rel > best_rel:
                best_rel = probe_rel
                ckpt_path = out_dir / "best_checkpoint.pt"
                torch.save(wrapper.gpt.state_dict(), ckpt_path)
                print(f"    [ckpt] best relation={best_rel:.4f} → {ckpt_path}")

            # early stopping: if relation drops significantly below baseline
            if probe_rel < baseline_rel - early_stop_rel:
                print(f"  [early stop] relation {probe_rel:.4f} dropped > {early_stop_rel} below baseline {baseline_rel:.4f}")
                break

    # ── final checkpoint ──────────────────────────────────────────────────────
    torch.save(wrapper.gpt.state_dict(), out_dir / "final_checkpoint.pt")

    # ── save metrics ──────────────────────────────────────────────────────────
    with open(out_dir / "train_metrics.jsonl", "w", encoding="utf-8") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    summary = {
        "run_name":       run_name,
        "objective":      objective,
        "total_steps":    step,
        "baseline_reward": baseline_r,
        "baseline_relation": baseline_rel,
        "best_relation":  best_rel,
        "final_mean_reward": metrics_log[-1]["mean_reward"] if metrics_log else 0.0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[train] done.  summary → {out_dir / 'summary.json'}")

    if wandb_run:
        wandb_run.log({"final/best_relation": best_rel})
        wandb_run.finish()


if __name__ == "__main__":
    main()
