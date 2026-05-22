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
    objects:  list
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
    def score_pil(self, pil_img, item: ShapeDataItem) -> dict:
        rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return verify_image_bgr(bgr, item.to_metadata())


# ══════════════════════════════════════════════════════════════════════════════
# GCPO-lite helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_per_token_lp_and_entropy(wrapper, tokens, c_indices, c_emb_masks):
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
            input_pos=None, targets=None, mask=None, valid=None,
        )
        logits = logits.float().nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
        log_p    = F.log_softmax(logits, dim=-1)
        token_lp = log_p.gather(-1, tokens_long.unsqueeze(-1)).squeeze(-1).clamp(min=-20.0)
        with torch.no_grad():
            p       = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1)
        return token_lp, entropy
    finally:
        wrapper.gpt.cls_embedding.uncond_prob = saved_uncond_prob
        for m in dropout_mods:
            m.train()


def build_gcpo_lite_weights(entropy: torch.Tensor, latent_size: int, alpha: float = 2.0) -> torch.Tensor:
    B, L = entropy.shape
    grid    = entropy.reshape(B, latent_size, latent_size)
    dy      = F.pad(grid[:, 1:, :] - grid[:, :-1, :], (0, 0, 0, 1))
    dx      = F.pad(grid[:, :, 1:] - grid[:, :, :-1], (0, 1, 0, 0))
    grad_flat = (dx**2 + dy**2).sqrt().reshape(B, L)
    gmin    = grad_flat.min(dim=1, keepdim=True).values
    gmax    = grad_flat.max(dim=1, keepdim=True).values
    grad_norm = (grad_flat - gmin) / (gmax - gmin + 1e-6)
    weights = 1.0 + alpha * grad_norm
    pos     = torch.arange(L, device=entropy.device, dtype=entropy.dtype)
    pos_w   = (L / (pos + 1)).sqrt()
    pos_w   = pos_w / pos_w.mean()
    weights = weights * pos_w.unsqueeze(0)
    return (weights / weights.mean(dim=1, keepdim=True)).detach()


# ══════════════════════════════════════════════════════════════════════════════
# One GRPO step
# ══════════════════════════════════════════════════════════════════════════════

def grpo_step(wrapper, reward_model, batch, cfg, global_step):
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    objective  = cfg.training.objective
    G          = cfg.training.num_samples
    beta       = cfg.training.beta
    margin_thr = cfg.training.get("margin_threshold", 0.05)
    gcpo_alpha = cfg.training.get("gcpo_alpha", 2.0)
    adv_eps    = cfg.training.get("advantage_eps", 1e-8)
    B          = len(batch)

    # ── conditioning ──────────────────────────────────────────────────────────
    with torch.no_grad():
        c_indices, c_emb_masks = wrapper._get_conditioning(batch, t5_cache=None)

    # ── generate G samples per prompt ─────────────────────────────────────────
    wrapper.gpt.eval()
    all_tokens = []
    all_pil    = []
    qzshape    = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]

    with torch.no_grad():
        for g in range(G):
            idx = generate(
                wrapper.gpt, c_indices, wrapper.latent_size ** 2, c_emb_masks,
                cfg_scale=wrapper.cfg_scale_train, temperature=wrapper.temperature,
                top_k=wrapper.top_k, top_p=wrapper.top_p, sample_logits=True,
            )
            all_tokens.append(idx)
            decoded   = wrapper.vq_model.decode_code(idx, qzshape)
            pil_batch = []
            for i in range(B):
                img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
                pil_batch.append(TF.to_pil_image(img_t.cpu()))
            all_pil.append(pil_batch)

    wrapper._disable_kv_cache()

    # ── score ─────────────────────────────────────────────────────────────────
    rewards = torch.zeros(B, G)
    comp_log = []
    for g in range(G):
        for b, item in enumerate(batch):
            result         = reward_model.score_pil(all_pil[g][b], item)
            rewards[b, g]  = result["reward"]
            comp_log.append({"id": item.id, "g": g, "reward": result["reward"],
                              "components": result.get("components", {})})

    # ── advantages ────────────────────────────────────────────────────────────
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r  = rewards.std(dim=1, keepdim=True) + adv_eps
    norm_adv = ((rewards - mean_r) / std_r).nan_to_num(nan=0.0)

    if "winner_only" in objective:
        advantages   = torch.zeros_like(rewards)
        best_g       = rewards.argmax(dim=1)
        n_winners    = 0
        for b in range(B):
            if rewards[b, best_g[b]].item() - mean_r[b, 0].item() >= margin_thr:
                advantages[b, best_g[b]] = 1.0
                n_winners += 1
    else:
        advantages = norm_adv
        n_winners  = B

    # ── stacked tokens ────────────────────────────────────────────────────────
    stacked_tokens = torch.stack(all_tokens, dim=1).reshape(B * G, -1)
    rep_c    = c_indices.repeat_interleave(G, dim=0)
    rep_mask = c_emb_masks.repeat_interleave(G, dim=0)
    flat_adv = advantages.reshape(-1).to(wrapper.device)

    wrapper.gpt.float()

    # ── reference log probs ───────────────────────────────────────────────────
    ref_lp = None
    if beta > 0.0:
        with torch.no_grad():
            ref_lp = wrapper._compute_log_probs_ref(stacked_tokens, rep_c, rep_mask, cfg_scale=1.0)

    # ── policy gradient ───────────────────────────────────────────────────────
    wrapper._optimizer.zero_grad()
    total_pg = 0.0
    total_kl = 0.0
    chunk    = B

    for start in range(0, B * G, chunk):
        end    = min(start + chunk, B * G)
        c_tok  = stacked_tokens[start:end]
        c_cond = rep_c[start:end]
        c_msk  = rep_mask[start:end]
        c_adv  = flat_adv[start:end]
        w      = (end - start) / (B * G)

        if objective == "winner_only_gcpo_lite":
            tok_lp, entropy = _compute_per_token_lp_and_entropy(wrapper, c_tok, c_cond, c_msk)
            gcpo_w  = build_gcpo_lite_weights(entropy, wrapper.latent_size, alpha=gcpo_alpha)
            seq_lp  = (tok_lp * gcpo_w).sum(dim=-1) / math.sqrt(tok_lp.shape[-1])
        else:
            seq_lp = wrapper._compute_log_probs(c_tok, c_cond, c_msk, cfg_scale=1.0)

        pg = -(c_adv * seq_lp).mean() * w

        if beta > 0.0 and ref_lp is not None:
            kl_chunk  = (seq_lp - ref_lp[start:end]).mean() * w
            chunk_loss = pg + beta * kl_chunk
            total_kl  += kl_chunk.item()
        else:
            chunk_loss = pg

        chunk_loss.backward()
        total_pg += pg.item()

        for p in wrapper.gpt.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in wrapper.gpt.parameters() if p.requires_grad],
        wrapper.max_grad_norm,
    ).item()
    wrapper._optimizer.step()
    wrapper._step += 1

    # per-component mean for logging
    comp_means = {}
    if comp_log:
        keys = list(comp_log[0]["components"].keys())
        for k in keys:
            vals = [c["components"].get(k, 0.0) for c in comp_log]
            comp_means[k] = float(np.mean(vals))

    metrics = {
        "step":       global_step,
        "pg_loss":    total_pg,
        "kl_loss":    total_kl,
        "grad_norm":  grad_norm,
        "mean_reward": float(rewards.mean()),
        "max_reward":  float(rewards.max()),
        "min_reward":  float(rewards.min()),
        "reward_std":  float(rewards.std()),
        "n_winners":   n_winners,
        "lr":          wrapper._optimizer.param_groups[0]["lr"],
        **{f"comp/{k}": v for k, v in comp_means.items()},
    }
    return metrics, comp_log


# ══════════════════════════════════════════════════════════════════════════════
# Mini-probe — 4 fixed val prompts, runs every step, saves images
# ══════════════════════════════════════════════════════════════════════════════

def run_mini_probe(wrapper, reward_model, items, step, out_dir):
    """Generate + score + save images for a fixed small set of val prompts.
    Same cost as one training batch. Images go to mini_probe/<item_id>/stepXXXX.png
    so you can open a folder and scroll through to watch the model change.
    """
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    wrapper.gpt.eval()
    B       = len(items)
    qzshape = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]

    with torch.no_grad():
        c_idx, c_msk = wrapper._get_conditioning(items, t5_cache=None)
        tokens = generate(
            wrapper.gpt, c_idx, wrapper.latent_size ** 2, c_msk,
            cfg_scale=wrapper.cfg_scale, temperature=wrapper.temperature,
            top_k=wrapper.top_k, top_p=wrapper.top_p, sample_logits=True,
        )
        wrapper._disable_kv_cache()
        decoded = wrapper.vq_model.decode_code(tokens, qzshape)

    scores = []
    for i, item in enumerate(items):
        img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
        pil   = TF.to_pil_image(img_t.cpu())

        # save to mini_probe/<item_id>/step_XXXX.png
        img_dir = Path(out_dir) / "mini_probe" / item.id
        img_dir.mkdir(parents=True, exist_ok=True)

        # write prompt once so you always know what the folder is about
        prompt_file = img_dir / "prompt.txt"
        if not prompt_file.exists():
            prompt_file.write_text(item.text, encoding="utf-8")

        pil.save(img_dir / f"step_{step:04d}.png")

        result = reward_model.score_pil(pil, item)
        scores.append(result["reward"])
        rel    = result.get("components", {}).get("relation", 0.0)
        print(f"    [mini] {item.id}  r={result['reward']:.3f}  rel={rel:.3f}  → {item.text[:50]}")

    print(f"    [mini] mean={float(np.mean(scores)):.3f}  step={step}")


# ══════════════════════════════════════════════════════════════════════════════
# Probe evaluation — generates + saves images
# ══════════════════════════════════════════════════════════════════════════════

def run_probe(wrapper, reward_model, val_items, cfg, step, out_dir, wandb_run=None):
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    probe_bs  = cfg.training.get("probe_batch_size", 8)
    n_save    = cfg.training.get("probe_save_images", 20)  # save first N val images

    img_dir   = Path(out_dir) / f"probe_step_{step:04d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    wrapper.gpt.eval()
    all_rewards   = []
    all_comp      = {}
    saved         = 0

    with torch.no_grad():
        for start in range(0, len(val_items), probe_bs):
            batch    = val_items[start:start + probe_bs]
            c_idx, c_msk = wrapper._get_conditioning(batch, t5_cache=None)
            B        = len(batch)
            qzshape  = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
            tokens   = generate(
                wrapper.gpt, c_idx, wrapper.latent_size ** 2, c_msk,
                cfg_scale=wrapper.cfg_scale,
                temperature=wrapper.temperature,
                top_k=wrapper.top_k, top_p=wrapper.top_p,
                sample_logits=True,
            )
            wrapper._disable_kv_cache()
            decoded  = wrapper.vq_model.decode_code(tokens, qzshape)

            for i, item in enumerate(batch):
                img_t  = (decoded[i].float().clamp(-1, 1) + 1) / 2
                pil    = TF.to_pil_image(img_t.cpu())

                # save image with prompt embedded in filename
                if saved < n_save:
                    safe_prompt = item.text[:60].replace(" ", "_").replace("/", "-")
                    fname = f"{item.id}__{safe_prompt}.png"
                    pil.save(img_dir / fname)
                    saved += 1

                result = reward_model.score_pil(pil, item)
                all_rewards.append(result["reward"])
                for k, v in result.get("components", {}).items():
                    all_comp.setdefault(k, []).append(v)

    mean_r   = float(np.mean(all_rewards))
    comp_means = {k: float(np.mean(v)) for k, v in all_comp.items()}

    # ── print table ───────────────────────────────────────────────────────────
    bar_width = 30
    print(f"\n  ┌─ probe step={step}  n={len(all_rewards)} ──────────────────────")
    print(f"  │  mean_reward : {mean_r:.4f}  {'█' * int(mean_r * bar_width)}")
    for k, v in comp_means.items():
        bar  = '█' * int(v * bar_width)
        flag = " ◄◄" if k == "relation" else ""
        print(f"  │  {k:<14}: {v:.4f}  {bar}{flag}")
    print(f"  │  images saved → {img_dir}")
    print(f"  └────────────────────────────────────────────────────\n")

    if wandb_run:
        wandb_run.log({
            "probe/mean_reward": mean_r,
            **{f"probe/{k}": v for k, v in comp_means.items()},
            "step": step,
        })

    return mean_r, comp_means.get("relation", 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Main — epoch-based loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg      = OmegaConf.load(args.config)
    run_name = args.run_name or cfg.get("run_name", Path(args.config).stem)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(cfg.paths.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    print(f"[train] run_name={run_name}")
    print(f"[train] out_dir ={out_dir}")

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
    batch_size  = cfg.training.batch_size
    num_epochs  = cfg.training.num_epochs
    steps_per_epoch = math.ceil(len(train_items) / batch_size)
    total_steps = num_epochs * steps_per_epoch

    print(f"[train] {len(train_items)} train prompts  {len(val_items)} val prompts")
    print(f"[train] batch_size={batch_size}  G={cfg.training.num_samples}  "
          f"epochs={num_epochs}  steps/epoch={steps_per_epoch}  total_steps={total_steps}")
    print(f"[train] objective={cfg.training.objective}")

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
    _ = wrapper.gpt
    _ = wrapper.vq_model

    reward_model = ShapeVerifierRewardModel()

    # ── probe settings ────────────────────────────────────────────────────────
    probe_every_steps = cfg.training.get("probe_every_steps", 10)
    early_stop_drop   = cfg.training.get("early_stop_relation_drop", 0.10)

    # ── fixed mini-probe prompts (shown every step, cheap) ────────────────────
    n_mini = cfg.training.get("mini_probe_n", 4)
    mini_items = val_items[:n_mini]   # always the same prompts so you can compare

    # ── baseline probe ────────────────────────────────────────────────────────
    baseline_r, baseline_rel = run_probe(wrapper, reward_model, val_items, cfg,
                                         step=0, out_dir=out_dir, wandb_run=wandb_run)
    best_rel     = baseline_rel
    global_step  = 0
    metrics_log  = []

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        epoch_items = train_items.copy()
        random.shuffle(epoch_items)
        epoch_rewards = []
        t_epoch = time.time()

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{num_epochs}")
        print(f"{'─'*60}")

        for step_in_epoch in range(steps_per_epoch):
            global_step += 1
            start_i = step_in_epoch * batch_size
            batch   = epoch_items[start_i: start_i + batch_size]
            if not batch:
                continue

            t0      = time.time()
            metrics, _ = grpo_step(wrapper, reward_model, batch, cfg, global_step)
            elapsed = time.time() - t0

            epoch_rewards.append(metrics["mean_reward"])
            metrics_log.append(metrics)

            # compact per-step line
            comp_str = "  ".join(
                f"{k.split('/')[1]}={v:.2f}"
                for k, v in metrics.items()
                if k.startswith("comp/")
            )
            print(f"  ep{epoch} step{step_in_epoch+1:3d}/{steps_per_epoch}  "
                  f"[global {global_step:4d}]  "
                  f"pg={metrics['pg_loss']:+.4f}  "
                  f"mean_r={metrics['mean_reward']:.4f}  "
                  f"max_r={metrics['max_reward']:.4f}  "
                  f"winners={metrics['n_winners']}/{len(batch)}  "
                  f"gnorm={metrics['grad_norm']:.3f}  "
                  f"({elapsed:.1f}s)")
            if comp_str:
                print(f"         components: {comp_str}")

            if wandb_run:
                wandb_run.log({f"train/{k}": v for k, v in metrics.items()})

            # mini-probe every step on 4 fixed val prompts (save images, cheap)
            run_mini_probe(wrapper, reward_model, mini_items, global_step, out_dir)

            # full val probe every probe_every_steps
            if global_step % probe_every_steps == 0:
                run_probe(wrapper, reward_model, val_items, cfg,
                          step=global_step, out_dir=out_dir, wandb_run=wandb_run)

        epoch_mean = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        print(f"\n  ── Epoch {epoch} summary: mean_reward={epoch_mean:.4f}  "
              f"time={time.time()-t_epoch:.0f}s")

        # probe at end of epoch always
        probe_r, probe_rel = run_probe(
            wrapper, reward_model, val_items, cfg,
            step=global_step, out_dir=out_dir, wandb_run=wandb_run,
        )
        if probe_rel > best_rel:
            best_rel  = probe_rel
            ckpt_path = out_dir / "best_checkpoint.pt"
            torch.save(wrapper.gpt.state_dict(), ckpt_path)
            print(f"  [ckpt] best relation={best_rel:.4f} → {ckpt_path}")

        if probe_rel < baseline_rel - early_stop_drop:
            print(f"  [early stop] relation {probe_rel:.4f} dropped >"
                  f" {early_stop_drop} below baseline {baseline_rel:.4f}")
            break

    # ── final save ────────────────────────────────────────────────────────────
    torch.save(wrapper.gpt.state_dict(), out_dir / "final_checkpoint.pt")

    with open(out_dir / "train_metrics.jsonl", "w", encoding="utf-8") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    summary = {
        "run_name":           run_name,
        "objective":          cfg.training.objective,
        "epochs_run":         epoch,
        "total_steps":        global_step,
        "baseline_reward":    baseline_r,
        "baseline_relation":  baseline_rel,
        "best_relation":      best_rel,
        "delta_relation":     best_rel - baseline_rel,
        "final_mean_reward":  metrics_log[-1]["mean_reward"] if metrics_log else 0.0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[train] done.  Δrelation={best_rel - baseline_rel:+.4f}")
    print(f"[train] summary → {out_dir / 'summary.json'}")

    if wandb_run:
        wandb_run.log({"final/best_relation": best_rel,
                       "final/delta_relation": best_rel - baseline_rel})
        wandb_run.finish()


if __name__ == "__main__":
    main()
