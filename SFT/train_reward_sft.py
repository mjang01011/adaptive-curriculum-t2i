"""
train_reward_sft.py — reward-guided SFT with weighted CE loss + LoRA.

Trains LlamaGen with per-sample weights derived from Qwen reward scores.
Loss = mean(weight_i * CE(logits_i, tokens_i))

Dataset format (from sample_and_score.py output):
  {"prompt": "...", "tokens_path": "...", "reward": 0.82, "weight": 0.64, ...}

Usage:
  python SFT/train_reward_sft.py \\
    --train-jsonl  outputs/reward_sft_data/reward_sft_dataset.jsonl \\
    --output-dir   outputs/reward_sft_lora_v1 \\
    --repo-root    LlamaGen \\
    --gpt-ckpt     pretrained/t2i_XL_stage1_256.pt \\
    --vq-ckpt      pretrained/vq_ds16_t2i.pt \\
    --t5-path      pretrained/t5-ckpt \\
    --train-lora --lora-r 16 --lora-alpha 32 \\
    --lora-targets wqkv wo w1 w2 w3 \\
    --num-epochs 5 --batch-size 8 --lr 5e-5 \\
    --wandb
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RewardSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, use_raw_caption: bool = False):
        self.jsonl_path = jsonl_path
        self.rows: list = []
        self._load()

    def _load(self):
        rows = []
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self.rows = rows
        print(f"[dataset] Loaded {len(rows)} reward-SFT samples", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "prompt":      row["prompt"],
            "tokens_path": row.get("tokens_path", ""),
            "reward":      float(row.get("reward", 0.5)),
            "weight":      float(row.get("weight", 1.0)),
            "category":    row.get("category", "unknown"),
        }


def _load_tokens(tokens_path: str, device) -> Optional[torch.Tensor]:
    p = Path(tokens_path)
    if p.exists():
        return torch.load(str(p), map_location="cpu").long().to(device)
    return None


# ---------------------------------------------------------------------------
# T5 encoding
# ---------------------------------------------------------------------------

def t5_encode(texts: List[str], t5_model, device: str) -> torch.Tensor:
    with torch.no_grad():
        embs, masks = t5_model.get_text_embeddings(texts)
    new_embs, new_masks = [], []
    for emb, mask in zip(embs, masks):
        valid = int(mask.sum().item())
        new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
        new_masks.append(torch.flip(mask, dims=[-1]))
    return (torch.stack(new_embs) * torch.stack(new_masks)[:, :, None]).float().to(device)


# ---------------------------------------------------------------------------
# Weighted CE loss
# ---------------------------------------------------------------------------

def weighted_ce_loss(logits: torch.Tensor, targets: torch.Tensor,
                     weights: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted cross-entropy. weights: [B]."""
    B = logits.shape[0]
    loss_per_tok = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )  # [B * seq_len]
    seq_len = loss_per_tok.shape[0] // B
    loss_per_sample = loss_per_tok.reshape(B, seq_len).mean(dim=1)  # [B]
    return (weights * loss_per_sample).mean()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_ckpt(path, model, optimizer, step, lora_only=True):
    if lora_only:
        state = {n: p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad}
    else:
        state = model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(), "step": step}, str(path))
    print(f"[train] Saved {path}", flush=True)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl",    required=True)
    p.add_argument("--val-jsonl",      default=None)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--repo-root",      required=True)
    p.add_argument("--gpt-ckpt",       required=True)
    p.add_argument("--vq-ckpt",        required=True)
    p.add_argument("--t5-path",        required=True)
    # LoRA
    p.add_argument("--train-lora",     action="store_true")
    p.add_argument("--lora-r",         type=int,   default=16)
    p.add_argument("--lora-alpha",     type=float, default=32.0)
    p.add_argument("--lora-targets",   nargs="+",  default=["wqkv", "wo", "w1", "w2", "w3"])
    p.add_argument("--resume-lora",    default=None, help="Path to LoRA checkpoint to resume from")
    # training
    p.add_argument("--num-epochs",     type=int,   default=5)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--weight-decay",   type=float, default=0.01)
    p.add_argument("--grad-clip",      type=float, default=1.0)
    p.add_argument("--warmup-steps",   type=int,   default=50)
    p.add_argument("--precision",      default="bf16")
    # logging
    p.add_argument("--eval-every",     type=int,   default=500)
    p.add_argument("--save-every",     type=int,   default=500)
    p.add_argument("--log-every",      type=int,   default=10)
    p.add_argument("--dl-workers",     type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)
    # val image logging
    p.add_argument("--val-prompts-jsonl", default=None,
                   help="Prompts file for val image generation (same format as sample_and_score)")
    p.add_argument("--val-n-prompts",  type=int,   default=8,
                   help="Number of val prompts to generate images for")
    p.add_argument("--val-gen-count",  type=int,   default=2,
                   help="Images per val prompt")
    p.add_argument("--cfg-scale",      type=float, default=2.0)
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="llamagen-reward-sft")
    p.add_argument("--wandb-entity",   default=None)
    p.add_argument("--run-name",       default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Val image generation
# ---------------------------------------------------------------------------

def _reset_kv_caches(model):
    """Set kv_cache=None on all attention layers so training forward works."""
    for module in model.modules():
        if hasattr(module, "kv_cache"):
            module.kv_cache = None


def _log_val_images(gpt, vq, t5, val_prompts, args, device, dtype,
                    step: int, wandb_run, prefix: str = "val"):
    """Generate images for val prompts and log to W&B."""
    import torchvision.transforms.functional as TF
    from autoregressive.models.generate import generate

    ls = vq._latent_size
    cb = vq._cb

    gpt.eval()
    wb_images: dict = {}

    with torch.no_grad():
        for row in val_prompts[:args.val_n_prompts]:
            prompt = row["prompt"]
            pid    = row.get("id", prompt[:20])

            embs, masks = t5.get_text_embeddings([prompt])
            emb, mask   = embs[0], masks[0]
            valid       = int(mask.sum().item())
            shifted     = torch.cat([emb[valid:], emb[:valid]])
            mask_s      = torch.flip(mask, dims=[-1])
            c_idx  = (shifted * mask_s[:, None]).to(device=device, dtype=dtype).unsqueeze(0)
            c_mask = mask_s.to(device=device, dtype=dtype).unsqueeze(0)

            G = args.val_gen_count
            torch.manual_seed(args.seed + abs(hash(pid)) % 100000)
            try:
                idx_all     = generate(gpt, c_idx.repeat(G, 1, 1), ls ** 2,
                                       c_mask.repeat(G, 1),
                                       cfg_scale=args.cfg_scale,
                                       temperature=1.0, top_k=2000, top_p=1.0,
                                       sample_logits=True)
                decoded_all = vq.decode_code(idx_all, [G, cb, ls, ls])
            except Exception as e:
                print(f"[val] gen failed for '{prompt[:40]}': {e}", flush=True)
                continue

            pils = []
            for si in range(G):
                img_t = (decoded_all[si].float().clamp(-1, 1) + 1) / 2
                pils.append(TF.to_pil_image(img_t.cpu()))

            wb_images[pid] = (prompt, pils)

    if wandb_run and wb_images:
        try:
            import wandb
            log_d = {}
            for pid, (prompt, pils) in wb_images.items():
                key = f"{prefix}/{pid}"
                log_d[key] = [wandb.Image(p, caption=f"[{prefix}] {prompt[:80]}") for p in pils]
            wandb_run.log(log_d, step=step)
            print(f"[val] logged {len(wb_images)} prompts to W&B as '{prefix}/*'", flush=True)
        except Exception as e:
            print(f"[val] W&B image log failed: {e}", flush=True)

    _reset_kv_caches(gpt)
    gpt.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity or None,
                name=args.run_name or out_dir.name, config=vars(args),
            )
            print(f"[train] W&B: {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[train] W&B init failed: {e}", flush=True)

    def _wb_log(d, step):
        if wandb_run:
            wandb_run.log(d, step=step)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── Models ────────────────────────────────────────────────────────────────
    print("[train] Loading LlamaGen ...", flush=True)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root, vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt, t5_path=args.t5_path, precision=args.precision,
        use_lora=args.train_lora,
        lora_config={
            "target_modules": args.lora_targets,
            "rank":  args.lora_r,
            "alpha": args.lora_alpha,
        } if args.train_lora else {},
    )
    gpt   = wrapper.gpt.to(device)
    vq    = wrapper.vq_model.to(device)
    t5    = wrapper.t5
    t5.model.to(device)
    dtype = wrapper.dtype
    ls    = wrapper.latent_size
    cb    = wrapper.codebook_embed_dim

    # Freeze everything, then unfreeze LoRA params
    for p in gpt.parameters():
        p.requires_grad = False
    for p in t5.model.parameters():
        p.requires_grad = False

    if args.train_lora:
        lora_params = [p for n, p in gpt.named_parameters()
                       if "lora_" in n]
        for p in lora_params:
            p.requires_grad = True
        trainable = sum(p.numel() for p in lora_params)
        print(f"[train] LoRA params: {trainable:,} trainable  "
              f"(r={args.lora_r}, alpha={args.lora_alpha}, "
              f"targets={args.lora_targets})", flush=True)
    else:
        # Full fine-tune (no LoRA) — unfreeze all GPT params
        for p in gpt.parameters():
            p.requires_grad = True
        print("[train] Full fine-tune (no LoRA)", flush=True)

    # Attach latent dims to vq so _log_val_images can find them
    vq._latent_size = ls
    vq._cb          = cb

    # ── Val prompts ───────────────────────────────────────────────────────────
    val_prompts: list = []
    if args.val_prompts_jsonl:
        with open(args.val_prompts_jsonl) as f:
            for line in f:
                if line.strip():
                    val_prompts.append(json.loads(line.strip()))
        print(f"[train] {len(val_prompts)} val prompts loaded", flush=True)

    # Resume LoRA weights
    start_step = 0
    if args.resume_lora and Path(args.resume_lora).exists():
        ckpt = torch.load(args.resume_lora, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = gpt.load_state_dict(state, strict=False)
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed LoRA from {args.resume_lora} at step {start_step}", flush=True)
        if unexpected:
            print(f"[train] Unexpected keys: {unexpected[:5]}", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in gpt.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Linear warmup scheduler
    def _lr_scale(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_scale)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = RewardSFTDataset(args.train_jsonl)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: x,
        num_workers=args.dl_workers, pin_memory=True,
        persistent_workers=args.dl_workers > 0,
    )
    print(f"[train] {len(ds)} training samples", flush=True)

    amp_ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else torch.autocast("cpu")

    # ── Training ──────────────────────────────────────────────────────────────
    step = start_step
    t0   = time.time()
    best_loss = float("inf")

    # Log base model images before any training
    if val_prompts and wandb_run:
        print("[train] Logging base model val images ...", flush=True)
        _log_val_images(gpt, vq, t5, val_prompts, args, device, dtype,
                        step=0, wandb_run=wandb_run, prefix="base")

    for epoch in range(args.num_epochs):
        gpt.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dl:
            # Gather valid samples
            token_list, caption_list, weight_list, reward_list = [], [], [], []
            for row in batch:
                toks = _load_tokens(row["tokens_path"], device)
                if toks is None:
                    continue
                token_list.append(toks)
                caption_list.append(row["prompt"])
                weight_list.append(row["weight"])
                reward_list.append(row["reward"])

            if not token_list:
                continue

            tokens  = torch.stack(token_list)   # [B, 256]
            weights = torch.tensor(weight_list, device=device, dtype=torch.float32)
            B       = tokens.shape[0]

            # T5 encode
            with torch.no_grad():
                c = t5_encode(caption_list, t5, device)  # [B, 120, 2048]

            if torch.isnan(c).any():
                continue

            gpt.cls_embedding.uncond_prob = 0.0

            # Forward
            with amp_ctx:
                logits, _ = gpt(
                    idx=tokens[:, :-1],
                    cond_idx=c.to(dtype=dtype),
                    input_pos=None, targets=tokens, mask=None, valid=None,
                )

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[train] WARNING: NaN/inf logits at step {step+1}, skipping", flush=True)
                continue

            # Weighted CE loss
            loss = weighted_ce_loss(logits, tokens, weights)

            if not torch.isfinite(loss):
                print(f"[train] WARNING: non-finite loss at step {step+1}, skipping", flush=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip).item()

            if math.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()

            step += 1
            epoch_loss  += float(loss)
            epoch_steps += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg_w   = float(weights.mean())
                avg_r   = sum(reward_list) / len(reward_list)
                lr_now  = optimizer.param_groups[0]["lr"]
                log_d   = {
                    "train/loss":       round(float(loss), 5),
                    "train/grad_norm":  round(grad_norm if math.isfinite(grad_norm) else -1, 4),
                    "train/avg_weight": round(avg_w, 4),
                    "train/avg_reward": round(avg_r, 4),
                    "train/lr":         lr_now,
                    "epoch":            epoch,
                }
                with open(log_path, "a") as lf:
                    lf.write(json.dumps({"step": step, **log_d}) + "\n")
                _wb_log(log_d, step)
                print(
                    f"[train] step={step}  loss={float(loss):.4f}"
                    f"  grad={grad_norm:.4f}  avg_w={avg_w:.3f}"
                    f"  avg_r={avg_r:.3f}  lr={lr_now:.2e}"
                    f"  elapsed={elapsed:.0f}s",
                    flush=True,
                )

            if step % args.save_every == 0:
                _save_ckpt(out_dir / f"ckpt_step{step}.pt", gpt, optimizer, step,
                           lora_only=args.train_lora)

            if step % args.eval_every == 0 and val_prompts and wandb_run:
                _log_val_images(gpt, vq, t5, val_prompts, args, device, dtype,
                                step=step, wandb_run=wandb_run, prefix="val")

        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        print(f"[train] Epoch {epoch} done  avg_loss={avg_epoch_loss:.4f}", flush=True)
        _wb_log({"epoch/loss": avg_epoch_loss}, step)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            _save_ckpt(out_dir / "best.pt", gpt, optimizer, step,
                       lora_only=args.train_lora)

    _save_ckpt(out_dir / f"final_step{step}.pt", gpt, optimizer, step,
               lora_only=args.train_lora)
    print(f"[train] Done. Step={step}  Best loss={best_loss:.4f}", flush=True)
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
