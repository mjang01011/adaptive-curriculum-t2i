"""
Train SlotResidualAdapter for LlamaGen with frozen base model.

Phase 1 (default): adapter-only — freeze LlamaGen + T5, train SlotResidualAdapter.
Phase 2 (--train-lora): adapter + LoRA on wqkv/wo.

Loss: CE(logits, tokens) + lambda_delta * ||gamma * delta||^2

Usage
-----
  python SFT/train_slot_adapter.py \\
    --train-jsonl $PROJ/data/gpic_slots_v1/dataset.jsonl \\
    --val-jsonl   $PROJ/data/attribute_binding/attribute_binding_val_20.jsonl \\
    --output-dir  $PROJ/outputs/slot_adapter_v1 \\
    --repo-root   $LLAMAGEN \\
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \\
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \\
    --t5-path     $PRETRAINED/t5-ckpt \\
    --freeze-llamagen \\
    --num-epochs 3 --batch-size 4 --lr 1e-4
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SlotDataset(Dataset):
    """
    Loads GPIC slot JSONL rows.
    Each row must have: canonical_prompt, slot_texts, tokens_path (optional).

    If tokens_path is present and valid, loads pre-computed VQ tokens.
    Otherwise returns None for tokens (caller must encode at runtime — slow).
    """

    def __init__(self, jsonl_path: str, max_slot_len: int = 12):
        self.rows         = []
        self.max_slot_len = max_slot_len
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "key":              row.get("key", str(idx)),
            "canonical_prompt": row["canonical_prompt"],
            "slot_texts":       row.get("slot_texts", [])[:self.max_slot_len],
            "tokens_path":      row.get("tokens_path"),
            "image_path":       row.get("image_path"),
        }


def _collate_fn(batch):
    return batch  # list of dicts; handled manually in training loop


# ---------------------------------------------------------------------------
# Slot T5 encoding helpers
# ---------------------------------------------------------------------------

def encode_slot_texts_t5(
    slot_texts_batch: List[List[str]],
    t5_model,
    device: str,
    max_len: int = 120,
    t5_dim: int  = 2048,
) -> tuple:
    """
    Encode a batch of slot_texts lists with frozen T5.
    Returns:
        slot_embs: [B, K, t5_dim]  float
        slot_mask: [B, K] bool      True = padding
    where K = max(len(slot_texts)) across batch, capped at max_len.
    """
    all_texts  = [st for row in slot_texts_batch for st in row]
    if not all_texts:
        B  = len(slot_texts_batch)
        K  = 1
        se = torch.zeros(B, K, t5_dim, dtype=torch.float32)
        sm = torch.ones(B, K, dtype=torch.bool)
        return se.to(device), sm.to(device)

    with torch.no_grad():
        embs, masks = t5_model.get_text_embeddings(all_texts)
    # embs: (sum_slots, seq_len, t5_dim), take first token or mean of valid
    # Use mean of valid tokens as the slot embedding
    slot_vecs = []
    for emb, mask in zip(embs, masks):
        valid_len = int(mask.sum().item())
        if valid_len > 0:
            slot_vecs.append(emb[:valid_len].mean(dim=0))   # (t5_dim,)
        else:
            slot_vecs.append(emb.mean(dim=0))

    # Split back into per-example groups
    K = max(len(st) for st in slot_texts_batch)
    B = len(slot_texts_batch)
    slot_embs = torch.zeros(B, K, t5_dim, dtype=torch.float32)
    slot_mask = torch.ones(B, K, dtype=torch.bool)   # True = PAD

    ptr = 0
    for i, row_texts in enumerate(slot_texts_batch):
        n = len(row_texts)
        for j in range(n):
            slot_embs[i, j] = slot_vecs[ptr].float().cpu()
            slot_mask[i, j] = False   # not padding
            ptr += 1

    return slot_embs.to(device), slot_mask.to(device)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _load_vq_tokens(tokens_path: str, device: str) -> Optional[torch.Tensor]:
    if tokens_path and Path(tokens_path).exists():
        t = torch.load(tokens_path, map_location="cpu")
        return t.long().to(device)
    return None


def _encode_pil_vq(pil_path: str, vq_model, device: str) -> Optional[torch.Tensor]:
    """Fallback: encode image from disk if no pre-computed tokens."""
    if not pil_path or not Path(pil_path).exists():
        return None
    try:
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        from PIL import Image
        img = Image.open(pil_path).convert("RGB")
        w, h = img.size
        scale = 256 / min(w, h)
        img  = img.resize((int(w * scale + .5), int(h * scale + .5)))
        img  = T.CenterCrop(256)(img)
        img_t = (TF.to_tensor(img) * 2 - 1).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(img_t)
        return indices.reshape(-1).long()
    except Exception:
        return None


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl",   required=True)
    p.add_argument("--val-jsonl",     default=None, help="Original val set for reward eval")
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--repo-root",     required=True)
    p.add_argument("--gpt-ckpt",      required=True)
    p.add_argument("--vq-ckpt",       required=True)
    p.add_argument("--t5-path",       required=True)
    p.add_argument("--qwen-model",    default=None)
    p.add_argument("--reward-mode",   default="grpo_attr_contrastive_rubric_v2")
    # adapter
    p.add_argument("--d-model",       type=int,   default=1280)
    p.add_argument("--t5-dim",        type=int,   default=2048)
    p.add_argument("--n-heads",       type=int,   default=8)
    # training
    p.add_argument("--freeze-llamagen", action="store_true", default=True)
    p.add_argument("--train-lora",    action="store_true", default=False)
    p.add_argument("--lora-r",        type=int,   default=32)
    p.add_argument("--lora-alpha",    type=float, default=64.0)
    p.add_argument("--lora-targets",  nargs="+",  default=["wqkv", "wo"])
    p.add_argument("--num-epochs",    type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--lambda-delta",  type=float, default=1e-5)
    p.add_argument("--eval-every",    type=int,   default=200)
    p.add_argument("--save-every",    type=int,   default=500)
    p.add_argument("--precision",     default="bf16")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── Load LlamaGen wrapper ─────────────────────────────────────────────────
    print("[train] Loading LlamaGen ...")
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        precision=args.precision,
        use_lora=args.train_lora,
        lora_config={
            "target_modules": args.lora_targets,
            "rank":  args.lora_r,
            "alpha": args.lora_alpha,
        } if args.train_lora else {},
    )
    gpt = wrapper.gpt          # triggers load; may inject LoRA
    t5  = wrapper.t5
    vq  = wrapper.vq_model

    # Freeze base LlamaGen if requested
    if args.freeze_llamagen and not args.train_lora:
        for p in gpt.parameters():
            p.requires_grad = False
        print("[train] LlamaGen frozen (adapter-only mode)")
    elif args.freeze_llamagen and args.train_lora:
        # LoRA already freezes base; lora params are unfrozen
        print("[train] LlamaGen frozen except LoRA params")

    # Freeze T5 always
    for p in t5.model.parameters():
        p.requires_grad = False

    # ── Build and attach SlotResidualAdapter ──────────────────────────────────
    from adaptive_curriculum.model.slot_adapter import SlotResidualAdapter, attach_slot_adapter
    adapter      = SlotResidualAdapter(
        d_model=args.d_model,
        t5_dim=args.t5_dim,
        n_heads=args.n_heads,
    ).to(device=device, dtype=wrapper.dtype)
    adapted_cls  = attach_slot_adapter(gpt, adapter)
    n_adapter    = count_parameters(adapter)
    print(f"[train] SlotResidualAdapter: {n_adapter:,} trainable params")

    # ── Optimizer (adapter params + optional LoRA) ────────────────────────────
    param_groups = [{"params": list(adapter.parameters()), "lr": args.lr}]
    if args.train_lora:
        lora_params = [p for n, p in gpt.named_parameters() if "lora_" in n and p.requires_grad]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr * 0.5})
            print(f"[train] LoRA params: {sum(p.numel() for p in lora_params):,}")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed from {args.resume} at step {start_step}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = SlotDataset(args.train_jsonl)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=0,
    )
    print(f"[train] Dataset: {len(train_ds)} examples")

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp  = args.precision in ("bf16", "fp16")
    amp_dtype = wrapper.dtype if use_amp else None

    # ── Val items (for reward eval) ───────────────────────────────────────────
    val_items = None
    reward_model = None
    if args.val_jsonl:
        from adaptive_curriculum.data.schemas import BucketItem
        val_items = []
        with open(args.val_jsonl) as f:
            for line in f:
                if line.strip():
                    val_items.append(BucketItem.from_dict(json.loads(line.strip())))
        print(f"[train] Val items: {len(val_items)}")

        qwen_id = args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
        try:
            from CARGO.scoring import CARGORewardModel
            reward_model = CARGORewardModel(model_id=qwen_id)
        except ImportError:
            from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
            reward_model = Qwen3VLRewardModel(model_id=qwen_id)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_reward = -float("inf")
    step = start_step
    t0   = time.time()

    for epoch in range(args.num_epochs):
        gpt.train()
        adapter.train()

        for batch in train_dl:
            step += 1

            # ── Prepare tokens ────────────────────────────────────────────────
            token_list = []
            slot_texts_batch = []
            skip_indices = []

            for i, row in enumerate(batch):
                toks = _load_vq_tokens(row["tokens_path"], device)
                if toks is None:
                    toks = _encode_pil_vq(row["image_path"], vq, device)
                if toks is None:
                    skip_indices.append(i)
                    continue
                token_list.append(toks)
                slot_texts_batch.append(row["slot_texts"])

            if not token_list:
                continue

            tokens = torch.stack(token_list, dim=0)  # [B, 256]
            B_eff  = tokens.shape[0]

            # ── T5 conditioning embeddings ────────────────────────────────────
            prompts       = [batch[i]["canonical_prompt"]
                             for i in range(len(batch)) if i not in skip_indices]
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            # Left-pad to cls_token_num=120 (LlamaGen convention)
            new_embs, new_masks = [], []
            for emb, mask in zip(caption_embs, emb_masks):
                valid = int(mask.sum().item())
                new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
                new_masks.append(torch.flip(mask, dims=[-1]))
            c_indices = (torch.stack(new_embs) * torch.stack(new_masks)[:, :, None]).to(device)

            # ── Slot T5 embeddings ────────────────────────────────────────────
            slot_embs, slot_mask = encode_slot_texts_t5(
                slot_texts_batch, t5, device, t5_dim=args.t5_dim
            )

            # ── Set slot context, forward, clear ─────────────────────────────
            gpt.cls_embedding.uncond_prob = 0.0
            adapted_cls.set_slot_context(
                slot_embs.to(wrapper.dtype),
                slot_mask,
            )

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = gpt(
                    idx=tokens[:, :-1],
                    cond_idx=c_indices,
                    input_pos=None,
                    targets=tokens,
                    mask=None,
                    valid=None,
                )
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.reshape(-1),
                )
                # Adapter regularisation
                info = adapted_cls.last_adapter_info or {}
                delta_norm  = info.get("delta_norm", 0.0)
                gamma       = abs(info.get("gamma",      0.0))
                reg_loss    = args.lambda_delta * (gamma * delta_norm) ** 2
                loss        = ce_loss + reg_loss

            adapted_cls.clear_slot_context()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for pg in optimizer.param_groups for p in pg["params"]],
                args.grad_clip,
            ).item()
            optimizer.step()

            # ── Logging ───────────────────────────────────────────────────────
            log = {
                "step":              step,
                "epoch":             epoch,
                "train/loss":        round(float(ce_loss), 5),
                "train/reg_loss":    round(float(reg_loss), 6),
                "train/adapter_gamma": round(float(gamma), 5),
                "train/delta_norm":  round(float(delta_norm), 4),
                "train/grad_norm":   round(grad_norm, 4),
                "train/delta_to_base_ratio": round(
                    float(delta_norm) / (info.get("base_norm", 1.0) + 1e-8), 5
                ),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log) + "\n")

            if step % 20 == 0:
                elapsed = time.time() - t0
                print(
                    f"[train] step={step}  loss={log['train/loss']:.4f}"
                    f"  gamma={log['train/adapter_gamma']:.4f}"
                    f"  delta/base={log['train/delta_to_base_ratio']:.4f}"
                    f"  elapsed={elapsed:.0f}s"
                )

            # ── Checkpoint ────────────────────────────────────────────────────
            if step % args.save_every == 0:
                _save_checkpoint(out_dir / f"ckpt_step{step}.pt", adapter, optimizer, step)

            # ── Validation ────────────────────────────────────────────────────
            if step % args.eval_every == 0 and val_items and reward_model:
                val_reward = run_val_eval(
                    val_items, wrapper, adapted_cls, adapter, reward_model,
                    args, device, n_gen=1,
                )
                log_val = {"step": step, "val/hard_reward": round(val_reward, 4)}
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_val) + "\n")
                print(f"[train] val hard_reward={val_reward:.4f}")

                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    _save_checkpoint(out_dir / "best.pt", adapter, optimizer, step)
                    print(f"[train] New best checkpoint at step {step}")
                gpt.train()
                adapter.train()

    # ── Final checkpoint ──────────────────────────────────────────────────────
    _save_checkpoint(out_dir / f"final_step{step}.pt", adapter, optimizer, step)
    print(f"[train] Done. Final step {step}. Best val reward: {best_val_reward:.4f}")


def _save_checkpoint(path, adapter, optimizer, step):
    torch.save({
        "adapter":   adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
    }, str(path))
    print(f"[train] Saved {path}")


def run_val_eval(val_items, wrapper, adapted_cls, adapter, reward_model, args, device, n_gen=1):
    """Generate n_gen images per val item, score with reward_model, return mean hard reward."""
    import contextlib
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    gpt = wrapper.gpt
    gpt.eval()
    adapter.eval()
    ls = wrapper.latent_size

    scores = []
    for item in val_items[:20]:  # limit to 20 items for speed
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])

        # No slot context for val (use raw prompt conditioning)
        adapted_cls.clear_slot_context()

        qzshape = [1, wrapper.codebook_embed_dim, ls, ls]
        pils = []
        with torch.no_grad():
            for _ in range(n_gen):
                idx = generate(
                    gpt, c_indices, ls ** 2, c_emb_masks,
                    cfg_scale=wrapper.cfg_scale, temperature=wrapper.temperature,
                    top_k=wrapper.top_k, top_p=wrapper.top_p, sample_logits=True,
                )
                decoded = wrapper.vq_model.decode_code(idx, qzshape)
                img_t   = (decoded[0].float().clamp(-1, 1) + 1) / 2
                pils.append(TF.to_pil_image(img_t.cpu()))

        results = reward_model.score_images_batch([(p, item) for p in pils], mode=args.reward_mode)
        item_scores = [float(r["score"]) for r in results]
        scores.append(max(item_scores))

    wrapper._disable_kv_cache()
    gpt.train()
    adapter.train()
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    main()
