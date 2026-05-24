"""
Train ImplicitCompositionAdapter for LlamaGen.

At inference, the adapter operates on any raw prompt with no parser.
During training, optionally adds a contrastive caption loss that teaches
the adapter that attribute/relation words matter.

Loss
----
  L = CE(image_tokens | canonical_caption)
    + λ_contrast * L_contrast                  (if --lambda-contrast > 0)
    + λ_delta    * ||γ * Δ||²                  (residual regularisation)

Contrastive loss
  For each example, a negative caption (attribute-swapped) is also available.
  L_contrast = -log σ(logp_correct - logp_neg)
  Only the positive pass produces gradients; the negative log-prob is computed
  with torch.no_grad() to keep memory cost bounded.

Usage
-----
  python SFT/train_implicit_adapter.py \\
    --train-jsonl $PROJ/data/gpic_slots_v1/dataset.jsonl \\
    --val-jsonl   $PROJ/data/attribute_binding/attribute_binding_val_20.jsonl \\
    --output-dir  $PROJ/outputs/implicit_adapter_v1 \\
    --repo-root $LLAMAGEN \\
    --gpt-ckpt  $PRETRAINED/t2i_XL_stage1_256.pt \\
    --vq-ckpt   $PRETRAINED/vq_ds16_t2i.pt \\
    --t5-path   $PRETRAINED/t5-ckpt \\
    --freeze-llamagen \\
    --num-epochs 3 --batch-size 4 --lr 1e-4
"""
import argparse
import json
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

class CompDataset(Dataset):
    """
    Loads GPIC mined JSONL.
    Required fields: canonical_caption, tokens_path or image_path.
    Optional fields: negative_captions (for contrastive loss).

    Supports live reload: call reload() at the start of each epoch to pick up
    rows appended by a concurrently running miner.
    """

    def __init__(self, jsonl_path: str, use_raw_caption: bool = False):
        self.jsonl_path      = jsonl_path
        self.use_raw_caption = use_raw_caption
        self.rows: list      = []
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
                        pass   # partial line written mid-flush — skip
        self.rows = rows

    def reload(self) -> int:
        """Re-read the JSONL. Returns number of new rows added."""
        prev = len(self.rows)
        self._load()
        return len(self.rows) - prev

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row      = self.rows[idx]
        caption  = row.get("raw_caption" if self.use_raw_caption else "canonical_caption") or ""
        caption  = caption or row.get("raw_caption", row.get("canonical_caption", ""))
        negatives = row.get("negative_captions", [])
        return {
            "key":         row.get("key", str(idx)),
            "caption":     caption,
            "negatives":   negatives,
            "tokens_path": row.get("tokens_path"),
            "image_path":  row.get("image_path"),
        }


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

def _load_vq_tokens(tokens_path: Optional[str], device: str) -> Optional[torch.Tensor]:
    if tokens_path and Path(tokens_path).exists():
        return torch.load(tokens_path, map_location="cpu").long().to(device)
    return None


def _encode_pil_vq(image_path: Optional[str], vq_model, device: str) -> Optional[torch.Tensor]:
    if not image_path or not Path(image_path).exists():
        return None
    try:
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        from PIL import Image
        img  = Image.open(image_path).convert("RGB")
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


# ---------------------------------------------------------------------------
# T5 encoding helper
# ---------------------------------------------------------------------------

def t5_encode(texts: List[str], t5_model, device: str, cls_token_num: int = 120) -> torch.Tensor:
    """
    Returns c_indices [B, cls_token_num, 2048] matching LlamaGen's left-pad convention.
    """
    with torch.no_grad():
        embs, masks = t5_model.get_text_embeddings(texts)
    new_embs, new_masks = [], []
    for emb, mask in zip(embs, masks):
        valid = int(mask.sum().item())
        new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
        new_masks.append(torch.flip(mask, dims=[-1]))
    # float32 to avoid bf16 overflow/NaN in conditioning tokens
    c_indices = (torch.stack(new_embs) * torch.stack(new_masks)[:, :, None]).float().to(device)
    return c_indices


def build_t5_cache(
    dataset,
    t5_model,
    device: str,
    encode_batch: int = 16,
) -> dict:
    """
    Pre-encode every unique caption (positive + negatives) in the dataset.
    Embeddings are stored on CPU; the training loop moves them to GPU per batch.
    Returns dict[text -> cpu tensor [120, 2048]].
    """
    all_texts: set = set()
    for row in dataset.rows:
        cap = row.get("canonical_caption") or row.get("raw_caption", "")
        if cap:
            all_texts.add(cap)
        for neg in row.get("negative_captions", []):
            if neg:
                all_texts.add(neg)

    all_texts = list(all_texts)
    cache: dict = {}
    print(f"[train] Building T5 cache for {len(all_texts)} unique texts ...", flush=True)
    for i in range(0, len(all_texts), encode_batch):
        batch = all_texts[i : i + encode_batch]
        embs = t5_encode(batch, t5_model, device)           # [B, 120, 2048] on GPU
        for text, emb in zip(batch, embs):
            cache[text] = emb.cpu()                         # keep on CPU
        if i % (encode_batch * 20) == 0:
            print(f"  [{i}/{len(all_texts)}]", flush=True)

    print(f"[train] T5 cache ready: {len(cache)} entries", flush=True)
    return cache


def t5_lookup(texts: List[str], cache: dict, t5_model, device: str) -> torch.Tensor:
    """Look up cached T5 embeddings; fall back to live encode for any cache miss."""
    hits, misses_idx, misses_text = [], [], []
    for i, text in enumerate(texts):
        if text in cache:
            hits.append((i, cache[text]))
        else:
            misses_idx.append(i)
            misses_text.append(text)

    result = [None] * len(texts)
    for i, emb in hits:
        result[i] = emb.to(device)
    if misses_text:
        live = t5_encode(misses_text, t5_model, device)
        for idx, emb in zip(misses_idx, live):
            result[idx] = emb
            cache[misses_text[misses_idx.index(idx)]] = emb.cpu()

    return torch.stack(result)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sequence_log_prob(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean per-token log-prob. Shape: scalar.
    Mean (not sum) keeps magnitude stable regardless of sequence length,
    preventing logsigmoid saturation in the contrastive loss.
    """
    return -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl",   required=True)
    p.add_argument("--val-jsonl",     default=None)
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--repo-root",     required=True)
    p.add_argument("--gpt-ckpt",      required=True)
    p.add_argument("--vq-ckpt",       required=True)
    p.add_argument("--t5-path",       required=True)
    p.add_argument("--qwen-model",    default=None)
    p.add_argument("--reward-mode",   default="grpo_attr_contrastive_rubric_v2")
    # adapter
    p.add_argument("--d-model",       type=int,   default=1280)
    p.add_argument("--n-comp-q",      type=int,   default=8,  help="Number of composition query tokens")
    p.add_argument("--n-heads",       type=int,   default=8)
    # training
    p.add_argument("--freeze-llamagen", action="store_true", default=True)
    p.add_argument("--train-lora",    action="store_true")
    p.add_argument("--lora-r",        type=int,   default=32)
    p.add_argument("--lora-alpha",    type=float, default=64.0)
    p.add_argument("--lora-targets",  nargs="+",  default=["wqkv", "wo"])
    p.add_argument("--use-raw-caption", action="store_true", help="Train on raw caption instead of canonical")
    p.add_argument("--num-epochs",    type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--lambda-contrast", type=float, default=0.1, help="0 = disable contrastive loss")
    p.add_argument("--tau-contrast",  type=float, default=0.1,  help="temperature for contrastive logp margin")
    p.add_argument("--lambda-delta",  type=float, default=1e-5)
    p.add_argument("--eval-every",    type=int,   default=500)
    p.add_argument("--save-every",    type=int,   default=500)
    p.add_argument("--dl-workers",    type=int,   default=2)
    p.add_argument("--precision",     default="bf16")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--resume",        default=None)
    p.add_argument("--min-rows",      type=int,   default=200,
                   help="Wait until dataset has at least this many rows before starting")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb-project", default="llamagen-implicit-adapter")
    p.add_argument("--wandb-entity",  default=None)
    p.add_argument("--run-name",      default=None)
    return p.parse_args()


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

    run_name = args.run_name or Path(args.output_dir).name
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=run_name,
                config=vars(args),
            )
            print(f"[train] W&B: {wandb_run.url}")
        except Exception as e:
            print(f"[train] W&B init failed (continuing without): {e}")

    def _wb_log(d, step):
        if wandb_run:
            wandb_run.log(d, step=step)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── Load LlamaGen ─────────────────────────────────────────────────────────
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
    gpt = wrapper.gpt
    t5  = wrapper.t5
    vq  = wrapper.vq_model

    if args.freeze_llamagen and not args.train_lora:
        for p in gpt.parameters():
            p.requires_grad = False
        print("[train] LlamaGen frozen (adapter-only mode)")
    elif args.freeze_llamagen and args.train_lora:
        print("[train] LlamaGen frozen except LoRA params")

    for p in t5.model.parameters():
        p.requires_grad = False

    # ── Attach ImplicitCompositionAdapter ─────────────────────────────────────
    from adaptive_curriculum.model.implicit_comp_adapter import (
        ImplicitCompositionAdapter, attach_implicit_adapter, count_adapter_params,
    )
    # Adapter stays in float32 — bf16 attention softmax can overflow
    adapter     = ImplicitCompositionAdapter(
        d_model=args.d_model, n_comp_q=args.n_comp_q, n_heads=args.n_heads,
    ).to(device=device)
    adapted_cls = attach_implicit_adapter(gpt, adapter)

    n_adapter = count_adapter_params(adapter)
    print(f"[train] ImplicitCompositionAdapter: {n_adapter:,} trainable params  n_comp_q={args.n_comp_q}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
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
    train_ds = CompDataset(args.train_jsonl, use_raw_caption=args.use_raw_caption)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda x: x, num_workers=args.dl_workers,
                          pin_memory=True, persistent_workers=args.dl_workers > 0)
    # ── Wait for enough rows (parallel mining support) ────────────────────────
    if len(train_ds) < args.min_rows:
        print(f"[train] Waiting for dataset to reach {args.min_rows} rows "
              f"(currently {len(train_ds)}) ...", flush=True)
        while len(train_ds) < args.min_rows:
            time.sleep(15)
            train_ds.reload()
        print(f"[train] Dataset ready: {len(train_ds)} rows", flush=True)

    print(f"[train] Dataset: {len(train_ds)} examples  "
          f"contrastive_lambda={args.lambda_contrast}")

    # ── T5 cache (encode all captions once, reuse every step) ─────────────────
    t5_cache = build_t5_cache(train_ds, t5, device)
    # T5 no longer needed on GPU — cache has everything; offload to free ~3GB
    t5.model.cpu()
    torch.cuda.empty_cache()
    print("[train] T5 offloaded to CPU after cache build", flush=True)

    use_amp   = args.precision in ("bf16", "fp16")
    amp_dtype = wrapper.dtype if use_amp else None

    # ── Val + reward model ────────────────────────────────────────────────────
    val_items    = None
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

            # ── Gather valid examples from batch ─────────────────────────────
            token_list  = []
            captions    = []
            neg_captions = []

            for row in batch:
                toks = _load_vq_tokens(row["tokens_path"], device)
                if toks is None:
                    toks = _encode_pil_vq(row["image_path"], vq, device)
                if toks is None:
                    continue
                token_list.append(toks)
                captions.append(row["caption"])
                negs = row.get("negatives", [])
                # negs[0] is the highest-priority negative (spatial > attr-swap > attr-drop)
                neg_captions.append(negs[0] if negs else None)

            if not token_list:
                continue

            tokens = torch.stack(token_list, dim=0)   # [B, 256]
            B_eff  = tokens.shape[0]

            # ── T5 conditioning (from cache — no T5 forward at train time) ──
            # Cast to model dtype (bf16) after lookup; cache stores float32
            c_indices = t5_lookup(captions, t5_cache, t5, device).to(dtype=wrapper.dtype)

            # Guard: skip batch if conditioning or tokens have NaN/inf
            if torch.isnan(c_indices).any() or torch.isinf(c_indices).any():
                print(f"[train] WARNING: NaN/inf in c_indices at step {step}, skipping batch", flush=True)
                continue
            if torch.isnan(tokens.float()).any():
                print(f"[train] WARNING: NaN in tokens at step {step}, skipping batch", flush=True)
                continue

            gpt.cls_embedding.uncond_prob = 0.0

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, _ = gpt(
                    idx=tokens[:, :-1],
                    cond_idx=c_indices,
                    input_pos=None, targets=tokens, mask=None, valid=None,
                )
                if torch.isnan(logits).any():
                    info = adapted_cls.last_adapter_info or {}
                    print(
                        f"[train] WARNING: NaN logits at step {step}"
                        f" | c_max={c_indices.abs().max():.2f}"
                        f" | gamma={info.get('gamma', float('nan')):.5f}"
                        f" | delta_norm={info.get('delta_norm', float('nan')):.4f}"
                        f" | eff_d/b={info.get('effective_delta_to_base', float('nan')):.5f}",
                        flush=True,
                    )
                    continue
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.reshape(-1),
                )

            # ── Contrastive caption loss ─────────────────────────────────────
            contrast_loss = torch.tensor(0.0, device=device)
            if args.lambda_contrast > 0:
                # Collect examples that have a negative caption
                neg_indices  = [i for i, n in enumerate(neg_captions) if n is not None]
                if neg_indices:
                    neg_texts    = [neg_captions[i] for i in neg_indices]
                    neg_toks     = tokens[neg_indices]
                    c_neg        = t5_lookup(neg_texts, t5_cache, t5, device).to(dtype=wrapper.dtype)

                    # logp under negative caption — no gradient (used as baseline)
                    with torch.no_grad():
                        logits_neg, _ = gpt(
                            idx=neg_toks[:, :-1],
                            cond_idx=c_neg,
                            input_pos=None, targets=neg_toks, mask=None, valid=None,
                        )
                        logp_neg = sequence_log_prob(logits_neg, neg_toks).detach()

                    # logp under correct caption — with gradient
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        logits_pos, _ = gpt(
                            idx=neg_toks[:, :-1],
                            cond_idx=c_indices[neg_indices],
                            input_pos=None, targets=neg_toks, mask=None, valid=None,
                        )
                        logp_pos = sequence_log_prob(logits_pos, neg_toks)

                    # -log sigmoid((logp_correct - logp_neg) / τ)
                    contrast_loss = -F.logsigmoid(
                        (logp_pos - logp_neg) / args.tau_contrast
                    ).mean()

            # ── Residual regularisation ──────────────────────────────────────
            info       = adapted_cls.last_adapter_info or {}
            delta_norm = info.get("delta_norm", 0.0)
            gamma      = abs(info.get("gamma",      0.0))
            reg_loss   = args.lambda_delta * (gamma * delta_norm) ** 2

            loss = ce_loss + args.lambda_contrast * contrast_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for pg in optimizer.param_groups for p in pg["params"]],
                args.grad_clip,
            ).item()
            optimizer.step()

            # ── Log ──────────────────────────────────────────────────────────
            log = {
                "step":             step,
                "epoch":            epoch,
                "train/loss":       round(float(ce_loss),       5),
                "train/contrast":   round(float(contrast_loss), 5),
                "train/reg":        round(float(reg_loss),      6),
                "train/gamma":      round(gamma,                5),
                "train/delta_norm": round(float(delta_norm),    4),
                "train/grad_norm":  round(grad_norm,            4),
                "train/delta_to_base":           round(info.get("delta_to_base",           0.0), 5),
                "train/effective_delta_to_base": round(info.get("effective_delta_to_base", 0.0), 6),
                "train/slot_entropy":            round(info.get("slot_attn_entropy",       0.0), 4),
            }
            with open(log_path, "a") as lf:
                lf.write(json.dumps(log) + "\n")
            _wb_log({k: v for k, v in log.items() if k != "step"}, step)

            if step % 20 == 0:
                elapsed = time.time() - t0
                print(
                    f"[train] step={step}  loss={log['train/loss']:.4f}"
                    f"  contrast={log['train/contrast']:.4f}"
                    f"  gamma={log['train/gamma']:.5f}"
                    f"  γΔ/base={log['train/effective_delta_to_base']:.5f}"
                    f"  elapsed={elapsed:.0f}s"
                )

            if step % args.save_every == 0:
                _save_ckpt(out_dir / f"ckpt_step{step}.pt", adapter, optimizer, step)

            if step % args.eval_every == 0 and val_items and reward_model:
                val_r = run_val_eval(val_items[:20], wrapper, reward_model, t5, args, device)
                # Restore gpt + T5 to GPU for continued training
                gpt.to(device=device, dtype=wrapper.dtype)
                t5.model.to(device)
                torch.cuda.empty_cache()

                log_v = {"step": step, "val/hard_reward": round(val_r, 4)}
                with open(log_path, "a") as lf:
                    lf.write(json.dumps(log_v) + "\n")
                _wb_log({"val/hard_reward": val_r}, step)
                print(f"[train] val hard_reward={val_r:.4f}")
                if val_r > best_val_reward:
                    best_val_reward = val_r
                    _save_ckpt(out_dir / "best.pt", adapter, optimizer, step)
                    print(f"[train] New best at step {step}")
                gpt.train()
                adapter.train()

    _save_ckpt(out_dir / f"final_step{step}.pt", adapter, optimizer, step)
    print(f"[train] Done. Step {step}. Best val reward: {best_val_reward:.4f}")
    if wandb_run:
        wandb_run.finish()


def _save_ckpt(path, adapter, optimizer, step):
    torch.save({
        "adapter":   adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
    }, str(path))
    print(f"[train] Saved {path}")


def run_val_eval(val_items, wrapper, reward_model, t5_model, args, device):
    """
    Two-phase val to avoid OOM on 48GB GPU:
      Phase 1 — generate images: gpt + T5 on GPU, Qwen on CPU
      Phase 2 — score images:    gpt + T5 on CPU, Qwen on GPU
    """
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    gpt = wrapper.gpt
    ls  = wrapper.latent_size

    # ── Phase 1: generate all images (gpt + T5 on GPU) ───────────────────────
    if hasattr(reward_model, "_model") and reward_model._model is not None:
        reward_model._model.cpu()
        torch.cuda.empty_cache()

    gpt.to(device=device, dtype=wrapper.dtype)
    t5_model.model.to(device)
    gpt.eval()

    pils = []
    for item in val_items:
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])
        qzshape = [1, wrapper.codebook_embed_dim, ls, ls]
        with torch.no_grad():
            idx     = generate(
                gpt, c_indices, ls ** 2, c_emb_masks,
                cfg_scale=wrapper.cfg_scale, temperature=wrapper.temperature,
                top_k=wrapper.top_k, top_p=wrapper.top_p, sample_logits=True,
            )
            decoded = wrapper.vq_model.decode_code(idx, qzshape)
            img_t   = (decoded[0].float().clamp(-1, 1) + 1) / 2
        pils.append(TF.to_pil_image(img_t.cpu()))
    wrapper._disable_kv_cache()

    # ── Phase 2: score with Qwen (gpt + T5 on CPU, Qwen on GPU) ──────────────
    gpt.cpu()
    t5_model.model.cpu()
    torch.cuda.empty_cache()

    if hasattr(reward_model, "_model") and reward_model._model is not None:
        reward_model._model.to(device)
    elif hasattr(reward_model, "_load"):
        reward_model._load()   # lazy-loads directly to GPU

    scores = []
    for pil, item in zip(pils, val_items):
        result = reward_model.score_images_batch([(pil, item)], mode=args.reward_mode)
        scores.append(float(result[0]["score"]))

    # Offload Qwen again so training can resume on GPU
    if hasattr(reward_model, "_model") and reward_model._model is not None:
        reward_model._model.cpu()
    torch.cuda.empty_cache()

    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    main()
