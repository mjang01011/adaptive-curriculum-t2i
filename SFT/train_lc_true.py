"""
True likelihood-contrastive training for ImplicitCompositionAdapter.

Gradient path
-------------
  positive pass: torch.autocast DISABLED → bf16 weights + f32 input → f32 computation
                 → numerically stable backward through all GPT attention layers → adapter
  negative pass: torch.no_grad() + bf16 autocast → cheap, no graph stored

Loss
----
  L = lambda_ce       * CE(tokens | adapter(pos_caption))
    + lambda_contrast * -logsigmoid((logp_pos - logp_neg) / tau)

The hard cap in HardCapAdaptedCaptionEmbedder is applied naturally during the
positive pass, so the gradient cannot drive delta beyond target_ratio.

Why float32 forward fixes the NaN issue
----------------------------------------
  bf16 attention softmax backward can produce NaN when attention logits are
  extreme (common in pretrained frozen models). Disabling autocast promotes all
  ops to float32 (bf16 param × f32 input → f32 output), giving stable gradients.

Usage
-----
  python SFT/train_lc_true.py \\
    --clean-jsonl  $PROJ/data/gpic_merged/dataset.jsonl \\
    --output-dir   $PROJ/outputs/lc_true_v1 \\
    --repo-root    $LLAMAGEN \\
    --gpt-ckpt     $PRETRAINED/t2i_XL_stage1_256.pt \\
    --vq-ckpt      $PRETRAINED/vq_ds16_t2i.pt \\
    --t5-path      $PRETRAINED/t5-ckpt \\
    --num-epochs 60 --batch-size 2 --lr 2e-6
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
# Dataset (same as train_likelihood_contrast.py)
# ---------------------------------------------------------------------------

class ContrastDataset(Dataset):
    def __init__(self, jsonl_path: str, use_raw_caption: bool = False):
        self.jsonl_path      = jsonl_path
        self.use_raw_caption = use_raw_caption
        self.rows: list      = []
        self._load()

    def _load(self):
        rows, skipped = [], 0
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if not row.get("negative_captions"):
                        skipped += 1
                        continue
                    rows.append(row)
                except json.JSONDecodeError:
                    pass
        if skipped:
            print(f"[dataset] Skipped {skipped} rows without negatives (kept {len(rows)})", flush=True)
        self.rows = rows

    def reload(self) -> int:
        prev = len(self.rows)
        self._load()
        return len(self.rows) - prev

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row       = self.rows[idx]
        caption   = row.get("raw_caption" if self.use_raw_caption else "canonical_caption") or ""
        caption   = caption or row.get("raw_caption", row.get("canonical_caption", ""))
        return {
            "key":            row.get("key", str(idx)),
            "caption":        caption,
            "negatives":      row.get("negative_captions", []),
            "negative_types": row.get("negative_types", []),
            "tokens_path":    row.get("tokens_path"),
            "image_path":     row.get("image_path"),
        }


# ---------------------------------------------------------------------------
# Token loading / VQ encode
# ---------------------------------------------------------------------------

def _load_vq_tokens(tokens_path, device):
    if tokens_path and Path(tokens_path).exists():
        return torch.load(tokens_path, map_location="cpu").long().to(device)
    return None


def _encode_pil_vq(image_path, vq_model, device):
    if not image_path or not Path(image_path).exists():
        return None
    try:
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        from PIL import Image
        img  = Image.open(image_path).convert("RGB")
        w, h = img.size
        scale = 256 / min(w, h)
        img   = img.resize((int(w * scale + .5), int(h * scale + .5)))
        img   = T.CenterCrop(256)(img)
        img_t = (TF.to_tensor(img) * 2 - 1).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(img_t)
        return indices.reshape(-1).long()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# T5 encoding — always returns float32
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
# Negative selection
# ---------------------------------------------------------------------------

_NEG_TYPE_PRIORITY = [
    "color_swap", "color_change", "relation_reversal",
    "material_change", "pattern_change", "size_change",
    "style_global", "other",
]
_NEG_TYPE_RANK = {t: i for i, t in enumerate(_NEG_TYPE_PRIORITY)}


def _best_negative(negs, types):
    if not negs:
        return None
    if not types or len(types) != len(negs):
        return negs[0]
    return negs[min(range(len(negs)), key=lambda i: _NEG_TYPE_RANK.get(types[i], 99))]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def sequence_log_prob(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean per-token log-prob (scalar)."""
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
    p.add_argument("--clean-jsonl",      required=True)
    p.add_argument("--val-jsonl",        default=None)
    p.add_argument("--output-dir",       required=True)
    p.add_argument("--repo-root",        required=True)
    p.add_argument("--gpt-ckpt",         required=True)
    p.add_argument("--vq-ckpt",          required=True)
    p.add_argument("--t5-path",          required=True)
    p.add_argument("--qwen-model",       default=None)
    p.add_argument("--reward-mode",      default="grpo_attr_contrastive_rubric_v2")
    # adapter
    p.add_argument("--d-model",          type=int,   default=1280)
    p.add_argument("--n-comp-q",         type=int,   default=8)
    p.add_argument("--n-heads",          type=int,   default=8)
    p.add_argument("--target-ratio",     type=float, default=0.05)
    # training
    p.add_argument("--use-raw-caption",  action="store_true")
    p.add_argument("--num-epochs",       type=int,   default=60)
    p.add_argument("--batch-size",       type=int,   default=2)
    p.add_argument("--lr",               type=float, default=2e-6)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--grad-clip",        type=float, default=0.5)
    p.add_argument("--lambda-ce",        type=float, default=1.0,
                   help="Weight on CE_pos term (keeps adapter from degrading generation)")
    p.add_argument("--lambda-contrast",  type=float, default=0.05)
    p.add_argument("--tau-contrast",     type=float, default=0.2)
    p.add_argument("--max-gamma",        type=float, default=0.01)
    p.add_argument("--eval-every",       type=int,   default=200)
    p.add_argument("--save-every",       type=int,   default=200)
    p.add_argument("--dl-workers",       type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--resume",           default=None)
    p.add_argument("--min-rows",         type=int,   default=1)
    p.add_argument("--wandb",            action="store_true")
    p.add_argument("--wandb-project",    default="llamagen-lc-true")
    p.add_argument("--wandb-entity",     default=None)
    p.add_argument("--run-name",         default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint / val / viz (mirrors train_likelihood_contrast.py)
# ---------------------------------------------------------------------------

def _save_ckpt(path, adapter, optimizer, step):
    torch.save({"adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, str(path))
    print(f"[train] Saved {path}", flush=True)


def run_val_eval(val_items, wrapper, reward_model, t5_model, args, device):
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    gpt = wrapper.gpt
    ls  = wrapper.latent_size

    if hasattr(reward_model, "_model") and reward_model._model is not None:
        reward_model._model.cpu()
        torch.cuda.empty_cache()

    gpt.to(device=device)   # keep float32
    t5_model.model.to(device)
    gpt.eval()

    pils = []
    for item in val_items:
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])
        qzshape = [1, wrapper.codebook_embed_dim, ls, ls]
        with torch.no_grad():
            idx     = generate(gpt, c_indices, ls ** 2, c_emb_masks,
                               cfg_scale=wrapper.cfg_scale, temperature=wrapper.temperature,
                               top_k=wrapper.top_k, top_p=wrapper.top_p, sample_logits=True)
            decoded = wrapper.vq_model.decode_code(idx, qzshape)
            img_t   = (decoded[0].float().clamp(-1, 1) + 1) / 2
        pils.append(TF.to_pil_image(img_t.cpu()))
    wrapper._disable_kv_cache()

    gpt.cpu()
    t5_model.model.cpu()
    torch.cuda.empty_cache()

    if hasattr(reward_model, "_model") and reward_model._model is not None:
        reward_model._model.to(device)
    elif hasattr(reward_model, "_load"):
        reward_model._load()

    scores = []
    for pil, item in zip(pils, val_items):
        result = reward_model.score_images_batch([(pil, item)], mode=args.reward_mode)
        scores.append(float(result[0]["score"]))

    if hasattr(reward_model, "_model") and reward_model._model is not None:
        reward_model._model.cpu()
    torch.cuda.empty_cache()
    return sum(scores) / len(scores) if scores else 0.0


_VIZ_PROMPTS = [
    "A red cube on top of a blue sphere.",
    "A small white cat sitting next to a large black dog.",
    "A green apple to the left of a red orange on a wooden table.",
    "A striped shirt hanging above a polka dot skirt.",
]


@torch.no_grad()
def _viz_generate(wrapper, adapted_cls, step, out_dir, wandb_run):
    import torchvision.transforms.functional as TF
    from PIL import Image, ImageDraw, ImageFont
    from autoregressive.models.generate import generate

    gpt    = wrapper.gpt
    vq     = wrapper.vq_model
    device = next(gpt.parameters()).device
    ls     = wrapper.latent_size

    caption_embs, emb_masks = wrapper.t5.get_text_embeddings(_VIZ_PROMPTS)
    new_embs = []
    for emb, mask in zip(caption_embs, emb_masks):
        valid = int(mask.sum().item())
        new_embs.append(torch.cat([emb[valid:], emb[:valid]]))
    c_indices   = (torch.stack(new_embs) * torch.flip(emb_masks, dims=[-1])[:, :, None]).to(device=device)  # float32
    c_emb_masks = torch.flip(emb_masks, dims=[-1]).to(device=device)
    qzshape     = [len(_VIZ_PROMPTS), wrapper.codebook_embed_dim, ls, ls]

    def _gen(enabled):
        adapted_cls._enabled = enabled
        torch.manual_seed(42)
        idx  = generate(gpt, c_indices, ls ** 2, c_emb_masks,
                        cfg_scale=7.5, temperature=1.0, top_k=2000, top_p=1.0, sample_logits=True)
        imgs = vq.decode_code(idx, qzshape)
        wrapper._disable_kv_cache()
        return [TF.to_pil_image(((s.float().clamp(-1, 1) + 1) / 2).cpu()) for s in imgs]

    imgs_off = _gen(False)
    imgs_on  = _gen(True)
    adapted_cls._enabled = True

    W, H   = imgs_off[0].size
    pad, lh = 4, 18
    canvas  = Image.new("RGB", (2 * W + pad, len(_VIZ_PROMPTS) * (H + lh) + lh), (20, 20, 20))
    draw    = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 2), "No adapter", fill=(200, 200, 200), font=font)
    draw.text((W + pad + 4, 2), f"LC-True step {step}", fill=(100, 220, 100), font=font)
    for i, (off, on, prompt) in enumerate(zip(imgs_off, imgs_on, _VIZ_PROMPTS)):
        y = lh + i * (H + lh)
        canvas.paste(off, (0, y + lh))
        canvas.paste(on,  (W + pad, y + lh))
        draw.text((4, y + 2), prompt[:55], fill=(160, 160, 255), font=font)

    out_path = out_dir / f"viz_step{step:05d}.png"
    canvas.save(out_path)
    if wandb_run:
        import wandb
        wandb_run.log({"viz/comparison": wandb.Image(str(out_path), caption=f"step {step}")}, step=step)
    print(f"[viz] {out_path}", flush=True)


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
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                                   name=args.run_name or out_dir.name, config=vars(args))
            print(f"[train] W&B: {wandb_run.url}")
        except Exception as e:
            print(f"[train] W&B init failed: {e}")

    def _wb_log(d, step):
        if wandb_run:
            wandb_run.log(d, step=step)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── Models ────────────────────────────────────────────────────────────────
    print("[train] Loading LlamaGen ...", flush=True)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(repo_root=args.repo_root, vq_ckpt=args.vq_ckpt,
                              gpt_ckpt=args.gpt_ckpt, t5_path=args.t5_path, precision="bf16")
    gpt = wrapper.gpt
    t5  = wrapper.t5
    vq  = wrapper.vq_model

    for p in gpt.parameters():
        p.requires_grad = False
    for p in t5.model.parameters():
        p.requires_grad = False
    print("[train] GPT + T5 frozen", flush=True)

    # ── Adapter ───────────────────────────────────────────────────────────────
    from adaptive_curriculum.model.implicit_comp_adapter import (
        ImplicitCompositionAdapter, count_adapter_params,
    )
    from adaptive_curriculum.model.implicit_comp_adapter_v2 import attach_hard_cap_adapter
    adapter     = ImplicitCompositionAdapter(
        d_model=args.d_model, n_comp_q=args.n_comp_q, n_heads=args.n_heads,
    ).to(device=device)
    adapted_cls = attach_hard_cap_adapter(gpt, adapter, target_ratio=args.target_ratio)
    print(f"[train] Adapter: {count_adapter_params(adapter):,} params  target_ratio={args.target_ratio}", flush=True)

    # Cast GPT to float32 so backward through attention is numerically stable.
    # _apply() override in AdaptedCaptionEmbedder keeps the adapter in f32
    # regardless of what dtype GPT is moved to, so this is safe.
    gpt.float()
    print("[train] GPT cast to float32 for stable backward", flush=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[train] Resumed from {args.resume} at step {start_step}", flush=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = ContrastDataset(args.clean_jsonl, use_raw_caption=args.use_raw_caption)
    while len(ds) < args.min_rows:
        print(f"[train] Waiting for {args.min_rows} rows (have {len(ds)}) ...", flush=True)
        time.sleep(15)
        ds.reload()
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x,
                    num_workers=args.dl_workers, pin_memory=True,
                    persistent_workers=args.dl_workers > 0)
    print(f"[train] Dataset: {len(ds)} examples", flush=True)

    # ── Val ───────────────────────────────────────────────────────────────────
    val_items = reward_model = None
    if args.val_jsonl:
        from adaptive_curriculum.data.schemas import BucketItem
        val_items = []
        with open(args.val_jsonl) as f:
            for line in f:
                if line.strip():
                    val_items.append(BucketItem.from_dict(json.loads(line.strip())))
        print(f"[train] Val: {len(val_items)} items")
        qwen_id = args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
        try:
            from CARGO.scoring import CARGORewardModel
            reward_model = CARGORewardModel(model_id=qwen_id)
        except ImportError:
            from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
            reward_model = Qwen3VLRewardModel(model_id=qwen_id)

    amp_dtype = torch.float32   # GPT is now float32 throughout

    # Force SDPA to use the math (reference) backend.
    # Flash and mem-efficient backends can produce NaN in float32 backward through
    # frozen causal-masked attention on some PyTorch/CUDA versions.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        print("[train] SDPA: using math backend (flash+memeff disabled)", flush=True)
    except Exception as e:
        print(f"[train] SDPA backend override failed (non-fatal): {e}", flush=True)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val   = -float("inf")
    step       = start_step
    t0         = time.time()
    nan_streak = 0
    _anomaly_steps = 3  # run anomaly detection on first N steps to locate NaN source

    for epoch in range(args.num_epochs):
        gpt.train()
        adapter.train()

        for batch in dl:
            step += 1

            # ── Gather batch ──────────────────────────────────────────────────
            token_list, captions, neg_captions = [], [], []
            for row in batch:
                toks = _load_vq_tokens(row["tokens_path"], device)
                if toks is None:
                    toks = _encode_pil_vq(row["image_path"], vq, device)
                if toks is None:
                    continue
                token_list.append(toks)
                captions.append(row["caption"])
                neg_captions.append(_best_negative(row["negatives"], row["negative_types"]))

            if not token_list:
                continue

            tokens = torch.stack(token_list)   # [B, 256]
            B      = tokens.shape[0]
            c_pos  = t5_encode(captions, t5, device)   # float32 [B, 120, 2048]

            if torch.isnan(c_pos).any():
                print(f"[train] WARNING: NaN in c_pos step {step}, skipping", flush=True)
                continue

            gpt.cls_embedding.uncond_prob = 0.0
            adapted_cls._enabled = True

            # ── Positive pass: GPT is float32 → stable attention backward ──────
            logits_pos, _ = gpt(
                idx=tokens[:, :-1],
                cond_idx=c_pos,          # float32, GPT is float32, no dtype mismatch
                input_pos=None, targets=tokens, mask=None, valid=None,
            )

            if torch.isnan(logits_pos).any() or torch.isinf(logits_pos).any():
                print(f"[train] WARNING: NaN/inf in logits_pos step {step}, skipping", flush=True)
                continue

            logp_pos = sequence_log_prob(logits_pos, tokens)   # has gradient

            info = adapted_cls._last_info or {}

            # ── Negative pass: no_grad + bf16 ────────────────────────────────
            logp_neg      = torch.tensor(0.0, device=device)
            contrast_loss = torch.tensor(0.0, device=device)
            logp_margin   = 0.0

            neg_indices = [i for i, n in enumerate(neg_captions) if n is not None]
            if neg_indices and args.lambda_contrast > 0:
                neg_texts = [neg_captions[i] for i in neg_indices]
                c_neg     = t5_encode(neg_texts, t5, device)

                with torch.no_grad():
                    logits_neg, _ = gpt(
                        idx=tokens[neg_indices, :-1],
                        cond_idx=c_neg,      # float32, matches GPT dtype
                        input_pos=None, targets=tokens[neg_indices],
                        mask=None, valid=None,
                    )
                    if torch.isnan(logits_neg).any():
                        print(f"[train] WARNING: NaN in logits_neg step {step}, skipping contrast", flush=True)
                        logp_neg = logp_pos[neg_indices].detach() if hasattr(logp_pos, '__getitem__') else logp_pos.detach()
                    else:
                        logp_neg = sequence_log_prob(logits_neg, tokens[neg_indices]).detach()

                # Compute logp_pos for the neg_indices rows only
                logp_pos_sub  = sequence_log_prob(logits_pos[neg_indices], tokens[neg_indices])
                margin_val    = (logp_pos_sub - logp_neg) / args.tau_contrast
                contrast_loss = -F.logsigmoid(margin_val).mean()
                logp_margin   = float(logp_pos_sub.detach().item()) - float(logp_neg.item())

            # ── Total loss ────────────────────────────────────────────────────
            loss = args.lambda_ce * (-logp_pos) + args.lambda_contrast * contrast_loss

            if not loss.requires_grad:
                print(f"[train] WARNING: loss has no grad at step {step}, skipping", flush=True)
                continue
            if not torch.isfinite(loss):
                print(f"[train] WARNING: non-finite loss={float(loss):.4f} step {step}", flush=True)
                continue

            optimizer.zero_grad()
            if step <= _anomaly_steps:
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
            else:
                loss.backward()

            # Gradient diagnostics
            nan_grad_params = [(n, p) for n, p in adapter.named_parameters()
                               if p.grad is not None and not torch.isfinite(p.grad).all()]
            n_nan_grad      = len(nan_grad_params)
            raw_grad_norm   = sum(p.grad.norm().item() ** 2 for _, p in adapter.named_parameters()
                                  if p.grad is not None and torch.isfinite(p.grad).all()) ** 0.5
            if n_nan_grad > 0:
                print(f"[train] WARNING: {n_nan_grad} NaN-grad params step {step}: "
                      f"{[n for n, _ in nan_grad_params[:5]]}", flush=True)

            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip).item()

            if not math.isfinite(grad_norm):
                print(f"[train] WARNING: non-finite grad_norm step {step}, zeroing", flush=True)
                optimizer.zero_grad()
            else:
                optimizer.step()
                if args.max_gamma is not None:
                    with torch.no_grad():
                        adapter.gamma.clamp_(0.0, args.max_gamma)

            # ── Early stop ────────────────────────────────────────────────────
            post_cap = float(info.get("post_cap_ratio", info.get("effective_delta_to_base", 0.0)))
            if post_cap > 0.5:
                nan_streak += 1
            else:
                nan_streak = 0
            if nan_streak >= 10:
                print(f"[train] EARLY STOP: post_cap_ratio={post_cap:.3f} > 0.5 for 10 steps", flush=True)
                _save_ckpt(out_dir / f"early_stop_step{step}.pt", adapter, optimizer, step)
                raise SystemExit(0)

            # ── Log ───────────────────────────────────────────────────────────
            gamma      = abs(info.get("gamma", 0.0))
            delta_norm = info.get("delta_norm", 0.0)
            pre_cap    = float(info.get("pre_cap_ratio",  info.get("effective_delta_to_base", 0.0)))
            cap_scale  = float(info.get("hard_cap_scale", 1.0))
            log = {
                "step":                  step,
                "epoch":                 epoch,
                "train/loss":            round(float(loss),              5),
                "train/ce":              round(float(-logp_pos.detach()), 5),
                "train/logp_pos":        round(float(logp_pos.detach()),  5),
                "train/logp_neg":        round(float(logp_neg),           5),
                "train/logp_margin":     round(logp_margin,               5),
                "train/contrast":        round(float(contrast_loss),      5),
                "train/gamma":           round(gamma,                     5),
                "train/delta_norm":      round(float(delta_norm),         4),
                "train/grad_norm":       round(grad_norm if math.isfinite(grad_norm) else -1, 4),
                "train/raw_grad_norm":   round(raw_grad_norm,             4),
                "train/n_nan_grad":      n_nan_grad,
                "train/pre_cap_ratio":   round(pre_cap,                   5),
                "train/post_cap_ratio":  round(post_cap,                  5),
                "train/hard_cap_scale":  round(cap_scale,                 4),
                "train/slot_entropy":    round(float(info.get("slot_attn_entropy", 0.0)), 4),
                "train/eff_delta":       round(float(info.get("effective_delta_to_base", 0.0)), 6),
            }
            with open(log_path, "a") as lf:
                lf.write(json.dumps(log) + "\n")
            _wb_log({k: v for k, v in log.items() if k != "step"}, step)

            if step % 20 == 0:
                elapsed = time.time() - t0
                print(
                    f"[train] step={step}"
                    f"  loss={log['train/loss']:.4f}"
                    f"  ce={log['train/ce']:.4f}"
                    f"  contrast={log['train/contrast']:.4f}"
                    f"  margin={log['train/logp_margin']:+.4f}"
                    f"  grad={log['train/grad_norm']:.4f}"
                    f"  raw_grad={log['train/raw_grad_norm']:.4f}"
                    f"  nan_grads={log['train/n_nan_grad']}"
                    f"  gamma={log['train/gamma']:.5f}"
                    f"  eff_delta={log['train/eff_delta']:.5f}"
                    f"  pre_cap={log['train/pre_cap_ratio']:.4f}"
                    f"  post_cap={log['train/post_cap_ratio']:.4f}"
                    f"  elapsed={elapsed:.0f}s",
                    flush=True,
                )

            if step % args.save_every == 0:
                _save_ckpt(out_dir / f"ckpt_step{step}.pt", adapter, optimizer, step)

            if step % args.eval_every == 0 and wandb_run:
                gpt.eval(); adapter.eval()
                _viz_generate(wrapper, adapted_cls, step, out_dir, wandb_run)
                gpt.train(); adapter.train()

            if step % args.eval_every == 0 and val_items and reward_model:
                val_r = run_val_eval(val_items[:20], wrapper, reward_model, t5, args, device)
                gpt.to(device=device)   # keep float32
                t5.model.to(device)
                torch.cuda.empty_cache()
                _wb_log({"val/hard_reward": val_r}, step)
                print(f"[train] val hard_reward={val_r:.4f}", flush=True)
                if val_r > best_val:
                    best_val = val_r
                    _save_ckpt(out_dir / "best.pt", adapter, optimizer, step)
                    print(f"[train] New best at step {step}", flush=True)
                gpt.train(); adapter.train()

        new_rows = ds.reload()
        if new_rows > 0:
            print(f"[train] Epoch {epoch}: +{new_rows} new rows → {len(ds)} total", flush=True)

    _save_ckpt(out_dir / f"final_step{step}.pt", adapter, optimizer, step)
    print(f"[train] Done. Step {step}. Best val: {best_val:.4f}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
