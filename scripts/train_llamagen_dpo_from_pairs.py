"""
DPO training for LlamaGen from pre-built G=2 pairs (one or more buckets).
Simplified DPO loss (no separate reference model — LoRA zeroed = reference).
Val progression: generates images + scores with Qwen every N steps, saves grid HTML.

Usage:
  python scripts/train_llamagen_dpo_from_pairs.py \
    --pairs-jsonl \
        /viscam/u/jj277/adaptive-curriculum-t2i/outputs_pairs/attr_g2_TSTAMP/pairs.jsonl \
        /viscam/u/jj277/adaptive-curriculum-t2i/outputs_pairs/spatial_g2_TSTAMP/pairs.jsonl \
    --val-jsonl \
        /viscam/u/jj277/adaptive-curriculum-t2i/data/attribute_binding/attribute_binding_val_20.jsonl \
        /viscam/u/jj277/adaptive-curriculum-t2i/data/spatial_relations_anchored/spatial_relations_anchored_val_20.jsonl \
    --output-dir /viscam/u/jj277/adaptive-curriculum-t2i/outputs_dpo/llamagen_dpo_both \
    --repo-root  /viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen \
    --gpt-ckpt   /viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/t2i_XL_stage1_256.pt \
    --vq-ckpt    /viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/vq_ds16_t2i.pt \
    --t5-path    /viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/t5-ckpt \
    --lora-r 8 --lora-alpha 16 \
    --lr 1e-5 --beta 0.1 --epochs 3 --batch-size 2 \
    --eval-every-steps 20 --val-seeds 0 1
"""
import argparse
import base64
import io
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

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from adaptive_curriculum.data.schemas import BucketItem


# ══════════════════════════════════════════════════════════════════════════════
# Minimal batch item (needed for _get_conditioning)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PairItem:
    id:     str
    text:   str
    bucket: str


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_val_items(jsonl_path, n=20) -> List[BucketItem]:
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))
    return items[:n]


# ══════════════════════════════════════════════════════════════════════════════
# DPO loss
# ══════════════════════════════════════════════════════════════════════════════

def dpo_loss_with_ref(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp,
                      pair_weights, beta):
    """Full DPO with reference (LoRA zeroed = base model)."""
    pi_lograt  = chosen_lp  - rejected_lp
    ref_lograt = ref_chosen_lp - ref_rejected_lp
    raw  = -F.logsigmoid(beta * (pi_lograt - ref_lograt))
    return (pair_weights * raw).mean()


def dpo_loss_no_ref(chosen_lp, rejected_lp, pair_weights, beta):
    """Simplified DPO (no reference). dpo_variant=simplified_no_reference."""
    raw = -F.logsigmoid(beta * (chosen_lp - rejected_lp))
    return (pair_weights * raw).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Val eval — generate + score + save grid HTML
# ══════════════════════════════════════════════════════════════════════════════

def _b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_val_eval(wrapper, reward_model, val_items_by_bucket, val_seeds, step,
                 out_dir, base_images=None, reward_mode="pseudo_soft_grpo_target_heavy",
                 wandb_run=None):
    """
    Generate val images for all buckets, score, save per-bucket and combined HTML.
    Returns (mean_reward, comp_means, new_images dict).
    """
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    eval_dir = Path(out_dir) / f"eval_step_{step:04d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "images").mkdir(exist_ok=True)

    wrapper.gpt.eval()
    all_rewards = []
    all_comps   = {}
    new_images  = {}   # (item_id, seed) → PIL
    scores_rows = []

    with torch.no_grad():
        for bucket, val_items in val_items_by_bucket.items():
            for seed in val_seeds:
                torch.manual_seed(seed)
                B       = len(val_items)
                qzshape = [B, wrapper.codebook_embed_dim,
                           wrapper.latent_size, wrapper.latent_size]

                c_idx, c_msk = wrapper._get_conditioning(val_items, t5_cache=None)
                tokens = generate(
                    wrapper.gpt, c_idx, wrapper.latent_size ** 2, c_msk,
                    cfg_scale=wrapper.cfg_scale, temperature=wrapper.temperature,
                    top_k=wrapper.top_k, top_p=wrapper.top_p, sample_logits=True,
                )
                wrapper._disable_kv_cache()
                decoded = wrapper.vq_model.decode_code(tokens, qzshape)

                for i, item in enumerate(val_items):
                    img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
                    pil   = TF.to_pil_image(img_t.cpu())
                    fname = f"{item.id}_seed{seed}.png"
                    pil.save(eval_dir / "images" / fname)
                    new_images[(item.id, seed)] = pil

                    result = reward_model.score_image(pil, item, mode=reward_mode)
                    r      = float(result["score"])
                    comps  = result.get("component_scores", {})
                    all_rewards.append(r)
                    for k, v in comps.items():
                        all_comps.setdefault(k, []).append(float(v))
                    scores_rows.append({"id": item.id, "bucket": bucket, "seed": seed,
                                        "reward": round(r, 4),
                                        "components": {k: round(float(v), 4)
                                                       for k, v in comps.items()}})

    mean_r     = float(np.mean(all_rewards)) if all_rewards else 0.0
    comp_means = {k: round(float(np.mean(v)), 4) for k, v in all_comps.items()}
    n          = len(all_rewards)
    se         = float(np.std(all_rewards) / math.sqrt(n)) if n > 1 else 0.0

    with open(eval_dir / "summary.json", "w") as f:
        json.dump({"step": step, "n": n, "mean_reward": round(mean_r, 4),
                   "se": round(se, 4), "components": comp_means}, f, indent=2)
    with open(eval_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for r in scores_rows:
            f.write(json.dumps(r) + "\n")

    # ── grid HTML ─────────────────────────────────────────────────────────────
    rows_html = ""
    for bucket, val_items in val_items_by_bucket.items():
        rows_html += f"<tr><td colspan='99' style='background:#222;color:#fa0;padding:6px'>{bucket}</td></tr>"
        for item in val_items:
            cols = f"<td style='color:#aef;font-size:10px;vertical-align:top;padding:4px;max-width:200px'>{item.text}</td>"
            for seed in val_seeds:
                cur   = new_images.get((item.id, seed))
                cur_r = next((s["reward"] for s in scores_rows
                              if s["id"] == item.id and s["seed"] == seed), 0)
                base  = base_images.get((item.id, seed)) if base_images else None
                if base:
                    cols += f"<td><div style='font-size:9px;color:#888'>base</div><img src='{_b64(base)}' style='width:110px'></td>"
                rcolor = "#6f6" if cur_r >= 0.6 else "#ff6" if cur_r >= 0.3 else "#f66"
                cols  += (f"<td><div style='font-size:9px;color:{rcolor}'>s{seed} r={cur_r:.3f}</div>"
                          f"<img src='{_b64(cur)}' style='width:110px'></td>")
            rows_html += f"<tr>{cols}</tr>"

    seed_hdrs = "".join(
        (f"<th>base s{s}</th><th>step{step} s{s}</th>" if base_images
         else f"<th>s{s} step{step}</th>")
        for s in val_seeds
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:monospace;background:#111;color:#eee;margin:16px}}
h1{{color:#7cf}} table{{border-collapse:collapse}} td,th{{border:1px solid #333;padding:3px}}</style>
</head><body>
<h1>LlamaGen DPO Val — step {step}</h1>
<p>mean_reward={mean_r:.4f}  se={se:.4f}  n={n}</p>
<p>{' '.join(f'{k}={v:.3f}' for k, v in comp_means.items())}</p>
<table><thead><tr><th>prompt</th>{seed_hdrs}</tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    (eval_dir / "grid.html").write_text(html, encoding="utf-8")

    print(f"  [eval step={step}] mean_r={mean_r:.4f}  se={se:.4f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in comp_means.items()))

    if wandb_run:
        import wandb as _wandb

        # ── scalar metrics ────────────────────────────────────────────────────
        log_dict = {
            "val/mean_reward": mean_r,
            "val/se":          se,
            "step":            step,
        }
        # per-component (overall + per-bucket)
        for k, v in comp_means.items():
            log_dict[f"val/comp/{k}"] = v

        # per-bucket mean reward
        for bucket, items in val_items_by_bucket.items():
            bkt_rewards = [s["reward"] for s in scores_rows if s["bucket"] == bucket]
            if bkt_rewards:
                log_dict[f"val/bucket/{bucket}/mean_reward"] = float(np.mean(bkt_rewards))
                for k in comp_means:
                    bkt_comp = [s["components"].get(k, 0) for s in scores_rows
                                if s["bucket"] == bucket]
                    if bkt_comp:
                        log_dict[f"val/bucket/{bucket}/{k}"] = float(np.mean(bkt_comp))

        # ── image progression panels ──────────────────────────────────────────
        # Log each fixed val prompt as a wandb.Image so you can scrub through
        # steps in the Media panel and see how images change over training.
        # Key format: "val_images/<bucket>/<item_id>/seed<s>"
        # wandb stacks same key across steps → progression view.
        for bucket, items in val_items_by_bucket.items():
            for item in items:
                for seed in val_seeds:
                    cur = new_images.get((item.id, seed))
                    if cur is None:
                        continue
                    cur_r = next((s["reward"] for s in scores_rows
                                  if s["id"] == item.id and s["seed"] == seed), 0)
                    base  = base_images.get((item.id, seed)) if base_images else None

                    caption = f"step={step}  r={cur_r:.3f}  {item.text[:60]}"

                    if base and step > 0:
                        # side-by-side: paste base | current into one image
                        from PIL import Image as _PIL, ImageDraw as _Draw
                        W, H = cur.width, cur.height
                        combined = _PIL.new("RGB", (W * 2 + 4, H + 20), (17, 17, 17))
                        combined.paste(base, (0, 20))
                        combined.paste(cur,  (W + 4, 20))
                        draw = _Draw.Draw(combined)
                        draw.text((2,  2), "base",        fill=(150, 150, 150))
                        draw.text((W + 6, 2), f"step {step}", fill=(100, 220, 100))
                        img_to_log = combined
                    else:
                        img_to_log = cur

                    key = f"val_images/{bucket}/{item.id}/seed{seed}"
                    log_dict[key] = _wandb.Image(img_to_log, caption=caption)

        wandb_run.log(log_dict)

    return mean_r, comp_means, new_images


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--pairs-jsonl",      nargs="+", required=True,
                        help="One pairs.jsonl per bucket (e.g. attr pairs + spatial pairs)")
    parser.add_argument("--val-jsonl",         nargs="+", required=True,
                        help="One val jsonl per bucket, same order as --pairs-jsonl")
    # model
    parser.add_argument("--repo-root",         required=True)
    parser.add_argument("--gpt-ckpt",          required=True)
    parser.add_argument("--vq-ckpt",           required=True)
    parser.add_argument("--t5-path",           required=True)
    parser.add_argument("--cfg-scale",         type=float, default=2.0)
    # lora
    parser.add_argument("--lora-r",            type=int,   default=8)
    parser.add_argument("--lora-alpha",        type=int,   default=16)
    parser.add_argument("--lora-target-modules", nargs="+", default=["wqkv", "wo"])
    # training
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--lr",                type=float, default=1e-5)
    parser.add_argument("--beta",              type=float, default=0.1)
    parser.add_argument("--epochs",            type=int,   default=3)
    parser.add_argument("--batch-size",        type=int,   default=2)
    parser.add_argument("--use-reference",     action="store_true",
                        help="Use LoRA-zeroed reference model in DPO loss (slower but more principled)")
    parser.add_argument("--max-grad-norm",     type=float, default=1.0)
    parser.add_argument("--seed",              type=int,   default=42)
    # eval
    parser.add_argument("--eval-every-steps",  type=int,   default=20)
    parser.add_argument("--val-prompts",       type=int,   default=20)
    parser.add_argument("--val-seeds",         type=int,   nargs="+", default=[0, 1])
    parser.add_argument("--reward-mode",       default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--qwen-model",        default="Qwen/Qwen3-VL-4B-Instruct")
    # logging
    parser.add_argument("--wandb-project",     default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpo_variant = "full_with_reference" if args.use_reference else "simplified_no_reference"
    print(f"[dpo] output      → {out_dir}")
    print(f"[dpo] dpo_variant = {dpo_variant}  beta={args.beta}  lr={args.lr}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        os.environ.setdefault("WANDB_API_KEY",
            "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")
        wandb_run = wandb.init(project=args.wandb_project, name=out_dir.name,
                               config=vars(args))

    # ── load pairs (all buckets combined) ─────────────────────────────────────
    all_pairs     = []
    pairs_dirs    = []
    bucket_names  = []
    for p in args.pairs_jsonl:
        rows = load_jsonl(p)
        # infer bucket from first row's id prefix
        bucket = rows[0]["id"].rsplit("_", 2)[0] if rows else Path(p).parent.name
        bucket_names.append(bucket)
        pairs_dirs.append(Path(p).parent)
        for r in rows:
            r["_bucket"]    = bucket
            r["_pairs_dir"] = Path(p).parent
        all_pairs.extend(rows)
    print(f"[dpo] {len(all_pairs)} pairs across {len(args.pairs_jsonl)} bucket(s): {bucket_names}")

    # ── load val items per bucket ─────────────────────────────────────────────
    val_items_by_bucket = {}
    for val_path in args.val_jsonl:
        items  = load_val_items(val_path, args.val_prompts)
        bucket = items[0].bucket if items else Path(val_path).stem
        val_items_by_bucket[bucket] = items
        print(f"[dpo] val  {bucket}: {len(items)} items")

    # ── load LlamaGen + LoRA ──────────────────────────────────────────────────
    sys.path.insert(0, args.repo_root)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=args.cfg_scale,
        use_lora=True,
        lora_config={
            "rank":           args.lora_r,
            "alpha":          args.lora_alpha,
            "dropout":        0.0,
            "target_modules": args.lora_target_modules,
            "start_layer":    0,
        },
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
    )
    _ = wrapper.gpt
    _ = wrapper.vq_model
    print("[dpo] LlamaGen + LoRA loaded.")

    # ── load Qwen ─────────────────────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(model_id=args.qwen_model)
    print("[dpo] Qwen loaded.\n")

    # ── step-0 baseline eval ──────────────────────────────────────────────────
    base_r, base_comps, base_images = run_val_eval(
        wrapper, reward_model, val_items_by_bucket, args.val_seeds,
        step=0, out_dir=out_dir, base_images=None,
        reward_mode=args.reward_mode, wandb_run=wandb_run,
    )
    best_val_r  = base_r
    global_step = 0
    metrics_log = []

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        random.shuffle(all_pairs)
        print(f"\n{'─'*60}\n  Epoch {epoch}/{args.epochs}  ({len(all_pairs)} pairs)\n{'─'*60}")
        epoch_losses = []

        for i in range(0, len(all_pairs), args.batch_size):
            batch = all_pairs[i: i + args.batch_size]
            if not batch:
                continue
            global_step += 1
            t0 = time.time()

            wrapper.gpt.train()

            # ── load tokens + conditioning ────────────────────────────────────
            chosen_tok_list   = []
            rejected_tok_list = []
            pair_w_list       = []
            pair_items        = []

            for p in batch:
                pd = p["_pairs_dir"]
                c_path  = pd / p["chosen_tokens_path"]   if p.get("chosen_tokens_path")  else None
                rj_path = pd / p["rejected_tokens_path"] if p.get("rejected_tokens_path") else None

                if not (c_path and rj_path and c_path.exists() and rj_path.exists()):
                    print(f"  [warn] missing tokens for {p['id']}, skipping")
                    continue

                chosen_tok_list.append(torch.load(c_path).long().cuda())
                rejected_tok_list.append(torch.load(rj_path).long().cuda())
                pair_w_list.append(p["pair_weight"])
                pair_items.append(PairItem(
                    id=p["id"], text=p["prompt"], bucket=p["_bucket"]
                ))

            if not pair_items:
                continue

            chosen_tokens   = torch.stack(chosen_tok_list)    # (B, seq_len)
            rejected_tokens = torch.stack(rejected_tok_list)  # (B, seq_len)
            pair_weights    = torch.tensor(pair_w_list, dtype=torch.float32).cuda()

            with torch.no_grad():
                c_indices, c_emb_masks = wrapper._get_conditioning(pair_items, t5_cache=None)

            # ── log probs ─────────────────────────────────────────────────────
            wrapper.gpt.float()
            chosen_lp   = wrapper._compute_log_probs(chosen_tokens,   c_indices, c_emb_masks)
            rejected_lp = wrapper._compute_log_probs(rejected_tokens, c_indices, c_emb_masks)

            if args.use_reference:
                with torch.no_grad():
                    ref_chosen_lp   = wrapper._compute_log_probs_ref(chosen_tokens,   c_indices, c_emb_masks)
                    ref_rejected_lp = wrapper._compute_log_probs_ref(rejected_tokens, c_indices, c_emb_masks)
                loss = dpo_loss_with_ref(chosen_lp, rejected_lp,
                                         ref_chosen_lp, ref_rejected_lp,
                                         pair_weights, beta=args.beta)
            else:
                loss = dpo_loss_no_ref(chosen_lp, rejected_lp, pair_weights, beta=args.beta)

            wrapper._optimizer.zero_grad()
            loss.backward()
            for p in wrapper.gpt.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in wrapper.gpt.parameters() if p.requires_grad],
                args.max_grad_norm,
            ).item()
            wrapper._optimizer.step()

            elapsed     = time.time() - t0
            loss_val    = loss.item()
            mean_margin = float(np.mean([p["margin"] for p in batch]))
            epoch_losses.append(loss_val)

            print(f"  ep{epoch} [{global_step:4d}]  "
                  f"loss={loss_val:.4f}  "
                  f"chosen_lp={chosen_lp.mean().item():.3f}  "
                  f"rej_lp={rejected_lp.mean().item():.3f}  "
                  f"margin={mean_margin:.3f}  "
                  f"gnorm={grad_norm:.3f}  ({elapsed:.1f}s)")

            m = {"step": global_step, "epoch": epoch, "loss": loss_val,
                 "chosen_lp": chosen_lp.mean().item(),
                 "rejected_lp": rejected_lp.mean().item(),
                 "mean_margin": mean_margin, "grad_norm": grad_norm}
            metrics_log.append(m)
            if wandb_run:
                wandb_run.log({f"train/{k}": v for k, v in m.items()})

            # ── periodic val eval ─────────────────────────────────────────────
            if global_step % args.eval_every_steps == 0:
                val_r, val_comps, _ = run_val_eval(
                    wrapper, reward_model, val_items_by_bucket, args.val_seeds,
                    step=global_step, out_dir=out_dir,
                    base_images=base_images,
                    reward_mode=args.reward_mode, wandb_run=wandb_run,
                )
                if val_r > best_val_r:
                    best_val_r = val_r
                    torch.save(wrapper.gpt.state_dict(), out_dir / "best_checkpoint.pt")
                    print(f"  [ckpt] best val_r={best_val_r:.4f}")
                wrapper.gpt.train()

        print(f"\n  ── Epoch {epoch} done  mean_loss={np.mean(epoch_losses):.4f}")

        # end-of-epoch eval
        val_r, _, _ = run_val_eval(
            wrapper, reward_model, val_items_by_bucket, args.val_seeds,
            step=global_step, out_dir=out_dir, base_images=base_images,
            reward_mode=args.reward_mode, wandb_run=wandb_run,
        )
        if val_r > best_val_r:
            best_val_r = val_r
            torch.save(wrapper.gpt.state_dict(), out_dir / "best_checkpoint.pt")
            print(f"  [ckpt] best val_r={best_val_r:.4f}")

    # ── final save ────────────────────────────────────────────────────────────
    torch.save(wrapper.gpt.state_dict(), out_dir / "final_checkpoint.pt")

    with open(out_dir / "train_metrics.jsonl", "w", encoding="utf-8") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    summary = {
        "dpo_variant":        dpo_variant,
        "buckets":            bucket_names,
        "total_pairs":        len(all_pairs),
        "beta":               args.beta,
        "lr":                 args.lr,
        "epochs":             args.epochs,
        "total_steps":        global_step,
        "baseline_reward":    round(base_r, 4),
        "best_val_reward":    round(best_val_r, 4),
        "delta_reward":       round(best_val_r - base_r, 4),
        "baseline_components": base_comps,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[dpo] done.  Δreward={best_val_r - base_r:+.4f}")
    print(f"[dpo] summary → {out_dir / 'summary.json'}")

    if wandb_run:
        wandb_run.log({"final/best_val_reward": best_val_r,
                       "final/delta_reward":    best_val_r - base_r})
        wandb_run.finish()


if __name__ == "__main__":
    main()
