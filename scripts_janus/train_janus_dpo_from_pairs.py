"""
DPO training for Janus-Pro-1B from pre-built G=2 pairs.
Uses weighted contrastive preference loss (simplified DPO, no separate reference model).
Logs val progression with saved image grids every N steps.

Usage:
  python scripts_janus/train_janus_dpo_from_pairs.py \
    --pairs-jsonl  /path/to/pairs.jsonl \
    --val-jsonl    /viscam/u/jj277/adaptive-curriculum-t2i/data/attribute_binding/attribute_binding_val_20.jsonl \
    --model-path   deepseek-ai/Janus-Pro-1B \
    --output-dir   /viscam/u/jj277/janus_project/outputs_janus_dpo/run1 \
    --lora-r 16 --lora-alpha 32 \
    --lr 1e-5 --beta 0.1 --epochs 2 --batch-size 1 \
    --eval-every-steps 25 --val-prompts 20 --val-seeds 0 1 \
    --cfg-weight 5.0 --temperature 1.0
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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from adaptive_curriculum.data.schemas import BucketItem


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_val_items(jsonl_path, n):
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

def dpo_loss(chosen_lp, rejected_lp, pair_weights, beta):
    """
    Simplified DPO (no reference model).
    dpo_variant = simplified_no_reference
    loss = -logsigmoid(beta * (chosen_lp - rejected_lp))  weighted by pair_weight
    """
    lograt = chosen_lp - rejected_lp
    raw    = -F.logsigmoid(beta * lograt)
    return (pair_weights * raw).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Validation eval — generates images + scores + saves grid HTML
# ══════════════════════════════════════════════════════════════════════════════

def _b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_val_eval(wrapper, reward_model, val_items, val_seeds, step, out_dir,
                 base_images=None, reward_mode="pseudo_soft_grpo_target_heavy",
                 wandb_run=None):
    """
    Generate val_items × val_seeds images, score, save grid HTML.
    base_images: dict {(item_id, seed): pil_img} from step=0, for before/after comparison.
    Returns (mean_reward, comp_means, new_images_dict).
    """
    eval_dir = Path(out_dir) / f"eval_step_{step:04d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    img_dir = eval_dir / "images"
    img_dir.mkdir(exist_ok=True)

    wrapper.model.eval()
    all_rewards = []
    all_comps   = {}
    new_images  = {}
    scores_rows = []

    with torch.no_grad():
        for item in val_items:
            for seed in val_seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                out = wrapper.generate_images([item.text], seeds=None)
                pil = out["images"][0]

                fname = f"{item.id}_seed{seed}.png"
                pil.save(img_dir / fname)
                new_images[(item.id, seed)] = pil

                result = reward_model.score_image(pil, item, mode=reward_mode)
                r      = float(result["score"])
                comps  = result.get("component_scores", {})
                all_rewards.append(r)
                for k, v in comps.items():
                    all_comps.setdefault(k, []).append(float(v))

                scores_rows.append({
                    "id": item.id, "seed": seed,
                    "reward": round(r, 4),
                    "components": {k: round(float(v), 4) for k, v in comps.items()},
                })

    mean_r     = float(np.mean(all_rewards))
    comp_means = {k: round(float(np.mean(v)), 4) for k, v in all_comps.items()}
    n = len(all_rewards)
    se = float(np.std(all_rewards) / math.sqrt(n)) if n > 1 else 0.0

    summary = {"step": step, "n_images": n, "mean_reward": round(mean_r, 4),
               "se": round(se, 4), "components": comp_means}
    with open(eval_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_dir / "scores.jsonl", "w", encoding="utf-8") as f:
        for row in scores_rows:
            f.write(json.dumps(row) + "\n")

    # ── grid HTML ─────────────────────────────────────────────────────────────
    rows_html = ""
    for item in val_items:
        cols = f"<td style='color:#aef;font-size:10px;vertical-align:top;padding:4px'>{item.text}</td>"
        for seed in val_seeds:
            cur = new_images.get((item.id, seed))
            cur_r = next((s["reward"] for s in scores_rows
                         if s["id"] == item.id and s["seed"] == seed), 0)
            base_img = base_images.get((item.id, seed)) if base_images else None

            if base_img:
                base_r = 0.0  # unknown for step-0 base
                cols += (f"<td><div style='font-size:9px;color:#888'>base</div>"
                         f"<img src='{_b64(base_img)}' style='width:120px'></td>")
            cols += (f"<td><div style='font-size:9px;color:#6f6'>step {step}  r={cur_r:.3f}</div>"
                     f"<img src='{_b64(cur)}' style='width:120px'></td>")
        rows_html += f"<tr>{cols}</tr>"

    seed_headers = "".join(
        (f"<th>base s{s}</th><th>step{step} s{s}</th>" if base_images
         else f"<th>step{step} s{s}</th>")
        for s in val_seeds
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:monospace;background:#111;color:#eee;margin:16px}}
h1{{color:#7cf}} table{{border-collapse:collapse}} td,th{{border:1px solid #333;padding:4px}}</style>
</head><body>
<h1>Val Eval — step {step}</h1>
<p>mean_reward={mean_r:.4f}  se={se:.4f}  n={n}</p>
<p>{' '.join(f'{k}={v:.3f}' for k, v in comp_means.items())}</p>
<table><thead><tr><th>prompt</th>{seed_headers}</tr></thead>
<tbody>{rows_html}</tbody></table>
</body></html>"""
    (eval_dir / "grid.html").write_text(html, encoding="utf-8")

    print(f"  [eval step={step}] mean_reward={mean_r:.4f}  se={se:.4f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in comp_means.items()))

    if wandb_run:
        wandb_run.log({"val/mean_reward": mean_r, "val/se": se,
                       **{f"val/{k}": v for k, v in comp_means.items()},
                       "step": step})

    return mean_r, comp_means, new_images


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-jsonl",      required=True)
    parser.add_argument("--val-jsonl",         required=True)
    parser.add_argument("--model-path",        default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--lora-r",            type=int,   default=16)
    parser.add_argument("--lora-alpha",        type=int,   default=32)
    parser.add_argument("--target-modules",    nargs="+",  default=["q_proj","k_proj","v_proj","o_proj"])
    parser.add_argument("--lr",                type=float, default=1e-5)
    parser.add_argument("--beta",              type=float, default=0.1)
    parser.add_argument("--epochs",            type=int,   default=2)
    parser.add_argument("--batch-size",        type=int,   default=1)
    parser.add_argument("--eval-every-steps",  type=int,   default=25)
    parser.add_argument("--qwen-model",        default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--val-prompts",       type=int,   default=20)
    parser.add_argument("--val-seeds",         type=int,   nargs="+", default=[0, 1])
    parser.add_argument("--cfg-weight",        type=float, default=5.0)
    parser.add_argument("--temperature",       type=float, default=1.0)
    parser.add_argument("--reward-mode",       default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--max-grad-norm",     type=float, default=1.0)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--wandb-project",     default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dpo] output → {out_dir}")
    print(f"[dpo] dpo_variant=simplified_no_reference  beta={args.beta}  lr={args.lr}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        os.environ.setdefault("WANDB_API_KEY",
            "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")
        wandb_run = wandb.init(project=args.wandb_project,
                               name=out_dir.name,
                               config=vars(args))

    # ── load pairs ────────────────────────────────────────────────────────────
    pairs_dir = Path(args.pairs_jsonl).parent
    pairs     = load_jsonl(args.pairs_jsonl)
    print(f"[dpo] {len(pairs)} pairs loaded")

    # ── load val items ────────────────────────────────────────────────────────
    val_items = load_val_items(args.val_jsonl, args.val_prompts)
    print(f"[dpo] {len(val_items)} val items  seeds={args.val_seeds}")

    # ── load model ────────────────────────────────────────────────────────────
    from scripts_janus.janus_wrapper import JanusProWrapper
    wrapper = JanusProWrapper(
        model_path=args.model_path,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        lora_config={
            "r":             args.lora_r,
            "alpha":         args.lora_alpha,
            "dropout":       0.0,
            "target_modules": args.target_modules,
            "target_scope":  "language_model",
        },
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
    )
    _ = wrapper.model
    print("[dpo] Janus + LoRA loaded.")

    # ── reward model ──────────────────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(model_id=args.qwen_model)
    print("[dpo] Qwen loaded.\n")

    # ── step 0 baseline eval ──────────────────────────────────────────────────
    base_r, base_comps, base_images = run_val_eval(
        wrapper, reward_model, val_items, args.val_seeds,
        step=0, out_dir=out_dir, base_images=None,
        reward_mode=args.reward_mode, wandb_run=wandb_run,
    )
    best_val_r = base_r
    global_step = 0

    # ── training loop ─────────────────────────────────────────────────────────
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        print(f"\n{'─'*60}\n  Epoch {epoch}/{args.epochs}\n{'─'*60}")
        epoch_losses = []

        for i in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[i: i + args.batch_size]
            if not batch_pairs:
                continue

            global_step += 1
            t0 = time.time()

            wrapper.model.train()

            # load tokens
            chosen_tokens_list   = []
            rejected_tokens_list = []
            pair_weights_list    = []
            valid_prompts        = []

            for p in batch_pairs:
                c_tok_path  = pairs_dir / p["chosen_tokens_path"]   if p.get("chosen_tokens_path")  else None
                rj_tok_path = pairs_dir / p["rejected_tokens_path"] if p.get("rejected_tokens_path") else None

                if c_tok_path and rj_tok_path and c_tok_path.exists() and rj_tok_path.exists():
                    chosen_tokens_list.append(torch.load(c_tok_path).cuda())
                    rejected_tokens_list.append(torch.load(rj_tok_path).cuda())
                    pair_weights_list.append(p["pair_weight"])
                    valid_prompts.append(p["prompt"])
                else:
                    print(f"  [warn] missing tokens for {p['id']}, skipping")

            if not valid_prompts:
                continue

            chosen_tokens   = torch.stack(chosen_tokens_list).cuda()    # (B, 576)
            rejected_tokens = torch.stack(rejected_tokens_list).cuda()  # (B, 576)
            pair_weights    = torch.tensor(pair_weights_list, dtype=torch.float32).cuda()

            # log probs
            chosen_lp   = wrapper._recompute_logprobs(valid_prompts, chosen_tokens)
            rejected_lp = wrapper._recompute_logprobs(valid_prompts, rejected_tokens)

            loss = dpo_loss(chosen_lp, rejected_lp, pair_weights, beta=args.beta)

            wrapper._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in wrapper.model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            wrapper._optimizer.step()

            loss_val = loss.item()
            elapsed  = time.time() - t0
            epoch_losses.append(loss_val)

            # margin stats for this batch
            margins  = [p["pair_weight"] for p in batch_pairs if "margin" in p]
            mean_margin = float(np.mean([p["margin"] for p in batch_pairs])) if batch_pairs else 0.0

            print(f"  ep{epoch} step{i//args.batch_size+1:3d}  "
                  f"[global {global_step:4d}]  "
                  f"loss={loss_val:.4f}  "
                  f"chosen_lp={chosen_lp.mean().item():.3f}  "
                  f"rej_lp={rejected_lp.mean().item():.3f}  "
                  f"margin={mean_margin:.3f}  "
                  f"({elapsed:.1f}s)")

            m = {"step": global_step, "epoch": epoch, "loss": loss_val,
                 "chosen_lp": chosen_lp.mean().item(),
                 "rejected_lp": rejected_lp.mean().item(),
                 "mean_margin": mean_margin}
            metrics_log.append(m)
            if wandb_run:
                wandb_run.log({f"train/{k}": v for k, v in m.items()})

            # periodic val eval
            if global_step % args.eval_every_steps == 0:
                val_r, val_comps, _ = run_val_eval(
                    wrapper, reward_model, val_items, args.val_seeds,
                    step=global_step, out_dir=out_dir,
                    base_images=base_images,
                    reward_mode=args.reward_mode, wandb_run=wandb_run,
                )
                if val_r > best_val_r:
                    best_val_r = val_r
                    ckpt = out_dir / "best_checkpoint.pt"
                    torch.save(wrapper.model.state_dict(), ckpt)
                    print(f"  [ckpt] best val_r={best_val_r:.4f} → {ckpt}")
                wrapper.model.train()

        epoch_mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"\n  ── Epoch {epoch} done  mean_loss={epoch_mean_loss:.4f}")

        # end-of-epoch eval
        val_r, val_comps, _ = run_val_eval(
            wrapper, reward_model, val_items, args.val_seeds,
            step=global_step, out_dir=out_dir,
            base_images=base_images,
            reward_mode=args.reward_mode, wandb_run=wandb_run,
        )
        if val_r > best_val_r:
            best_val_r = val_r
            torch.save(wrapper.model.state_dict(), out_dir / "best_checkpoint.pt")
            print(f"  [ckpt] best val_r={best_val_r:.4f}")

    # ── final eval + save ────────────────────────────────────────────────────
    torch.save(wrapper.model.state_dict(), out_dir / "final_checkpoint.pt")

    with open(out_dir / "train_metrics.jsonl", "w", encoding="utf-8") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")

    summary = {
        "dpo_variant":       "simplified_no_reference",
        "beta":              args.beta,
        "lr":                args.lr,
        "epochs":            args.epochs,
        "total_steps":       global_step,
        "baseline_reward":   round(base_r, 4),
        "best_val_reward":   round(best_val_r, 4),
        "delta_reward":      round(best_val_r - base_r, 4),
        "baseline_components": base_comps,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[dpo] done.  Δreward={best_val_r - base_r:+.4f}")
    print(f"[dpo] summary → {out_dir / 'summary.json'}")

    if wandb_run:
        wandb_run.log({"final/best_val_reward": best_val_r,
                       "final/delta_reward": best_val_r - base_r})
        wandb_run.finish()


if __name__ == "__main__":
    main()
