"""
GRPO training script for LlamaGen, adapted from AR-GRPO.

Key differences from original adaptive_curriculum GRPO:
  - PPO clipped loss (not plain REINFORCE)
  - Reverse KL: exp(lp_ref - lp) - (lp_ref - lp) - 1
  - Per-token log probs (not sequence-level)
  - GRPO loss reduction: mean(sum_tokens(loss_t) / len_t)
  - G=8 samples per prompt (vs G=4/6 previously)

Usage (cluster):
  python GRPO/train.py \
    --train-jsonl $PROJ/data/attribute_binding/attribute_binding_train_500.jsonl \
    --val-jsonl   $PROJ/data/attribute_binding/attribute_binding_val_20.jsonl \
    --output-dir  $PROJ/outputs/grpo_attr \
    --repo-root   $PROJ/LlamaGen \
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path     $PRETRAINED/t5-ckpt \
    --num-steps   500 \
    --batch-size  4 \
    --num-generations 8 \
    --beta 1.0 \
    --epsilon 0.2 \
    --lr 1e-5 \
    --cfg-scale-train 4.0 \
    --reward-mode pseudo_soft_grpo_target_heavy \
    --wandb-project llamagen-grpo
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for LlamaGen (AR-GRPO style)")

    # ── data ────────────────────────────────────────────────────────────────
    p.add_argument("--train-jsonl",     required=True)
    p.add_argument("--val-jsonl",       default=None)
    p.add_argument("--num-val-prompts", type=int, default=20)

    # ── model paths ─────────────────────────────────────────────────────────
    p.add_argument("--repo-root",  required=True)
    p.add_argument("--gpt-ckpt",   required=True)
    p.add_argument("--vq-ckpt",    required=True)
    p.add_argument("--t5-path",    required=True)

    # ── LoRA ────────────────────────────────────────────────────────────────
    p.add_argument("--lora-r",              type=int,   default=16)
    p.add_argument("--lora-alpha",          type=int,   default=32)
    p.add_argument("--lora-dropout",        type=float, default=0.0)
    p.add_argument("--lora-target-modules", nargs="+",  default=["wqkv", "wo"])
    p.add_argument("--lora-start-layer",    type=int,   default=0)

    # ── GRPO hypers ──────────────────────────────────────────────────────────
    p.add_argument("--num-generations",  type=int,   default=8,    help="G: samples per prompt")
    p.add_argument("--num-iterations",  type=int,   default=3,    help="PPO update iterations per rollout (>1 makes clipping active)")
    p.add_argument("--beta",            type=float, default=1.0,  help="Reverse-KL penalty coefficient")
    p.add_argument("--epsilon",         type=float, default=0.2,  help="PPO clip ratio")
    p.add_argument("--no-scale-rewards", action="store_true",     help="Subtract mean only, don't divide by std")
    p.add_argument("--batch-size",      type=int,   default=4,    help="Prompts per GRPO step")
    p.add_argument("--num-steps",       type=int,   default=500)
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--max-grad-norm",   type=float, default=1.0)

    # ── generation ───────────────────────────────────────────────────────────
    p.add_argument("--cfg-scale-train", type=float, default=4.0)
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--top-k",           type=int,   default=1000)
    p.add_argument("--top-p",           type=float, default=1.0)

    # ── reward ───────────────────────────────────────────────────────────────
    p.add_argument("--reward-mode",     default="pseudo_soft_grpo_target_heavy")
    p.add_argument("--qwen-model",      default=None, help="HF model ID or local path for Qwen3-VL reward (default: Qwen/Qwen3-VL-4B-Instruct)")

    # ── output / logging ─────────────────────────────────────────────────────
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--save-every",      type=int, default=50)
    p.add_argument("--val-every",        type=int, default=50)
    p.add_argument("--log-every",        type=int, default=10)
    p.add_argument("--num-panel-items",  type=int, default=8,
                   help="Fixed val items logged as W&B image panel every val step")
    p.add_argument("--wandb-project",   default=None)
    p.add_argument("--run-name",        default=None)
    p.add_argument("--seed",            type=int, default=42)

    # ── warm start ───────────────────────────────────────────────────────────
    p.add_argument("--init-checkpoint", default=None, help="LoRA checkpoint to resume from")

    # ── precision ────────────────────────────────────────────────────────────
    p.add_argument("--precision",       default="bf16", choices=["none", "fp16", "bf16"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str):
    from adaptive_curriculum.data.schemas import BucketItem
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))
    return items


# ---------------------------------------------------------------------------
# Validation evaluation + W&B image progression panel
# ---------------------------------------------------------------------------

def _decode_to_pil(decoded_tensor):
    """Convert a single decoded VQ image tensor (C,H,W) in [-1,1] to PIL."""
    import torchvision.transforms.functional as TF
    img_t = (decoded_tensor.float().clamp(-1, 1) + 1) / 2
    return TF.to_pil_image(img_t.cpu())


def _annotate_image(pil_img, caption_lines, font_size=14):
    """
    Paste a white caption strip below the image.
    Uses PIL's default bitmap font — no external font files needed.
    """
    from PIL import Image, ImageDraw, ImageFont
    line_h = font_size + 4
    strip_h = line_h * len(caption_lines) + 6
    w, h = pil_img.size
    canvas = Image.new("RGB", (w, h + strip_h), (255, 255, 255))
    canvas.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    for i, line in enumerate(caption_lines):
        draw.text((4, h + 3 + i * line_h), line, fill=(20, 20, 20), font=font)
    return canvas


def run_val(
    wrapper,
    reward_model,
    panel_items,          # fixed list of items, same every call
    reward_mode,          # gated mode for detailed component scores
    output_dir,
    step,
    use_wandb=False,
):
    """
    Generate one image per panel item (batched), score with both hard_target and
    reward_mode (batched). Saves images to disk. Logs annotated images + component
    score scalars to W&B. Returns (mean_hard_reward, per_item_results).
    """
    import contextlib
    from autoregressive.models.generate import generate
    from torchvision.utils import save_image

    out_dir = Path(output_dir) / "val_images" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(panel_items)
    wrapper.gpt.eval()

    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _sdpa_ctx = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                                         SDPBackend.EFFICIENT_ATTENTION,
                                         SDPBackend.MATH])
    except Exception:
        _sdpa_ctx = contextlib.nullcontext

    # ── batched generation (all N panel items in one forward pass) ────────
    with torch.no_grad():
        c_indices, c_emb_masks = wrapper._get_conditioning(panel_items)

    qzshape = [N, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
    with torch.no_grad(), _sdpa_ctx():
        tokens = generate(
            wrapper.gpt, c_indices, wrapper.latent_size ** 2,
            c_emb_masks,
            cfg_scale=wrapper.cfg_scale,
            temperature=wrapper.temperature,
            top_k=wrapper.top_k,
            top_p=wrapper.top_p,
            sample_logits=True,
        )
    wrapper._disable_kv_cache()

    with torch.no_grad():
        decoded = wrapper.vq_model.decode_code(tokens, qzshape)

    # ── decode to PIL + save (order matches panel_items exactly) ─────────
    pil_imgs = []
    for idx, item in enumerate(panel_items):
        pil_img = _decode_to_pil(decoded[idx])
        pil_imgs.append(pil_img)
        save_image(decoded[idx:idx + 1], str(out_dir / f"{item.id}.png"),
                   normalize=True, value_range=(-1, 1))

    # ── batched scoring (two passes: hard_target + reward_mode) ──────────
    pairs = list(zip(pil_imgs, panel_items))
    hard_results = reward_model.score_images_batch(pairs, mode="hard_target")
    grpo_results = reward_model.score_images_batch(pairs, mode=reward_mode)

    # ── assemble results (indices guaranteed to align with panel_items) ───
    results = []
    wandb_images = {}
    comp_accum = {}

    for idx, (item, pil_img, hard_result, grpo_result) in enumerate(
        zip(panel_items, pil_imgs, hard_results, grpo_results)
    ):
        hard_score  = hard_result["score"]
        grpo_score  = grpo_result["score"]
        comp_scores = grpo_result.get("component_scores", {})

        results.append({
            "item_id":     item.id,
            "prompt":      item.text,
            "hard_score":  hard_score,
            "grpo_score":  grpo_score,
            "comp_scores": comp_scores,
        })

        for k, v in comp_scores.items():
            comp_accum.setdefault(k, []).append(v)

        if use_wandb:
            import wandb
            prompt_short = item.text[:55] + ("..." if len(item.text) > 55 else "")
            comp_str = "  ".join(f"{k}={v:.2f}" for k, v in sorted(comp_scores.items()))
            caption_lines = [
                f"[step {step}] {prompt_short}",
                f"hard={hard_score:.2f}  grpo={grpo_score:.2f}",
                comp_str[:72] if comp_str else "",
            ]
            annotated = _annotate_image(pil_img, [l for l in caption_lines if l])
            wandb_images[f"val/panel_{idx:02d}"] = wandb.Image(
                annotated,
                caption=f"step={step}  hard={hard_score:.2f}  {prompt_short}",
            )

    hard_scores = [r["hard_score"] for r in results]
    mean_hard   = sum(hard_scores) / max(len(hard_scores), 1)

    if use_wandb:
        import wandb
        log_dict = {**wandb_images, "val/mean_hard_reward": mean_hard}
        for k, vs in comp_accum.items():
            log_dict[f"val/comp/{k}"] = sum(vs) / len(vs)
        wandb.log(log_dict, step=step)

    return mean_hard, results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── sys.path for LlamaGen ────────────────────────────────────────────────
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb
        run_name = args.run_name or f"grpo_{Path(args.output_dir).name}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[train] Loading train data from {args.train_jsonl}")
    train_items = load_jsonl(args.train_jsonl)
    print(f"[train] {len(train_items)} train items")

    val_items = []
    panel_items = []
    if args.val_jsonl:
        val_items = load_jsonl(args.val_jsonl)
        print(f"[train] {len(val_items)} val items")
        # Fixed panel: first N items (deterministic, same every run)
        panel_items = val_items[:min(args.num_panel_items, len(val_items))]
        print(f"[train] {len(panel_items)} panel items for W&B image progression")

    # ── Build LlamaGenWrapper ─────────────────────────────────────────────────
    print("[train] Loading LlamaGenWrapper ...")
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    lora_config = {
        "rank":           args.lora_r,
        "alpha":          args.lora_alpha,
        "dropout":        args.lora_dropout,
        "target_modules": args.lora_target_modules,
        "start_layer":    args.lora_start_layer,
    }
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=2.0,                  # used during val generation
        cfg_scale_train=args.cfg_scale_train,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        precision=args.precision,
        use_lora=True,
        lora_config=lora_config,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
    )
    # Trigger lazy load
    _ = wrapper.gpt
    _ = wrapper.vq_model
    _ = wrapper.t5

    if args.init_checkpoint:
        print(f"[train] Loading init checkpoint from {args.init_checkpoint}")
        wrapper.load_checkpoint(args.init_checkpoint)

    # ── Build GRPOTrainer ─────────────────────────────────────────────────────
    from GRPO.trainer import GRPOTrainer
    trainer = GRPOTrainer(
        wrapper=wrapper,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        scale_rewards=not args.no_scale_rewards,
        max_grad_norm=args.max_grad_norm,
    )

    # ── Build reward model ────────────────────────────────────────────────────
    print("[train] Loading reward model ...")
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model_id = args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
    reward_model = Qwen3VLRewardModel(model_id=reward_model_id)

    # ── Save config ───────────────────────────────────────────────────────────
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_log_path = out_dir / "train_log.jsonl"
    reward_details_path = out_dir / "reward_details.jsonl"

    best_val_reward = float("-inf")
    best_ckpt_path = str(out_dir / "best_checkpoint.pt")
    latest_ckpt_path = str(out_dir / "latest_checkpoint.pt")

    # Save step-0 checkpoint (untrained baseline for val comparison)
    trainer.save_checkpoint(best_ckpt_path)
    trainer.save_checkpoint(latest_ckpt_path)

    step = 0
    t0_total = time.time()

    print(f"[train] Starting GRPO training: {args.num_steps} steps, "
          f"B={args.batch_size}, G={args.num_generations}, "
          f"beta={args.beta}, epsilon={args.epsilon}, lr={args.lr}")

    while step < args.num_steps:
        # ── sample batch ──────────────────────────────────────────────────────
        batch = random.sample(train_items, min(args.batch_size, len(train_items)))

        # ── GRPO step ─────────────────────────────────────────────────────────
        metrics = trainer.train_step(batch, reward_model, reward_mode=args.reward_mode)
        step = metrics["step"]

        # ── log reward details ────────────────────────────────────────────────
        with open(reward_details_path, "a") as f:
            for d in metrics.pop("sample_details"):
                d["global_step"] = step
                f.write(json.dumps(d) + "\n")

        # ── console + file log ────────────────────────────────────────────────
        if step % args.log_every == 0 or step == 1:
            print(
                f"[step {step:5d}] loss={metrics['loss']:.4f}  "
                f"r={metrics['mean_reward']:.3f}±{metrics['reward_std']:.3f}  "
                f"kl={metrics['mean_kl']:.4f}  "
                f"grad={metrics['grad_norm']:.3f}  "
                f"clip={metrics['clip_frac']:.3f}  "
                f"zeroed={metrics['percent_groups_zeroed_low_std']:.1f}%  "
                f"adv={metrics['mean_abs_advantage']:.3f}"
            )
            print(
                f"         [timing] gen={metrics['t_gen_s']:.1f}s  "
                f"score={metrics['t_score_s']:.1f}s  "
                f"logp={metrics['t_logp_s']:.1f}s  "
                f"update*{args.num_iterations}={metrics['t_update_total_s']:.1f}s  "
                f"step={metrics['t_step_s']:.1f}s"
            )
            with open(train_log_path, "a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")

        if use_wandb:
            import wandb
            wandb.log({k: v for k, v in metrics.items()
                        if isinstance(v, (int, float))}, step=step)

        # ── checkpoint ───────────────────────────────────────────────────────
        if step % args.save_every == 0:
            trainer.save_checkpoint(latest_ckpt_path)
            print(f"[step {step}] saved latest checkpoint → {latest_ckpt_path}")

        # ── val eval + W&B image panel ────────────────────────────────────────
        if panel_items and (step % args.val_every == 0 or step == args.num_steps):
            print(f"[step {step}] running val eval ({len(panel_items)} panel items) ...")
            val_reward, val_results = run_val(
                wrapper, reward_model, panel_items,
                reward_mode=args.reward_mode,
                output_dir=str(out_dir),
                step=step,
                use_wandb=use_wandb,
            )
            print(f"[step {step}] val_hard_reward={val_reward:.4f}")

            with open(train_log_path, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "val_reward": val_reward,
                    "val_results": [{k: v for k, v in r.items() if k != "comp_scores"}
                                    for r in val_results],
                }) + "\n")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                trainer.save_checkpoint(best_ckpt_path)
                print(f"[step {step}] NEW BEST val_reward={val_reward:.4f} → {best_ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    trainer.save_checkpoint(latest_ckpt_path)
    elapsed_total = time.time() - t0_total

    summary = {
        "total_steps": step,
        "best_val_reward": best_val_reward,
        "elapsed_min": elapsed_total / 60,
        "best_checkpoint": best_ckpt_path,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[train] Done. {step} steps in {elapsed_total/60:.1f} min")
    print(f"  best_val_reward = {best_val_reward:.4f}")
    print(f"  best checkpoint → {best_ckpt_path}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
