"""
GRPO training of LlamaGen with PARM as the frozen reward model.

Reuses LlamaGenWrapper.train_grpo_step() from the existing Qwen-GRPO path.
PARM replaces Qwen-VL as the reward model; everything else (LoRA, KL, PPO
clipping in the wrapper, checkpointing) is unchanged.

Algorithm per step:
  1. Sample B prompts from the training prompt file.
  2. Generate G=16 candidate images per prompt (LlamaGen policy).
  3. Score each with PARM → parm_norm_yes_prob in [0,1].
  4. Group-normalize rewards within each prompt group → advantages.
  5. Groups with std(rewards) < min_group_std get zero advantages (no update).
  6. GRPO loss (REINFORCE with baseline) + reverse-KL penalty.
  7. Clip gradients, step optimizer (LoRA params only).
  8. Log to W&B, save visual grids, checkpoint LoRA weights.

Usage:
  python train_grpo_parm.py \\
    --prompt-file Image-Generation-CoT/geneval/prompts/generation_prompts.txt \\
    --output-dir  outputs/grpo_llamagen_parm_g16_lora_all_v1 \\
    --repo-root   LlamaGen \\
    --gpt-ckpt    /path/to/t2i_XL_stage1_256.pt \\
    --vq-ckpt     /path/to/vq_ds16_t2i.pt \\
    --t5-path     /path/to/t5-ckpt \\
    --parm-repo   /path/to/Image-Generation-CoT \\
    --parm-ckpt   /path/to/Image-Generation-CoT/ckpts/.../parm \\
    --G 16 --batch-size 1 --lr 3e-6 --beta 0.1 --max-steps 200 --wandb
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
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GRPO LlamaGen + PARM reward")

    # ── Prompts ──────────────────────────────────────────────────────────────
    p.add_argument("--prompt-file",   required=True,
                   help="Text file with one GenEval-style prompt per line")
    p.add_argument("--val-prompt-file", default=None,
                   help="Separate val prompt file (defaults to same as train)")
    p.add_argument("--num-val-prompts", type=int, default=16,
                   help="Fixed validation prompts (1 image generated per prompt at eval)")

    # ── LlamaGen model paths ─────────────────────────────────────────────────
    p.add_argument("--repo-root",     required=True)
    p.add_argument("--gpt-ckpt",      required=True)
    p.add_argument("--vq-ckpt",       required=True)
    p.add_argument("--t5-path",       required=True)

    # ── PARM paths ───────────────────────────────────────────────────────────
    p.add_argument("--parm-repo",     required=True,
                   help="Path to Image-Generation-CoT repo")
    p.add_argument("--parm-ckpt",     required=True,
                   help="Path to PARM checkpoint folder")
    p.add_argument("--parm-batch-size", type=int, default=4,
                   help="Images per PARM forward pass within one prompt group")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    p.add_argument("--use-lora",      action="store_true", default=True)
    p.add_argument("--lora-r",        type=int,   default=16)
    p.add_argument("--lora-alpha",    type=int,   default=32)
    p.add_argument("--lora-dropout",  type=float, default=0.05)
    p.add_argument("--lora-targets",  nargs="+",
                   default=["wqkv", "wo", "w1", "w2", "w3"],
                   help="LlamaGen module names to wrap with LoRA")
    p.add_argument("--lora-start-layer", type=int, default=0,
                   help="Only inject LoRA on layers >= this index (0 = all layers)")
    p.add_argument("--init-checkpoint", default=None,
                   help="LoRA checkpoint to warm-start from")

    # ── GRPO hypers ──────────────────────────────────────────────────────────
    p.add_argument("--G",             type=int,   default=16,
                   help="Candidate images per prompt per GRPO step")
    p.add_argument("--batch-size",    type=int,   default=1,
                   help="Prompts per GRPO step (B). G=16,B=1 ≈ 16 PARM calls per step")
    p.add_argument("--beta",          type=float, default=0.1,
                   help="Reverse-KL penalty weight (β). Use 0.05–0.1 for sparse rewards")
    p.add_argument("--advantage-eps", type=float, default=1e-4,
                   help="Epsilon added to group std when normalising advantages")
    p.add_argument("--min-group-std", type=float, default=0.02,
                   help="Groups with reward std < this are skipped (zero advantages)")
    p.add_argument("--max-steps",     type=int,   default=200)
    p.add_argument("--lr",            type=float, default=3e-6)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # ── Generation ───────────────────────────────────────────────────────────
    p.add_argument("--cfg-scale",     type=float, default=2.0,
                   help="CFG scale for both training generation and eval")
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--top-k",         type=int,   default=1000)
    p.add_argument("--top-p",         type=float, default=1.0)

    # ── Output / logging ─────────────────────────────────────────────────────
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--save-every",    type=int, default=50,
                   help="Save LoRA checkpoint every N steps")
    p.add_argument("--eval-every",    type=int, default=25,
                   help="Run validation grid every N steps")
    p.add_argument("--log-every",     type=int, default=5)
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb-project", default="llamagen-grpo-parm")
    p.add_argument("--run-name",      default=None)
    p.add_argument("--seed",          type=int, default=42)

    # ── Precision ─────────────────────────────────────────────────────────────
    p.add_argument("--precision",     default="bf16",
                   choices=["none", "fp16", "bf16"])

    # ── Safety ────────────────────────────────────────────────────────────────
    p.add_argument("--zero-reward-steps-max", type=int, default=5,
                   help="Stop if all rewards are exactly zero for this many consecutive steps")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def make_item(prompt: str, idx: int, bucket: str = "parm"):
    """Create a minimal BucketItem-compatible object from a plain prompt string."""
    from adaptive_curriculum.data.schemas import BucketItem
    return BucketItem(
        id=f"parm_{idx:06d}",
        bucket=bucket,
        prompt=prompt,
        target_questions=[],
        grpo_reward_questions=[],
    )


def sample_batch(prompts: list, batch_size: int) -> list:
    chosen = random.sample(prompts, min(batch_size, len(prompts)))
    return chosen


# ---------------------------------------------------------------------------
# PARM-aware advantage computation (adds group-std filtering on top of wrapper)
# ---------------------------------------------------------------------------

def compute_parm_group_stats(rewards_tensor: torch.Tensor, min_group_std: float):
    """
    rewards_tensor: (B, G) float
    Returns:
        advantages:         (B, G) — zero for groups below min_group_std
        pct_groups_skipped: float in [0,1]
        pct_groups_all_zero: float in [0,1]
        mean_group_std:     float
        best_of_G:          float  (mean over prompts of max reward in group)
        random_reward:      float  (mean of all rewards)
    """
    B, G = rewards_tensor.shape
    mean_r = rewards_tensor.mean(dim=1, keepdim=True)
    std_r  = rewards_tensor.std(dim=1, keepdim=True)

    all_zero  = (rewards_tensor == 0.0).all(dim=1)          # (B,)
    low_std   = std_r.squeeze(1) < min_group_std             # (B,)
    skip_mask = low_std                                       # (B,)

    advantages = torch.zeros_like(rewards_tensor)
    active = ~skip_mask
    if active.any():
        advantages[active] = (
            (rewards_tensor[active] - mean_r[active])
            / (std_r[active] + 1e-8)
        ).nan_to_num(nan=0.0)

    return {
        "advantages":           advantages,
        "pct_groups_skipped":   skip_mask.float().mean().item(),
        "pct_groups_all_zero":  all_zero.float().mean().item(),
        "mean_group_std":       std_r.squeeze(1).mean().item(),
        "best_of_G_reward":     rewards_tensor.max(dim=1).values.mean().item(),
        "random_reward":        rewards_tensor.mean().item(),
    }


# ---------------------------------------------------------------------------
# Validation: generate 1 image per fixed prompt, score, save visual grid
# ---------------------------------------------------------------------------

def run_val(wrapper, reward_model, val_items, out_dir, step, use_wandb=False):
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF
    from PIL import Image as PILImage, ImageDraw, ImageFont

    val_dir = Path(out_dir) / "val_grids" / f"step_{step:06d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    wrapper.gpt.eval()
    scores = []
    pil_imgs = []
    prompts  = []

    print(f"[eval] Generating {len(val_items)} val images ...", flush=True)
    for item in val_items:
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])
        qzshape = [1, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
        with torch.no_grad():
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
        img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
        pil = TF.to_pil_image(img_t.cpu())
        pil_imgs.append(pil)
        prompts.append(item.text)

    # Score with PARM
    for pil, item in zip(pil_imgs, val_items):
        r = reward_model.score_image(pil, item)
        scores.append(r["score"])
    reward_model.clear_cache()

    mean_score = sum(scores) / max(len(scores), 1)
    print(f"[eval] step={step}  val mean PARM score: {mean_score:.3f}", flush=True)

    # Build annotated grid
    def _annotate(img, text, score):
        w, h = img.size
        strip = 20
        canvas = PILImage.new("RGB", (w, h + strip), (255, 255, 255))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
        label = f"{score:.2f} | {text[:40]}"
        draw.text((2, h + 2), label, fill=(20, 20, 20), font=font)
        return canvas

    cols = 4
    rows = (len(val_items) + cols - 1) // cols
    ann_imgs = [_annotate(img, pr, sc) for img, pr, sc in zip(pil_imgs, prompts, scores)]
    w, h = ann_imgs[0].size
    grid = PILImage.new("RGB", (cols * w, rows * h), (180, 180, 180))
    for gi, ann in enumerate(ann_imgs):
        r, c = divmod(gi, cols)
        grid.paste(ann, (c * w, r * h))
    grid_path = val_dir / "grid.jpg"
    grid.save(str(grid_path))
    print(f"[eval] Saved grid → {grid_path}", flush=True)

    if use_wandb:
        import wandb
        wandb_imgs = {}
        for gi, (ann, prompt, sc) in enumerate(zip(ann_imgs, prompts, scores)):
            wandb_imgs[f"val/img_{gi:02d}"] = wandb.Image(
                ann, caption=f"step={step}  score={sc:.2f}  {prompt[:50]}"
            )
        wandb.log({
            **wandb_imgs,
            "val/mean_parm_score": mean_score,
        }, step=step)

    return mean_score


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── W&B ─────────────────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        run_name = args.run_name or f"parm_grpo_{Path(args.output_dir).name}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print(f"[train] W&B run: {run_name}", flush=True)

    # ── Load prompts ─────────────────────────────────────────────────────────
    print(f"[train] Loading prompts from {args.prompt_file}", flush=True)
    all_prompts = load_prompts(args.prompt_file)
    print(f"[train] {len(all_prompts)} prompts loaded", flush=True)
    print(f"[train] First 3 prompts:")
    for pr in all_prompts[:3]:
        print(f"  - {pr}")

    val_prompt_file = args.val_prompt_file or args.prompt_file
    val_prompts_raw = load_prompts(val_prompt_file)
    random.seed(args.seed + 1)
    val_prompts_raw = random.sample(
        val_prompts_raw, min(args.num_val_prompts, len(val_prompts_raw))
    )
    val_items = [make_item(pr, i, "parm_val") for i, pr in enumerate(val_prompts_raw)]
    print(f"[train] {len(val_items)} fixed val prompts", flush=True)

    # ── Load LlamaGen wrapper ────────────────────────────────────────────────
    print("[train] Loading LlamaGen ...", flush=True)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    from adaptive_curriculum.model.lora_utils import count_trainable_parameters

    lora_config = {
        "rank":           args.lora_r,
        "alpha":          args.lora_alpha,
        "dropout":        args.lora_dropout,
        "target_modules": args.lora_targets,
        "start_layer":    args.lora_start_layer,
    }

    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=args.cfg_scale,
        cfg_scale_train=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        precision=args.precision,
        use_lora=args.use_lora,
        lora_config=lora_config,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
    )

    # Force lazy load of all three models
    print("[train] Loading GPT ...", flush=True)
    _ = wrapper.gpt
    print("[train] Loading VQ  ...", flush=True)
    _ = wrapper.vq_model
    print("[train] Loading T5  ...", flush=True)
    _ = wrapper.t5
    print("[train] All models loaded.", flush=True)

    n_trainable = count_trainable_parameters(wrapper.gpt)
    n_total     = sum(p.numel() for p in wrapper.gpt.parameters())
    print(f"[train] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)", flush=True)

    # Print LoRA-injected modules
    lora_modules = [n for n, m in wrapper.gpt.named_modules()
                    if hasattr(m, "lora_A")]
    print(f"[train] LoRA injected into {len(lora_modules)} modules:")
    for nm in lora_modules[:8]:
        print(f"  {nm}")
    if len(lora_modules) > 8:
        print(f"  ... ({len(lora_modules)-8} more)")

    # Warm-start LoRA weights
    if args.init_checkpoint:
        from adaptive_curriculum.model.lora_utils import load_lora_weights
        print(f"[train] Loading LoRA checkpoint: {args.init_checkpoint}", flush=True)
        load_lora_weights(wrapper.gpt, args.init_checkpoint)

    # ── Load PARM ────────────────────────────────────────────────────────────
    print("[train] Loading PARM reward model ...", flush=True)
    from adaptive_curriculum.rewards.parm_reward import PARMRewardModel
    reward_model = PARMRewardModel(
        parm_repo=args.parm_repo,
        parm_ckpt=args.parm_ckpt,
        score_batch_size=args.parm_batch_size,
    )
    print("[train] PARM loaded.", flush=True)

    # ── Save first batch of images as sanity check ───────────────────────────
    _save_first_images(wrapper, all_prompts[:4], out_dir)

    # ── Training loop ────────────────────────────────────────────────────────
    score_log_path = out_dir / "parm_scores_train.jsonl"
    score_log_fh   = open(score_log_path, "w")

    consecutive_zero_steps = 0
    best_val_score = -1.0

    print(f"\n[train] Starting GRPO training for {args.max_steps} steps "
          f"(B={args.batch_size}, G={args.G}, β={args.beta})\n", flush=True)

    for step in range(1, args.max_steps + 1):
        t0 = time.time()

        # Sample batch
        batch_prompts = sample_batch(all_prompts, args.batch_size)
        batch_items   = [make_item(pr, random.randint(0, 999999)) for pr in batch_prompts]

        # GRPO step via wrapper (handles generation + scoring + LoRA update)
        metrics = wrapper.train_grpo_step(
            batch=batch_items,
            reward_model=reward_model,
            num_samples=args.G,
            beta=args.beta,
            reward_mode="parm_norm_yes_prob",  # passed to score_image, ignored by PARM
            advantage_eps=args.advantage_eps,
        )
        reward_model.clear_cache()

        # Pull reward tensor from wrapper internals for PARM-specific stats
        # train_grpo_step stores per-sample details in wrapper._last_sample_details
        sample_details = getattr(wrapper, "_last_sample_details", [])
        soft_rewards   = [d["soft_reward"] for d in sample_details]

        rewards_tensor = torch.tensor(soft_rewards).reshape(args.batch_size, args.G)
        parm_stats     = compute_parm_group_stats(rewards_tensor, args.min_group_std)

        # Safety: track consecutive all-zero steps
        if parm_stats["random_reward"] == 0.0:
            consecutive_zero_steps += 1
        else:
            consecutive_zero_steps = 0

        if consecutive_zero_steps >= args.zero_reward_steps_max:
            print(f"\n[train] WARNING: All rewards have been exactly 0.0 for "
                  f"{consecutive_zero_steps} consecutive steps. "
                  f"PARM may not be scoring these images correctly. Stopping.", flush=True)
            break

        # Write per-step score JSONL
        for d in sample_details:
            score_log_fh.write(json.dumps({
                "step":              step,
                "prompt_id":         d.get("prompt_id", ""),
                "prompt":            d.get("prompt", ""),
                "sample":            d.get("sample", 0),
                "parm_norm_yes_prob":d.get("soft_reward", 0.0),
                "component_scores":  d.get("component_scores", {}),
            }) + "\n")
        score_log_fh.flush()

        # Logging
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            log = {
                "step":                    step,
                "train/loss":              metrics.get("loss", 0.0),
                "train/pg_loss":           metrics.get("pg_loss", 0.0),
                "train/kl":                metrics.get("kl", 0.0),
                "train/grad_norm":         metrics.get("grad_norm", 0.0),
                "train/lora_grad_norm":    metrics.get("lora_grad_norm", 0.0),
                "train/reward_mean":       parm_stats["random_reward"],
                "train/reward_max":        parm_stats["best_of_G_reward"],
                "train/best_of_G_reward":  parm_stats["best_of_G_reward"],
                "train/random_reward":     parm_stats["random_reward"],
                "train/mean_group_std":    parm_stats["mean_group_std"],
                "train/pct_groups_all_zero":   100 * parm_stats["pct_groups_all_zero"],
                "train/pct_groups_skipped":    100 * parm_stats["pct_groups_skipped"],
                "train/seconds_per_step":  elapsed,
            }
            log_str = (
                f"step={step:4d}  loss={log['train/loss']:.4f}  "
                f"kl={log['train/kl']:.4f}  "
                f"r_mean={log['train/reward_mean']:.3f}  "
                f"r_max={log['train/reward_max']:.3f}  "
                f"grp_std={log['train/mean_group_std']:.3f}  "
                f"skip={log['train/pct_groups_skipped']:.0f}%  "
                f"zero={log['train/pct_groups_all_zero']:.0f}%  "
                f"({elapsed:.1f}s)"
            )
            print(log_str, flush=True)
            if use_wandb:
                import wandb
                wandb.log(log, step=step)

        # Validation
        if step % args.eval_every == 0 or step == args.max_steps:
            val_score = run_val(
                wrapper, reward_model, val_items, str(out_dir), step, use_wandb
            )
            if val_score > best_val_score:
                best_val_score = val_score
                _save_checkpoint(wrapper, out_dir, "best", step)
                print(f"[train] New best val score: {best_val_score:.3f} → saved best ckpt",
                      flush=True)

        # Regular checkpoint
        if step % args.save_every == 0:
            _save_checkpoint(wrapper, out_dir, f"step_{step:06d}", step)

    score_log_fh.close()
    _save_checkpoint(wrapper, out_dir, "final", args.max_steps)
    print(f"\n[train] Done. Final checkpoint saved. Best val score: {best_val_score:.3f}",
          flush=True)
    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(wrapper, out_dir, tag, step):
    from adaptive_curriculum.model.lora_utils import save_lora_weights
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"lora_{tag}.pt"
    save_lora_weights(wrapper.gpt, str(path))
    # Also save a small metadata file alongside
    meta = {"step": step, "tag": tag}
    (ckpt_dir / f"lora_{tag}_meta.json").write_text(json.dumps(meta))
    print(f"[ckpt] Saved {path}", flush=True)


def _save_first_images(wrapper, prompts, out_dir):
    """Generate and save the very first batch of images (sanity check, no scoring)."""
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    sanity_dir = Path(out_dir) / "sanity_first_batch"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    wrapper.gpt.eval()
    print("[train] Saving sanity images from base model ...", flush=True)
    for i, prompt in enumerate(prompts):
        item = make_item(prompt, i, "sanity")
        try:
            with torch.no_grad():
                c_indices, c_emb_masks = wrapper._get_conditioning([item])
            qzshape = [1, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
            with torch.no_grad():
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
            img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
            pil = TF.to_pil_image(img_t.cpu())
            pil.save(str(sanity_dir / f"sanity_{i:02d}.png"))
        except Exception as exc:
            print(f"[train] Sanity image {i} failed: {exc}", flush=True)

    print(f"[train] Sanity images saved to {sanity_dir}", flush=True)


if __name__ == "__main__":
    main()
