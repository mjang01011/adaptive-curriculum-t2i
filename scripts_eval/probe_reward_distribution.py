"""
Probe reward distribution before GRPO training.

Generates G images per prompt for N prompts, scores with the target reward mode,
and reports whether the reward has enough signal to train on.

Pass criteria (default thresholds):
  best_of_G_delta       >= 0.08  (spatial: >= 0.05)
  mean_group_reward_std >= 0.05
  percent_groups_std_lt_0.03 <= 40%

Usage:
  python scripts_eval/probe_reward_distribution.py \
    --train-jsonl data/attribute_binding/attribute_binding_train_500.jsonl \
    --reward-mode grpo_attr_presence_gated_v2 \
    --num-prompts 50 --num-samples 8 \
    --repo-root   $PROJ/LlamaGen \
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path     $PRETRAINED/t5-ckpt \
    --out         outputs_reward_probe/attr_v2
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl",   required=True)
    p.add_argument("--reward-mode",   required=True)
    p.add_argument("--num-prompts",   type=int, default=50)
    p.add_argument("--num-samples",   type=int, default=8,  help="G: images per prompt")
    p.add_argument("--out",           required=True)
    p.add_argument("--seed",          type=int, default=42)
    # model
    p.add_argument("--repo-root",     required=True)
    p.add_argument("--gpt-ckpt",      required=True)
    p.add_argument("--vq-ckpt",       required=True)
    p.add_argument("--t5-path",       required=True)
    p.add_argument("--cfg-scale",     type=float, default=4.0)
    p.add_argument("--temperature",   type=float, default=1.0)
    p.add_argument("--top-k",         type=int,   default=1000)
    p.add_argument("--top-p",         type=float, default=1.0)
    p.add_argument("--precision",     default="bf16")
    # reward
    p.add_argument("--qwen-model",    default=None)
    # optional LoRA (for probing a trained checkpoint mid-run)
    p.add_argument("--lora-r",              type=int,   default=16)
    p.add_argument("--lora-alpha",          type=int,   default=32)
    p.add_argument("--lora-target-modules", nargs="+",  default=["wqkv", "wo"])
    p.add_argument("--lora-start-layer",    type=int,   default=0)
    p.add_argument("--init-checkpoint",     default=None,
                   help="LoRA checkpoint to load (omit to probe base model)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── Load data ────────────────────────────────────────────────────────────
    from adaptive_curriculum.data.schemas import BucketItem
    items = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))

    rng = random.Random(args.seed)
    sampled = rng.sample(items, min(args.num_prompts, len(items)))
    print(f"[probe] {len(sampled)} prompts  G={args.num_samples}  mode={args.reward_mode}")

    # ── Load model ───────────────────────────────────────────────────────────
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    from autoregressive.models.generate import generate

    use_lora = args.init_checkpoint is not None
    lora_config = {
        "rank":           args.lora_r,
        "alpha":          args.lora_alpha,
        "dropout":        0.0,
        "target_modules": args.lora_target_modules,
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
        use_lora=use_lora,
        lora_config=lora_config if use_lora else {},
    )
    _ = wrapper.gpt
    _ = wrapper.vq_model
    _ = wrapper.t5

    if args.init_checkpoint:
        print(f"[probe] Loading checkpoint: {args.init_checkpoint}")
        wrapper.load_checkpoint(args.init_checkpoint)

    # ── Load reward model ────────────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(
        model_id=args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
    )

    # ── Generate + score ─────────────────────────────────────────────────────
    import time
    import torchvision.transforms.functional as TF

    group_rewards   = []   # (N, G) — target reward mode
    group_hard      = []   # (N, G) — hard_target
    group_comp      = []   # (N, G, dict) — component scores

    # timing accumulators
    t_gen_all   = []   # per-image generation seconds
    t_score_all = []   # per-image scoring seconds (both modes)

    wrapper.gpt.eval()
    details_rows = []

    t_total_start = time.perf_counter()

    for pi, item in enumerate(sampled):
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])

        rewards_i = []
        hard_i    = []
        comp_i    = []

        qzshape = [1, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]

        for s in range(args.num_samples):
            t_gen_start = time.perf_counter()
            with torch.no_grad():
                tokens = generate(
                    wrapper.gpt, c_indices, wrapper.latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample_logits=True,
                )
                wrapper._disable_kv_cache()
                decoded = wrapper.vq_model.decode_code(tokens, qzshape)
            t_gen_all.append(time.perf_counter() - t_gen_start)

            img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
            pil_img = TF.to_pil_image(img_t.cpu())

            # Save image
            img_path = out_dir / f"p{pi:03d}_s{s:02d}_{item.id}.png"
            pil_img.save(str(img_path))

            # Score
            t_score_start = time.perf_counter()
            r = reward_model.score_image(pil_img, item, mode=args.reward_mode)
            h = reward_model.score_image(pil_img, item, mode="hard_target")
            t_score_all.append(time.perf_counter() - t_score_start)

            rewards_i.append(r["score"])
            hard_i.append(h["score"])
            comp_i.append(r.get("component_scores", {}))

            details_rows.append({
                "prompt_id": item.id,
                "prompt": item.prompt,
                "sample": s,
                "reward": r["score"],
                "hard": h["score"],
                "component_scores": r.get("component_scores", {}),
                "reward_debug": r.get("reward_debug", {}),
            })

        group_rewards.append(rewards_i)
        group_hard.append(hard_i)
        group_comp.append(comp_i)

    # ── Compute metrics ──────────────────────────────────────────────────────
    import statistics

    all_rewards  = [r for g in group_rewards for r in g]
    all_hard     = [h for g in group_hard     for h in g]

    random_mean  = statistics.mean(all_rewards)
    hard_random  = statistics.mean(all_hard)

    best_rewards = [max(g) for g in group_rewards]
    best_hard    = []
    best_comp    = {}

    for gi, (g_r, g_h, g_c) in enumerate(zip(group_rewards, group_hard, group_comp)):
        best_idx = g_r.index(max(g_r))
        best_hard.append(g_h[best_idx])
        for k, v in g_c[best_idx].items():
            best_comp.setdefault(k, []).append(v)

    best_of_G_mean  = statistics.mean(best_rewards)
    best_of_G_delta = best_of_G_mean - random_mean
    hard_best       = statistics.mean(best_hard)

    group_stds = [
        statistics.stdev(g) if len(g) > 1 else 0.0
        for g in group_rewards
    ]
    mean_group_std = statistics.mean(group_stds)
    pct_lt_003     = 100.0 * sum(1 for s in group_stds if s < 0.03) / max(len(group_stds), 1)

    # Per-component means (all samples vs best)
    comp_all = {}
    for g_c in group_comp:
        for sample_comp in g_c:
            for k, v in sample_comp.items():
                comp_all.setdefault(k, []).append(v)
    component_random = {k: statistics.mean(vs) for k, vs in comp_all.items()}
    component_best   = {k: statistics.mean(vs) for k, vs in best_comp.items()}

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Reward probe: {args.reward_mode}")
    print(f"  N={len(sampled)}  G={args.num_samples}")
    print(f"{'='*60}")
    print(f"  random_mean          = {random_mean:.4f}")
    print(f"  best_of_G_mean       = {best_of_G_mean:.4f}")
    print(f"  best_of_G_delta      = {best_of_G_delta:.4f}  (need >= 0.08, spatial >= 0.05)")
    print(f"  mean_group_std       = {mean_group_std:.4f}  (need >= 0.05)")
    print(f"  pct_groups_std<0.03  = {pct_lt_003:.1f}%    (need <= 40%)")
    print(f"  hard_target_random   = {hard_random:.4f}")
    print(f"  hard_target_best     = {hard_best:.4f}")
    print(f"\n  Component scores (random | best-of-G):")
    all_keys = sorted(set(list(component_random) + list(component_best)))
    for k in all_keys:
        r_val = component_random.get(k, float("nan"))
        b_val = component_best.get(k, float("nan"))
        print(f"    {k:<22} {r_val:.3f}  |  {b_val:.3f}")

    # Pass/fail
    print(f"\n{'='*60}")
    pass_delta = best_of_G_delta >= 0.05   # relaxed threshold for display
    pass_std   = mean_group_std >= 0.05
    pass_pct   = pct_lt_003 <= 40.0
    pass_hard  = hard_best > hard_random
    overall    = pass_delta and pass_std and pass_pct
    print(f"  best_of_G_delta >= 0.05   {'PASS' if pass_delta else 'FAIL'}")
    print(f"  mean_group_std  >= 0.05   {'PASS' if pass_std   else 'FAIL'}")
    print(f"  pct_lt_0.03     <= 40%    {'PASS' if pass_pct   else 'FAIL'}")
    print(f"  hard improves           {'PASS' if pass_hard  else 'FAIL'}")
    print(f"\n  OVERALL: {'✓ PROCEED TO GRPO' if overall else '✗ DO NOT TRAIN — reward too flat'}")
    print(f"{'='*60}\n")

    # Compute timing stats (used in report and summary)
    mean_gen_s    = statistics.mean(t_gen_all)   if t_gen_all   else float("nan")
    mean_score_s  = statistics.mean(t_score_all) if t_score_all else float("nan")
    t_total       = time.perf_counter() - t_total_start
    est_rollout_s = mean_gen_s * args.num_samples + mean_score_s * len(sampled) * args.num_samples
    print(f"  Timing  (N={len(sampled)}, G={args.num_samples})")
    print(f"    mean_gen_per_image   = {mean_gen_s:.2f}s")
    print(f"    mean_score_per_image = {mean_score_s:.2f}s  (both modes)")
    print(f"    est_rollout_per_step = {est_rollout_s:.1f}s  (G imgs + N*G scores)")
    print(f"    total_probe_time     = {t_total/60:.1f} min  ({t_total:.0f}s)")
    print()
    # ── Save results ─────────────────────────────────────────────────────────
    summary = {
        "reward_mode":             args.reward_mode,
        "num_prompts":             len(sampled),
        "num_samples":             args.num_samples,
        "random_mean":             random_mean,
        "best_of_G_mean":          best_of_G_mean,
        "best_of_G_delta":         best_of_G_delta,
        "mean_group_reward_std":   mean_group_std,
        "percent_groups_std_lt_0.03": pct_lt_003,
        "hard_target_random":      hard_random,
        "hard_target_best":        hard_best,
        "component_random":        component_random,
        "component_best":          component_best,
        "pass_overall":            overall,
        "timing": {
            "mean_gen_per_image_s":   statistics.mean(t_gen_all)   if t_gen_all   else None,
            "mean_score_per_image_s": statistics.mean(t_score_all) if t_score_all else None,
            "total_s":                t_total,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "details.jsonl", "w") as f:
        for row in details_rows:
            f.write(json.dumps(row) + "\n")

    print(f"  Results saved → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
