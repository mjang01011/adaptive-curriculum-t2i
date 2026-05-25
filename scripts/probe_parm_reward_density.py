"""
Probe PARM reward density before starting GRPO training.

Generates G images per prompt from the base LlamaGen model, scores them with
PARM, and reports statistics that predict whether GRPO can learn from this
reward signal.

Viability thresholds (from PARM paper heuristics):
  pct_groups_all_zero  <= 60%   (if higher: most groups give zero gradient)
  mean_group_std       >= 0.02  (if lower:  within-group variation too small)
  best_of_G_delta      >= 0.05  (best-of-G must beat random by at least 5pp)

Usage:
  python scripts/probe_parm_reward_density.py \\
    --prompt-file Image-Generation-CoT/geneval/prompts/generation_prompts.txt \\
    --repo-root   LlamaGen \\
    --gpt-ckpt    /path/to/t2i_XL_stage1_256.pt \\
    --vq-ckpt     /path/to/vq_ds16_t2i.pt \\
    --t5-path     /path/to/t5-ckpt \\
    --parm-repo   /path/to/Image-Generation-CoT \\
    --parm-ckpt   /path/to/Image-Generation-CoT/ckpts/.../parm \\
    --num-prompts 50 \\
    --G 16 \\
    --output-dir  outputs/parm_probe
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()

    # Prompt file
    p.add_argument("--prompt-file",  required=True,
                   help="Text file with one prompt per line")
    p.add_argument("--num-prompts",  type=int, default=50,
                   help="Number of prompts to probe (randomly sampled)")
    p.add_argument("--seed",         type=int, default=42)

    # LlamaGen paths
    p.add_argument("--repo-root",    required=True)
    p.add_argument("--gpt-ckpt",     required=True)
    p.add_argument("--vq-ckpt",      required=True)
    p.add_argument("--t5-path",      required=True)

    # PARM paths
    p.add_argument("--parm-repo",    required=True)
    p.add_argument("--parm-ckpt",    required=True)
    p.add_argument("--parm-batch-size", type=int, default=4,
                   help="Images per PARM forward pass (all same prompt → true batching)")

    # Generation settings
    p.add_argument("--G",            type=int,   default=16,
                   help="Candidate images per prompt")
    p.add_argument("--cfg-scale",    type=float, default=2.0)
    p.add_argument("--temperature",  type=float, default=1.0)
    p.add_argument("--top-k",        type=int,   default=1000)
    p.add_argument("--top-p",        type=float, default=1.0)

    # Output
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--save-images",  action="store_true",
                   help="Save generated PNG grids to output-dir")

    return p.parse_args()


def load_prompts(path: str, n: int, seed: int):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(seed)
    if len(lines) > n:
        lines = random.sample(lines, n)
    return lines


def generate_candidates(wrapper, prompts, G, output_dir=None, save=False):
    """
    Generate G PIL images per prompt.
    Returns: list of (prompt, list_of_pil_images)
    """
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    results = []
    for idx, prompt in enumerate(prompts):
        print(f"  Generating prompt {idx+1}/{len(prompts)}: {prompt[:60]}...", flush=True)

        # Build a minimal BucketItem-compatible object
        class _Item:
            id = f"probe_{idx:05d}"
            text = prompt
            bucket = "probe"

        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([_Item()])

        B = 1
        qzshape = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]
        pil_images = []

        wrapper.gpt.eval()
        with torch.no_grad():
            for _ in range(G):
                tokens = generate(
                    wrapper.gpt, c_indices, wrapper.latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=wrapper.cfg_scale,
                    temperature=wrapper.temperature,
                    top_k=wrapper.top_k,
                    top_p=wrapper.top_p,
                    sample_logits=True,
                )  # (1, seq_len)
                decoded = wrapper.vq_model.decode_code(tokens, qzshape)
                img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
                pil_images.append(TF.to_pil_image(img_t.cpu()))

        wrapper._disable_kv_cache()

        if save and output_dir:
            from PIL import Image as PILImage
            grid_path = Path(output_dir) / "images" / f"probe_{idx:05d}_grid.jpg"
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            # 4 columns
            cols = 4
            rows = (G + cols - 1) // cols
            w, h = pil_images[0].size
            grid = PILImage.new("RGB", (cols * w, rows * h), (200, 200, 200))
            for gi, pil in enumerate(pil_images):
                r, c = divmod(gi, cols)
                grid.paste(pil, (c * w, r * h))
            grid.save(str(grid_path))

        results.append((prompt, pil_images))

    return results


def compute_stats(all_scores):
    """
    all_scores: list of lists — all_scores[i] is the G scores for prompt i.
    Returns a dict of statistics.
    """
    import statistics

    group_stds     = []
    group_means    = []
    best_of_G      = []
    random_scores  = []
    all_flat       = []
    n_all_zero     = 0
    n_low_std      = 0

    for scores in all_scores:
        if not scores:
            continue
        g_mean = sum(scores) / len(scores)
        g_max  = max(scores)
        g_std  = statistics.stdev(scores) if len(scores) > 1 else 0.0

        group_means.append(g_mean)
        group_stds.append(g_std)
        best_of_G.append(g_max)
        random_scores.extend(scores)
        all_flat.extend(scores)

        if all(s == 0.0 for s in scores):
            n_all_zero += 1
        if g_std < 0.02:
            n_low_std += 1

    N = len(all_scores)
    if N == 0:
        return {}

    random_mean  = sum(random_scores) / len(random_scores) if random_scores else 0.0
    best_mean    = sum(best_of_G) / len(best_of_G) if best_of_G else 0.0
    best_delta   = best_mean - random_mean
    mean_std     = sum(group_stds) / len(group_stds) if group_stds else 0.0

    nonzero      = [s for s in all_flat if s > 0.0]
    above_05     = [s for s in all_flat if s > 0.5]
    above_08     = [s for s in all_flat if s > 0.8]

    return {
        "num_prompts":             N,
        "G":                       len(all_scores[0]) if all_scores else 0,
        "random_mean":             round(random_mean, 4),
        "best_of_G_mean":          round(best_mean, 4),
        "best_of_G_delta":         round(best_delta, 4),
        "mean_group_std":          round(mean_std, 4),
        "pct_groups_all_zero":     round(100 * n_all_zero / N, 1),
        "pct_groups_std_below_002":round(100 * n_low_std / N, 1),
        "pct_samples_nonzero":     round(100 * len(nonzero) / max(len(all_flat), 1), 1),
        "pct_samples_above_0.5":   round(100 * len(above_05) / max(len(all_flat), 1), 1),
        "pct_samples_above_0.8":   round(100 * len(above_08) / max(len(all_flat), 1), 1),
    }


def print_viability(stats):
    print("\n" + "="*60)
    print("PARM Reward Density Probe Results")
    print("="*60)
    for k, v in stats.items():
        print(f"  {k:<35} {v}")

    print("\nViability check:")
    ok = True

    if stats.get("pct_groups_all_zero", 100) > 60:
        print("  [WARN] pct_groups_all_zero > 60% — most groups give zero gradient")
        ok = False
    else:
        print(f"  [OK]   pct_groups_all_zero = {stats['pct_groups_all_zero']}% (threshold 60%)")

    if stats.get("mean_group_std", 0) < 0.02:
        print("  [WARN] mean_group_std < 0.02 — within-group variation too small for GRPO")
        ok = False
    else:
        print(f"  [OK]   mean_group_std = {stats['mean_group_std']:.4f} (threshold 0.02)")

    if stats.get("best_of_G_delta", 0) < 0.05:
        print("  [WARN] best_of_G_delta < 0.05 — best-of-G barely beats random")
        ok = False
    else:
        print(f"  [OK]   best_of_G_delta = {stats['best_of_G_delta']:.4f} (threshold 0.05)")

    if ok:
        print("\n  => Reward signal looks viable. Proceed with GRPO training.")
    else:
        print("\n  => One or more thresholds failed. GRPO may not learn from this reward.")
        print("     Consider: larger G, different prompts, or a warmer starting checkpoint.")
    print("="*60 + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts ────────────────────────────────────────────────────────
    print(f"[probe] Loading prompts from {args.prompt_file}")
    prompts = load_prompts(args.prompt_file, args.num_prompts, args.seed)
    print(f"[probe] Using {len(prompts)} prompts (G={args.G} each = {len(prompts)*args.G} total images)")

    # ── Load LlamaGen ───────────────────────────────────────────────────────
    print("[probe] Loading LlamaGen wrapper ...")
    sys.path.insert(0, args.repo_root)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_lora=False,
    )
    # Force lazy load
    _ = wrapper.gpt
    _ = wrapper.vq_model
    _ = wrapper.t5
    print("[probe] LlamaGen loaded.")

    # ── Load PARM ───────────────────────────────────────────────────────────
    print("[probe] Loading PARM ...")
    from adaptive_curriculum.rewards.parm_reward import PARMRewardModel
    reward_model = PARMRewardModel(
        parm_repo=args.parm_repo,
        parm_ckpt=args.parm_ckpt,
        score_batch_size=args.parm_batch_size,
    )
    print("[probe] PARM loaded.")

    # ── Generate candidates ─────────────────────────────────────────────────
    print(f"\n[probe] Generating {args.G} images per prompt ...")
    candidates = generate_candidates(
        wrapper, prompts, args.G,
        output_dir=str(out_dir) if args.save_images else None,
        save=args.save_images,
    )

    # ── Score with PARM ─────────────────────────────────────────────────────
    print("[probe] Scoring with PARM ...")
    all_scores = []
    score_log  = []

    for idx, (prompt, pil_images) in enumerate(candidates):
        print(f"  Scoring prompt {idx+1}/{len(candidates)} ...", flush=True)
        results = reward_model.score_grouped(prompt, pil_images, chunk_size=args.parm_batch_size)
        scores  = [r["score"] for r in results]
        all_scores.append(scores)

        for gi, (r, sc) in enumerate(zip(results, scores)):
            score_log.append({
                "prompt_idx":         idx,
                "prompt":             prompt,
                "sample":             gi,
                "parm_norm_yes_prob": sc,
                "parm_yes_prob":      r["component_scores"].get("parm_yes_prob", 0.0),
                "parm_selected":      r["component_scores"].get("parm_selected", False),
            })

        print(f"    scores: min={min(scores):.3f}  max={max(scores):.3f}  "
              f"mean={sum(scores)/len(scores):.3f}", flush=True)

    # ── Stats + viability check ─────────────────────────────────────────────
    stats = compute_stats(all_scores)
    print_viability(stats)

    # ── Save outputs ────────────────────────────────────────────────────────
    stats_path = out_dir / "probe_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[probe] Stats saved to {stats_path}")

    scores_path = out_dir / "probe_scores.jsonl"
    with open(scores_path, "w") as f:
        for row in score_log:
            f.write(json.dumps(row) + "\n")
    print(f"[probe] Per-image scores saved to {scores_path}")


if __name__ == "__main__":
    main()
