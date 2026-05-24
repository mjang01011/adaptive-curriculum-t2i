"""
Evaluate ImplicitCompositionAdapter in four modes.

Modes
-----
  identity           — verify adapter with gamma=0 is identical to base LlamaGen
  counterfactual     — raw-prompt counterfactuals (attribute swap / spatial flip)
  val_metrics        — score on original val set, report per-component rewards
  ablation           — compare base / adapter-only / adapter+LoRA side by side

Usage
-----
  # Identity check (before/without training)
  python SFT/eval_implicit_adapter.py --mode identity \\
    --output-dir $PROJ/outputs/eval_identity \\
    <model args>

  # Counterfactual grid (after training)
  python SFT/eval_implicit_adapter.py --mode counterfactual \\
    --checkpoint $PROJ/outputs/implicit_adapter_v1/best.pt \\
    --output-dir $PROJ/outputs/eval_cf \\
    <model args>

  # Validation metrics
  python SFT/eval_implicit_adapter.py --mode val_metrics \\
    --val-jsonl $PROJ/data/attribute_binding/attribute_binding_val_20.jsonl \\
    --checkpoint $PROJ/outputs/implicit_adapter_v1/best.pt \\
    --output-dir $PROJ/outputs/eval_val \\
    <model args>
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       required=True,
                   choices=["identity", "counterfactual", "val_metrics", "ablation"])
    p.add_argument("--checkpoint", default=None,  help="Adapter checkpoint (.pt)")
    p.add_argument("--output-dir", required=True)
    # model
    p.add_argument("--repo-root",  required=True)
    p.add_argument("--gpt-ckpt",   required=True)
    p.add_argument("--vq-ckpt",    required=True)
    p.add_argument("--t5-path",    required=True)
    p.add_argument("--qwen-model", default=None)
    p.add_argument("--reward-mode", default="grpo_attr_contrastive_rubric_v2")
    p.add_argument("--precision",  default="bf16")
    # generation
    p.add_argument("--num-gen",    type=int,   default=1)
    p.add_argument("--cfg-scale",  type=float, default=4.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--best-of-g",  type=int,   default=1)
    # adapter arch
    p.add_argument("--d-model",    type=int,   default=1280)
    p.add_argument("--n-comp-q",   type=int,   default=8)
    p.add_argument("--n-heads",    type=int,   default=8)
    # mode-specific
    p.add_argument("--val-jsonl",  default=None)
    p.add_argument("--prompt",     default="A red cube and a blue sphere.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, device: str):
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root, vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,  t5_path=args.t5_path,
        cfg_scale=args.cfg_scale, precision=args.precision, use_lora=False,
    )
    gpt = wrapper.gpt
    _   = wrapper.t5
    _   = wrapper.vq_model

    from adaptive_curriculum.model.implicit_comp_adapter import (
        ImplicitCompositionAdapter, attach_implicit_adapter,
    )
    adapter     = ImplicitCompositionAdapter(
        d_model=args.d_model, n_comp_q=args.n_comp_q, n_heads=args.n_heads,
    ).to(device=device, dtype=wrapper.dtype)
    adapted_cls = attach_implicit_adapter(gpt, adapter)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter"])
        step = ckpt.get("step", 0)
        print(f"[eval] Loaded adapter from {args.checkpoint}  step={step}"
              f"  gamma={float(adapter.gamma.item()):.4f}")
    else:
        print("[eval] No checkpoint — zero-initialized adapter (identity mode)")

    return wrapper, adapted_cls, adapter


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_pils(wrapper, prompt: str, n: int = 1, seed: Optional[int] = None) -> list:
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    if seed is not None:
        torch.manual_seed(seed)

    gpt = wrapper.gpt
    gpt.eval()
    ls  = wrapper.latent_size
    device = wrapper.device

    with torch.no_grad():
        caption_embs, emb_masks = wrapper.t5.get_text_embeddings([prompt])
    emb, mask = caption_embs[0], emb_masks[0]
    valid = int(mask.sum().item())
    new_emb  = torch.cat([emb[valid:], emb[:valid]])
    new_mask = torch.flip(mask, dims=[-1])
    c_indices = (new_emb * new_mask[:, None]).unsqueeze(0).to(device)

    qzshape = [1, wrapper.codebook_embed_dim, ls, ls]
    pils = []
    with torch.no_grad():
        for _ in range(n):
            idx = generate(
                gpt, c_indices, ls ** 2, None,
                cfg_scale=wrapper.cfg_scale,
                temperature=wrapper.temperature,
                top_k=wrapper.top_k,
                top_p=wrapper.top_p,
                sample_logits=True,
            )
            decoded = wrapper.vq_model.decode_code(idx, qzshape)
            img_t   = (decoded[0].float().clamp(-1, 1) + 1) / 2
            pils.append(TF.to_pil_image(img_t.cpu()))

    wrapper._disable_kv_cache()
    return pils


def _annotate(pil_img, label: str):
    from PIL import Image, ImageDraw, ImageFont
    W, H = pil_img.size
    strip = Image.new("RGB", (W, 18), (20, 20, 20))
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    draw.text((2, 2), label[:52], fill=(220, 220, 100), font=font)
    out = Image.new("RGB", (W, H + 18))
    out.paste(pil_img, (0, 0))
    out.paste(strip, (0, H))
    return out


def _hstack(images):
    from PIL import Image
    h = max(i.size[1] for i in images)
    w = sum(i.size[0] for i in images)
    out = Image.new("RGB", (w, h), (40, 40, 40))
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.size[0]
    return out


# ---------------------------------------------------------------------------
# Mode A: identity
# ---------------------------------------------------------------------------

def run_identity(args, wrapper, adapted_cls, adapter, device, out_dir):
    import numpy as np
    from PIL import Image
    import torchvision.transforms.functional as TF

    print("[eval:identity] Checking identity at gamma=0 ...")

    saved_gamma = float(adapter.gamma.item())
    adapter.gamma.data.fill_(0.0)

    pil_a = generate_pils(wrapper, args.prompt, n=1, seed=args.seed)[0]
    pil_b = generate_pils(wrapper, args.prompt, n=1, seed=args.seed)[0]

    t_a = TF.to_tensor(pil_a)
    t_b = TF.to_tensor(pil_b)
    diff = (t_a - t_b).abs()
    max_diff  = float(diff.max())
    mean_diff = float(diff.mean())

    pil_a.save(str(out_dir / "run1.png"))
    pil_b.save(str(out_dir / "run2.png"))
    diff_img = (diff.permute(1, 2, 0).numpy() * 255 * 20).clip(0, 255).astype("uint8")
    Image.fromarray(diff_img).save(str(out_dir / "diff_x20.png"))

    result = {
        "max_abs_diff_pixels":  max_diff,
        "mean_abs_diff_pixels": mean_diff,
        "identity_ok":          max_diff < 1e-3,
        "gamma_forced_zero":    True,
        "original_gamma":       saved_gamma,
    }
    with open(out_dir / "identity_result.json", "w") as f:
        json.dump(result, f, indent=2)

    adapter.gamma.data.fill_(saved_gamma)
    status = "PASS" if result["identity_ok"] else "FAIL"
    print(f"[eval:identity] {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    return result


# ---------------------------------------------------------------------------
# Mode B: counterfactual grid
# ---------------------------------------------------------------------------

# (prompt, [variant_prompts]) — the adapter sees the full raw prompt, no slots
_CF_GRIDS = [
    # attribute binding
    ("A cube and a sphere", [
        "A red cube and a blue sphere.",
        "A blue cube and a red sphere.",
        "A green cube and a yellow sphere.",
        "A wooden cube and a metal sphere.",
    ]),
    # spatial relations
    ("A red apple and a blue book", [
        "A red apple to the left of a blue book.",
        "A red apple to the right of a blue book.",
        "A red apple above a blue book.",
        "A red apple below a blue book.",
    ]),
]


def run_counterfactual(args, wrapper, reward_model, device, out_dir):
    print("[eval:counterfactual] Building counterfactual grids ...")

    all_results = {}
    for base_label, variant_prompts in _CF_GRIDS:
        grid_label = base_label.replace(" ", "_")[:30]
        panels, scores_row = [], []
        for vp in variant_prompts:
            pils = generate_pils(wrapper, vp, n=1, seed=args.seed)
            score = 0.0
            if reward_model:
                class _FakeItem:
                    text = vp
                    id   = "cf"
                    def __getattr__(self, k): return [] if "questions" in k.lower() else None
                score = float(reward_model.score_images_batch(
                    [(pils[0], _FakeItem())], mode=args.reward_mode
                )[0]["score"])
            pils[0].save(str(out_dir / f"cf_{grid_label}_{len(panels)}.png"))
            scores_row.append({"prompt": vp, "score": score})
            panels.append(_annotate(pils[0], f"r={score:.3f}  {vp[:40]}"))

        _hstack(panels).save(str(out_dir / f"cf_{grid_label}_row.png"))
        all_results[base_label] = scores_row

    with open(out_dir / "counterfactual_scores.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[eval:counterfactual] Saved {len(all_results)} grids to {out_dir}")


# ---------------------------------------------------------------------------
# Mode C: val_metrics
# ---------------------------------------------------------------------------

def run_val_metrics(args, wrapper, reward_model, device, out_dir):
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF
    from adaptive_curriculum.data.schemas import BucketItem

    assert args.val_jsonl, "--val-jsonl required for val_metrics mode"
    val_items = []
    with open(args.val_jsonl) as f:
        for line in f:
            if line.strip():
                val_items.append(BucketItem.from_dict(json.loads(line.strip())))

    print(f"[eval:val_metrics] {len(val_items)} items  G={args.best_of_g}")

    gpt = wrapper.gpt
    gpt.eval()
    ls  = wrapper.latent_size
    all_results = []

    for item in val_items:
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])
        qzshape = [1, wrapper.codebook_embed_dim, ls, ls]
        pils = []
        with torch.no_grad():
            for _ in range(args.best_of_g):
                idx = generate(
                    gpt, c_indices, ls ** 2, c_emb_masks,
                    cfg_scale=wrapper.cfg_scale,
                    temperature=wrapper.temperature,
                    top_k=wrapper.top_k,
                    top_p=wrapper.top_p,
                    sample_logits=True,
                )
                decoded = wrapper.vq_model.decode_code(idx, qzshape)
                img_t   = (decoded[0].float().clamp(-1, 1) + 1) / 2
                pils.append(TF.to_pil_image(img_t.cpu()))
        wrapper._disable_kv_cache()

        scored  = reward_model.score_images_batch([(p, item) for p in pils], mode=args.reward_mode)
        best    = max(scored, key=lambda r: r["score"])
        all_results.append({
            "id":               item.id,
            "score":            best["score"],
            "component_scores": best.get("component_scores", {}),
        })

    def _avg(field):
        vals = [r["component_scores"].get(field, 0) for r in all_results if r["component_scores"]]
        return sum(vals) / len(vals) if vals else 0.0

    hard_rewards = [r["score"] for r in all_results]
    summary = {
        "G":                  args.best_of_g,
        "n_items":            len(all_results),
        "checkpoint":         args.checkpoint,
        "val_hard_reward":    sum(hard_rewards) / len(hard_rewards),
        "object_presence":    _avg("object_presence"),
        "target_binding":     _avg("target_binding"),
        "swapped_binding":    _avg("swapped_binding"),
        "attribute_subscore": _avg("attribute_subscore"),
        "relation_effective": _avg("relation_effective"),
        "image_quality":      _avg("image_quality"),
        "prompt_alignment":   _avg("prompt_alignment"),
    }

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "val_per_item.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"[eval:val_metrics] hard_reward={summary['val_hard_reward']:.4f}"
          f"  presence={summary['object_presence']:.4f}"
          f"  quality={summary['image_quality']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Mode D: ablation  (base vs adapter — same seed)
# ---------------------------------------------------------------------------

def run_ablation(args, wrapper, adapted_cls, adapter, reward_model, device, out_dir):
    """Compare base LlamaGen vs trained adapter on a fixed set of prompts."""
    test_prompts = [
        "A red cube and a blue sphere.",
        "A wooden chair and a metal lamp.",
        "A striped cat sitting on a plaid rug.",
        "A black dog to the left of a white fence.",
        args.prompt,
    ]

    rows = []
    for prompt in test_prompts:
        safe = prompt.replace(" ", "_").replace(".", "")[:30]

        # Base (gamma forced to 0)
        saved_gamma = float(adapter.gamma.item())
        adapter.gamma.data.fill_(0.0)
        pil_base = generate_pils(wrapper, prompt, n=1, seed=args.seed)[0]
        adapter.gamma.data.fill_(saved_gamma)

        # Adapter
        pil_adp  = generate_pils(wrapper, prompt, n=1, seed=args.seed)[0]

        pil_base.save(str(out_dir / f"base_{safe}.png"))
        pil_adp.save( str(out_dir / f"adp_{safe}.png"))

        score_base = score_adp = 0.0
        if reward_model:
            class _FakeItem:
                text = prompt
                id   = "abl"
                def __getattr__(self, k): return [] if "questions" in k.lower() else None
            fi = _FakeItem()
            score_base = float(reward_model.score_images_batch([(pil_base, fi)], mode=args.reward_mode)[0]["score"])
            score_adp  = float(reward_model.score_images_batch([(pil_adp,  fi)], mode=args.reward_mode)[0]["score"])

        panel = _hstack([
            _annotate(pil_base, f"BASE  r={score_base:.3f}"),
            _annotate(pil_adp,  f"ADAPTER  r={score_adp:.3f}"),
        ])
        panel.save(str(out_dir / f"compare_{safe}.png"))
        rows.append({"prompt": prompt, "base_score": score_base, "adapter_score": score_adp,
                     "delta": score_adp - score_base})
        print(f"  {prompt[:45]}  base={score_base:.3f}  adapter={score_adp:.3f}"
              f"  Δ={score_adp - score_base:+.3f}")

    with open(out_dir / "ablation_scores.json", "w") as f:
        json.dump(rows, f, indent=2)

    mean_delta = sum(r["delta"] for r in rows) / len(rows) if rows else 0.0
    print(f"[eval:ablation] Mean score delta = {mean_delta:+.4f}")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper, adapted_cls, adapter = load_model(args, device)

    reward_model = None
    if args.mode in ("counterfactual", "val_metrics", "ablation"):
        qwen_id = args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
        try:
            from CARGO.scoring import CARGORewardModel
            reward_model = CARGORewardModel(model_id=qwen_id)
        except ImportError:
            from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
            reward_model = Qwen3VLRewardModel(model_id=qwen_id)

    if   args.mode == "identity":
        run_identity(args, wrapper, adapted_cls, adapter, device, out_dir)
    elif args.mode == "counterfactual":
        run_counterfactual(args, wrapper, reward_model, device, out_dir)
    elif args.mode == "val_metrics":
        run_val_metrics(args, wrapper, reward_model, device, out_dir)
    elif args.mode == "ablation":
        run_ablation(args, wrapper, adapted_cls, adapter, reward_model, device, out_dir)


if __name__ == "__main__":
    main()
