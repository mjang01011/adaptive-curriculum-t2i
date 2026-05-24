"""
Evaluate SlotResidualAdapter in four modes.

Modes
-----
  identity           — verify adapter with gamma=0 gives identical output to base
  slot_ablation      — correct vs empty vs swapped slots on same prompt
  counterfactual_grid — vary slot attributes/relations; save image grid
  val_metrics        — evaluate on original val set; report per-component rewards

Usage
-----
  # Identity test
  python SFT/eval_slot_adapter.py \\
    --mode identity \\
    --checkpoint $PROJ/outputs/slot_adapter_v1/best.pt \\
    --prompt "A red cube and a blue sphere." \\
    --output-dir $PROJ/outputs/slot_eval/identity \\
    <llm / model args>

  # Counterfactual grid
  python SFT/eval_slot_adapter.py \\
    --mode counterfactual_grid \\
    --checkpoint $PROJ/outputs/slot_adapter_v1/best.pt \\
    --prompt "A cube and a sphere." \\
    --output-dir $PROJ/outputs/slot_eval/cf_grid \\
    <model args>

  # Validation metrics
  python SFT/eval_slot_adapter.py \\
    --mode val_metrics \\
    --val-jsonl $PROJ/data/attribute_binding/attribute_binding_val_20.jsonl \\
    --checkpoint $PROJ/outputs/slot_adapter_v1/best.pt \\
    --output-dir $PROJ/outputs/slot_eval/val \\
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
    p.add_argument("--mode",         required=True, choices=["identity", "slot_ablation", "counterfactual_grid", "val_metrics"])
    p.add_argument("--checkpoint",   default=None,  help="Path to adapter checkpoint (.pt)")
    p.add_argument("--output-dir",   required=True)
    # model
    p.add_argument("--repo-root",    required=True)
    p.add_argument("--gpt-ckpt",     required=True)
    p.add_argument("--vq-ckpt",      required=True)
    p.add_argument("--t5-path",      required=True)
    p.add_argument("--qwen-model",   default=None)
    p.add_argument("--reward-mode",  default="grpo_attr_contrastive_rubric_v2")
    p.add_argument("--precision",    default="bf16")
    # generation
    p.add_argument("--num-gen",      type=int,   default=1,    help="Samples per condition")
    p.add_argument("--cfg-scale",    type=float, default=4.0)
    p.add_argument("--seed",         type=int,   default=42)
    # adapter architecture (must match training)
    p.add_argument("--d-model",      type=int,   default=1280)
    p.add_argument("--t5-dim",       type=int,   default=2048)
    p.add_argument("--n-heads",      type=int,   default=8)
    # mode-specific
    p.add_argument("--prompt",       default="A red cube and a blue sphere.")
    p.add_argument("--val-jsonl",    default=None)
    p.add_argument("--best-of-g",    type=int,   default=1,    help="Best-of-G samples for val_metrics")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_wrapper_and_adapter(args, device: str):
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=args.cfg_scale,
        precision=args.precision,
        use_lora=False,
    )
    gpt = wrapper.gpt
    t5  = wrapper.t5
    _   = wrapper.vq_model

    from adaptive_curriculum.model.slot_adapter import SlotResidualAdapter, attach_slot_adapter
    adapter     = SlotResidualAdapter(
        d_model=args.d_model, t5_dim=args.t5_dim, n_heads=args.n_heads,
    ).to(device=device, dtype=wrapper.dtype)
    adapted_cls = attach_slot_adapter(gpt, adapter)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        adapter.load_state_dict(ckpt["adapter"])
        print(f"[eval] Loaded adapter from {args.checkpoint}  "
              f"gamma={float(adapter.gamma.item()):.4f}")
    else:
        print("[eval] No checkpoint — using zero-initialized adapter (identity mode)")

    return wrapper, adapted_cls, adapter


# ---------------------------------------------------------------------------
# Shared generation helper
# ---------------------------------------------------------------------------

def generate_with_slots(
    wrapper,
    adapted_cls,
    t5_model,
    prompt:      str,
    slot_texts:  List[str],
    n:           int   = 1,
    seed:        Optional[int] = None,
    device:      str   = "cuda",
    t5_dim:      int   = 2048,
) -> list:
    """Returns list of n PIL images generated with given slot context."""
    import contextlib
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    if seed is not None:
        torch.manual_seed(seed)

    gpt = wrapper.gpt
    gpt.eval()
    ls  = wrapper.latent_size

    # T5 conditioning
    with torch.no_grad():
        caption_embs, emb_masks = t5_model.get_text_embeddings([prompt])
    emb, mask = caption_embs[0], emb_masks[0]
    valid = int(mask.sum().item())
    new_emb  = torch.cat([emb[valid:], emb[:valid]])
    new_mask = torch.flip(mask, dims=[-1])
    c_indices = (new_emb * new_mask[:, None]).unsqueeze(0).to(device)

    # Slot T5 embeddings
    if slot_texts:
        with torch.no_grad():
            s_embs, s_masks = t5_model.get_text_embeddings(slot_texts)
        slot_vecs = []
        for se, sm in zip(s_embs, s_masks):
            vl = int(sm.sum().item())
            slot_vecs.append(se[:vl].mean(0) if vl > 0 else se.mean(0))
        K = len(slot_vecs)
        slot_emb_t = torch.stack(slot_vecs, dim=0).unsqueeze(0).to(device)   # [1, K, 2048]
        slot_mask_t = torch.zeros(1, K, dtype=torch.bool, device=device)      # no padding
        adapted_cls.set_slot_context(
            slot_emb_t.to(wrapper.dtype),
            slot_mask_t,
        )
    else:
        adapted_cls.clear_slot_context()

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

    adapted_cls.clear_slot_context()
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
    draw.text((2, 2), label[:50], fill=(220, 220, 100), font=font)
    canvas = Image.new("RGB", (W, H + 18))
    canvas.paste(pil_img, (0, 0))
    canvas.paste(strip,   (0, H))
    return canvas


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

def run_identity(args, wrapper, adapted_cls, adapter, t5_model, device, out_dir):
    from PIL import Image
    print("[eval:identity] Checking adapter identity at gamma=0 ...")

    saved_gamma = float(adapter.gamma.item())
    adapter.gamma.data.fill_(0.0)

    pils_base  = generate_with_slots(wrapper, adapted_cls, t5_model, args.prompt,
                                      slot_texts=[], n=1, seed=args.seed, device=device)
    pils_adapt = generate_with_slots(wrapper, adapted_cls, t5_model, args.prompt,
                                      slot_texts=["entity: red cube", "entity: blue sphere"],
                                      n=1, seed=args.seed, device=device)

    import torchvision.transforms.functional as TF
    import numpy as np

    t_base  = TF.to_tensor(pils_base[0])
    t_adapt = TF.to_tensor(pils_adapt[0])
    diff    = (t_base - t_adapt).abs()
    max_diff  = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    result = {
        "max_abs_diff_pixels": max_diff,
        "mean_abs_diff_pixels": mean_diff,
        "identity_ok": max_diff < 1e-3,
        "gamma_forced_zero": True,
        "checkpoint_gamma": saved_gamma,
    }

    pils_base[0].save(str(out_dir / "base.png"))
    pils_adapt[0].save(str(out_dir / "adapter_gamma0.png"))

    diff_np = (diff.permute(1, 2, 0).numpy() * 255 * 20).clip(0, 255).astype("uint8")
    Image.fromarray(diff_np).save(str(out_dir / "diff_x20.png"))

    with open(out_dir / "identity_diff.json", "w") as f:
        json.dump(result, f, indent=2)

    adapter.gamma.data.fill_(saved_gamma)
    status = "PASS" if result["identity_ok"] else "FAIL"
    print(f"[eval:identity] {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    return result


# ---------------------------------------------------------------------------
# Mode B: slot_ablation
# ---------------------------------------------------------------------------

def run_slot_ablation(args, wrapper, adapted_cls, adapter, t5_model, reward_model, device, out_dir):
    print("[eval:slot_ablation] Correct vs empty vs swapped slots ...")

    # Derive swapped from prompt heuristically (swap first two attr-noun pairs)
    import re
    # Simple: correct = extract from prompt, swapped = swap attrs
    cap = args.prompt
    entities_raw = re.findall(r"((?:red|blue|green|yellow|black|white|wooden|metal|plastic)\s+\w+)", cap, re.I)

    if len(entities_raw) >= 2:
        correct_slots  = [f"entity: {e}" for e in entities_raw[:2]]
        # Swap attributes between two entities
        words0 = entities_raw[0].split()
        words1 = entities_raw[1].split()
        sw0 = words1[0] + " " + " ".join(words0[1:])
        sw1 = words0[0] + " " + " ".join(words1[1:])
        swapped_slots  = [f"entity: {sw0}", f"entity: {sw1}"]
    else:
        correct_slots  = [f"entity: {cap}"]
        swapped_slots  = [f"entity: wrong object other thing"]

    conditions = {
        "correct_slots": correct_slots,
        "empty_slots":   [],
        "swapped_slots": swapped_slots,
    }

    results   = {}
    panels    = []
    for cond_name, slot_texts in conditions.items():
        pils = generate_with_slots(wrapper, adapted_cls, t5_model, args.prompt,
                                    slot_texts=slot_texts, n=args.num_gen,
                                    seed=args.seed, device=device)
        if reward_model:
            # Build a minimal item for scoring
            class _FakeItem:
                text = args.prompt
                id   = "ablation"
                def __getattr__(self, k): return [] if "questions" in k.lower() else None
            fake_item = _FakeItem()
            scores = [float(r["score"]) for r in
                      reward_model.score_images_batch([(p, fake_item) for p in pils], mode=args.reward_mode)]
            results[cond_name] = {"mean_score": sum(scores) / len(scores), "scores": scores}
        else:
            results[cond_name] = {"scores": []}

        pils[0].save(str(out_dir / f"{cond_name}.png"))
        label = f"{cond_name}  r={results[cond_name]['mean_score']:.3f}" if reward_model else cond_name
        panels.append(_annotate(pils[0], label))

    _hstack(panels).save(str(out_dir / "slot_ablation_panel.png"))
    with open(out_dir / "slot_ablation_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    if reward_model:
        correct_r = results["correct_slots"]["mean_score"]
        empty_r   = results["empty_slots"]["mean_score"]
        swapped_r = results["swapped_slots"]["mean_score"]
        print(f"[eval:slot_ablation] correct={correct_r:.3f}  empty={empty_r:.3f}  swapped={swapped_r:.3f}")
        if correct_r > empty_r:
            print("[eval:slot_ablation] PASS: correct > empty")
        else:
            print("[eval:slot_ablation] WARN: correct <= empty (adapter may not be effective)")


# ---------------------------------------------------------------------------
# Mode C: counterfactual_grid
# ---------------------------------------------------------------------------

_CF_ATTRIBUTE_CONDITIONS = [
    ("A red cube and a blue sphere.",
     [("entity: red cube", "entity: blue sphere"),
      ("entity: blue cube", "entity: red sphere"),
      ("entity: green cube", "entity: yellow sphere"),
      ()]),
]

_CF_SPATIAL_CONDITIONS = [
    ("A red cube and a blue sphere.",
     [("entity: red cube", "entity: blue sphere", "relation: red cube left_of blue sphere"),
      ("entity: red cube", "entity: blue sphere", "relation: red cube right_of blue sphere"),
      ("entity: red cube", "entity: blue sphere", "relation: red cube above blue sphere"),
      ("entity: red cube", "entity: blue sphere")]),
]


def run_counterfactual_grid(args, wrapper, adapted_cls, adapter, t5_model, reward_model, device, out_dir):
    print("[eval:counterfactual_grid] Building attribute and spatial counterfactual grids ...")
    from PIL import Image

    for grid_name, conditions in [
        ("attribute", _CF_ATTRIBUTE_CONDITIONS),
        ("spatial",   _CF_SPATIAL_CONDITIONS),
    ]:
        for prompt_template, slot_variants in conditions:
            base_prompt = prompt_template
            row_panels  = []
            scores_grid = []

            for variant_slots in slot_variants:
                slot_texts = list(variant_slots)
                pils = generate_with_slots(wrapper, adapted_cls, t5_model, base_prompt,
                                            slot_texts=slot_texts, n=1,
                                            seed=args.seed, device=device)

                if reward_model:
                    class _FakeItem:
                        text = base_prompt
                        id   = "cf"
                        def __getattr__(self, k): return [] if "questions" in k.lower() else None
                    score = float(reward_model.score_images_batch(
                        [(pils[0], _FakeItem())], mode=args.reward_mode
                    )[0]["score"])
                else:
                    score = 0.0

                scores_grid.append({"slots": slot_texts, "score": score})
                label = (", ".join(slot_texts[:2]) or "no slots") + f"  r={score:.3f}"
                row_panels.append(_annotate(pils[0], label[:55]))

            # stitch into row
            _hstack(row_panels).save(str(out_dir / f"cf_{grid_name}.png"))
            with open(out_dir / f"cf_{grid_name}_scores.json", "w") as f:
                json.dump(scores_grid, f, indent=2)
            print(f"[eval:cf] {grid_name} grid saved")


# ---------------------------------------------------------------------------
# Mode D: val_metrics
# ---------------------------------------------------------------------------

def run_val_metrics(args, wrapper, adapted_cls, adapter, t5_model, reward_model, device, out_dir):
    import contextlib
    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    assert args.val_jsonl, "--val-jsonl required for val_metrics mode"
    from adaptive_curriculum.data.schemas import BucketItem
    val_items = []
    with open(args.val_jsonl) as f:
        for line in f:
            if line.strip():
                val_items.append(BucketItem.from_dict(json.loads(line.strip())))

    print(f"[eval:val_metrics] {len(val_items)} val items, G={args.best_of_g}")

    gpt = wrapper.gpt
    gpt.eval()
    ls  = wrapper.latent_size

    all_results = []
    for item in val_items:
        with torch.no_grad():
            c_indices, c_emb_masks = wrapper._get_conditioning([item])

        # val runs without slot context (generalization to original prompts)
        adapted_cls.clear_slot_context()

        qzshape = [1, wrapper.codebook_embed_dim, ls, ls]
        pils    = []
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

        scored = reward_model.score_images_batch([(p, item) for p in pils], mode=args.reward_mode)
        best_result = max(scored, key=lambda r: r["score"])
        all_results.append({
            "id":               item.id,
            "score":            best_result["score"],
            "component_scores": best_result.get("component_scores", {}),
        })

    # Aggregate
    def _mean_field(field):
        vals = [r["component_scores"].get(field, 0) for r in all_results if r["component_scores"]]
        return sum(vals) / len(vals) if vals else 0.0

    hard_rewards = [r["score"] for r in all_results]
    summary = {
        "G":                args.best_of_g,
        "n_items":          len(all_results),
        "val_hard_reward":  sum(hard_rewards) / len(hard_rewards),
        "object_presence":  _mean_field("object_presence"),
        "target_binding":   _mean_field("target_binding"),
        "swapped_binding":  _mean_field("swapped_binding"),
        "attribute_subscore": _mean_field("attribute_subscore"),
        "relation_effective": _mean_field("relation_effective"),
        "image_quality":    _mean_field("image_quality"),
        "prompt_alignment": _mean_field("prompt_alignment"),
    }

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "val_per_item.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"[eval:val_metrics] G={args.best_of_g}")
    print(f"  hard_reward    = {summary['val_hard_reward']:.4f}")
    print(f"  object_presence= {summary['object_presence']:.4f}")
    comp_key = "target_binding" if summary["target_binding"] > 0 else "relation_effective"
    print(f"  {comp_key:<16} = {summary[comp_key]:.4f}")
    print(f"  image_quality  = {summary['image_quality']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper, adapted_cls, adapter = load_wrapper_and_adapter(args, device)
    t5_model = wrapper.t5

    reward_model = None
    if args.mode in ("slot_ablation", "counterfactual_grid", "val_metrics"):
        qwen_id = args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
        try:
            from CARGO.scoring import CARGORewardModel
            reward_model = CARGORewardModel(model_id=qwen_id)
        except ImportError:
            from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
            reward_model = Qwen3VLRewardModel(model_id=qwen_id)

    if args.mode == "identity":
        run_identity(args, wrapper, adapted_cls, adapter, t5_model, device, out_dir)

    elif args.mode == "slot_ablation":
        run_slot_ablation(args, wrapper, adapted_cls, adapter, t5_model, reward_model, device, out_dir)

    elif args.mode == "counterfactual_grid":
        run_counterfactual_grid(args, wrapper, adapted_cls, adapter, t5_model, reward_model, device, out_dir)

    elif args.mode == "val_metrics":
        run_val_metrics(args, wrapper, adapted_cls, adapter, t5_model, reward_model, device, out_dir)


if __name__ == "__main__":
    main()
