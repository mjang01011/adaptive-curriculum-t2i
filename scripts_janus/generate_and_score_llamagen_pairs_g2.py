"""
Generate G=2 images per prompt with LlamaGen, score with Qwen, build DPO pairs.
All pairs saved (no margin filtering). pair_weight = f(margin).

Usage (svl env):
  python scripts_janus/generate_and_score_llamagen_pairs_g2.py \
    --input-jsonl /viscam/u/jj277/adaptive-curriculum-t2i/data/attribute_binding/attribute_binding_train_500.jsonl \
    --output-dir  /viscam/u/jj277/adaptive-curriculum-t2i/outputs_pairs/attr_g2_llamagen \
    --num-prompts 32 --seeds 0 1 \
    --repo-root   /viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen \
    --gpt-ckpt    /viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/t2i_XL_stage1_256.pt \
    --vq-ckpt     /viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/vq_ds16_t2i.pt \
    --t5-path     /viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models/t5-ckpt \
    --reward-mode pseudo_soft_grpo_target_heavy \
    --save-tokens
"""
import argparse
import base64
import io
import json
import random
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from adaptive_curriculum.data.schemas import BucketItem


# ══════════════════════════════════════════════════════════════════════════════
# Pair weighting
# ══════════════════════════════════════════════════════════════════════════════

def compute_pair_weight(margin: float) -> float:
    return 0.25 + 0.75 * min(margin / 0.30, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_items(jsonl_path, num_prompts, seed=42):
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:num_prompts]


# ══════════════════════════════════════════════════════════════════════════════
# Audit HTML
# ══════════════════════════════════════════════════════════════════════════════

def _b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_audit_html(pair_records, out_path):
    sorted_records = sorted(pair_records, key=lambda r: r["margin"], reverse=True)
    top    = sorted_records[:20]
    bottom = sorted_records[-20:]
    mid    = sorted_records[20:-20]
    random.shuffle(mid)
    sections = [("Largest margin", top), ("Smallest margin (noisy)", bottom), ("Random", mid[:20])]

    def card(r):
        c_img  = r.get("_chosen_pil")
        rj_img = r.get("_rejected_pil")
        ci  = f'<img src="{_b64(c_img)}"  style="width:160px">' if c_img  else ""
        ri  = f'<img src="{_b64(rj_img)}" style="width:160px">' if rj_img else ""
        mc  = "#6f6" if r["margin"] >= 0.3 else "#ff6" if r["margin"] >= 0.1 else "#f66"
        return f"""
<div style="background:#1a1a1a;border:1px solid #444;border-radius:6px;padding:8px;margin-bottom:8px">
  <div style="color:#aef;font-size:10px;margin-bottom:4px">{r['prompt']}</div>
  <div style="display:flex;gap:8px">
    <div><div style="color:#6f6;font-size:9px">chosen r={r['chosen_reward']:.3f}</div>{ci}</div>
    <div><div style="color:#f66;font-size:9px">rejected r={r['rejected_reward']:.3f}</div>{ri}</div>
    <div style="font-size:11px;padding-top:16px">
      <span style="color:{mc}">margin={r['margin']:.3f}</span><br>weight={r['pair_weight']:.3f}
    </div>
  </div>
</div>"""

    body = ""
    for title, records in sections:
        if not records:
            continue
        body += f"<h2 style='color:#fa0'>{title} ({len(records)})</h2>"
        body += "".join(card(r) for r in records)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:monospace;background:#111;color:#eee;margin:16px}}
h1{{color:#7cf}}</style></head>
<body><h1>LlamaGen DPO Pair Audit</h1>
<p>{len(pair_records)} pairs  accept_all=True</p>{body}</body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"[audit] → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl",  required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--num-prompts",  type=int, default=32)
    parser.add_argument("--seeds",        type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--data-seed",    type=int, default=42)
    # LlamaGen paths
    parser.add_argument("--repo-root",    required=True)
    parser.add_argument("--gpt-ckpt",     required=True)
    parser.add_argument("--vq-ckpt",      required=True)
    parser.add_argument("--t5-path",      required=True)
    parser.add_argument("--cfg-scale",    type=float, default=2.0)
    # Qwen
    parser.add_argument("--reward-mode",  default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--qwen-model",   default="Qwen/Qwen3-VL-4B-Instruct")
    # output
    parser.add_argument("--save-tokens",     action="store_true")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Path to best_checkpoint.pt from previous DPO round.")
    # LoRA config — must match the training run that produced --init-checkpoint
    parser.add_argument("--lora-r",            type=int, default=16)
    parser.add_argument("--lora-alpha",        type=int, default=32)
    parser.add_argument("--lora-target-modules", nargs="+", default=["wqkv", "wo"])
    args = parser.parse_args()

    assert len(args.seeds) == 2, "G=2 requires exactly 2 seeds"

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    tok_dir = out_dir / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    if args.save_tokens:
        tok_dir.mkdir(exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    items = load_items(args.input_jsonl, args.num_prompts, seed=args.data_seed)
    print(f"[gen] {len(items)} prompts  seeds={args.seeds}  reward_mode={args.reward_mode}")

    # ── load LlamaGen ─────────────────────────────────────────────────────────
    sys.path.insert(0, args.repo_root)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper

    lora_cfg = None
    if args.init_checkpoint:
        lora_cfg = {
            "rank":           args.lora_r,
            "alpha":          args.lora_alpha,
            "dropout":        0.0,
            "target_modules": args.lora_target_modules,
            "start_layer":    0,
        }

    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root,
        vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt,
        t5_path=args.t5_path,
        cfg_scale=args.cfg_scale,
        use_lora=bool(args.init_checkpoint),
        lora_config=lora_cfg,
    )
    _ = wrapper.gpt
    _ = wrapper.vq_model
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cuda")
        wrapper.gpt.load_state_dict(ckpt)
        print(f"[gen] loaded init checkpoint: {args.init_checkpoint}")
    print("[gen] LlamaGen loaded.")

    # ── load Qwen ─────────────────────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(model_id=args.qwen_model)
    print("[gen] Qwen loaded.\n")

    # ── generate seed by seed ─────────────────────────────────────────────────
    import torchvision.transforms.functional as TF
    from autoregressive.models.generate import generate

    def _batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i: i + bs]

    samples_by_id = {item.id: {} for item in items}
    all_samples   = []

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n── seed={seed} ({seed_idx+1}/2) ───────────────────────────────")
        for batch in _batches(items, args.batch_size):
            prompts = [it.text for it in batch]
            B       = len(batch)
            qzshape = [B, wrapper.codebook_embed_dim, wrapper.latent_size, wrapper.latent_size]

            t0 = time.time()
            wrapper.gpt.eval()
            with torch.no_grad():
                c_indices, c_emb_masks = wrapper._get_conditioning(
                    batch, t5_cache=None
                )
                tokens = generate(
                    wrapper.gpt, c_indices, wrapper.latent_size ** 2, c_emb_masks,
                    cfg_scale=wrapper.cfg_scale,
                    temperature=wrapper.temperature,
                    top_k=wrapper.top_k, top_p=wrapper.top_p,
                    sample_logits=True,
                )
                wrapper._disable_kv_cache()
                decoded = wrapper.vq_model.decode_code(tokens, qzshape)

            elapsed = time.time() - t0

            for i, item in enumerate(batch):
                img_t = (decoded[i].float().clamp(-1, 1) + 1) / 2
                pil   = TF.to_pil_image(img_t.cpu())

                fname = f"{item.id}_seed{seed}.png"
                pil.save(img_dir / fname)

                tok_path = None
                if args.save_tokens:
                    tok_fname = f"{item.id}_seed{seed}.pt"
                    torch.save(tokens[i].cpu(), tok_dir / tok_fname)
                    tok_path = f"tokens/{tok_fname}"

                result = reward_model.score_image(pil, item, mode=args.reward_mode)
                r      = float(result["score"])
                comps  = result.get("component_scores", {})

                sample = {
                    "id":          item.id,
                    "prompt":      item.text,
                    "seed":        seed,
                    "image_path":  f"images/{fname}",
                    "tokens_path": tok_path,
                    "reward":      round(r, 4),
                    "components":  {k: round(float(v), 4) for k, v in comps.items()},
                    "qwen_answers": result.get("question_scores", []),
                    "_pil":        pil,
                }
                all_samples.append(sample)
                samples_by_id[item.id][seed_idx] = sample
                print(f"  {item.id}  seed={seed}  r={r:.3f}  ({elapsed/B:.1f}s/img)")

    # ── build pairs ───────────────────────────────────────────────────────────
    print("\n── building pairs ────────────────────────────────────────────────")
    pair_records = []
    n_skipped    = 0

    for item in items:
        s = samples_by_id[item.id]
        if 0 not in s or 1 not in s:
            n_skipped += 1
            continue

        s0, s1 = s[0], s[1]
        chosen, rejected = (s0, s1) if s0["reward"] >= s1["reward"] else (s1, s0)
        margin      = abs(chosen["reward"] - rejected["reward"])
        pair_weight = compute_pair_weight(margin)

        pair = {
            "id":     item.id,
            "prompt": item.text,
            "chosen_image_path":    chosen["image_path"],
            "rejected_image_path":  rejected["image_path"],
            "chosen_tokens_path":   chosen["tokens_path"],
            "rejected_tokens_path": rejected["tokens_path"],
            "chosen_reward":        chosen["reward"],
            "rejected_reward":      rejected["reward"],
            "margin":               round(margin, 4),
            "pair_weight":          round(pair_weight, 4),
            "chosen_components":    chosen["components"],
            "rejected_components":  rejected["components"],
            "chosen_qwen_answers":  chosen["qwen_answers"],
            "rejected_qwen_answers": rejected["qwen_answers"],
            "_chosen_pil":          chosen["_pil"],
            "_rejected_pil":        rejected["_pil"],
        }
        pair_records.append(pair)
        print(f"  {item.id}  chosen={chosen['reward']:.3f}  rej={rejected['reward']:.3f}  "
              f"margin={margin:.3f}  weight={pair_weight:.3f}")

    # ── write outputs ─────────────────────────────────────────────────────────
    with open(out_dir / "all_samples.jsonl", "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}) + "\n")

    with open(out_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pair_records:
            f.write(json.dumps({k: v for k, v in p.items() if not k.startswith("_")}) + "\n")

    all_rewards = [s["reward"] for s in all_samples]
    margins     = [p["margin"]      for p in pair_records]
    chosen_r    = [p["chosen_reward"]   for p in pair_records]
    rejected_r  = [p["rejected_reward"] for p in pair_records]

    def bucket_margins(ms):
        b = {"0.00_0.05": 0, "0.05_0.10": 0, "0.10_0.20": 0, "0.20_0.30": 0, "0.30_plus": 0}
        for m in ms:
            if   m < 0.05: b["0.00_0.05"] += 1
            elif m < 0.10: b["0.05_0.10"] += 1
            elif m < 0.20: b["0.10_0.20"] += 1
            elif m < 0.30: b["0.20_0.30"] += 1
            else:          b["0.30_plus"] += 1
        return b

    summary = {
        "num_prompts":         len(items),
        "num_images":          len(all_samples),
        "num_pairs":           len(pair_records),
        "num_skipped":         n_skipped,
        "model":               "llamagen",
        "reward_mode":         args.reward_mode,
        "mean_reward":         round(sum(all_rewards)/len(all_rewards), 4) if all_rewards else 0,
        "mean_chosen_reward":  round(sum(chosen_r)/len(chosen_r), 4) if chosen_r else 0,
        "mean_rejected_reward":round(sum(rejected_r)/len(rejected_r), 4) if rejected_r else 0,
        "mean_margin":         round(sum(margins)/len(margins), 4) if margins else 0,
        "median_margin":       round(sorted(margins)[len(margins)//2], 4) if margins else 0,
        "margin_buckets":      bucket_margins(margins),
        "mean_pair_weight":    round(sum(p["pair_weight"] for p in pair_records)/len(pair_records), 4) if pair_records else 0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_audit_html(pair_records, out_dir / "audit.html")

    print(f"\n[summary] pairs={len(pair_records)}  mean_reward={summary['mean_reward']}  "
          f"mean_margin={summary['mean_margin']}")
    print(f"[summary] margin_buckets={summary['margin_buckets']}")
    print(f"[gen] done → {out_dir}")


if __name__ == "__main__":
    main()
