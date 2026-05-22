"""
Generate G=2 images per prompt with Janus-Pro-1B, score with Qwen, build DPO pairs.
All pairs are saved (no margin filtering). Pair weight = f(margin).

Usage:
  python scripts_janus/generate_and_score_janus_pairs_g2.py \
    --input-jsonl /viscam/u/jj277/adaptive-curriculum-t2i/data/attribute_binding/attribute_binding_train_500.jsonl \
    --output-dir  /viscam/u/jj277/janus_project/outputs_janus_pairs/attr_g2_all \
    --num-prompts 200 \
    --seeds 0 1 \
    --cfg-weight 5.0 \
    --reward-mode pseudo_soft_grpo_target_heavy \
    --save-tokens \
    --accept-all-pairs
"""
import argparse
import base64
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))

from adaptive_curriculum.data.schemas import BucketItem
from adaptive_curriculum.reward.vlm_reward import RewardModel as VLMRewardModel


# ══════════════════════════════════════════════════════════════════════════════
# Pair weighting
# ══════════════════════════════════════════════════════════════════════════════

def compute_pair_weight(margin: float) -> float:
    """
    margin in [0, 1].
    0.00 → 0.25  (near-tie: still used, very low weight)
    0.10 → 0.50
    0.20 → 0.75
    0.30+ → 1.00
    """
    return 0.25 + 0.75 * min(margin / 0.30, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_items(jsonl_path: str, num_prompts: int, seed: int = 42) -> list:
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
# HTML audit
# ══════════════════════════════════════════════════════════════════════════════

def _b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_audit_html(pair_records, out_path):
    """Sort by margin (largest first, then smallest, then random middle)."""
    sorted_records = sorted(pair_records, key=lambda r: r["margin"], reverse=True)
    top    = sorted_records[:30]
    bottom = sorted_records[-30:]
    mid    = sorted_records[30:-30]
    random.shuffle(mid)
    sections = [("Largest margin (high confidence)", top),
                ("Smallest margin (near-tie / noisy)", bottom),
                ("Random sample", mid[:30])]

    def card(r):
        c_img = r.get("_chosen_pil")
        rj_img = r.get("_rejected_pil")
        ci  = f'<img src="{_b64(c_img)}" style="width:180px">'   if c_img  else "<div style='width:180px;height:180px;background:#333'></div>"
        ri  = f'<img src="{_b64(rj_img)}" style="width:180px">'  if rj_img else "<div style='width:180px;height:180px;background:#333'></div>"
        margin = r["margin"]
        mcolor = "#6f6" if margin >= 0.3 else "#ff6" if margin >= 0.1 else "#f66"
        comp_chosen = "  ".join(f"{k}={v:.2f}" for k, v in r.get("chosen_components", {}).items())
        comp_rej    = "  ".join(f"{k}={v:.2f}" for k, v in r.get("rejected_components", {}).items())
        return f"""
<div style="background:#1a1a1a;border:1px solid #444;border-radius:6px;padding:10px;margin-bottom:10px">
  <div style="color:#aef;font-size:11px;margin-bottom:6px">{r['prompt']}</div>
  <div style="display:flex;gap:8px;align-items:flex-start">
    <div><div style="color:#6f6;font-size:10px">chosen  r={r['chosen_reward']:.3f}</div>{ci}
      <div style="font-size:9px;color:#888;margin-top:2px">{comp_chosen}</div></div>
    <div><div style="color:#f66;font-size:10px">rejected r={r['rejected_reward']:.3f}</div>{ri}
      <div style="font-size:9px;color:#888;margin-top:2px">{comp_rej}</div></div>
    <div style="font-size:11px;padding-top:20px">
      <span style="color:{mcolor}">margin={margin:.3f}</span><br>
      weight={r['pair_weight']:.3f}
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
h1{{color:#7cf}} p{{color:#aaa}}</style></head>
<body><h1>Janus DPO Pair Audit</h1>
<p>{len(pair_records)} pairs  accept_all=True</p>{body}</body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"[audit] → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl",   required=True)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--num-prompts",   type=int, default=200)
    parser.add_argument("--model-path",    default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--seeds",         type=int, nargs="+", default=[0, 1])
    parser.add_argument("--cfg-weight",    type=float, default=5.0)
    parser.add_argument("--temperature",   type=float, default=1.0)
    parser.add_argument("--reward-mode",   default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--save-tokens",   action="store_true")
    parser.add_argument("--accept-all-pairs", action="store_true", default=True)
    parser.add_argument("--batch-size",    type=int, default=4)
    parser.add_argument("--data-seed",     type=int, default=42)
    args = parser.parse_args()

    assert len(args.seeds) == 2, "G=2 requires exactly 2 seeds"

    out_dir   = Path(args.output_dir)
    img_dir   = out_dir / "images"
    tok_dir   = out_dir / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    if args.save_tokens:
        tok_dir.mkdir(exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    items = load_items(args.input_jsonl, args.num_prompts, seed=args.data_seed)
    print(f"[gen] {len(items)} prompts  seeds={args.seeds}  reward_mode={args.reward_mode}")

    # ── load models ───────────────────────────────────────────────────────────
    from scripts_janus.janus_wrapper import JanusProWrapper
    wrapper = JanusProWrapper(
        model_path=args.model_path,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    _ = wrapper.model
    print("[gen] Janus loaded.")

    # Qwen reward model — reuse VLMRewardModel with a minimal config shim
    from omegaconf import OmegaConf
    reward_cfg = OmegaConf.create({
        "model": {"vlm_model": "Qwen/Qwen2-VL-2B-Instruct"},
        "evaluation": {"reward_model": "vlm"},
    })
    reward_model = VLMRewardModel(reward_cfg)
    print("[gen] Qwen loaded.\n")

    # ── generate + score ──────────────────────────────────────────────────────
    all_samples  = []   # one entry per (item, seed)
    pair_records = []   # one entry per item

    def _batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i + bs]

    # generate seed-by-seed so memory stays bounded
    # samples_by_id[item.id][seed_idx] = {image_path, tokens_path, reward, ...}
    samples_by_id = {item.id: {} for item in items}

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n── seed={seed} ({seed_idx+1}/2) ────────────────────────────────")
        for batch_items in _batches(items, args.batch_size):
            prompts = [it.text for it in batch_items]
            t0 = time.time()
            out = wrapper.generate_images(
                prompts, seeds=[seed] * len(batch_items),
                return_tokens=args.save_tokens,
            )
            elapsed = time.time() - t0

            for i, item in enumerate(batch_items):
                pil = out["images"][i]

                # save image
                fname  = f"{item.id}_seed{seed}.png"
                fpath  = img_dir / fname
                pil.save(fpath)

                # save tokens
                tok_path = None
                if args.save_tokens:
                    tok_fname = f"{item.id}_seed{seed}.pt"
                    tok_path  = tok_dir / tok_fname
                    torch.save(out["generated_tokens"][i], tok_path)

                # score
                result = reward_model.score_image(pil, item, mode=args.reward_mode)
                reward = float(result["score"])
                comps  = result.get("component_scores", {})

                sample = {
                    "id":           item.id,
                    "prompt":       item.text,
                    "seed":         seed,
                    "image_path":   f"images/{fname}",
                    "tokens_path":  f"tokens/{tok_fname}" if tok_path else None,
                    "reward":       round(reward, 4),
                    "components":   {k: round(float(v), 4) for k, v in comps.items()},
                    "qwen_answers": result.get("question_scores", []),
                }
                all_samples.append(sample)
                samples_by_id[item.id][seed_idx] = sample

                print(f"  {item.id}  seed={seed}  r={reward:.3f}  ({elapsed/len(batch_items):.1f}s/img)")

    # ── build pairs ───────────────────────────────────────────────────────────
    print("\n── building pairs ───────────────────────────────────────────────")
    n_skipped = 0

    for item in items:
        s = samples_by_id[item.id]
        if 0 not in s or 1 not in s:
            n_skipped += 1
            print(f"  [skip] {item.id}: missing one generation")
            continue

        s0, s1 = s[0], s[1]
        if s0["reward"] >= s1["reward"]:
            chosen, rejected = s0, s1
        else:
            chosen, rejected = s1, s0

        margin      = abs(chosen["reward"] - rejected["reward"])
        pair_weight = compute_pair_weight(margin)

        pair = {
            "id":     item.id,
            "prompt": item.text,

            "chosen_image_path":    chosen["image_path"],
            "rejected_image_path":  rejected["image_path"],
            "chosen_tokens_path":   chosen["tokens_path"],
            "rejected_tokens_path": rejected["tokens_path"],

            "chosen_reward":   chosen["reward"],
            "rejected_reward": rejected["reward"],
            "margin":          round(margin, 4),
            "pair_weight":     round(pair_weight, 4),

            "chosen_components":   chosen["components"],
            "rejected_components": rejected["components"],

            "chosen_qwen_answers":   chosen["qwen_answers"],
            "rejected_qwen_answers": rejected["qwen_answers"],
        }
        pair_records.append(pair)
        print(f"  {item.id}  chosen_r={chosen['reward']:.3f}  rej_r={rejected['reward']:.3f}  "
              f"margin={margin:.3f}  weight={pair_weight:.3f}")

    # ── write outputs ─────────────────────────────────────────────────────────
    with open(out_dir / "all_samples.jsonl", "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps({k: v for k, v in s.items() if k != "_pil"}) + "\n")

    with open(out_dir / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pair_records:
            f.write(json.dumps({k: v for k, v in p.items() if not k.startswith("_")}) + "\n")

    # summary stats
    all_rewards  = [s["reward"] for s in all_samples]
    margins      = [p["margin"] for p in pair_records]
    weights      = [p["pair_weight"] for p in pair_records]
    chosen_r     = [p["chosen_reward"]   for p in pair_records]
    rejected_r   = [p["rejected_reward"] for p in pair_records]

    def bucket_margins(ms):
        b = {"0.00_0.05": 0, "0.05_0.10": 0, "0.10_0.20": 0, "0.20_0.30": 0, "0.30_plus": 0}
        for m in ms:
            if m < 0.05:   b["0.00_0.05"] += 1
            elif m < 0.10: b["0.05_0.10"] += 1
            elif m < 0.20: b["0.10_0.20"] += 1
            elif m < 0.30: b["0.20_0.30"] += 1
            else:          b["0.30_plus"] += 1
        return b

    summary = {
        "num_prompts":       len(items),
        "num_images":        len(all_samples),
        "num_pairs":         len(pair_records),
        "num_skipped":       n_skipped,
        "accept_all_pairs":  True,
        "reward_mode":       args.reward_mode,
        "mean_reward":       round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0,
        "mean_chosen_reward":  round(sum(chosen_r) / len(chosen_r), 4) if chosen_r else 0,
        "mean_rejected_reward": round(sum(rejected_r) / len(rejected_r), 4) if rejected_r else 0,
        "mean_margin":       round(sum(margins) / len(margins), 4) if margins else 0,
        "median_margin":     round(sorted(margins)[len(margins) // 2], 4) if margins else 0,
        "margin_buckets":    bucket_margins(margins),
        "mean_pair_weight":  round(sum(weights) / len(weights), 4) if weights else 0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[summary]")
    print(f"  pairs={summary['num_pairs']}  skipped={n_skipped}")
    print(f"  mean_reward={summary['mean_reward']}  mean_margin={summary['mean_margin']}")
    print(f"  margin_buckets={summary['margin_buckets']}")

    # audit HTML (attach PIL images for embedding)
    pil_cache = {}
    for s in all_samples:
        pil_cache[(s["id"], s["seed"])] = img_dir / Path(s["image_path"]).name

    for p in pair_records:
        from PIL import Image
        c_path  = out_dir / p["chosen_image_path"]
        rj_path = out_dir / p["rejected_image_path"]
        p["_chosen_pil"]   = Image.open(c_path)  if c_path.exists()  else None
        p["_rejected_pil"] = Image.open(rj_path) if rj_path.exists() else None

    make_audit_html(pair_records, out_dir / "audit.html")

    print(f"\n[gen] done → {out_dir}")


if __name__ == "__main__":
    main()
