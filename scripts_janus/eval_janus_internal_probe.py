"""
Janus-Pro-1B base internal probe.
Generates images for N prompts × K seeds, scores with VLM reward, reports summary.

Usage:
  python scripts_janus/eval_janus_internal_probe.py \
    --data-root /viscam/u/jj277/adaptive-curriculum-t2i/data \
    --bucket attribute_binding \
    --num-prompts 8 \
    --seeds 0 1 2 3 \
    --model-path deepseek-ai/Janus-Pro-1B \
    --cfg-weight 5.0 \
    --temperature 1.0 \
    --out /viscam/u/jj277/janus_project/outputs_janus_grpo/janus_attribute_base_probe.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",    required=True)
    parser.add_argument("--bucket",       default="attribute_binding")
    parser.add_argument("--num-prompts",  type=int, default=8)
    parser.add_argument("--seeds",        type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--model-path",   default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--cfg-weight",   type=float, default=5.0)
    parser.add_argument("--temperature",  type=float, default=1.0)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--reward-model-path", default=None,
                        help="Path or HF name for Qwen VLM reward model")
    args = parser.parse_args()

    # ── load data ──────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    val_file = data_root / args.bucket / f"{args.bucket}_val_20.jsonl"
    if not val_file.exists():
        raise FileNotFoundError(f"Val file not found: {val_file}")

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from adaptive_curriculum.data.schemas import BucketItem
    raw = load_jsonl(val_file)
    items = [BucketItem.from_dict(d) for d in raw[:args.num_prompts]]
    print(f"[probe] Loaded {len(items)} items from {val_file}")

    # ── load reward model ──────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(model_path=args.reward_model_path)
    print("[probe] Reward model loaded.")

    # ── load Janus ─────────────────────────────────────────────────────
    from scripts_janus.janus_wrapper import JanusProWrapper
    wrapper = JanusProWrapper(
        model_path=args.model_path,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
    )
    _ = wrapper.model  # trigger load
    print("[probe] Janus model loaded.")

    # ── generate & score ───────────────────────────────────────────────
    prompts = [item.text for item in items]
    all_records = []
    hard_scores = []

    # component accumulators
    comp_sums = {}
    comp_counts = {}

    for seed in args.seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        out = wrapper.generate_images(prompts, seeds=None, return_tokens=False, return_logprobs=False)
        images = out["images"]

        for item, pil_img in zip(items, images):
            result = reward_model.score_image(pil_img, item, mode="hard_target")
            score = float(result["score"])
            hard_scores.append(score)

            # collect component scores
            for qs in result.get("question_scores", []):
                qt = qs.get("q_type", qs.get("type", "unknown"))
                comp_sums[qt]   = comp_sums.get(qt, 0.0) + float(qs.get("correct", 0))
                comp_counts[qt] = comp_counts.get(qt, 0) + 1

            all_records.append({
                "prompt_id": item.id,
                "prompt": item.text,
                "seed": seed,
                "hard_score": score,
                "question_scores": result.get("question_scores", []),
            })

    n = len(hard_scores)
    mean = sum(hard_scores) / n
    se   = (sum((s - mean) ** 2 for s in hard_scores) / (n * (n - 1))) ** 0.5 if n > 1 else 0.0

    component_means = {qt: comp_sums[qt] / comp_counts[qt] for qt in comp_sums}

    summary = {
        "bucket": args.bucket,
        "model": args.model_path,
        "num_prompts": len(items),
        "seeds": args.seeds,
        "n": n,
        "mean": round(mean, 4),
        "se": round(se, 4),
        "component_means": {k: round(v, 4) for k, v in component_means.items()},
        "records": all_records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[probe] mean={mean:.4f}  se={se:.4f}  n={n}")
    print(f"[probe] components: {component_means}")
    print(f"[probe] Saved → {out_path}")


if __name__ == "__main__":
    main()
