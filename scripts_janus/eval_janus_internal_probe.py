"""
Janus-Pro-1B base internal probe — two-phase: generate then score.

Phase 1 (januspro_venv): generate images
  python3 scripts_janus/eval_janus_internal_probe.py \
    --phase generate \
    --data-root $PROJ/data --bucket attribute_binding \
    --num-prompts 8 --seeds 0 1 2 3 \
    --images-dir $JANUS/outputs_janus_grpo/probe_images

Phase 2 (svl env): score images
  python3 scripts_janus/eval_janus_internal_probe.py \
    --phase score \
    --data-root $PROJ/data --bucket attribute_binding \
    --num-prompts 8 --seeds 0 1 2 3 \
    --images-dir $JANUS/outputs_janus_grpo/probe_images \
    --out $JANUS/outputs_janus_grpo/janus_attribute_base_probe.json
"""
import argparse
import json
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


def img_path(images_dir, prompt_idx, seed):
    return Path(images_dir) / f"prompt{prompt_idx:03d}_seed{seed}.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",          choices=["generate", "score", "both"], default="both")
    parser.add_argument("--data-root",      required=True)
    parser.add_argument("--bucket",         default="attribute_binding")
    parser.add_argument("--num-prompts",    type=int, default=8)
    parser.add_argument("--seeds",          type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--model-path",     default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--cfg-weight",     type=float, default=5.0)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--images-dir",     required=True)
    parser.add_argument("--out",            default=None,
                        help="Output JSON path (required for score/both phase)")
    parser.add_argument("--reward-model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from adaptive_curriculum.data.schemas import BucketItem

    # ── load data ──────────────────────────────────────────────────────
    val_file = Path(args.data_root) / args.bucket / f"{args.bucket}_val_20.jsonl"
    if not val_file.exists():
        raise FileNotFoundError(f"Val file not found: {val_file}")
    raw = load_jsonl(val_file)
    items = [BucketItem.from_dict(d) for d in raw[:args.num_prompts]]
    print(f"[probe] {len(items)} items, seeds={args.seeds}, phase={args.phase}")

    images_dir = Path(args.images_dir)

    # ── phase: generate ────────────────────────────────────────────────
    if args.phase in ("generate", "both"):
        images_dir.mkdir(parents=True, exist_ok=True)
        from scripts_janus.janus_wrapper import JanusProWrapper
        wrapper = JanusProWrapper(
            model_path=args.model_path,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
        _ = wrapper.model
        print("[probe] Janus loaded. Generating...")

        prompts = [item.text for item in items]
        for seed in args.seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            out = wrapper.generate_images(prompts, seeds=None)
            for i, pil_img in enumerate(out["images"]):
                pil_img.save(img_path(images_dir, i, seed))
            print(f"  seed {seed} done")
        print(f"[probe] Images saved → {images_dir}")

    # ── phase: score ───────────────────────────────────────────────────
    if args.phase in ("score", "both"):
        if args.out is None:
            raise ValueError("--out is required for score/both phase")

        import PIL.Image
        from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
        reward_model = Qwen3VLRewardModel(model_id=args.reward_model_id)
        print("[probe] Scoring images...")

        hard_scores = []
        comp_sums = {}
        comp_counts = {}
        all_records = []

        for seed in args.seeds:
            for i, item in enumerate(items):
                path = img_path(images_dir, i, seed)
                if not path.exists():
                    raise FileNotFoundError(f"Image not found: {path} — run generate phase first")
                pil_img = PIL.Image.open(path).convert("RGB")
                result = reward_model.score_image(pil_img, item, mode="hard_target")
                score = float(result["score"])
                hard_scores.append(score)
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
            print(f"  seed {seed} scored")

        n    = len(hard_scores)
        mean = sum(hard_scores) / n
        se   = (sum((s - mean) ** 2 for s in hard_scores) / (n * (n - 1))) ** 0.5 if n > 1 else 0.0
        component_means = {qt: round(comp_sums[qt] / comp_counts[qt], 4) for qt in comp_sums}

        summary = {
            "bucket": args.bucket,
            "model": args.model_path,
            "num_prompts": len(items),
            "seeds": args.seeds,
            "n": n,
            "mean": round(mean, 4),
            "se": round(se, 4),
            "component_means": component_means,
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
