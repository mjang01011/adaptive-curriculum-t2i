"""
Best-of-G reward ranker diagnostic.

For each validation prompt, generates G images with fixed seeds, scores each with
both the GRPO reward and hard_target, then reports whether the GRPO reward can
rank images by hard_target correctness.

Decision table:
  best_by_grpo beats random by >= +0.05  → reward ranking usable, proceed to training
  best_by_target_component beats random but best_by_grpo_total does not → stabilizers dilute
  neither beats random                   → reward misaligned; implement logit reward first
  oracle >> random but reward-best is not → good images exist but reward can't find them

Usage:
  python scripts/best_of_g_reward_ranker.py \
    --data-root data \
    --bucket attribute_binding \
    --num-prompts 20 \
    --num-generations 6 \
    --reward-mode pseudo_soft_grpo_target_heavy \
    --seeds 0 1 2 3 4 5 \
    --output-dir outputs/best_of_g_attribute_v3_debug \
    --repo-root LlamaGen \
    --gpt-ckpt /path/to/t2i_XL_stage1_256.pt \
    --vq-ckpt  /path/to/vq_ds16_t2i.pt \
    --t5-path  /path/to/t5-ckpt \
    --t5-cache-dir data/t5_cache
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-9 or dy < 1e-9:
        return 0.0
    return num / (dx * dy)


def _rank(xs: List[float]) -> List[int]:
    sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0] * len(xs)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


# ---------------------------------------------------------------------------
# Component aggregation
# ---------------------------------------------------------------------------

TARGET_QTYPES = {"relation", "attribute", "count"}
ANTI_QTYPES = {"anti_relation", "anti_swap", "anti_count"}
PRESENCE_QTYPES = {"object_presence"}
QUALITY_QTYPES = {"image_quality"}
ALIGNMENT_QTYPES = {"prompt_alignment"}


def _aggregate_components(question_scores: List[dict]) -> dict:
    buckets: Dict[str, List[float]] = {
        "target": [], "anti": [], "presence": [], "quality": [], "alignment": [], "other": []
    }
    for qs in question_scores:
        qt = qs.get("q_type", "")
        if qt in TARGET_QTYPES:
            buckets["target"].append(qs["score"])
        elif qt in ANTI_QTYPES:
            buckets["anti"].append(qs["score"])
        elif qt in PRESENCE_QTYPES:
            buckets["presence"].append(qs["score"])
        elif qt in QUALITY_QTYPES:
            buckets["quality"].append(qs["score"])
        elif qt in ALIGNMENT_QTYPES:
            buckets["alignment"].append(qs["score"])
        else:
            buckets["other"].append(qs["score"])
    return {
        k: (sum(v) / len(v)) if v else float("nan")
        for k, v in buckets.items()
    }


# ---------------------------------------------------------------------------
# Image grid
# ---------------------------------------------------------------------------

def _save_grid(prompt_id: str, images_info: List[dict], out_path: Path):
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
    except ImportError:
        return

    cell_w, cell_h, pad = 256, 296, 4
    n = len(images_info)
    grid_w = n * (cell_w + pad) + pad
    grid_h = cell_h + 2 * pad
    grid = PILImage.new("RGB", (grid_w, grid_h), (40, 40, 40))

    for i, info in enumerate(images_info):
        img_path = info.get("image_path", "")
        x0 = pad + i * (cell_w + pad)
        try:
            img = PILImage.open(img_path).convert("RGB").resize((cell_w, 256))
        except Exception:
            img = PILImage.new("RGB", (cell_w, 256), (80, 80, 80))
        grid.paste(img, (x0, pad))

        draw = ImageDraw.Draw(grid)
        label = (
            f"G={info.get('grpo_total_score', 0):.2f} "
            f"H={info.get('hard_target_score', 0):.2f} "
            f"Rg={info.get('grpo_rank', '?')} Rh={info.get('hard_rank', '?')}"
        )
        draw.text((x0 + 2, pad + 256 + 2), label, fill=(240, 240, 60))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=6)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--reward-mode", type=str, default="pseudo_soft_grpo_target_heavy")
    parser.add_argument("--hard-reward-mode", type=str, default="hard_target")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=str, required=True)
    # model args
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--vq-ckpt", type=str, default=None)
    parser.add_argument("--t5-path", type=str, default=None)
    parser.add_argument("--t5-cache-dir", type=str, default=None)
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--no-grids", action="store_true")
    args = parser.parse_args()

    # auto-detect repo root
    if args.repo_root is None:
        for cand in ["LlamaGen", "../LlamaGen", "/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen"]:
            if Path(cand).exists():
                args.repo_root = str(Path(cand).resolve())
                break
    if args.repo_root:
        sys.path.insert(0, args.repo_root)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    grids_dir = out_dir / "best_of_g_grids"
    if not args.no_grids:
        grids_dir.mkdir(exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    split_file = data_root / args.bucket / f"{args.bucket}_{args.split}_20.jsonl"
    if not split_file.exists():
        # try _500 for train
        split_file = data_root / args.bucket / f"{args.bucket}_train_500.jsonl"
    if not split_file.exists():
        print(f"[ERROR] Data file not found: {split_file}")
        sys.exit(1)

    from adaptive_curriculum.data.schemas import BucketItem
    all_items: List[BucketItem] = []
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_items.append(BucketItem.from_dict(json.loads(line)))
    items = all_items[: args.num_prompts]
    print(f"[best_of_g] bucket={args.bucket}  prompts={len(items)}  G={args.num_generations}")

    # ── Load reward model ────────────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(model_id=args.vlm_model)

    # ── Load T5 cache ────────────────────────────────────────────────────────
    t5_cache = None
    if args.t5_cache_dir:
        from adaptive_curriculum.data.t5_cache import load_t5_cache
        t5_cache = load_t5_cache(args.t5_cache_dir, [args.bucket])

    # ── Load generative model ────────────────────────────────────────────────
    model = None
    if args.gpt_ckpt and args.vq_ckpt and args.t5_path:
        from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
        model = LlamaGenWrapper(
            repo_root=args.repo_root,
            vq_ckpt=args.vq_ckpt,
            gpt_ckpt=args.gpt_ckpt,
            gpt_model="GPT-XL",
            image_size=256,
            t5_path=args.t5_path,
            t5_model_type="flan-t5-xl",
            t5_feature_max_len=120,
            cfg_scale=args.cfg_scale,
            cfg_scale_train=args.cfg_scale,
            precision="bf16",
            use_lora=False,
        )
        print("[best_of_g] Model loaded (no LoRA, base weights).")
    else:
        print("[best_of_g] WARNING: no model args provided — using dummy scores (0.5 everywhere).")

    seeds = args.seeds[: args.num_generations]

    # ── Per-prompt scoring ───────────────────────────────────────────────────
    all_rows: List[dict] = []
    hard_first_list, hard_random_list = [], []
    hard_best_grpo_list, hard_best_target_comp_list, hard_oracle_list = [], [], []

    for item in items:
        prompt_rows = []
        cached_embs = t5_cache.bucket_embeddings(args.bucket) if t5_cache else None

        for k, seed in enumerate(seeds):
            if model is not None:
                img_paths = model.generate_images(
                    prompts=[item.text],
                    out_dir=str(samples_dir),
                    prompt_ids=[f"{item.id}_s{k}"],
                    bucket_names=[item.bucket],
                    num_samples_per_prompt=1,
                    seed=seed,
                    cached_embeddings=cached_embs,
                )
                img_path = img_paths[0] if img_paths else ""
            else:
                img_path = ""

            from PIL import Image as PILImage
            try:
                pil_img = PILImage.open(img_path).convert("RGB") if img_path else None
            except Exception:
                pil_img = None

            # GRPO reward
            if pil_img is not None:
                grpo_result = reward_model.score_image(pil_img, item, mode=args.reward_mode)
                hard_result = reward_model.score_image(pil_img, item, mode=args.hard_reward_mode)
            else:
                grpo_result = {"score": 0.5, "question_scores": [], "component_scores": {}}
                hard_result = {"score": 0.5, "question_scores": []}

            grpo_q_scores = grpo_result.get("question_scores", [])
            hard_q_scores = hard_result.get("question_scores", [])
            comps = _aggregate_components(grpo_q_scores)
            uncertain_grpo = sum(1 for q in grpo_q_scores if q.get("predicted") == "uncertain")
            uncertain_hard = sum(1 for q in hard_q_scores if q.get("predicted") == "uncertain")

            row = {
                "bucket": args.bucket,
                "prompt_id": item.id,
                "prompt": item.text,
                "sample_index": k,
                "seed": seed,
                "image_path": img_path,
                "grpo_total_score": float(grpo_result["score"]),
                "hard_target_score": float(hard_result["score"]),
                "target_component_score": comps.get("target", float("nan")),
                "presence_component_score": comps.get("presence", float("nan")),
                "anti_component_score": comps.get("anti", float("nan")),
                "quality_component_score": comps.get("quality", float("nan")),
                "alignment_component_score": comps.get("alignment", float("nan")),
                "num_grpo_questions": len(grpo_q_scores),
                "num_target_questions": len(hard_q_scores),
                "uncertain_count_grpo": uncertain_grpo,
                "uncertain_count_target": uncertain_hard,
                "grpo_answers": [
                    {"question": q["question"], "expected": q["expected"],
                     "answer": q.get("predicted", ""), "score": q["score"], "weight": q.get("weight", 0)}
                    for q in grpo_q_scores
                ],
                "target_answers": [
                    {"question": q["question"], "expected": q["expected"],
                     "answer": q.get("predicted", ""), "score": q["score"]}
                    for q in hard_q_scores
                ],
            }
            prompt_rows.append(row)
            all_rows.append(row)

        if not prompt_rows:
            continue

        grpo_scores = [r["grpo_total_score"] for r in prompt_rows]
        hard_scores = [r["hard_target_score"] for r in prompt_rows]
        target_comp_scores = [r["target_component_score"] for r in prompt_rows]

        # rank annotations
        grpo_order = sorted(range(len(grpo_scores)), key=lambda i: -grpo_scores[i])
        hard_order = sorted(range(len(hard_scores)), key=lambda i: -hard_scores[i])
        for idx, r in enumerate(prompt_rows):
            r["grpo_rank"] = grpo_order.index(idx) + 1
            r["hard_rank"] = hard_order.index(idx) + 1

        best_grpo_idx = grpo_order[0]
        best_target_comp_idx = sorted(range(len(target_comp_scores)), key=lambda i: -target_comp_scores[i])[0]
        random_idx = random.randint(0, len(prompt_rows) - 1)
        oracle_idx = hard_order[0]

        hard_first_list.append(hard_scores[0])
        hard_random_list.append(hard_scores[random_idx])
        hard_best_grpo_list.append(hard_scores[best_grpo_idx])
        hard_best_target_comp_list.append(hard_scores[best_target_comp_idx])
        hard_oracle_list.append(hard_scores[oracle_idx])

        if not args.no_grids:
            _save_grid(item.id, prompt_rows, grids_dir / f"{item.id}_grid.png")

        print(
            f"  {item.id:30s}  "
            f"first={hard_scores[0]:.2f}  "
            f"best_grpo={hard_scores[best_grpo_idx]:.2f}  "
            f"oracle={hard_scores[oracle_idx]:.2f}"
        )

    # ── Correlations ─────────────────────────────────────────────────────────
    all_grpo = [r["grpo_total_score"] for r in all_rows]
    all_hard = [r["hard_target_score"] for r in all_rows]
    all_target_comp = [r["target_component_score"] for r in all_rows if not math.isnan(r["target_component_score"])]
    all_hard_for_target = [r["hard_target_score"] for r in all_rows if not math.isnan(r["target_component_score"])]

    pearson_total = _pearson(all_grpo, all_hard)
    spearman_total = _spearman(all_grpo, all_hard)
    pearson_target = _pearson(all_target_comp, all_hard_for_target)
    spearman_target = _spearman(all_target_comp, all_hard_for_target)

    n = len(items)
    mean_first = sum(hard_first_list) / n
    mean_random = sum(hard_random_list) / n
    mean_best_grpo = sum(hard_best_grpo_list) / n
    mean_best_target_comp = sum(hard_best_target_comp_list) / n
    mean_oracle = sum(hard_oracle_list) / n

    delta_grpo_vs_first = mean_best_grpo - mean_first
    delta_grpo_vs_random = mean_best_grpo - mean_random
    delta_target_comp_vs_first = mean_best_target_comp - mean_first

    uncertain_grpo_total = sum(r["uncertain_count_grpo"] for r in all_rows)
    uncertain_hard_total = sum(r["uncertain_count_target"] for r in all_rows)
    total_grpo_q = sum(r["num_grpo_questions"] for r in all_rows) or 1
    total_hard_q = sum(r["num_target_questions"] for r in all_rows) or 1

    # ── Decision ─────────────────────────────────────────────────────────────
    MIN_DELTA = 0.05 if args.bucket != "spatial_relations_anchored" else 0.03
    success = delta_grpo_vs_random >= MIN_DELTA
    if delta_grpo_vs_random < MIN_DELTA and delta_grpo_vs_random >= 0:
        verdict = "WEAK — reward barely ranks; consider logit reward before training"
    elif delta_grpo_vs_random < 0:
        verdict = "FAIL — reward anti-correlates with hard target; fix reward before training"
    else:
        verdict = "PASS — reward ranks usably; proceed to training"

    if mean_best_target_comp > mean_random and mean_best_grpo <= mean_random + 0.02:
        verdict += " | NOTE: target_component ranks better than total — stabilizers diluting reward"

    summary = {
        "bucket": args.bucket,
        "reward_mode": args.reward_mode,
        "num_prompts": n,
        "num_generations": args.num_generations,
        "mean_hard_first": round(mean_first, 4),
        "mean_hard_random": round(mean_random, 4),
        "mean_hard_best_by_grpo_total": round(mean_best_grpo, 4),
        "mean_hard_best_by_target_component": round(mean_best_target_comp, 4),
        "mean_hard_oracle": round(mean_oracle, 4),
        "delta_best_grpo_vs_first": round(delta_grpo_vs_first, 4),
        "delta_best_grpo_vs_random": round(delta_grpo_vs_random, 4),
        "delta_best_target_component_vs_first": round(delta_target_comp_vs_first, 4),
        "pearson_grpo_total_vs_hard": round(pearson_total, 4),
        "spearman_grpo_total_vs_hard": round(spearman_total, 4),
        "pearson_target_component_vs_hard": round(pearson_target, 4),
        "spearman_target_component_vs_hard": round(spearman_target, 4),
        "uncertain_rate_grpo": round(uncertain_grpo_total / total_grpo_q, 4),
        "uncertain_rate_target": round(uncertain_hard_total / total_hard_q, 4),
        "success": success,
        "verdict": verdict,
    }

    # ── Save outputs ─────────────────────────────────────────────────────────
    jsonl_path = out_dir / "best_of_g_samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    summary_path = out_dir / "best_of_g_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Best-of-G Reward Ranker — {args.bucket}")
    print("=" * 60)
    print(f"  hard(first):            {mean_first:.4f}")
    print(f"  hard(random):           {mean_random:.4f}")
    print(f"  hard(best_by_grpo):     {mean_best_grpo:.4f}  Δ_vs_random={delta_grpo_vs_random:+.4f}")
    print(f"  hard(best_by_target):   {mean_best_target_comp:.4f}  Δ_vs_first={delta_target_comp_vs_first:+.4f}")
    print(f"  hard(oracle):           {mean_oracle:.4f}")
    print(f"  pearson(grpo, hard):    {pearson_total:.4f}  spearman={spearman_total:.4f}")
    print(f"  pearson(target, hard):  {pearson_target:.4f}  spearman={spearman_target:.4f}")
    print(f"  uncertain_rate (grpo):  {uncertain_grpo_total/total_grpo_q:.3f}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 60)
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
