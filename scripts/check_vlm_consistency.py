"""
Check Qwen VLM answer stability: ask the same questions twice for the same images
and measure agreement rate. Low agreement = noisy reward signal.

Usage:
  python scripts/check_vlm_consistency.py \
    --image-dir  outputs/<run>/evals/step_000000/spatial_relations/generations \
    --data-root  /path/to/data \
    --bucket     spatial_relations \
    --num-images 50 \
    --out        outputs/<run>/vlm_consistency.json
"""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir",  required=True,
                        help="Directory containing generated .png images")
    parser.add_argument("--data-root",  required=True)
    parser.add_argument("--bucket",     required=True)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--out",        required=True)
    # optional: path to run config for reward model settings
    parser.add_argument("--run-dir",    default=None)
    args = parser.parse_args()

    # load reward model
    if args.run_dir:
        from omegaconf import OmegaConf
        config = OmegaConf.load(str(Path(args.run_dir) / "config_resolved.yaml"))
    else:
        # minimal config for reward model
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            "reward": {"model": "Qwen/Qwen2-VL-7B-Instruct", "device": "cuda"}
        })

    from adaptive_curriculum.reward.vlm_reward import build_reward_model
    reward_model = build_reward_model(config)

    # load val items to get target_questions
    from adaptive_curriculum.data.bucket_dataset import load_bucket_datasets
    datasets = load_bucket_datasets(
        data_root=args.data_root,
        bucket_names=[args.bucket],
    )
    val_items = datasets[args.bucket].val_items
    items_by_id = {item.id: item for item in val_items}

    # find images
    img_dir = Path(args.image_dir)
    all_images = sorted(img_dir.glob("*.png"))
    if not all_images:
        print(f"No .png files found in {img_dir}")
        return

    random.seed(args.seed)
    sampled = random.sample(all_images, min(args.num_images, len(all_images)))
    print(f"[consistency] Testing {len(sampled)} images from {img_dir}")

    results = []
    agree_scores = []
    agree_answers = []

    for img_path in sampled:
        # try to find matching item from filename
        stem = img_path.stem  # e.g. "attr_001_s0"
        item = None
        for candidate_id in items_by_id:
            if candidate_id in stem:
                item = items_by_id[candidate_id]
                break
        if item is None:
            # fallback: use first val item's questions (for structure)
            item = val_items[0]

        # run VLM twice
        answers_1 = reward_model.answer_all_questions_once(str(img_path), item.target_questions)
        answers_2 = reward_model.answer_all_questions_once(str(img_path), item.target_questions)

        # score both
        result_1 = reward_model.score_image(str(img_path), item, mode="hard_target")
        result_2 = reward_model.score_image(str(img_path), item, mode="hard_target")

        answer_match = sum(a1 == a2 for a1, a2 in zip(answers_1, answers_2)) / max(len(answers_1), 1)
        score_match = float(result_1["score"] == result_2["score"])

        agree_answers.append(answer_match)
        agree_scores.append(score_match)

        results.append({
            "image": str(img_path),
            "answers_1": answers_1,
            "answers_2": answers_2,
            "score_1": result_1["score"],
            "score_2": result_2["score"],
            "answer_agreement": answer_match,
            "score_agreement": score_match,
        })

    mean_answer_agree = sum(agree_answers) / len(agree_answers)
    mean_score_agree = sum(agree_scores) / len(agree_scores)

    uncertain_rate = sum(
        1 for r in results
        if "uncertain" in r["answers_1"] or "uncertain" in r["answers_2"]
    ) / len(results)

    print(f"\n{'='*50}")
    print(f"VLM Consistency — bucket: {args.bucket}")
    print(f"{'='*50}")
    print(f"  images tested     : {len(results)}")
    print(f"  answer agreement  : {mean_answer_agree:.2%}  "
          f"{'✓ stable' if mean_answer_agree > 0.85 else '⚠ moderate' if mean_answer_agree > 0.70 else '✗ unstable'}")
    print(f"  score agreement   : {mean_score_agree:.2%}")
    print(f"  uncertain rate    : {uncertain_rate:.2%}")

    interp = (
        "STABLE — VLM answers are consistent; reward noise is low"
        if mean_answer_agree > 0.85 else
        "MODERATE — some VLM instability; reward has noise but may be usable"
        if mean_answer_agree > 0.70 else
        "UNSTABLE — VLM answers vary significantly; reward signal is unreliable"
    )
    print(f"\n  Interpretation: {interp}")

    out = {
        "bucket": args.bucket,
        "n_images": len(results),
        "mean_answer_agreement": mean_answer_agree,
        "mean_score_agreement": mean_score_agree,
        "uncertain_rate": uncertain_rate,
        "interpretation": interp,
        "per_image": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
