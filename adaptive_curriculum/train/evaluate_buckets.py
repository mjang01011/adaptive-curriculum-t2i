"""
Bucket evaluation: generate images for validation prompts and score them.
"""
import json
import statistics
from pathlib import Path
from typing import List, Dict, Optional

from adaptive_curriculum.data.schemas import BucketItem
from adaptive_curriculum.utils.jsonl import write_json


def evaluate_bucket(
    model,
    reward_model,
    val_items: List[BucketItem],
    out_dir: str,
    num_samples_per_prompt: int = 1,
    seed: Optional[int] = None,
    t5_cache=None,
    reward_mode: str = "hard_target",
) -> dict:
    """
    Generate images for validation items, score them, return bucket-level summary.
    Works in no-GPU mode when model is None (uses heuristic reward with fake paths).
    """
    out_path = Path(out_dir)
    gen_dir = out_path / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    bucket_name = val_items[0].bucket if val_items else "unknown"

    # generate images (or use fake paths for no-GPU testing)
    if model is not None:
        prompt_ids = [item.id for item in val_items]
        prompts = [item.text for item in val_items]
        bucket_names = [item.bucket for item in val_items]
        # use pre-extracted embeddings if available (eliminates T5 from eval hot path)
        cached_embs = None
        if t5_cache is not None:
            bucket_name = val_items[0].bucket
            cached_embs = t5_cache.bucket_embeddings(bucket_name)

        image_paths = model.generate_images(
            prompts=prompts,
            out_dir=str(gen_dir),
            prompt_ids=prompt_ids,
            bucket_names=bucket_names,
            num_samples_per_prompt=num_samples_per_prompt,
            seed=seed,
            cached_embeddings=cached_embs,
        )
    else:
        # no-GPU mode: use placeholder paths for heuristic reward
        image_paths = []
        for item in val_items:
            for k in range(num_samples_per_prompt):
                image_paths.append(str(gen_dir / f"{item.bucket}_{item.id}_sample{k}.png"))

    # group image paths by item
    # image_paths order: [item0_sample0, item0_sample1, ..., item1_sample0, ...]
    # but generate_images returns [item0_s0, item1_s0, ..., itemN_s0, item0_s1, ...]
    # let's handle both orderings by re-mapping
    n_items = len(val_items)
    n_samples = num_samples_per_prompt

    rewards_records = []
    prompt_rewards = []
    per_question_correct: Dict[str, List[bool]] = {}
    per_qtype_correct: Dict[str, List[bool]] = {}
    sample_image_paths: List[str] = []  # up to 4 images for W&B logging

    for item_idx, item in enumerate(val_items):
        item_image_paths = []
        for k in range(n_samples):
            path_idx = k * n_items + item_idx
            if path_idx < len(image_paths):
                item_image_paths.append(image_paths[path_idx])

        image_scores = []
        for img_path in item_image_paths:
            result = reward_model.score_image(img_path, item, mode=reward_mode)
            image_scores.append(result["score"])
            for q_result in result.get("question_scores", []):
                q = q_result["question"]
                q_type = q_result.get("q_type", "unknown")
                correct = q_result["correct"]
                per_question_correct.setdefault(q, []).append(correct)
                per_qtype_correct.setdefault(q_type, []).append(correct)
            rewards_records.append({
                "item_id": item.id,
                "bucket": item.bucket,
                "prompt": item.text,
                "image_path": img_path,
                "score": result["score"],
                "question_scores": result.get("question_scores", []),
            })
            if len(sample_image_paths) < 4:
                sample_image_paths.append(img_path)

        prompt_reward = sum(image_scores) / len(image_scores) if image_scores else 0.0
        prompt_rewards.append(prompt_reward)

    rewards_path = out_path / "rewards.jsonl"
    with open(rewards_path, "w", encoding="utf-8") as f:
        for r in rewards_records:
            f.write(json.dumps(r) + "\n")

    mean_reward = sum(prompt_rewards) / len(prompt_rewards) if prompt_rewards else 0.0
    std_reward = statistics.stdev(prompt_rewards) if len(prompt_rewards) > 1 else 0.0

    per_q_accuracy = {q: sum(v) / len(v) for q, v in per_question_correct.items()}
    per_qtype_accuracy = {qt: sum(v) / len(v) for qt, v in per_qtype_correct.items()}

    summary = {
        "bucket": bucket_name,
        "mean_raw_reward": mean_reward,
        "std_raw_reward": std_reward,
        "num_prompts": len(val_items),
        "num_images": len(rewards_records),
        "per_question_accuracy": per_q_accuracy,
        "per_qtype_accuracy": per_qtype_accuracy,
        "sample_image_paths": sample_image_paths,
        "reward_distribution": prompt_rewards,
    }
    write_json(str(out_path / "summary.json"), summary)
    return summary


def evaluate_all_buckets(
    model,
    reward_model,
    datasets: dict,
    out_dir: str,
    curriculum_step: int,
    num_samples_per_prompt: int = 1,
    seed: Optional[int] = None,
    t5_cache=None,
    reward_mode: str = "hard_target",
) -> Dict[str, dict]:
    results = {}
    for bucket_name, dataset in datasets.items():
        bucket_out = Path(out_dir) / f"step_{curriculum_step:06d}" / bucket_name
        summary = evaluate_bucket(
            model=model,
            reward_model=reward_model,
            val_items=dataset.val_items,
            out_dir=str(bucket_out),
            num_samples_per_prompt=num_samples_per_prompt,
            seed=seed,
            t5_cache=t5_cache,
            reward_mode=reward_mode,
        )
        results[bucket_name] = summary
    return results
