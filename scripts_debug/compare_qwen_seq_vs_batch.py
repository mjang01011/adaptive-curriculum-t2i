"""
Parity test: sequential score_image vs batched score_images_batch.

Uses already-saved images from a probe run — no generation needed.

Usage:
  python scripts_debug/compare_qwen_seq_vs_batch.py \
    --train-jsonl  $PROJ/data/attribute_binding/attribute_binding_train_500.jsonl \
    --images-dir   $PROJ/outputs_reward_probe/attr_v2 \
    --modes        grpo_attr_presence_gated_v2 hard_target \
    --batch-sizes  1 4 8 \
    --num-items    20 \
    --out          $PROJ/outputs_debug/qwen_batch_parity

Images are expected at: {images_dir}/p{pi:03d}_s{s:02d}_{item_id}.png
Item ordering must match the probe run (same seed, same jsonl).
Alternatively pass --details-jsonl to load exact items/image paths.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl",    required=True)
    p.add_argument("--images-dir",     required=True, help="Probe output dir containing saved PNGs")
    p.add_argument("--modes",          nargs="+", default=["grpo_attr_presence_gated_v2", "hard_target"])
    p.add_argument("--batch-sizes",    nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--num-items",      type=int, default=20)
    p.add_argument("--sample-idx",     type=int, default=0, help="Which probe sample (s) to use for images")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--qwen-model",     default=None)
    p.add_argument("--out",            required=True)
    return p.parse_args()


def load_items(jsonl_path, n, seed):
    import random
    from adaptive_curriculum.data.schemas import BucketItem
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(BucketItem.from_dict(json.loads(line)))
    rng = random.Random(seed)
    return rng.sample(items, min(n, len(items)))


def find_image(images_dir: Path, pi: int, s: int, item_id: str):
    """Match probe naming: p{pi:03d}_s{s:02d}_{item_id}.png"""
    path = images_dir / f"p{pi:03d}_s{s:02d}_{item_id}.png"
    if path.exists():
        return path
    # Fallback: glob by item_id
    matches = list(images_dir.glob(f"*_{item_id}.png"))
    return matches[0] if matches else None


def scores_equal(a: dict, b: dict, tol: float = 1e-6) -> tuple:
    """
    Returns (ok: bool, issues: list[str]).
    Checks schema, score, component_scores, question_scores labels.
    """
    issues = []

    # Schema
    if a.keys() != b.keys():
        issues.append(f"key mismatch: {a.keys()} vs {b.keys()}")

    # Score
    da = abs(a["score"] - b["score"])
    if da > tol:
        issues.append(f"score diff={da:.6f}  seq={a['score']:.4f}  bat={b['score']:.4f}")

    # Component scores
    ac = a.get("component_scores", {})
    bc = b.get("component_scores", {})
    if ac.keys() != bc.keys():
        issues.append(f"component key mismatch: {ac.keys()} vs {bc.keys()}")
    for k in ac:
        if k in bc:
            dc = abs(ac[k] - bc[k])
            if dc > tol:
                issues.append(f"component[{k}] diff={dc:.6f}")

    # Question-level labels (predicted answers)
    aqs = a.get("question_scores", [])
    bqs = b.get("question_scores", [])
    if len(aqs) != len(bqs):
        issues.append(f"question_scores length mismatch: {len(aqs)} vs {len(bqs)}")
    else:
        for qi, (aq, bq) in enumerate(zip(aqs, bqs)):
            if aq["predicted"] != bq["predicted"]:
                issues.append(
                    f"q{qi} predicted mismatch: seq={aq['predicted']} bat={bq['predicted']}"
                    f" | q={aq['question'][:40]}"
                )

    return len(issues) == 0, issues


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent))

    items = load_items(args.train_jsonl, args.num_items, args.seed)
    images_dir = Path(args.images_dir)
    s = args.sample_idx

    print(f"[parity] {len(items)} items  sample_idx={s}  modes={args.modes}  batch_sizes={args.batch_sizes}")

    # Load images
    pil_imgs = []
    valid_items = []
    from PIL import Image
    for pi, item in enumerate(items):
        img_path = find_image(images_dir, pi, s, item.id)
        if img_path is None:
            print(f"  [warn] no image found for pi={pi} item={item.id}, skipping")
            continue
        pil_imgs.append(Image.open(str(img_path)).convert("RGB"))
        valid_items.append(item)

    print(f"[parity] {len(valid_items)} images found on disk")
    if not valid_items:
        print("No images found — check --images-dir and --sample-idx")
        return

    # Load reward model
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(
        model_id=args.qwen_model or "Qwen/Qwen3-VL-4B-Instruct"
    )

    results = {}  # mode -> batch_size -> list of (ok, issues)

    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")
        results[mode] = {}

        # Sequential baseline
        print("  Running sequential baseline...")
        seq_scores = []
        for img, item in zip(pil_imgs, valid_items):
            seq_scores.append(reward_model.score_image(img, item, mode=mode))

        for bs in args.batch_sizes:
            print(f"  Testing batch_size={bs}...")
            pairs = list(zip(pil_imgs, valid_items))

            # Run batched in chunks of bs
            bat_scores = []
            for start in range(0, len(pairs), bs):
                chunk = pairs[start:start + bs]
                bat_scores.extend(reward_model.score_images_batch(chunk, mode=mode))

            # Compare
            ok_count = 0
            fail_count = 0
            all_issues = []
            score_diffs = []
            label_flips = 0

            for i, (seq, bat) in enumerate(zip(seq_scores, bat_scores)):
                ok, issues = scores_equal(seq, bat, tol=1e-6)
                score_diffs.append(abs(seq["score"] - bat["score"]))
                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    for iss in issues:
                        if "predicted mismatch" in iss:
                            label_flips += 1
                    all_issues.append({
                        "item_idx": i,
                        "item_id": valid_items[i].id,
                        "issues": issues,
                    })

            mean_diff = sum(score_diffs) / max(len(score_diffs), 1)
            max_diff  = max(score_diffs) if score_diffs else 0.0

            status = "PASS" if fail_count == 0 else "FAIL"
            print(f"    batch_size={bs}: {status}  "
                  f"exact_match={ok_count}/{len(seq_scores)}  "
                  f"label_flips={label_flips}  "
                  f"mean_score_diff={mean_diff:.6f}  "
                  f"max_score_diff={max_diff:.6f}")

            if all_issues:
                print(f"    First failure:")
                fi = all_issues[0]
                print(f"      item={fi['item_id']}")
                for iss in fi["issues"][:3]:
                    print(f"      {iss}")

            results[mode][bs] = {
                "ok_count":      ok_count,
                "fail_count":    fail_count,
                "label_flips":   label_flips,
                "mean_score_diff": mean_diff,
                "max_score_diff":  max_diff,
                "failures":      all_issues,
            }

    # Save
    with open(out_dir / "parity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for mode in args.modes:
        for bs in args.batch_sizes:
            r = results[mode][bs]
            ok = r["fail_count"] == 0
            all_pass = all_pass and ok
            print(f"  {mode}  bs={bs:2d}  {'PASS' if ok else 'FAIL'}  "
                  f"label_flips={r['label_flips']}  "
                  f"mean_diff={r['mean_score_diff']:.2e}")

    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES — check parity_results.json'}")
    print(f"  Results saved → {out_dir}/parity_results.json")


if __name__ == "__main__":
    main()
