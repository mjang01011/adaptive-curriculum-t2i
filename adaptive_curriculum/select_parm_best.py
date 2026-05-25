"""
Select the best-of-N candidate per prompt using PARM yes_prob scores.

Input: scored JSONL from parm_score_llamagen.py
Output:
  <output-dir>/selected_images/<id>_parm_best.png
  <output-dir>/selected.jsonl

Usage:
  python adaptive_curriculum/select_parm_best.py \
    --scores-jsonl  outputs/parm_scores/scores.jsonl \
    --output-dir    outputs/parm_reranked
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-jsonl", required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--id-key",       default="id")
    args = p.parse_args()

    rows = load_jsonl(args.scores_jsonl)
    groups: dict = defaultdict(list)
    for row in rows:
        groups[row[args.id_key]].append(row)

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "selected_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Use norm_yes_prob (yes/(yes+no)) as selection key — denser signal than raw yes_prob.
    # Falls back to parm_yes_prob for backwards-compat with older score files.
    def _score(r):
        if "parm_norm_yes_prob" in r:
            return r["parm_norm_yes_prob"]
        return r.get("parm_yes_prob", 0.0)

    selected_rows = []
    n_all_zero = 0

    for pid, group in groups.items():
        best = max(group, key=_score)
        if all(_score(r) == 0.0 for r in group):
            n_all_zero += 1

        src = Path(best["image_path"])
        dst = img_dir / f"{pid}_parm_best.png"
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"[warn] source image missing: {src}")

        out = dict(best)
        out["selected_image_path"] = str(dst)
        out["num_candidates"] = len(group)
        out["mean_norm_yes_prob"] = sum(_score(r) for r in group) / len(group)
        out["max_norm_yes_prob"]  = _score(best)
        selected_rows.append(out)

    with open(out_dir / "selected.jsonl", "w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row) + "\n")

    mean_score = sum(r["max_norm_yes_prob"] for r in selected_rows) / max(len(selected_rows), 1)
    print(f"Selected {len(selected_rows)} images → {img_dir}")
    print(f"Mean best norm_yes_prob: {mean_score:.3f}")
    if n_all_zero:
        print(f"[warn] {n_all_zero} prompts had all zero scores (scoring may have failed)")


if __name__ == "__main__":
    main()
