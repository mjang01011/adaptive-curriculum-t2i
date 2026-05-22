"""
Summarize T2I-CompBench evaluation results across categories.

Reads each category's vqa_result.json and computes mean score, N, and missing IDs.

Usage:
  python scripts_compbench/summarize_compbench_results.py \
    --run-dir /path/to/<RUN> \
    --categories color shape texture spatial non_spatial complex \
    --out /path/to/<RUN>/compbench_1sample_summary.json
"""
import argparse
import json
import math
from pathlib import Path


def _load_vqa_result(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # some evaluators wrap in a dict
        for key in ("results", "annotations", "data"):
            if key in data:
                return data[key]
        return list(data.values())
    raise ValueError(f"Unexpected vqa_result format in {path}: {type(data)}")


def _score(entry: dict) -> float:
    val = entry.get("answer", entry.get("score", entry.get("value", None)))
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir",    required=True)
    parser.add_argument("--categories", nargs="+",
                        default=["color", "shape", "texture", "spatial", "non_spatial", "complex"])
    parser.add_argument("--out",        required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = {"num_samples_per_prompt": 1}

    all_means = []

    for cat in args.categories:
        result_path = run_dir / cat / "vqa_result.json"
        if not result_path.exists():
            print(f"  MISSING  {cat}: {result_path}")
            summary[cat] = {"mean": None, "n": 0, "missing": [], "status": "missing"}
            continue

        entries = _load_vqa_result(result_path)
        scores = []
        missing = []
        for e in entries:
            s = _score(e)
            if math.isnan(s):
                missing.append(e.get("question_id", "?"))
            else:
                scores.append(s)

        mean = sum(scores) / len(scores) if scores else float("nan")
        summary[cat] = {
            "mean": round(mean, 6) if not math.isnan(mean) else None,
            "n": len(scores),
            "missing_ids": missing[:20],   # cap to avoid bloat
            "status": "ok",
        }
        if not math.isnan(mean):
            all_means.append(mean)
        print(f"  {cat:15s}  mean={mean:.4f}  n={len(scores)}  missing={len(missing)}")

    overall = sum(all_means) / len(all_means) if all_means else None
    summary["overall_mean"] = round(overall, 6) if overall is not None else None
    summary["note"] = "T2I-CompBench, 1 image per prompt (single-sample evaluation variant)"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[compbench_summary] overall_mean={overall:.4f}" if overall else
          "\n[compbench_summary] overall_mean=N/A (some categories missing)")
    print(f"[compbench_summary] Saved → {out_path}")


if __name__ == "__main__":
    main()
