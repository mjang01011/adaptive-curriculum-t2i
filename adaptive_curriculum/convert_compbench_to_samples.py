"""
Convert CompBench metadata.jsonl files to samples.jsonl for PARM scoring.

Usage:
  python adaptive_curriculum/convert_compbench_to_samples.py \
    --compbench-dir outputs_compbench_vanilla/llamagen_base_10sample \
    --output-jsonl  outputs/parm_scores/samples.jsonl \
    --categories    color shape texture spatial non_spatial complex
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compbench-dir", required=True,
                   help="Root dir with <category>/samples/metadata.jsonl")
    p.add_argument("--output-jsonl",  required=True)
    p.add_argument("--categories",    nargs="+",
                   default=["color", "shape", "texture", "spatial", "non_spatial", "complex"])
    args = p.parse_args()

    root = Path(args.compbench_dir)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for cat in args.categories:
            meta_path = root / cat / "samples" / "metadata.jsonl"
            if not meta_path.exists():
                print(f"[warn] missing: {meta_path}")
                continue

            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    img_path = root / cat / "samples" / row["filename"]
                    out_row = {
                        "id":           f"{cat}_{row['flat_id']:06d}",
                        "prompt":       row["prompt"],
                        "image_path":   str(img_path.resolve()),
                        "category":     cat,
                        "question_id":  row["question_id"],
                        "sample_index": row["sample_index"],
                        "flat_id":      row["flat_id"],
                    }
                    fout.write(json.dumps(out_row) + "\n")
                    total += 1

    print(f"Wrote {total} rows → {out_path}")


if __name__ == "__main__":
    main()
