"""
Rewrite compositional prompts into structured visual-plan prompts.

Uses bucket-specific templates derived from the metadata fields that are
already present in each jsonl row (objects, relation, count, etc.).
Falls back to a safe generic wrapper for any row whose metadata is incomplete.

Usage:
  python scripts_cot/create_structured_prompts.py \
    --data-root data \
    --buckets attribute_binding spatial_relations_anchored counting complex_composition \
    --splits train val \
    --output-root data_cot_structured
"""
import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Relation → human phrase table for spatial bucket
# (subject_placement, reference_placement, relation_phrase)
# ---------------------------------------------------------------------------
_SPATIAL_PHRASES = {
    "left_of":     ("on the left side",          "on the right side",         "to the left of"),
    "right_of":    ("on the right side",         "on the left side",          "to the right of"),
    "above":       ("at the top of the image",   "at the bottom of the image","above"),
    "below":       ("at the bottom of the image","at the top of the image",   "below"),
    "inside":      ("inside the other object",   "as the outer container",    "inside"),
    "in_front_of": ("in the foreground",         "in the background",         "in front of"),
    "behind":      ("in the background",         "in the foreground",         "behind"),
    "on_top_of":   ("resting on top",            "at the base",               "on top of"),
    "next_to":     ("on the left side",          "on the right side",         "next to"),
    "near":        ("close together",            "close together",            "near"),
}


# ---------------------------------------------------------------------------
# Bucket-specific rewrite functions
# ---------------------------------------------------------------------------

def _generic(raw_prompt: str) -> str:
    return (
        "Create a simple clean image that clearly satisfies the following prompt: "
        f"{raw_prompt} "
        "Make every object clearly visible. Make all colors and attributes unambiguous. "
        "Keep the layout simple and avoid extra objects."
    )


def _rewrite_attribute(row: dict) -> str:
    meta = row.get("metadata", {})
    objects = meta.get("objects", [])
    if len(objects) == 2:
        o1, a1 = objects[0]["name"], objects[0]["attribute"]
        o2, a2 = objects[1]["name"], objects[1]["attribute"]
        return (
            "A simple clean image with exactly two main objects. "
            f"The first object is a {a1} {o1}. "
            f"The second object is a {a2} {o2}. "
            f"The {o1} should clearly appear {a1}. "
            f"The {o2} should clearly appear {a2}. "
            "Keep both objects visually separated and easy to identify. "
            "Do not swap the attributes between the objects."
        )
    return _generic(row["prompt"])


def _rewrite_spatial(row: dict) -> str:
    meta    = row.get("metadata", {})
    subj    = meta.get("subject", {})
    ref     = meta.get("reference", {})
    relation = meta.get("relation", "")

    subj_name = f"{subj.get('attribute', '')} {subj.get('object', '')}".strip()
    ref_name  = f"{ref.get('attribute',  '')} {ref.get('object',  '')}".strip()

    if not subj_name or not ref_name:
        return _generic(row["prompt"])

    if relation in _SPATIAL_PHRASES:
        s_pos, r_pos, rel_phrase = _SPATIAL_PHRASES[relation]
        return (
            "A simple clean image with two main objects on a plain background. "
            f"Place the {subj_name} {s_pos}. "
            f"Place the {ref_name} {r_pos}. "
            f"The {subj_name} must be {rel_phrase} the {ref_name}. "
            "Keep both objects clearly visible and well separated. "
            "Do not swap the positions of the objects."
        )
    # unknown relation — use generic but with object names
    return (
        "A simple clean image with two main objects on a plain background. "
        f"Show a {subj_name} and a {ref_name}. "
        f"The spatial arrangement must satisfy: {row['prompt']} "
        "Keep both objects clearly visible."
    )


def _rewrite_counting(row: dict) -> str:
    meta   = row.get("metadata", {})
    count  = meta.get("count")
    plural = meta.get("object_plural") or meta.get("object", "objects")
    if count is not None:
        return (
            f"A simple clean image showing exactly {count} {plural}. "
            f"Each of the {plural} should be clearly separated and individually distinguishable. "
            f"Do not include more or fewer than {count} {plural}. "
            f"Do not add other similar objects that could be mistaken for {plural}."
        )
    return _generic(row["prompt"])


def _rewrite_complex(row: dict) -> str:
    meta  = row.get("metadata", {})
    parts = []
    # try to reconstruct constraint descriptions from whatever metadata exists
    if "objects" in meta:
        objs_str = ", ".join(
            f"{o.get('attribute','')} {o.get('name','')}".strip()
            for o in meta["objects"]
        )
        parts.append(f"Objects and attributes: {objs_str}.")
    if "count" in meta:
        plural = meta.get("object_plural", meta.get("object", "objects"))
        parts.append(f"Count: exactly {meta['count']} {plural}.")
    if "relation" in meta:
        subj = meta.get("subject", {})
        ref  = meta.get("reference", {})
        s_name = f"{subj.get('attribute','')} {subj.get('object','')}".strip()
        r_name = f"{ref.get('attribute','')} {ref.get('object','')}".strip()
        rel    = _SPATIAL_PHRASES.get(meta["relation"], (None, None, meta["relation"]))[2]
        parts.append(f"Spatial constraint: {s_name} must be {rel} {r_name}.")

    if parts:
        constraints = " ".join(parts)
        return (
            "A simple clean image following this visual plan. "
            f"{constraints} "
            "All required objects should be clearly visible, separated, and easy to identify. "
            "Do not add extra confusing objects."
        )
    return _generic(row["prompt"])


_REWRITERS = {
    "attribute_binding":          _rewrite_attribute,
    "spatial_relations_anchored": _rewrite_spatial,
    "counting":                   _rewrite_counting,
    "complex_composition":        _rewrite_complex,
}


def rewrite_row(row: dict, bucket: str) -> dict:
    raw_prompt = row.get("prompt") or row.get("caption") or ""
    rewriter   = _REWRITERS.get(bucket, lambda r: _generic(r["prompt"]))
    structured = rewriter(row)
    out = dict(row)
    out["raw_prompt"]       = raw_prompt
    out["structured_prompt"] = structured
    out["prompt_style"]     = "structured_visual_plan"
    return out


# ---------------------------------------------------------------------------
# File resolution helpers
# ---------------------------------------------------------------------------

def _find_split_file(data_root: Path, bucket: str, split: str) -> Path:
    """Resolve the jsonl path for a given bucket and split."""
    candidates = list((data_root / bucket).glob(f"{bucket}_{split}_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No jsonl found for bucket={bucket} split={split} in {data_root / bucket}"
        )
    # prefer the one without 'structured' in the name
    non_structured = [p for p in candidates if "structured" not in p.name]
    return (non_structured or candidates)[0]


def _out_filename(src_name: str) -> str:
    """e.g. attribute_binding_train_500.jsonl → attribute_binding_train_500_structured.jsonl"""
    stem = src_name.replace(".jsonl", "")
    return f"{stem}_structured.jsonl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",    default="data")
    parser.add_argument("--buckets",      nargs="+",
                        default=["attribute_binding", "spatial_relations_anchored",
                                 "counting", "complex_composition"])
    parser.add_argument("--splits",       nargs="+", default=["train", "val"])
    parser.add_argument("--output-root",  default="data_cot_structured")
    args = parser.parse_args()

    data_root  = Path(args.data_root)
    out_root   = Path(args.output_root)

    for bucket in args.buckets:
        for split in args.splits:
            try:
                src_path = _find_split_file(data_root, bucket, split)
            except FileNotFoundError as e:
                print(f"  SKIP  {bucket}/{split}: {e}")
                continue

            out_dir  = out_root / bucket
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / _out_filename(src_path.name)

            rows_in = 0
            rows_out = 0
            with open(src_path, encoding="utf-8") as f_in, \
                 open(out_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows_in += 1
                    out_row = rewrite_row(row, bucket)
                    f_out.write(json.dumps(out_row) + "\n")
                    rows_out += 1

            print(f"  {bucket:35s}  {split:5s}  {rows_in} → {rows_out}  → {out_path}")

    print(f"\n[create_structured_prompts] Done.  Output root: {out_root}")


if __name__ == "__main__":
    main()
