"""
Second-pass filter for mined GPIC dataset.

Four stages (cheapest first):
  1. Fast     — non-visual content, compositional signal, attr_correctness,
                contrastive negatives, malformed canonical patterns
  2. Type     — classify each negative by type (color_swap, relation_reversal, …)
                optionally require / exclude specific types
  3. Grammar  — Qwen text-only logit check on raw_caption
  4. Margin   — VLM positive-vs-negative margin on saved image

Usage
-----
  # Standard clean run
  python scripts_data/filter_gpic_dataset.py \\
    --input-jsonl   data/gpic_slots_v2/dataset.jsonl \\
    --output-jsonl  data/gpic_slots_v2_clean/dataset.jsonl \\
    --margin 0.50 --neg-threshold 0.35 --image-quality 0.90

  # Strict first-adapter run (high-confidence compositional only)
  python scripts_data/filter_gpic_dataset.py \\
    --input-jsonl   data/gpic_slots_v2/dataset.jsonl \\
    --output-jsonl  data/gpic_slots_v2_strict/dataset.jsonl \\
    --strict \\
    --prefer-types  color_swap,color_change,relation_reversal,material_change

Negative types
--------------
  color_swap          same colors in both captions but swapped between objects  (BEST)
  color_change        a color word differs between pos and neg                  (BEST)
  relation_reversal   spatial/action relation changed                           (BEST)
  material_change     material/texture word differs                             (GOOD)
  pattern_change      pattern word (striped, spotted…) differs                 (GOOD)
  size_change         size word (large, small, tall…) differs                  (OK)
  style_global        global image style change (b&w, grayscale…)              (SKIP)
  other               no visually labellable attribute detected                (SKIP)
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


# ── Compositional signal constants ───────────────────────────────────────────

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "purple", "pink", "brown", "gray", "grey", "teal", "gold", "silver",
    "maroon", "navy", "cyan", "magenta", "beige", "ivory", "crimson",
}
MATERIAL_WORDS = {
    "wooden", "metal", "metallic", "plastic", "glass", "stone",
    "leather", "bamboo", "ceramic", "clay", "rubber", "fabric",
    "concrete", "steel", "iron", "wooden", "chrome", "brass",
}
PATTERN_WORDS = {
    "striped", "plaid", "spotted", "dotted", "checkered", "patterned",
    "floral", "geometric", "printed",
}
SIZE_WORDS = {
    "large", "small", "tall", "short", "big", "tiny", "huge", "wide",
    "narrow", "thick", "thin", "long",
}

ATTRIBUTE_WORDS = COLOR_WORDS | MATERIAL_WORDS | PATTERN_WORDS | SIZE_WORDS

SPATIAL_RELATIONS = {
    "on top of", "under", "below", "above", "next to", "beside",
    "to the left of", "to the right of", "left of", "right of",
    "in front of", "behind", "inside", "holding", "wearing",
    "riding", "standing on", "sitting on",
}
_RELATION_KEYS = sorted(SPATIAL_RELATIONS, key=len, reverse=True)

# Priority ordering for filtering (best first)
NEGATIVE_TYPE_PRIORITY = [
    "color_swap", "color_change", "relation_reversal", "material_change",
    "pattern_change", "size_change", "style_global", "other",
]

_STYLE_WORDS = frozenset({"black-and-white", "grayscale", "sepia", "monochrome", "colorized"})

# Non-visual content patterns
_NON_VISUAL = [
    re.compile(r'\b\d+\s*°\s*[FfCc]\b'),
    re.compile(r'\$\s*\d+|\bprice[sd]?\b|\bpricing\b', re.I),
    re.compile(r'\bmenu\b|\border online\b|\brestaurant menu\b', re.I),
    re.compile(r'\bweather (forecast|report)\b|\btemperature[s]?\b', re.I),
    re.compile(r'\btweet[s]?\b|\binstagram\b|\bfacebook\b', re.I),
    re.compile(r'\btext reads?\b|\bcaption reads?\b|\blabel reads?\b', re.I),
    re.compile(r'\bscoreboard\b|\bscoreline\b|\bsubtitle[s]?\b', re.I),
]

# Canonical captions that indicate the rule-based rewriter produced a broken template
_BAD_CANONICAL = [
    # Place-as-subject spatial templates
    re.compile(r'^A (mall|shopping mall|building|church|stadium|airport|station|'
               r'bridge|road|street|highway|park|forest|ocean|lake|river|mountain|'
               r'sky|crowd|city|town|village|campus|courtyard)\b', re.I),
    # Physically impossible: person/object "on" clothing
    re.compile(r'(standing|sitting|lying|placed?) on a (shirt|dress|hat|shoe|'
               r'sock|pants|jeans|jacket|coat|sweater)', re.I),
    # Trailing verb after second-entity template: "A X next to a Y stands."
    re.compile(r'a [a-z ]{2,25}\.\s*\w+[sde]+\.$', re.I),
    # Vague non-visual subject
    re.compile(r'^A (pair|group|set|bunch|collection|number|array) (in front of|behind|'
               r'next to|to the left of|to the right of)', re.I),
    # Grammar: "a orange" (missing h) from article bug
    re.compile(r'\ba [aeiou]', re.I),
]

_VLM_SUFFIX = "\nAnswer with only 'yes', 'no', or 'uncertain'."


# ── Negative type classifier ──────────────────────────────────────────────────

def classify_negative_type(pos: str, neg: str) -> str:
    """
    Rule-based classification of what changed between positive and negative caption.
    Operates on word-level attribute/relation sets — no NLP model needed.
    """
    pos_words = set(pos.lower().split())
    neg_words = set(neg.lower().split())
    pos_low, neg_low = pos.lower(), neg.lower()

    pos_colors = pos_words & COLOR_WORDS
    neg_colors = neg_words & COLOR_WORDS
    pos_mats   = pos_words & MATERIAL_WORDS
    neg_mats   = neg_words & MATERIAL_WORDS
    pos_pats   = pos_words & PATTERN_WORDS
    neg_pats   = neg_words & PATTERN_WORDS
    pos_sizes  = pos_words & SIZE_WORDS
    neg_sizes  = neg_words & SIZE_WORDS
    pos_rels   = {r for r in _RELATION_KEYS if r in pos_low}
    neg_rels   = {r for r in _RELATION_KEYS if r in neg_low}

    # Relation reversal: any spatial/action relation changed
    if pos_rels != neg_rels:
        return "relation_reversal"

    # Color swap: same color words in both but arrangement changed
    # e.g. "red cube + blue sphere" → "blue cube + red sphere"
    if pos_colors and pos_colors == neg_colors:
        return "color_swap"

    # Color change: different color words
    if pos_colors != neg_colors:
        return "color_change"

    # Material / texture change
    if pos_mats != neg_mats:
        return "material_change"

    # Pattern change
    if pos_pats != neg_pats:
        return "pattern_change"

    # Size change
    if pos_sizes != neg_sizes:
        return "size_change"

    # Global style change (black-and-white, colorized, etc.)
    pos_style = any(s in pos_low for s in _STYLE_WORDS)
    neg_style = any(s in neg_low for s in _STYLE_WORDS)
    if pos_style != neg_style:
        return "style_global"

    return "other"


def annotate_negative_types(row: dict) -> dict:
    """Add a 'negative_types' list aligned with 'negative_captions'."""
    raw  = row.get("raw_caption", "")
    negs = row.get("negative_captions", [])
    row["negative_types"] = [classify_negative_type(raw, neg) for neg in negs]
    return row


# ── Stage 1: fast filter ──────────────────────────────────────────────────────

def fast_filter(
    row: dict,
    image_quality_threshold: float,
    obj_presence_threshold:  float,
) -> Tuple[bool, str]:
    raw    = row.get("raw_caption", "").strip()
    canon  = row.get("canonical_caption", "").strip()
    negs   = row.get("negative_captions", [])
    verif  = row.get("verification", {})

    if not negs:
        return False, "no_negatives"
    if not raw:
        return False, "empty_caption"

    low = raw.lower()

    # 1a. Reject non-visual content
    for pat in _NON_VISUAL:
        if pat.search(raw):
            return False, "non_visual_content"

    # 1b. Require at least one attribute word or spatial relation
    has_attr = any(a in low.split() for a in ATTRIBUTE_WORDS)
    has_rel  = any(rp in low for rp in _RELATION_KEYS)
    if not has_attr and not has_rel:
        return False, "no_compositional_signal"

    # 1c. Verification thresholds
    iq = verif.get("image_quality",   0.5)
    op = verif.get("object_presence", 0.5)
    if iq < image_quality_threshold:
        return False, f"low_image_quality={iq:.3f}"
    if op < obj_presence_threshold:
        return False, f"low_object_presence={op:.3f}"

    # 1d. attr_correctness 0.5 = uncertain/default — require spatial relation evidence
    attr_corr = verif.get("attr_correctness", 0.5)
    if attr_corr <= 0.5 and not has_rel:
        return False, "attr_uncertain_no_spatial_relation"

    # 1e. At least one negative changes a visually checkable word
    any_contrastive = False
    for neg in negs:
        neg_low = neg.lower()
        for a in ATTRIBUTE_WORDS:
            if (a in low.split()) != (a in neg_low.split()):
                any_contrastive = True
                break
        if not any_contrastive:
            for rp in _RELATION_KEYS:
                if (rp in low) != (rp in neg_low):
                    any_contrastive = True
                    break
        if any_contrastive:
            break
    if not any_contrastive:
        return False, "negatives_not_visually_contrastive"

    # 1f. Reject known-bad canonical patterns
    if canon:
        for pat in _BAD_CANONICAL:
            if pat.search(canon):
                return False, "malformed_canonical"

    return True, "ok"


# ── Stage 2: type-based filter ────────────────────────────────────────────────

def type_filter(
    row: dict,
    prefer: Optional[List[str]],
    exclude: Optional[List[str]],
) -> Tuple[bool, str]:
    """
    If --prefer-types: keep only rows that have at least one negative of a preferred type.
    If --exclude-types: reject rows where ALL negatives are excluded types.
    """
    types = row.get("negative_types", [])
    if not types:
        return True, "ok"

    if prefer:
        if not any(t in prefer for t in types):
            best = types[0] if types else "none"
            return False, f"no_preferred_type (best={best})"

    if exclude:
        if all(t in exclude for t in types):
            return False, f"all_excluded_types ({types})"

    return True, "ok"


# ── Stage 3: grammar check (Qwen text-only logit scoring) ────────────────────

_GRAMMAR_PROMPT = (
    'Is the following image caption grammatically correct and does it describe '
    'a real, visually concrete scene (not prices, temperatures, or physically '
    'impossible situations)? Caption: "{caption}"'
)


def grammar_check_batch(captions: List[str], verifier) -> List[bool]:
    verifier._load()
    tok = verifier._processor.tokenizer

    yes_ids = [tok.encode(s, add_special_tokens=False)[0]
               for s in ["yes", "Yes", "YES"] if tok.encode(s, add_special_tokens=False)]
    no_ids  = [tok.encode(s, add_special_tokens=False)[0]
               for s in ["no", "No", "NO"]   if tok.encode(s, add_special_tokens=False)]

    all_texts = []
    for cap in captions:
        msg = [{"role": "user", "content":
                _GRAMMAR_PROMPT.format(caption=cap) + _VLM_SUFFIX}]
        text_in = verifier._processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
        )
        all_texts.append(text_in)

    inputs = verifier._processor(
        text=all_texts, return_tensors="pt", padding=True,
    ).to(verifier._model.device)

    with torch.inference_mode():
        out = verifier._model(**inputs)

    batch_logits = out.logits[:, -1, :]
    results = []
    for i in range(len(captions)):
        lrow = batch_logits[i]
        y_l  = max((lrow[j].item() for j in yes_ids), default=-1e9)
        n_l  = max((lrow[j].item() for j in no_ids),  default=-1e9)
        base = max(y_l, n_l)
        p_yes = math.exp(y_l - base) / (math.exp(y_l - base) + math.exp(n_l - base) + 1e-12)
        results.append(p_yes >= 0.55)
    return results


# ── Stage 4: VLM margin filter ────────────────────────────────────────────────

def margin_filter(
    row: dict,
    verifier,
    pos_threshold: float,
    neg_threshold: float,
    margin: float,
) -> Tuple[bool, str, dict]:
    """
    Score raw_caption vs each negative against the saved image.
    All captions batched in one Qwen forward pass per image.
    """
    image_path = row.get("image_path")
    if not image_path or not Path(image_path).exists():
        return True, "no_image_skip", {}

    try:
        from PIL import Image as PILImage
        pil = PILImage.open(image_path).convert("RGB")
    except Exception:
        return True, "image_load_error_skip", {}

    raw  = row.get("raw_caption", "")
    negs = row.get("negative_captions", [])

    stmts: Dict[str, str] = {
        "pos": f'The image visually matches this description: "{raw}".',
    }
    for i, neg in enumerate(negs):
        stmts[f"neg_{i}"] = f'The image visually matches this description: "{neg}".'

    raw_scores = verifier.score_statements(pil, stmts)

    pos_score  = raw_scores["pos"]
    neg_scores = [raw_scores[f"neg_{i}"] for i in range(len(negs))]
    neg_max    = max(neg_scores) if neg_scores else 0.0
    gap        = pos_score - neg_max

    scores = {
        "pos_score":  round(pos_score, 4),
        "neg_scores": [round(s, 4) for s in neg_scores],
        "neg_max":    round(neg_max,   4),
        "margin":     round(gap,       4),
    }

    if pos_score < pos_threshold:
        return False, f"low_pos={pos_score:.3f}", scores
    if neg_max > neg_threshold:
        return False, f"high_neg={neg_max:.3f}", scores
    if gap < margin:
        return False, f"small_margin={gap:.3f}", scores

    return True, "ok", scores


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl",    required=True)
    p.add_argument("--output-jsonl",   required=True)
    p.add_argument("--qwen-model",     default="Qwen/Qwen3-VL-4B-Instruct")

    # Thresholds (individual flags — --strict overrides these)
    p.add_argument("--pos-threshold",  type=float, default=0.80,
                   help="Minimum positive caption match score")
    p.add_argument("--neg-threshold",  type=float, default=0.35,
                   help="Maximum negative caption match score")
    p.add_argument("--margin",         type=float, default=0.50,
                   help="Minimum pos - neg_max margin")
    p.add_argument("--image-quality",  type=float, default=0.90,
                   help="Minimum verification.image_quality")
    p.add_argument("--obj-presence",   type=float, default=0.75,
                   help="Minimum verification.object_presence")

    # Preset for first strict adapter run
    p.add_argument("--strict", action="store_true",
                   help="Override thresholds: margin=0.80 image_quality=0.95 obj_presence=0.85")

    # Negative type filtering
    p.add_argument("--prefer-types",   default=None,
                   help="Comma-separated types to require. Keep only rows with at least one "
                        "negative of these types. "
                        "e.g. color_swap,color_change,relation_reversal,material_change")
    p.add_argument("--exclude-types",  default=None,
                   help="Comma-separated types to exclude. Reject rows where ALL negatives "
                        "are of these types. e.g. style_global,other")

    # Stage toggles
    p.add_argument("--no-grammar-check", action="store_true")
    p.add_argument("--no-vlm-margin",    action="store_true")
    p.add_argument("--grammar-batch-size", type=int, default=16)

    # Output
    p.add_argument("--max-keep",  type=int, default=None)
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.time()

    # Apply --strict overrides
    if args.strict:
        args.margin        = 0.80
        args.image_quality = 0.95
        args.obj_presence  = 0.85
        print("[filter] --strict mode: margin=0.80  image_quality=0.95  obj_presence=0.85")

    prefer_types  = [t.strip() for t in args.prefer_types.split(",")]  if args.prefer_types  else None
    exclude_types = [t.strip() for t in args.exclude_types.split(",")] if args.exclude_types else None

    rows: List[dict] = []
    with open(args.input_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[filter] Loaded {len(rows)} rows")

    stats = {
        "input": len(rows),
        "after_fast": 0, "after_type": 0, "after_grammar": 0, "after_margin": 0,
        "type_distribution": {},
        "reject_reasons": {},
    }

    def _reject(reason: str):
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1

    # ── Stage 1: fast filter ──────────────────────────────────────────────────
    fast_pass = []
    for row in rows:
        keep, reason = fast_filter(row, args.image_quality, args.obj_presence)
        if keep:
            fast_pass.append(row)
        else:
            _reject(f"fast:{reason}")
    stats["after_fast"] = len(fast_pass)
    print(f"[filter] Stage 1 (fast):    {len(fast_pass)}/{len(rows)} passed  "
          f"({time.time()-t0:.0f}s)")

    # ── Stage 2: type annotation + filter ────────────────────────────────────
    for row in fast_pass:
        annotate_negative_types(row)

    # Tally type distribution
    for row in fast_pass:
        for t in row.get("negative_types", []):
            stats["type_distribution"][t] = stats["type_distribution"].get(t, 0) + 1

    type_pass = []
    for row in fast_pass:
        keep, reason = type_filter(row, prefer_types, exclude_types)
        if keep:
            type_pass.append(row)
        else:
            _reject(f"type:{reason}")
    stats["after_type"] = len(type_pass)
    print(f"[filter] Stage 2 (type):    {len(type_pass)}/{len(fast_pass)} passed  "
          f"({time.time()-t0:.0f}s)")
    print(f"[filter] Type distribution: " +
          "  ".join(f"{k}={v}" for k, v in
                    sorted(stats["type_distribution"].items(),
                           key=lambda x: NEGATIVE_TYPE_PRIORITY.index(x[0])
                           if x[0] in NEGATIVE_TYPE_PRIORITY else 99)))

    # ── Load verifier ─────────────────────────────────────────────────────────
    verifier = None
    need_model = (not args.no_grammar_check) or (not args.no_vlm_margin)
    if need_model and type_pass:
        sys.path.insert(0, str(Path(__file__).parent))
        from mine_gpic_slots import QwenVLMVerifier
        print(f"[filter] Loading Qwen ({args.qwen_model}) ...")
        verifier = QwenVLMVerifier(model_id=args.qwen_model)
        verifier._load()
        print(f"[filter] Qwen loaded  ({time.time()-t0:.0f}s)")

    # ── Stage 3: grammar check ────────────────────────────────────────────────
    if args.no_grammar_check or verifier is None:
        grammar_pass = type_pass
    else:
        grammar_pass = []
        bs = args.grammar_batch_size
        for i in range(0, len(type_pass), bs):
            batch    = type_pass[i:i+bs]
            captions = [r.get("raw_caption", "") for r in batch]
            results  = grammar_check_batch(captions, verifier)
            for row, ok in zip(batch, results):
                if ok:
                    grammar_pass.append(row)
                else:
                    _reject("grammar:failed")
            if (i // bs + 1) % 5 == 0:
                print(f"[filter] Stage 3 grammar: {min(i+bs, len(type_pass))}/{len(type_pass)} "
                      f"checked, {len(grammar_pass)} kept  ({time.time()-t0:.0f}s)")
    stats["after_grammar"] = len(grammar_pass)
    print(f"[filter] Stage 3 (grammar): {len(grammar_pass)}/{len(type_pass)} passed  "
          f"({time.time()-t0:.0f}s)")

    # ── Stage 4: VLM margin filter ────────────────────────────────────────────
    if args.no_vlm_margin or verifier is None:
        margin_pass = grammar_pass
    else:
        has_images = any(
            r.get("image_path") and Path(r["image_path"]).exists()
            for r in grammar_pass
        )
        if not has_images:
            print("[filter] Stage 4: no image_path found — skipping margin check")
            margin_pass = grammar_pass
        else:
            margin_pass = []
            for i, row in enumerate(grammar_pass):
                keep, reason, scores = margin_filter(
                    row, verifier,
                    pos_threshold=args.pos_threshold,
                    neg_threshold=args.neg_threshold,
                    margin=args.margin,
                )
                if keep:
                    row = {**row, "margin_scores": scores} if scores else row
                    margin_pass.append(row)
                    if args.verbose:
                        types = row.get("negative_types", [])
                        print(f"  KEEP   {row.get('key','?'):20s}  types={types}  {scores}")
                else:
                    _reject(f"margin:{reason}")
                    if args.verbose:
                        print(f"  REJECT {row.get('key','?'):20s}  {reason}")

                if (i + 1) % 100 == 0:
                    print(f"[filter] Stage 4 margin: {i+1}/{len(grammar_pass)} scored, "
                          f"{len(margin_pass)} kept  ({time.time()-t0:.0f}s)")

                if args.max_keep and len(margin_pass) >= args.max_keep:
                    print(f"[filter] Reached --max-keep {args.max_keep}, stopping.")
                    break

    stats["after_margin"] = len(margin_pass)

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in margin_pass:
            f.write(json.dumps(row) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n[filter] ── Summary ──────────────────────────────────")
    print(f"  Input:            {stats['input']}")
    print(f"  After fast:       {stats['after_fast']}")
    print(f"  After type:       {stats['after_type']}")
    print(f"  After grammar:    {stats['after_grammar']}")
    print(f"  After margin:     {stats['after_margin']}")
    print(f"  Written to:       {out_path}")
    print(f"  Elapsed:          {elapsed:.0f}s")
    print(f"\n  Reject breakdown:")
    for reason, count in sorted(stats["reject_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print(f"\n  Final type distribution:")
    final_type_dist: dict = {}
    for row in margin_pass:
        for t in row.get("negative_types", []):
            final_type_dist[t] = final_type_dist.get(t, 0) + 1
    for t in NEGATIVE_TYPE_PRIORITY:
        if t in final_type_dist:
            print(f"    {t}: {final_type_dist[t]}")


if __name__ == "__main__":
    main()
