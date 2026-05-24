"""
Second-pass filter for mined GPIC dataset.

Four stages (cheapest first):
  1. Fast  — reject non-visual content (temperatures, prices, menus, weather)
             require at least one attribute or spatial relation
             require attr_correctness > 0.5 OR a concrete spatial relation
             require at least one negative that changes a visually checkable word
  2. Grammar — Qwen text-only logit check: "is this caption grammatical and visual?"
  3. VLM margin — score raw_caption (positive) vs each negative against the image
                  keep only if pos >= 0.80, neg_max <= 0.40, margin >= 0.30

Usage
-----
  python scripts_data/filter_gpic_dataset.py \\
    --input-jsonl   data/gpic_slots_v2/dataset.jsonl \\
    --output-jsonl  data/gpic_slots_v2_clean/dataset.jsonl \\
    --qwen-model    Qwen/Qwen3-VL-4B-Instruct \\
    --pos-threshold 0.80 \\
    --neg-threshold 0.40 \\
    --margin        0.30 \\
    --max-keep      5000
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
}
MATERIAL_WORDS = {
    "wooden", "metal", "metallic", "plastic", "glass", "stone",
    "leather", "bamboo", "ceramic", "clay", "rubber", "fabric",
}
PATTERN_WORDS = {
    "striped", "plaid", "spotted", "dotted", "checkered", "patterned",
}
SIZE_WORDS = {"large", "small", "tall", "short", "big", "tiny", "huge", "wide"}

ATTRIBUTE_WORDS = COLOR_WORDS | MATERIAL_WORDS | PATTERN_WORDS | SIZE_WORDS

SPATIAL_RELATIONS = {
    "on top of", "under", "below", "above", "next to", "beside",
    "to the left of", "to the right of", "left of", "right of",
    "in front of", "behind", "inside", "holding", "wearing",
    "riding", "standing on", "sitting on",
}
_RELATION_KEYS = sorted(SPATIAL_RELATIONS, key=len, reverse=True)

# Patterns that indicate non-visual or unreliable content
_NON_VISUAL = [
    re.compile(r'\b\d+\s*°\s*[FfCc]\b'),                          # temperatures
    re.compile(r'\$\s*\d+|\bprice[sd]?\b|\bpricing\b', re.I),     # prices
    re.compile(r'\bmenu\b|\border online\b|\brestaurant menu\b', re.I),
    re.compile(r'\bweather (forecast|report)\b|\btemperature[s]?\b', re.I),
    re.compile(r'\btweet[s]?\b|\binstagram\b|\bfacebook\b', re.I), # social media text
    re.compile(r'\btext reads?\b|\bcaption reads?\b|\blabel reads?\b', re.I),
    re.compile(r'\bscoreboard\b|\bscoreline\b|\bsubtitle[s]?\b', re.I),
]

# Canonical patterns that indicate the rule-based rewriter went wrong
_BAD_CANONICAL = [
    # place-as-subject: "A shopping mall behind a white van"
    re.compile(r'^A (mall|building|church|stadium|airport|station|bridge|road|street|highway|park|forest|ocean|lake|river|mountain|sky|crowd)\b', re.I),
    # impossible physics: "standing on a shirt/dress/hat"
    re.compile(r'standing on a (shirt|dress|hat|shoe|sock|pants|jeans|jacket)', re.I),
    # trailing verb after second entity: "A X next to a Y stands."
    re.compile(r'a [a-z ]{3,30}\.\s*\w+s\.$', re.I),
]

_VLM_SUFFIX = "\nAnswer with only 'yes', 'no', or 'uncertain'."


# ── Stage 1: fast filter ──────────────────────────────────────────────────────

def fast_filter(row: dict) -> Tuple[bool, str]:
    raw      = row.get("raw_caption", "").strip()
    canon    = row.get("canonical_caption", "").strip()
    negs     = row.get("negative_captions", [])
    verif    = row.get("verification", {})

    if not negs:
        return False, "no_negatives"
    if not raw:
        return False, "empty_caption"

    low = raw.lower()

    # 1a. Reject non-visual content
    for pat in _NON_VISUAL:
        if pat.search(raw):
            return False, "non_visual_content"

    # 1b. Require at least one attribute or spatial relation
    has_attr = any(a in low.split() for a in ATTRIBUTE_WORDS)
    has_rel  = any(rp in low for rp in _RELATION_KEYS)
    if not has_attr and not has_rel:
        return False, "no_compositional_signal"

    # 1c. attr_correctness 0.5 means uncertain/default — require relation evidence
    attr_corr = verif.get("attr_correctness", 0.5)
    obj_pres  = verif.get("object_presence",  0.5)
    if obj_pres < 0.75:
        return False, "low_object_presence"
    if attr_corr <= 0.5 and not has_rel:
        return False, "attr_uncertain_no_spatial_relation"

    # 1d. At least one negative must flip a visually checkable attribute or relation
    any_contrastive = False
    for neg in negs:
        neg_low = neg.lower()
        for a in ATTRIBUTE_WORDS:
            if (a in low) != (a in neg_low):
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

    # 1e. Reject known-bad canonical patterns
    if canon:
        for pat in _BAD_CANONICAL:
            if pat.search(canon):
                return False, "malformed_canonical"

    return True, "ok"


# ── Stage 2: grammar check (Qwen text-only logit scoring) ────────────────────

_GRAMMAR_PROMPT = (
    'Is the following image caption grammatically correct and does it describe '
    'a real, visually concrete scene (not prices, temperatures, or physically '
    'impossible situations)? Caption: "{caption}"'
)


def grammar_check_batch(
    captions: List[str],
    verifier,
) -> List[bool]:
    """
    Returns a bool per caption: True = grammatical and visual.
    Uses Qwen text-only logit scoring (no image). Batched.
    """
    verifier._load()
    tok = verifier._processor.tokenizer

    yes_ids = [tok.encode(s, add_special_tokens=False)[0]
               for s in ["yes", "Yes", "YES"] if tok.encode(s, add_special_tokens=False)]
    no_ids  = [tok.encode(s, add_special_tokens=False)[0]
               for s in ["no",  "No",  "NO"]  if tok.encode(s, add_special_tokens=False)]

    all_texts = []
    for cap in captions:
        msg = [{"role": "user", "content": _GRAMMAR_PROMPT.format(caption=cap) + _VLM_SUFFIX}]
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
        y_l = max((lrow[j].item() for j in yes_ids), default=-1e9)
        n_l = max((lrow[j].item() for j in no_ids),  default=-1e9)
        base = max(y_l, n_l)
        ey   = math.exp(y_l - base)
        en   = math.exp(n_l - base)
        p_yes = ey / (ey + en + 1e-12)
        results.append(p_yes >= 0.55)   # lean yes = keep
    return results


# ── Stage 3: VLM margin filter ────────────────────────────────────────────────

def margin_filter(
    row: dict,
    verifier,
    pos_threshold: float,
    neg_threshold: float,
    margin: float,
) -> Tuple[bool, str, dict]:
    """
    Score raw_caption (positive) vs each negative against the saved image.
    All captions are batched in one Qwen forward pass per image.
    Returns (keep, reason, score_dict).
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

    # Batch all captions for this image into one forward pass
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
        "pos_score":  round(pos_score,  4),
        "neg_scores": [round(s, 4) for s in neg_scores],
        "neg_max":    round(neg_max,    4),
        "margin":     round(gap,        4),
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
    # VLM margin thresholds
    p.add_argument("--pos-threshold",  type=float, default=0.80)
    p.add_argument("--neg-threshold",  type=float, default=0.40)
    p.add_argument("--margin",         type=float, default=0.30)
    # Stage toggles
    p.add_argument("--no-grammar-check",  action="store_true", help="Skip Qwen grammar check")
    p.add_argument("--no-vlm-margin",     action="store_true", help="Skip VLM margin check")
    p.add_argument("--grammar-batch-size", type=int, default=16)
    # Output cap
    p.add_argument("--max-keep",       type=int,   default=None)
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    t0     = time.time()

    rows: List[dict] = []
    with open(args.input_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[filter] Loaded {len(rows)} rows from {args.input_jsonl}")

    stats = {
        "input": len(rows),
        "after_fast": 0,
        "after_grammar": 0,
        "after_margin": 0,
        "reject_reasons": {},
    }

    def _reject(reason: str):
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1

    # ── Stage 1: fast filter ──────────────────────────────────────────────────
    fast_pass = []
    for row in rows:
        keep, reason = fast_filter(row)
        if keep:
            fast_pass.append(row)
        else:
            _reject(f"fast:{reason}")
    stats["after_fast"] = len(fast_pass)
    print(f"[filter] Stage 1 (fast):    {len(fast_pass)}/{len(rows)} passed  "
          f"({time.time()-t0:.0f}s)")

    # ── Load verifier (shared by grammar + margin stages) ─────────────────────
    verifier = None
    need_model = (not args.no_grammar_check) or (not args.no_vlm_margin)
    if need_model:
        sys.path.insert(0, str(Path(__file__).parent))
        from mine_gpic_slots import QwenVLMVerifier
        print(f"[filter] Loading Qwen ({args.qwen_model}) ...")
        verifier = QwenVLMVerifier(model_id=args.qwen_model)
        verifier._load()
        print(f"[filter] Qwen loaded  ({time.time()-t0:.0f}s)")

    # ── Stage 2: grammar check ────────────────────────────────────────────────
    if args.no_grammar_check or verifier is None:
        grammar_pass = fast_pass
    else:
        grammar_pass = []
        bs = args.grammar_batch_size
        for i in range(0, len(fast_pass), bs):
            batch    = fast_pass[i:i+bs]
            # Check raw_caption (what we train on) not canonical
            captions = [r.get("raw_caption", "") for r in batch]
            results  = grammar_check_batch(captions, verifier)
            for row, ok in zip(batch, results):
                if ok:
                    grammar_pass.append(row)
                else:
                    _reject("grammar:failed_visual_grammar_check")
            if (i // bs + 1) % 5 == 0:
                print(f"[filter] Stage 2 grammar: {i+bs}/{len(fast_pass)} checked, "
                      f"{len(grammar_pass)} kept  ({time.time()-t0:.0f}s)")

    stats["after_grammar"] = len(grammar_pass)
    print(f"[filter] Stage 2 (grammar): {len(grammar_pass)}/{len(fast_pass)} passed  "
          f"({time.time()-t0:.0f}s)")

    # ── Stage 3: VLM margin filter ────────────────────────────────────────────
    if args.no_vlm_margin or verifier is None:
        margin_pass = grammar_pass
    else:
        # Only run if rows actually have image_path
        has_images = any(
            r.get("image_path") and Path(r["image_path"]).exists()
            for r in grammar_pass
        )
        if not has_images:
            print("[filter] Stage 3: no image_path found in rows — skipping margin check")
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
                    if scores:
                        row = {**row, "margin_scores": scores}
                    margin_pass.append(row)
                    if args.verbose:
                        print(f"  KEEP   {row.get('key','?'):20s}  {scores}")
                else:
                    _reject(f"margin:{reason}")
                    if args.verbose:
                        print(f"  REJECT {row.get('key','?'):20s}  {reason}  {scores}")

                if (i + 1) % 100 == 0:
                    print(f"[filter] Stage 3 margin: {i+1}/{len(grammar_pass)} scored, "
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
    print(f"\n[filter] ── Summary ────────────────────────────")
    print(f"  Input:            {stats['input']}")
    print(f"  After fast:       {stats['after_fast']}")
    print(f"  After grammar:    {stats['after_grammar']}")
    print(f"  After VLM margin: {stats['after_margin']}")
    print(f"  Written to:       {out_path}")
    print(f"  Elapsed:          {elapsed:.0f}s")
    print(f"\n[filter] Reject breakdown:")
    for reason, count in sorted(stats["reject_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
