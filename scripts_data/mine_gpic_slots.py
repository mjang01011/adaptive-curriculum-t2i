"""
Mine compositional image-caption training data from GPIC tar archives.

Outputs per accepted example
-----------------------------
  raw_caption        — original GPIC caption
  canonical_caption  — LLM/rule-cleaned short prompt (optional)
  negative_captions  — attribute/relation-swapped negatives for contrastive loss
  image_tokens       — pre-encoded VQ token path  (requires --save-tokens)
  image_path         — saved image                (requires --save-images)
  verification       — VLM scores used for filtering
  slot_schema        — optional metadata for analysis/debugging (NOT required at inference)

The resulting dataset trains an implicit composition adapter that works on
any raw text prompt at inference — no parser, no explicit slot schema.

Usage
-----
  python scripts_data/mine_gpic_slots.py \\
    --tar-path /path/to/gpic/000000.tar \\
    --output-dir $PROJ/data/gpic_slots_v1 \\
    --vq-ckpt   $PRETRAINED/vq_ds16_t2i.pt \\
    --repo-root $LLAMAGEN \\
    --max-records 5000 --max-keep 1000 \\
    --save-images --save-tokens
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

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
ATTRIBUTE_WORDS = COLOR_WORDS | MATERIAL_WORDS | PATTERN_WORDS

RELATION_MAP = {
    "on top of":       "on top of",
    "under":           "under",
    "below":           "below",
    "above":           "above",
    "next to":         "next to",
    "beside":          "next to",
    "left of":         "to the left of",
    "to the left of":  "to the left of",
    "right of":        "to the right of",
    "to the right of": "to the right of",
    "in front of":     "in front of",
    "behind":          "behind",
    "inside":          "inside",
    "holding":         "holding",
    "wearing":         "wearing",
    "riding":          "riding",
    "standing on":     "standing on",
    "sitting on":      "sitting on",
    "with":            "with",
}
_RELATION_KEYS_SORTED = sorted(RELATION_MAP, key=len, reverse=True)

_ENTITY_PAT = re.compile(
    r"\b(?:a|an|the)\s+"
    r"((?:(?:" + "|".join(re.escape(a) for a in sorted(ATTRIBUTE_WORDS, key=len, reverse=True)) + r")\s+){0,4})"
    r"([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*)?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tar pairing
# ---------------------------------------------------------------------------

def iter_gpic_records(tar_path: str):
    """Yield (key, meta_dict, img_name, img_bytes) pairing by key."""
    json_by_key  = {}
    image_by_key = {}

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            f    = tf.extractfile(member)
            if f is None:
                continue

            if ext == "json":
                try:
                    meta = json.loads(f.read().decode("utf-8"))
                except Exception:
                    continue
                key = meta.get("key", stem)
                json_by_key[key] = meta
            elif ext in {"jpg", "jpeg", "png", "webp"}:
                image_by_key[stem] = (name, f.read())

            ready = set(json_by_key) & set(image_by_key)
            for key in list(ready):
                meta                = json_by_key.pop(key)
                img_name, img_bytes = image_by_key.pop(key)
                yield key, meta, img_name, img_bytes


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------

def prefilter_caption(meta: dict) -> tuple:
    """Returns (pass: bool, reject_reason: str)."""
    cap_type = meta.get("caption_type", "")
    caption  = (meta.get("caption") or "").strip()
    w        = meta.get("img_width",  0)
    h        = meta.get("img_height", 0)

    if cap_type not in {"short", "medium"}:
        return False, "caption_type"
    if w < 256 or h < 256:
        return False, "image_size"
    words = caption.split()
    if len(words) < 4:
        return False, "too_short"
    if len(words) > 40:
        return False, "too_long"

    low = caption.lower()
    has_attr = any(a in low.split() for a in ATTRIBUTE_WORDS)
    has_rel  = any(rp in low for rp in RELATION_MAP)
    if not (has_attr or has_rel):
        return False, "no_compositional_signal"

    return True, "ok"


# ---------------------------------------------------------------------------
# Entity extraction (for canonical prompt and negatives — not runtime slot schema)
# ---------------------------------------------------------------------------

def _extract_entities(caption: str) -> List[dict]:
    """Extract [(attrs, noun)] pairs from caption — used offline only."""
    low     = caption.lower().strip().rstrip(".")
    entities = []
    seen_spans = []
    for m in _ENTITY_PAT.finditer(low):
        if any(s <= m.start() <= e or s <= m.end() <= e for s, e in seen_spans):
            continue
        seen_spans.append((m.start(), m.end()))
        attrs = [a for a in m.group(1).split() if a in ATTRIBUTE_WORDS]
        noun  = m.group(2).strip()
        if noun and len(noun) <= 30:
            entities.append({"attrs": attrs, "noun": noun})
        if len(entities) >= 4:
            break
    return entities


def build_canonical_caption(caption: str) -> str:
    """Rule-based canonical form. Used as a cleaner training input."""
    entities = _extract_entities(caption)
    if not entities:
        return caption
    parts = []
    for e in entities:
        attr_str = " ".join(e["attrs"])
        parts.append((attr_str + " " + e["noun"]).strip())

    low = caption.lower()
    relation_phrase = None
    for rp in _RELATION_KEYS_SORTED:
        if rp in low:
            relation_phrase = RELATION_MAP[rp]
            break

    if len(parts) >= 2 and relation_phrase and relation_phrase != "with":
        return f"A {parts[0]} {relation_phrase} a {parts[1]}."
    elif len(parts) >= 2:
        return f"A {parts[0]} and a {parts[1]}."
    elif len(parts) == 1:
        return f"A {parts[0]}."
    return caption


_SPATIAL_FLIP = {
    "to the left of":  "to the right of",
    "to the right of": "to the left of",
    "above":           "below",
    "below":           "above",
    "on top of":       "under",
    "under":           "on top of",
    "in front of":     "behind",
    "behind":          "in front of",
}
_SPATIAL_FLIP_KEYS = sorted(_SPATIAL_FLIP, key=len, reverse=True)


def generate_negatives(caption: str, n_negatives: int = 3) -> List[str]:
    """
    Generate corrupted captions for contrastive training.

    Priority order (most informative first so negs[0] is the hardest negative):
      1. Spatial relation reversal  (left↔right, above↔below, in front of↔behind …)
      2. Attribute swap             (red cube + blue sphere → blue cube + red sphere)
      3. Attribute drop             (red cube and blue sphere → cube and sphere)
    """
    entities = _extract_entities(caption)
    low      = caption.lower()
    candidates: List[str] = []

    # ── 1. Spatial relation reversal ────────────────────────────────────────
    for rel in _SPATIAL_FLIP_KEYS:
        if rel in low:
            flipped = low.replace(rel, _SPATIAL_FLIP[rel], 1)
            # Restore rough capitalisation
            flipped = flipped[0].upper() + flipped[1:]
            if not flipped.endswith("."):
                flipped += "."
            candidates.append(flipped)
            break

    # ── 2. Attribute swap between first two entities ────────────────────────
    if len(entities) >= 2:
        e0, e1 = entities[0], entities[1]
        if e0["attrs"] and e1["attrs"]:
            sw0 = (" ".join(e1["attrs"]) + " " + e0["noun"]).strip()
            sw1 = (" ".join(e0["attrs"]) + " " + e1["noun"]).strip()
            candidates.append(f"A {sw0} and a {sw1}.")
        elif e0["attrs"] or e1["attrs"]:
            # Only one has attrs — swap entity order as fallback
            p0 = (" ".join(e0["attrs"]) + " " + e0["noun"]).strip()
            p1 = (" ".join(e1["attrs"]) + " " + e1["noun"]).strip()
            candidates.append(f"A {p1} and a {p0}.")

    # ── 3. Attribute drop ────────────────────────────────────────────────────
    if len(entities) >= 2:
        e0, e1 = entities[0], entities[1]
        bare = [e["noun"] for e in entities[:2]]
        if bare[0] != entities[0]["noun"] or bare[1] != entities[1]["noun"]:
            candidates.append(f"A {bare[0]} and a {bare[1]}.")

    # Deduplicate and remove anything identical to canonical
    canonical = build_canonical_caption(caption).lower()
    seen: set = set()
    negatives: List[str] = []
    for n in candidates:
        nl = n.lower()
        if nl != canonical and nl not in seen:
            seen.add(nl)
            negatives.append(n)

    return negatives[:n_negatives]


# ---------------------------------------------------------------------------
# Metadata slot schema (optional — saved for debugging/analysis only)
# ---------------------------------------------------------------------------

def _build_slot_schema(caption: str) -> dict:
    """Offline metadata — not used at inference."""
    entities  = _extract_entities(caption)
    relations = []
    low = caption.lower()
    for rp in _RELATION_KEYS_SORTED:
        if rp in low and len(entities) >= 2:
            relations.append({
                "subject": "e1", "predicate": RELATION_MAP[rp], "object": "e2"
            })
            break
    return {
        "entities": [
            {"id": f"e{i+1}", "name": e["noun"], "attributes": e["attrs"],
             "count": 1, "role": "subject" if i == 0 else "object"}
            for i, e in enumerate(entities[:4])
        ],
        "relations": relations,
    }


# ---------------------------------------------------------------------------
# VLM verification
# ---------------------------------------------------------------------------

_VLM_SUFFIX = "\nAnswer with only 'yes', 'no', or 'uncertain'."

_VERIFICATION_STATEMENTS = {
    "image_quality": (
        "The image is visually coherent enough to identify the main requested objects, "
        "without severe distortion or unreadable blobs."
    ),
    "overall_match": None,  # built per-example from canonical_caption
}


class QwenVLMVerifier:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct"):
        self.model_id   = model_id
        self._model     = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._processor.tokenizer.padding_side = "left"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto",
            attn_implementation=attn_impl,
        )
        self._model.eval()

    def score_statements(self, pil_image, statements: dict) -> dict:
        """
        Batch all statements for this image into a single forward pass.
        padding_side='left' guarantees position -1 is the real last token for
        every row in the batch, so logits[i, -1] is safe to read directly.
        """
        import math
        self._load()
        from qwen_vl_utils import process_vision_info

        tok     = self._processor.tokenizer
        yes_ids = [tok.encode(s, add_special_tokens=False)[0]
                   for s in ["yes", "Yes", "YES"] if tok.encode(s, add_special_tokens=False)]
        no_ids  = [tok.encode(s, add_special_tokens=False)[0]
                   for s in ["no", "No", "NO"]   if tok.encode(s, add_special_tokens=False)]
        unc_ids = [tok.encode(s, add_special_tokens=False)[0]
                   for s in ["uncertain", "Uncertain"] if tok.encode(s, add_special_tokens=False)]

        # Build one chat message per statement, all referencing the same image.
        # We track key order explicitly to avoid any dict-ordering ambiguity.
        keys        = list(statements.keys())
        all_texts   = []
        all_images  = []
        for key in keys:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": statements[key] + _VLM_SUFFIX},
            ]}]
            text_in = self._processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(msg)
            all_texts.append(text_in)
            all_images.extend(image_inputs)   # process_vision_info returns a list

        inputs = self._processor(
            text=all_texts, images=all_images,
            return_tensors="pt", padding=True,
        ).to(self._model.device)

        with torch.inference_mode():
            out = self._model(**inputs)
        # out.logits: [N, seq_len, vocab] — with left-padding, -1 is always real last token
        batch_logits = out.logits[:, -1, :]   # [N, vocab]

        def _best(logits_row, ids):
            return max((logits_row[i].item() for i in ids), default=-1e9)

        results = {}
        for i, key in enumerate(keys):
            lrow = batch_logits[i]
            y_l, n_l, u_l = _best(lrow, yes_ids), _best(lrow, no_ids), _best(lrow, unc_ids)
            base = max(y_l, n_l, u_l)
            ey, en, eu = math.exp(y_l - base), math.exp(n_l - base), math.exp(u_l - base)
            tot = ey + en + eu + 1e-12
            results[key] = (ey / tot) + 0.5 * (eu / tot)

        return results


def run_vlm_verification(pil_image, canonical_caption: str, entities: List[dict], verifier) -> dict:
    stmts = {
        "image_quality": (
            "The image is visually coherent enough to identify the main requested objects, "
            "without severe distortion or unreadable blobs."
        ),
        "overall_match": (
            f'The image visually matches this description: "{canonical_caption}".'
        ),
    }
    for e in entities[:3]:
        attrs = " ".join(e.get("attrs", []))
        phrase = (attrs + " " + e["noun"]).strip()
        stmts[f"presence_{e.get('id', phrase)}"] = f"A {phrase} is visible and recognizable in the image."
        if e.get("attrs"):
            stmts[f"attr_{e.get('id', phrase)}"] = f"The {e['noun']} appears to be {', '.join(e['attrs'])}."

    raw  = verifier.score_statements(pil_image, stmts)
    pres = [v for k, v in raw.items() if k.startswith("presence_")]
    attr = [v for k, v in raw.items() if k.startswith("attr_")]
    overall = (
        0.30 * raw.get("overall_match",   0.5) +
        0.30 * (sum(pres) / len(pres) if pres else 0.5) +
        0.20 * (sum(attr) / len(attr) if attr else 0.5) +
        0.20 * raw.get("image_quality",   0.5)
    )
    return {
        "image_quality":    raw.get("image_quality",   0.5),
        "overall_match":    raw.get("overall_match",   0.5),
        "object_presence":  sum(pres) / len(pres) if pres else 0.5,
        "attr_correctness": sum(attr) / len(attr) if attr else 0.5,
        "overall_keep_score": overall,
    }


def passes_keep_criteria(verification: dict) -> tuple:
    v = verification
    if v["image_quality"]     < 0.80: return False, "image_quality"
    if v["object_presence"]   < 0.75: return False, "object_presence"
    if v["overall_keep_score"]< 0.75: return False, "overall_keep_score"
    return True, "ok"


# ---------------------------------------------------------------------------
# VQ token encoding
# ---------------------------------------------------------------------------

def encode_vq_tokens(pil_image, vq_model, device: str = "cuda") -> torch.Tensor:
    """Center-crop to 256×256, VQ-encode. Returns int64 tensor (256,)."""
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    w, h   = pil_image.size
    scale  = 256 / min(w, h)
    img    = pil_image.resize((int(w * scale + .5), int(h * scale + .5)), resample=3)
    img    = T.CenterCrop(256)(img)
    img_t  = (TF.to_tensor(img.convert("RGB")) * 2 - 1).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, [_, _, indices] = vq_model.encode(img_t)
    return indices.reshape(-1).cpu().to(torch.int64)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar-path",      required=True, nargs="+")
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--vq-ckpt",       default=None)
    p.add_argument("--repo-root",     default=None)
    p.add_argument("--qwen-model",    default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--caption-types", nargs="+",  default=["short", "medium"])
    p.add_argument("--max-records",   type=int,   default=5000)
    p.add_argument("--max-keep",      type=int,   default=1000)
    p.add_argument("--n-negatives",   type=int,   default=2, help="Negative captions per example")
    p.add_argument("--save-images",   action="store_true")
    p.add_argument("--save-tokens",   action="store_true")
    p.add_argument("--save-rejected", action="store_true")
    p.add_argument("--no-vlm-verify", action="store_true")
    p.add_argument("--save-slot-meta", action="store_true", help="Save slot_schema as debug metadata")
    return p.parse_args()


def _load_vq_model(vq_ckpt: str, repo_root: str, device: str = "cuda"):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tokenizer.tokenizer_image.vq_model import VQ_models
    m = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    m.to(device).eval()
    ckpt = torch.load(vq_ckpt, map_location="cpu")
    m.load_state_dict(ckpt["model"])
    del ckpt
    for p in m.parameters():
        p.requires_grad = False
    return m


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out    = Path(args.output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    if args.save_rejected:
        (out / "rejected").mkdir(parents=True, exist_ok=True)

    vq_model = None
    verifier = None

    if args.save_tokens:
        assert args.vq_ckpt and args.repo_root, "--vq-ckpt and --repo-root required for --save-tokens"
        print("[mine] Loading VQ model ...")
        vq_model = _load_vq_model(args.vq_ckpt, args.repo_root, device)

    if not args.no_vlm_verify:
        print(f"[mine] Loading VLM verifier ({args.qwen_model}) ...")
        verifier = QwenVLMVerifier(model_id=args.qwen_model)

    stats = {
        "num_records_seen":    0,
        "num_prefilter_pass":  0,
        "num_images_decoded":  0,
        "num_vlm_verified":    0,
        "num_vq_encoded":      0,
        "num_final_kept":      0,
        "reject_reasons":      {},
    }

    rejected_rows = []
    t0            = time.time()

    out_jsonl = out / "dataset.jsonl"
    _jsonl_f  = open(out_jsonl, "w")   # opened now, rows written as they are accepted

    def _reject(reason, row=None):
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1
        if args.save_rejected and row:
            rejected_rows.append({**row, "reject_reason": reason})

    for tar_path in args.tar_path:
        print(f"[mine] Processing {tar_path} ...")
        n_this_tar = 0

        for key, meta, img_name, img_bytes in iter_gpic_records(tar_path):
            if stats["num_final_kept"] >= args.max_keep:
                break
            if n_this_tar >= args.max_records:
                break
            stats["num_records_seen"] += 1
            n_this_tar += 1

            if stats["num_records_seen"] % 100 == 0:
                elapsed = time.time() - t0
                print(
                    f"[mine] seen={stats['num_records_seen']}"
                    f"  kept={stats['num_final_kept']}"
                    f"  prefilter_pass={stats['num_prefilter_pass']}"
                    f"  elapsed={elapsed:.0f}s",
                    flush=True,
                )

            ok, reason = prefilter_caption(meta)
            if not ok:
                _reject(reason)
                continue
            stats["num_prefilter_pass"] += 1

            caption = meta["caption"].strip()

            # ── Decode image ────────────────────────────────────────────────
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                _reject("image_decode_error")
                continue
            stats["num_images_decoded"] += 1

            # ── Build canonical + negatives (cheap, no LLM) ─────────────────
            canonical        = build_canonical_caption(caption)
            negative_captions = generate_negatives(caption, n_negatives=args.n_negatives)

            # Require at least one clean entity for the canonical to be useful
            entities = _extract_entities(caption)
            if not entities:
                _reject("no_entities")
                continue

            # ── VLM verification ─────────────────────────────────────────────
            verification = {
                "image_quality": 1.0, "overall_match": 1.0,
                "object_presence": 1.0, "attr_correctness": 1.0,
                "overall_keep_score": 1.0,
            }
            if not args.no_vlm_verify and verifier is not None:
                verification = run_vlm_verification(pil_img, canonical, entities, verifier)
                keep, reason = passes_keep_criteria(verification)
                if not keep:
                    _reject(reason, {"key": key, "caption": caption})
                    continue
                stats["num_vlm_verified"] += 1

            # ── Save image ────────────────────────────────────────────────────
            img_ext  = img_name.rsplit(".", 1)[-1].lower() if "." in img_name else "jpg"
            img_path = str(out / "images" / f"{key}.{img_ext}")
            if args.save_images:
                pil_img.save(img_path)

            # ── VQ encoding ────────────────────────────────────────────────────
            tokens_path = str(out / "tokens" / f"{key}.pt")
            if args.save_tokens and vq_model is not None:
                try:
                    tokens = encode_vq_tokens(pil_img, vq_model, device)
                    torch.save(tokens, tokens_path)
                    stats["num_vq_encoded"] += 1
                except Exception as e:
                    _reject(f"vq_encode:{e}")
                    continue

            # ── Build output row ──────────────────────────────────────────────
            row = {
                "source":             "gpic",
                "key":                key,
                "caption_type":       meta.get("caption_type"),
                "raw_caption":        caption,
                "canonical_caption":  canonical,
                "negative_captions":  negative_captions,
                "verification":       verification,
                "image_path":         img_path if args.save_images  else None,
                "tokens_path":        tokens_path if args.save_tokens else None,
            }
            if args.save_slot_meta:
                row["slot_schema"] = _build_slot_schema(caption)

            _jsonl_f.write(json.dumps(row) + "\n")
            _jsonl_f.flush()
            stats["num_final_kept"] += 1

            if stats["num_final_kept"] % 50 == 0:
                elapsed = time.time() - t0
                print(f"[mine]  kept={stats['num_final_kept']}  seen={stats['num_records_seen']}"
                      f"  elapsed={elapsed:.0f}s")

    # ── Finalise outputs ──────────────────────────────────────────────────────
    _jsonl_f.close()

    if args.save_rejected and rejected_rows:
        with open(out / "rejected" / "rejected_examples.jsonl", "w") as f:
            for row in rejected_rows:
                f.write(json.dumps(row) + "\n")

    with open(out / "mining_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[mine] Done.  kept={stats['num_final_kept']}  elapsed={elapsed:.0f}s")
    print(f"[mine] Output: {out_jsonl}")


if __name__ == "__main__":
    main()
