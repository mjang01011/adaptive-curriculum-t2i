"""
sample_and_score.py — generate G images per prompt, score all of them, save everything.

Saves ALL generated samples with full Qwen scores to all_samples.jsonl.
No filtering is done here — use filter_sft_dataset.py to apply thresholds later.

For the first --visualize-n prompts (default 10), generates a comparison grid
showing all 8 images ranked by reward so you can sanity-check the scorer.

Output layout:
  {output_dir}/
    all_samples.jsonl        ← one line per (prompt, sample_i) with all scores
    images/{id}_s{i}.png     ← all generated images
    tokens/{id}_s{i}.pt      ← VQ token indices for SFT training
    viz/{id}_grid.png        ← reward-ranked grid (first --visualize-n prompts)

Usage:
  python SFT/sample_and_score.py \\
    --prompts-jsonl data/easy_compositional_prompts_500.jsonl \\
    --output-dir    outputs/reward_sft_data \\
    --repo-root     LlamaGen \\
    --gpt-ckpt      pretrained/t2i_XL_stage1_256.pt \\
    --vq-ckpt       pretrained/vq_ds16_t2i.pt \\
    --t5-path       pretrained/t5-ckpt \\
    --qwen-model    Qwen/Qwen3-VL-4B-Instruct \\
    --gen-count 8 --visualize-n 10 --resume
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict

import torch


# ---------------------------------------------------------------------------
# Rubric templates
# ---------------------------------------------------------------------------

RUBRIC_TEMPLATE = (
    'You are an expert evaluator for a text-to-image compositionality benchmark.\n\n'
    'Your task is to decide whether the image visually supports one specific statement '
    'about the requested scene.\n\n'
    'Original text prompt:\n"{prompt}"\n\n'
    'Evaluation rules:\n'
    '1. Judge only what is clearly visible in the image.\n'
    '2. Do not infer hidden, cropped, tiny, or ambiguous objects.\n'
    '3. Do not give credit for broad semantic similarity if the specific statement is false.\n'
    '4. If an object is missing, merged, unrecognizable, or only suggested, answer "no".\n'
    '5. If the object is visible but the attribute is unclear, answer "uncertain".\n'
    '6. For attribute binding, verify the attribute belongs to the correct object.\n'
    '7. For spatial relations, verify the relation between two distinct objects.\n'
    '8. For counting, count only clearly visible, distinct instances.\n'
    '9. For interactions, verify both objects are visible and the interaction is supported.\n'
    '10. Minor stylization or blur is acceptable if the object/attribute remains recognizable.\n'
    '11. Use "yes" only when the statement is clearly supported.\n'
    '12. Use "no" when the statement is false, contradicted, or depends on a missing object.\n'
    '13. Use "uncertain" when the statement may be true but the image is too unclear.\n\n'
    'Statement to verify:\n{statement}\n\n'
    'Answer with exactly one word: yes, no, or uncertain.'
)

MATCH_TEMPLATE = (
    'You are an expert evaluator for text-to-image alignment.\n\n'
    'Your task is to decide whether the image visually matches the caption.\n\n'
    'Caption:\n"{caption}"\n\n'
    'Evaluation rules:\n'
    '1. Judge only visible evidence in the image.\n'
    '2. The image must match the main objects, attributes, counts, and relations.\n'
    '3. Do not infer hidden or ambiguous objects.\n'
    '4. If a key object is missing or unrecognizable, answer "no".\n'
    '5. If a key attribute is assigned to the wrong object, answer "no".\n'
    '6. If a spatial or interaction relation is contradicted, answer "no".\n'
    '7. Use "yes" only if the image clearly supports the caption.\n'
    '8. Use "uncertain" only if ambiguous but not clearly wrong.\n\n'
    'Answer with exactly one word: yes, no, or uncertain.'
)


# ---------------------------------------------------------------------------
# Statement builders
# ---------------------------------------------------------------------------

_SPATIAL_HUMAN = {
    "under": "under", "on_top_of": "on top of", "on top of": "on top of",
    "left_of": "to the left of", "right_of": "to the right of",
    "left of": "to the left of", "right of": "to the right of",
    "in front of": "in front of", "behind": "behind",
    "above": "above", "below": "below",
}
_SPATIAL_OPPOSITE = {
    "under": "on top of", "on_top_of": "under", "on top of": "under",
    "left_of": "to the right of", "right_of": "to the left of",
    "left of": "to the right of", "right of": "to the left of",
    "in front of": "behind", "behind": "in front of",
    "above": "below", "below": "above",
}
_INTERACTION_NEG = {
    "wearing": "not wearing", "carrying": "not carrying", "holding": "not holding",
    "next_to": "far away from", "next to": "far away from",
    "near": "far away from", "beside": "far away from",
    "touching": "not touching", "pushing": "not pushing",
    "eating": "not eating", "riding": "not riding",
}


def _entity_desc(e: dict) -> str:
    attrs = " ".join(e.get("attributes", []))
    obj   = e.get("object", "object")
    count = e.get("count", 1)
    cword = {1: "a", 2: "two", 3: "three", 4: "four"}.get(count, str(count))
    plural = "s" if count > 1 else ""
    return f"{cword} {attrs} {obj}{plural}" if attrs else f"{cword} {obj}{plural}"


def build_statements(row: dict) -> Dict[str, str]:
    category = row["category"]
    entities = row["entities"]
    relations = row.get("relations", [])
    eid = {e["id"]: e for e in entities}

    stmts: Dict[str, str] = {}
    obj_list = " and ".join(_entity_desc(e) for e in entities)
    stmts["presence"] = (
        f"The image contains the main requested objects: {obj_list}, "
        "each separately identifiable."
    )
    stmts["quality"] = (
        "The image is visually coherent enough to identify the main requested objects, "
        "without severe distortion or unreadable blobs."
    )
    stmts["alignment"] = (
        "The image broadly depicts the original text prompt without changing the requested scene."
    )

    if category == "attribute_binding":
        e1, e2 = entities[0], entities[1]
        a1 = " ".join(e1.get("attributes", []))
        a2 = " ".join(e2.get("attributes", []))
        o1, o2 = e1["object"], e2["object"]
        stmts["target_binding"] = (
            f"The image contains a visible {a1} {o1} and a visible {a2} {o2}, "
            "with the attributes bound to the correct objects."
        )
        stmts["swapped_binding"] = (
            f"The image instead shows a {a2} {o1} and a {a1} {o2} "
            "(attributes on wrong objects)."
        )
        stmts["obj1_attr"] = f"The {o1} is visible and recognizable, and the {o1} is {a1}."
        stmts["obj2_attr"] = f"The {o2} is visible and recognizable, and the {o2} is {a2}."

    elif category == "spatial_relation" and relations:
        rel    = relations[0]
        se, oe = eid[rel["subject"]], eid[rel["object"]]
        pred   = rel["predicate"]
        human  = _SPATIAL_HUMAN.get(pred, pred.replace("_", " "))
        opp    = _SPATIAL_OPPOSITE.get(pred, f"not {human}")
        a1     = " ".join(se.get("attributes", []))
        a2     = " ".join(oe.get("attributes", []))
        sd     = f"{a1} {se['object']}" if a1 else se["object"]
        od     = f"{a2} {oe['object']}" if a2 else oe["object"]
        stmts["distinct_objects"] = (
            f"The {sd} and the {od} are distinct objects, not merged or ambiguous."
        )
        stmts["target_relation"]   = f"The {sd} is clearly {human} the {od}."
        stmts["opposite_relation"] = f"The {sd} is clearly {opp} the {od}."
        stmts["relation_clarity"]  = (
            f"The spatial arrangement between the {sd} and the {od} is clear enough to determine."
        )
        if se.get("attributes"):
            stmts["obj1_attr"] = f"The {se['object']} is {' '.join(se['attributes'])}."
        if oe.get("attributes"):
            stmts["obj2_attr"] = f"The {oe['object']} is {' '.join(oe['attributes'])}."

    elif category == "counting":
        count_desc = " and ".join(
            f"exactly {e.get('count',1)} {' '.join(e.get('attributes',[]))} "
            f"{e['object']}{'s' if e.get('count',1)>1 else ''}"
            for e in entities
        )
        stmts["target_count"] = f"The image contains {count_desc}."
        stmts["wrong_count"] = (
            "The image contains the wrong number of objects (too many, too few, or missing)."
        )
        stmts["distinct_instances"] = (
            "Each counted instance is distinct and separately visible, "
            "not a shadow, reflection, or fragment."
        )

    elif category in ("interaction_relation", "mixed") and relations:
        rel    = relations[0]
        se, oe = eid[rel["subject"]], eid[rel["object"]]
        pred   = rel["predicate"]
        human  = pred.replace("_", " ")
        neg_p  = _INTERACTION_NEG.get(pred, _INTERACTION_NEG.get(human, f"not {human}"))
        a1     = " ".join(se.get("attributes", []))
        a2     = " ".join(oe.get("attributes", []))
        sd     = f"{a1} {se['object']}" if a1 else se["object"]
        od     = f"{a2} {oe['object']}" if a2 else oe["object"]
        stmts["target_interaction"]   = f"The {sd} is clearly {human} the {od}."
        stmts["negative_interaction"] = f"The {od} is visible but the {sd} is {neg_p} the {od}."
        if se.get("attributes"):
            stmts["obj1_attr"] = f"The {se['object']} is {' '.join(se['attributes'])}."
        if oe.get("attributes"):
            stmts["obj2_attr"] = f"The {oe['object']} is {' '.join(oe['attributes'])}."
    else:
        # fallback
        for i, e in enumerate(entities[:2], 1):
            a = " ".join(e.get("attributes", []))
            if a:
                stmts[f"obj{i}_attr"] = f"The {e['object']} is {a}."

    return stmts


# ---------------------------------------------------------------------------
# Reward formulas
# ---------------------------------------------------------------------------

def _soft(p: dict) -> float:
    return p["yes"] + 0.5 * p["uncertain"]

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _contrast(t: float, n: float, tau: float = 0.20) -> float:
    return _sigmoid((t - n) / tau)

def compute_reward(category: str, scores: Dict[str, float]) -> float:
    g = scores.get
    if category == "attribute_binding":
        attr_sub = 0.5 * g("obj1_attr", 0.5) + 0.5 * g("obj2_attr", 0.5)
        return (
            0.25 * g("presence", 0.5) +
            0.40 * _contrast(g("target_binding", 0.5), g("swapped_binding", 0.5)) +
            0.15 * attr_sub +
            0.10 * g("alignment", 0.5) +
            0.10 * g("quality", 0.5)
        )
    elif category == "spatial_relation":
        pres  = g("presence", 0.5)
        dist  = g("distinct_objects", 0.5)
        eff   = pres * dist * _contrast(g("target_relation", 0.5), g("opposite_relation", 0.5))
        return (
            0.30 * pres + 0.20 * dist + 0.35 * eff +
            0.05 * g("relation_clarity", 0.5) +
            0.05 * g("alignment", 0.5) + 0.05 * g("quality", 0.5)
        )
    elif category == "counting":
        return (
            0.25 * g("presence", 0.5) +
            0.45 * _contrast(g("target_count", 0.5), g("wrong_count", 0.5)) +
            0.20 * g("distinct_instances", 0.5) +
            0.10 * g("quality", 0.5)
        )
    else:  # interaction_relation / mixed
        attr_sub = 0.5 * g("obj1_attr", 0.5) + 0.5 * g("obj2_attr", 0.5)
        return (
            0.25 * g("presence", 0.5) +
            0.40 * _contrast(g("target_interaction", 0.5), g("negative_interaction", 0.5)) +
            0.15 * attr_sub +
            0.10 * g("alignment", 0.5) +
            0.10 * g("quality", 0.5)
        )

def compute_weight(reward: float) -> float:
    return float(min(1.0, max(0.1, (reward - 0.5) / 0.5)))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_reward_grid(pil_images, scores_list, rewards, prompt: str,
                     neg_score_list, out_path: Path,
                     reward_thresh: float = 0.65, neg_thresh: float = 0.40):
    """
    Create a grid of all G images ranked by reward.
    Green border = passes threshold. Red border = fails.
    Component scores printed below each image.
    """
    from PIL import Image, ImageDraw, ImageFont

    G      = len(pil_images)
    order  = sorted(range(G), key=lambda i: rewards[i], reverse=True)
    cols   = 4
    rows   = math.ceil(G / cols)
    W, H   = pil_images[0].size   # 256×256
    pad    = 6
    text_h = 80
    title_h = 30

    canvas_w = cols * (W + 2 * pad) + pad
    canvas_h = title_h + rows * (H + 2 * pad + text_h) + pad
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw     = ImageDraw.Draw(canvas)

    try:
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_md = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font_sm = font_md = ImageFont.load_default()

    # Title
    draw.text((pad, 4), f"Prompt: {prompt[:90]}", fill=(220, 220, 220), font=font_sm)

    for rank, orig_i in enumerate(order):
        col = rank % cols
        row = rank // cols
        x   = pad + col * (W + 2 * pad)
        y   = title_h + row * (H + 2 * pad + text_h)

        r       = rewards[orig_i]
        neg_s   = neg_score_list[orig_i]
        passes  = (r >= reward_thresh and neg_s <= neg_thresh)
        border_color = (50, 220, 80) if passes else (220, 60, 60)

        # Draw border
        draw.rectangle([x - pad, y - pad,
                        x + W + pad - 1, y + H + pad - 1],
                       outline=border_color, width=3)

        # Paste image
        canvas.paste(pil_images[orig_i], (x, y))

        # Rank + reward header
        draw.text((x, y - pad + 1),
                  f"#{rank+1}  R={r:.3f}  neg={neg_s:.2f}",
                  fill=border_color, font=font_md)

        # Component scores below image
        sc   = scores_list[orig_i]
        keys = [k for k in sc if k != "neg_caption_score"][:6]
        lines = [f"{k[:16]}={sc[k]:.2f}" for k in keys]
        for li, line in enumerate(lines):
            draw.text((x, y + H + pad + li * 12),
                      line, fill=(160, 180, 160), font=font_sm)

    canvas.save(str(out_path))


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-jsonl",  required=True)
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--repo-root",      required=True)
    p.add_argument("--gpt-ckpt",       required=True)
    p.add_argument("--vq-ckpt",        required=True)
    p.add_argument("--t5-path",        required=True)
    p.add_argument("--qwen-model",     default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--gen-count",      type=int,   default=8)
    p.add_argument("--cfg-scale",      type=float, default=7.5)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--top-k-gen",      type=int,   default=2000)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--visualize-n",    type=int,   default=10,
                   help="Generate reward grid for first N prompts")
    p.add_argument("--viz-reward-thresh", type=float, default=0.65,
                   help="Threshold used for green/red border in viz grids")
    p.add_argument("--viz-neg-thresh",    type=float, default=0.40)
    p.add_argument("--precision",      default="bf16")
    p.add_argument("--resume",         action="store_true",
                   help="Skip prompts already in all_samples.jsonl")
    p.add_argument("--start-idx",      type=int,   default=0)
    p.add_argument("--end-idx",        type=int,   default=None)
    p.add_argument("--score-chunk-size", type=int, default=32,
                   help="Max (image, statement) pairs per Qwen forward pass")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    out_dir  = Path(args.output_dir)
    img_dir  = out_dir / "images"
    tok_dir  = out_dir / "tokens"
    viz_dir  = out_dir / "viz"
    all_jsonl = out_dir / "all_samples.jsonl"

    for d in (out_dir, img_dir, tok_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Resume: collect already-processed prompt IDs
    done_ids: set = set()
    if args.resume and all_jsonl.exists():
        with open(all_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r["prompt_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} prompts already done", flush=True)

    # Load prompts
    prompts = []
    with open(args.prompts_jsonl) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    end_idx = args.end_idx or len(prompts)
    prompts = prompts[args.start_idx:end_idx]
    print(f"[data] {len(prompts)} prompts  gen_count={args.gen_count}", flush=True)

    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # ── Load LlamaGen ─────────────────────────────────────────────────────────
    print("[model] Loading LlamaGen ...", flush=True)
    from adaptive_curriculum.model.llamagen_wrapper import LlamaGenWrapper
    wrapper = LlamaGenWrapper(
        repo_root=args.repo_root, vq_ckpt=args.vq_ckpt,
        gpt_ckpt=args.gpt_ckpt, t5_path=args.t5_path, precision=args.precision,
    )
    gpt   = wrapper.gpt.to(device)
    vq    = wrapper.vq_model.to(device)
    t5    = wrapper.t5
    t5.model.to(device)
    ls    = wrapper.latent_size
    cb    = wrapper.codebook_embed_dim
    dtype = wrapper.dtype

    # ── Load Qwen scorer ──────────────────────────────────────────────────────
    print("[model] Loading Qwen3-VL scorer ...", flush=True)
    try:
        from CARGO.scoring import CARGORewardModel
        scorer = CARGORewardModel(model_id=args.qwen_model)
    except ImportError:
        from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
        scorer = Qwen3VLRewardModel(model_id=args.qwen_model)

    from autoregressive.models.generate import generate
    import torchvision.transforms.functional as TF

    t0 = time.time()
    n_total = 0

    with open(all_jsonl, "a") as out_f:
        for pi, row in enumerate(prompts):
            pid      = row["id"]
            prompt   = row["prompt"]
            category = row["category"]
            neg_prompts = row.get("negative_prompts", [])

            if pid in done_ids:
                continue

            print(f"\n[{pi+1}/{len(prompts)}] {pid}  '{prompt}'", flush=True)

            # ── Step 1: Generate G images ──────────────────────────────────
            gpt.eval()
            with torch.no_grad():
                embs, masks = t5.get_text_embeddings([prompt])
                emb, mask   = embs[0], masks[0]
                valid       = int(mask.sum().item())
                shifted     = torch.cat([emb[valid:], emb[:valid]])
                mask_s      = torch.flip(mask, dims=[-1])
                c_idx       = (shifted * mask_s[:, None]).to(device=device, dtype=dtype).unsqueeze(0)
                c_mask      = mask_s.to(device=device, dtype=dtype).unsqueeze(0)

            qzshape  = [1, cb, ls, ls]
            pil_list = []
            tok_list = []

            for si in range(args.gen_count):
                torch.manual_seed(args.seed + pi * 1000 + si)
                try:
                    with torch.no_grad():
                        idx     = generate(gpt, c_idx, ls**2, c_mask,
                                           cfg_scale=args.cfg_scale,
                                           temperature=args.temperature,
                                           top_k=args.top_k_gen, top_p=1.0,
                                           sample_logits=True)
                        decoded = vq.decode_code(idx, qzshape)
                    img_t = (decoded[0].float().clamp(-1, 1) + 1) / 2
                    pil   = TF.to_pil_image(img_t.cpu())
                    toks  = idx.reshape(-1).cpu()
                    pil_list.append(pil)
                    tok_list.append(toks)
                except Exception as e:
                    print(f"  [warn] gen sample {si} failed: {e}", flush=True)
                    pil_list.append(None)
                    tok_list.append(None)

            # ── Step 2: Score all generated images with Qwen ──────────────
            # Swap to Qwen GPU, GPT to CPU
            gpt.cpu(); t5.model.cpu(); torch.cuda.empty_cache()
            scorer._load()

            statements = build_statements(row)
            stmt_keys  = list(statements.keys())
            stmt_vals  = list(statements.values())
            n_rubric   = len(stmt_vals)
            n_neg      = len(neg_prompts)
            n_per_img  = n_rubric + n_neg

            # Build ONE mega-batch: all valid images × all statements/negatives.
            # This lets Qwen process the whole prompt's worth in a single generate
            # call (chunked by --score-chunk-size) instead of G separate calls.
            valid_si = [si for si, pil in enumerate(pil_list) if pil is not None]
            mega_pairs: list = []
            for si in valid_si:
                pil = pil_list[si]
                mega_pairs += [
                    (pil, RUBRIC_TEMPLATE.format(prompt=prompt, statement=s))
                    for s in stmt_vals
                ]
                if neg_prompts:
                    mega_pairs += [
                        (pil, MATCH_TEMPLATE.format(caption=cap))
                        for cap in neg_prompts
                    ]

            print(
                f"  [score] {len(valid_si)} images × {n_per_img} stmts "
                f"= {len(mega_pairs)} pairs  chunk={args.score_chunk_size}",
                flush=True,
            )
            all_probs = (
                scorer._forward_probs_batch(mega_pairs, chunk_size=args.score_chunk_size)
                if mega_pairs else []
            )

            # Unpack results back to per-image lists
            all_scores_list: list = [{} for _ in pil_list]
            all_rewards:     list = [0.0] * len(pil_list)
            all_neg_scores:  list = [1.0 if pil is None else 0.0
                                     for pil in pil_list]

            for vi, si in enumerate(valid_si):
                start      = vi * n_per_img
                comp_probs = all_probs[start : start + n_rubric]
                neg_probs_ = all_probs[start + n_rubric : start + n_per_img]

                comp_scores = {k: round(_soft(p), 4)
                               for k, p in zip(stmt_keys, comp_probs)}
                neg_score   = round(
                    max((_soft(p) for p in neg_probs_), default=0.0), 4
                )
                comp_scores["neg_caption_score"] = neg_score

                reward = round(compute_reward(category, comp_scores), 4)

                all_scores_list[si] = comp_scores
                all_rewards[si]     = reward
                all_neg_scores[si]  = neg_score

                print(
                    f"  s{si}  reward={reward:.3f}  neg={neg_score:.3f}  "
                    f"presence={comp_scores.get('presence', 0):.2f}",
                    flush=True,
                )

            # ── Step 3: Save images, tokens, records ──────────────────────
            # Bring GPT back for next iteration
            gpt.to(device); t5.model.to(device)
            if hasattr(scorer, "_model") and scorer._model is not None:
                scorer._model.cpu()
            torch.cuda.empty_cache()

            for si, (pil, toks) in enumerate(zip(pil_list, tok_list)):
                if pil is None or toks is None:
                    continue
                sample_id = f"{pid}_s{si}"
                img_path  = img_dir / f"{sample_id}.png"
                tok_path  = tok_dir / f"{sample_id}.pt"

                pil.save(str(img_path))
                torch.save(toks, str(tok_path))

                record = {
                    "id":               sample_id,
                    "prompt_id":        pid,
                    "sample_index":     si,
                    "category":         category,
                    "prompt":           prompt,
                    "negative_prompts": neg_prompts,
                    "tokens_path":      str(tok_path.resolve()),
                    "image_path":       str(img_path.resolve()),
                    "reward":           all_rewards[si],
                    "weight":           round(compute_weight(all_rewards[si]), 4),
                    "scores":           all_scores_list[si],
                }
                out_f.write(json.dumps(record) + "\n")
                n_total += 1

            out_f.flush()

            # ── Step 4: Visualization grid for first N prompts ────────────
            valid_pils = [p for p in pil_list if p is not None]
            if pi < args.visualize_n and valid_pils:
                valid_idx  = [i for i, p in enumerate(pil_list) if p is not None]
                viz_pils   = [pil_list[i] for i in valid_idx]
                viz_scores = [all_scores_list[i] for i in valid_idx]
                viz_rews   = [all_rewards[i]     for i in valid_idx]
                viz_negs   = [all_neg_scores[i]  for i in valid_idx]

                viz_path = viz_dir / f"{pid}_grid.png"
                try:
                    make_reward_grid(
                        viz_pils, viz_scores, viz_rews, prompt, viz_negs,
                        viz_path,
                        reward_thresh=args.viz_reward_thresh,
                        neg_thresh=args.viz_neg_thresh,
                    )
                    print(f"  [viz] saved {viz_path}", flush=True)
                except Exception as e:
                    print(f"  [viz] failed: {e}", flush=True)

            elapsed = time.time() - t0
            best_r  = max(all_rewards) if all_rewards else 0.0
            worst_r = min(all_rewards) if all_rewards else 0.0
            print(
                f"  best={best_r:.3f}  worst={worst_r:.3f}  "
                f"total_saved={n_total}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    print(f"\n[done] {n_total} samples saved to {all_jsonl}", flush=True)
    print(f"  Run filter_sft_dataset.py to apply reward thresholds.", flush=True)


if __name__ == "__main__":
    main()
