"""
Qwen localization smoke test.

Asks Qwen-VL to return bounding boxes for objects in generated images,
converts boxes to 16×16 token masks, saves overlays, and measures timing
vs the existing yes/no scorer.

Usage:
  python scripts_qwen/smoke_test_qwen_localization.py \
    --samples-jsonl  /path/to/all_samples.jsonl \
    --images-root    /path/to/pairs_dir \
    --out-dir        /path/to/outputs_qwen_localization/smoke_$(date +%Y%m%d_%H%M%S) \
    --num-samples 20 \
    --grid-size 16 \
    --mask-dilate 1 \
    --run-modes attr_phrase plain_object \
    --compare-yesno \
    --save-overlays \
    --source-jsonl /path/to/attribute_binding_train_500.jsonl
"""
import argparse
import json
import math
import re
import sys
import time
from math import ceil, floor
from pathlib import Path

from PIL import Image, ImageDraw

_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO))


# ── bbox → token mask ─────────────────────────────────────────────────────────

def bbox_to_token_mask(bbox_norm, grid_size=16, dilate=0):
    if bbox_norm is None:
        return None
    x1, y1, x2, y2 = bbox_norm
    gx1 = max(0, floor(x1 * grid_size))
    gy1 = max(0, floor(y1 * grid_size))
    gx2 = min(grid_size, ceil(x2 * grid_size))
    gy2 = min(grid_size, ceil(y2 * grid_size))
    mask = [[0] * grid_size for _ in range(grid_size)]
    for gy in range(gy1, gy2):
        for gx in range(gx1, gx2):
            mask[gy][gx] = 1
    if dilate > 0:
        dilated = [row[:] for row in mask]
        for gy in range(grid_size):
            for gx in range(grid_size):
                if mask[gy][gx]:
                    for dy in range(-dilate, dilate + 1):
                        for dx in range(-dilate, dilate + 1):
                            ny, nx = gy + dy, gx + dx
                            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                                dilated[ny][nx] = 1
        mask = dilated
    return mask


def num_tokens(mask):
    if mask is None:
        return 0
    return sum(c for row in mask for c in row)


# ── bbox validation ───────────────────────────────────────────────────────────

def validate_bbox(bbox):
    if bbox is None:
        return False, "null"
    if len(bbox) != 4:
        return False, "wrong_length"
    x1, y1, x2, y2 = bbox
    if not all(isinstance(v, (int, float)) for v in bbox):
        return False, "non_numeric"
    if x2 <= x1 or y2 <= y1:
        return False, "inverted"
    if any(v < 0 or v > 1 for v in bbox):
        return False, "out_of_range"
    area = (x2 - x1) * (y2 - y1)
    if area > 0.75:
        return True, "too_large"
    if area < 0.01:
        return True, "too_small"
    return True, "ok"


# ── Qwen localization prompt ──────────────────────────────────────────────────

def build_localization_prompt(prompt, obj_phrases):
    obj_list = "\n".join(f'{i + 1}. "{p}"' for i, p in enumerate(obj_phrases))
    return f"""You are given an image and a text prompt.

Prompt: "{prompt}"

Find the visible regions corresponding to the following objects:
{obj_list}

Return ONLY valid JSON with this schema:
{{
  "objects": [
    {{
      "label": "object phrase",
      "visible": true or false,
      "confidence": number from 0 to 1,
      "bbox": [x1, y1, x2, y2],
      "notes": "short explanation"
    }}
  ]
}}

Bounding boxes must be normalized coordinates in [0, 1], where:
x1,y1 = top-left corner
x2,y2 = bottom-right corner

If an object is not visible, set visible=false, confidence=0, and bbox=null.
Do not include markdown. Do not include any text outside JSON."""


def parse_localization_response(raw_text, num_objects):
    text = raw_text.strip()
    data = None

    for pattern in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r"(\{.*\})", t, re.DOTALL).group(1)),
    ]:
        if data is not None:
            break
        try:
            data = pattern(text)
        except Exception:
            pass

    if data is None or "objects" not in data:
        return None, False

    repair_used = False
    result = []
    for obj in list(data["objects"])[:num_objects]:
        bbox = obj.get("bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                bbox = [float(v) for v in bbox]
                if any(v > 1.5 for v in bbox):
                    bbox = [v / 256.0 for v in bbox]
                    repair_used = True
            except Exception:
                bbox = None
        else:
            bbox = None

        result.append({
            "label":      obj.get("label", ""),
            "visible":    bool(obj.get("visible", False)),
            "confidence": float(obj.get("confidence", 0.0)),
            "bbox_norm":  bbox,
            "notes":      obj.get("notes", ""),
        })

    while len(result) < num_objects:
        result.append({"label": "", "visible": False, "confidence": 0.0, "bbox_norm": None, "notes": ""})
        repair_used = True

    return result, repair_used


# ── overlay visualization ─────────────────────────────────────────────────────

_COLORS = [(255, 80, 80), (80, 180, 255), (80, 255, 80), (255, 200, 0)]


def draw_overlay(pil_img, obj_results, queries, grid_size=16):
    img = pil_img.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    cell_w = W / grid_size
    cell_h = H / grid_size
    for i in range(1, grid_size):
        draw.line([(int(i * cell_w), 0), (int(i * cell_w), H)], fill=(60, 60, 60), width=1)
        draw.line([(0, int(i * cell_h)), (W, int(i * cell_h))], fill=(60, 60, 60), width=1)

    for idx, (obj, query) in enumerate(zip(obj_results, queries)):
        color = _COLORS[idx % len(_COLORS)]
        bbox  = obj.get("bbox_norm")
        conf  = obj.get("confidence", 0.0)

        if bbox and obj.get("visible"):
            x1 = int(bbox[0] * W)
            y1 = int(bbox[1] * H)
            x2 = int(bbox[2] * W)
            y2 = int(bbox[3] * H)
            if idx == 0:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            else:
                seg = 8
                for x in range(x1, x2, seg * 2):
                    draw.line([(x, y1), (min(x + seg, x2), y1)], fill=color, width=2)
                    draw.line([(x, y2), (min(x + seg, x2), y2)], fill=color, width=2)
                for y in range(y1, y2, seg * 2):
                    draw.line([(x1, y), (x1, min(y + seg, y2))], fill=color, width=2)
                    draw.line([(x2, y), (x2, min(y + seg, y2))], fill=color, width=2)
            draw.text((x1 + 2, y1 + 2), f"{query} ({conf:.2f})", fill=color)
        else:
            draw.text((4, 4 + idx * 14), f"{query}: not visible", fill=color)

    return img


def draw_grid_mask(obj_results, grid_size=16, dilate=0):
    cell = 16
    size = grid_size * cell
    img  = Image.new("RGB", (size, size), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    for idx, obj in enumerate(obj_results):
        if not obj.get("visible") or obj.get("bbox_norm") is None:
            continue
        color = _COLORS[idx % len(_COLORS)]
        mask  = bbox_to_token_mask(obj["bbox_norm"], grid_size, dilate)
        if mask is None:
            continue
        for gy in range(grid_size):
            for gx in range(grid_size):
                if mask[gy][gx]:
                    x0 = gx * cell
                    y0 = gy * cell
                    draw.rectangle(
                        [x0, y0, x0 + cell - 1, y0 + cell - 1],
                        fill=(color[0] // 2, color[1] // 2, color[2] // 2),
                        outline=color,
                    )

    for i in range(grid_size + 1):
        draw.line([(i * cell, 0), (i * cell, size)], fill=(80, 80, 80), width=1)
        draw.line([(0, i * cell), (size, i * cell)], fill=(80, 80, 80), width=1)

    return img


# ── object phrase extraction ──────────────────────────────────────────────────

def extract_objects(sample, source_map):
    if "objects" in sample:
        return sample["objects"]
    if "obj1" in sample:
        objs = [{"name": sample["obj1"], "attribute": ""}]
        if "obj2" in sample:
            objs.append({"name": sample["obj2"], "attribute": ""})
        return objs
    base_id = sample["id"].rsplit("_seed", 1)[0] if "_seed" in sample["id"] else sample["id"]
    if base_id in source_map:
        meta = source_map[base_id].get("metadata", {})
        if "objects" in meta:
            return meta["objects"]
    return None


def get_phrases(objects, mode):
    phrases = []
    for obj in objects:
        name = obj.get("name", "")
        attr = obj.get("attribute", "")
        if mode == "attr_phrase" and attr:
            phrases.append(f"{attr} {name}")
        else:
            phrases.append(name)
    return phrases


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-jsonl",             required=True)
    parser.add_argument("--images-root",               required=True)
    parser.add_argument("--out-dir",                   required=True)
    parser.add_argument("--num-samples",               type=int, default=20)
    parser.add_argument("--grid-size",                 type=int, default=16)
    parser.add_argument("--mask-dilate",               type=int, default=1)
    parser.add_argument("--run-modes",                 nargs="+",
                        default=["attr_phrase", "plain_object"],
                        choices=["attr_phrase", "plain_object"])
    parser.add_argument("--compare-yesno",             action="store_true")
    parser.add_argument("--save-overlays",             action="store_true")
    parser.add_argument("--source-jsonl",              default=None)
    parser.add_argument("--infer-objects-from-prompt", action="store_true")
    parser.add_argument("--qwen-model",                default="Qwen/Qwen3-VL-4B-Instruct")
    args = parser.parse_args()

    out_dir     = Path(args.out_dir)
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    images_root = Path(args.images_root)

    # ── load samples ──────────────────────────────────────────────────────────
    samples = []
    with open(args.samples_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    valid_samples = []
    for s in samples:
        if (images_root / s["image_path"]).exists():
            valid_samples.append(s)
        if len(valid_samples) >= args.num_samples:
            break
    print(f"[smoke] {len(valid_samples)} samples with images")

    # ── load source metadata ──────────────────────────────────────────────────
    source_map = {}
    if args.source_jsonl:
        with open(args.source_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    source_map[d["id"]] = d
        print(f"[smoke] loaded {len(source_map)} source items")

    # ── load Qwen ─────────────────────────────────────────────────────────────
    from adaptive_curriculum.reward.vlm_reward import Qwen3VLRewardModel
    reward_model = Qwen3VLRewardModel(model_id=args.qwen_model)
    reward_model._load()
    print("[smoke] Qwen loaded\n")

    raw_outputs  = []
    parsed_rows  = []
    timing_rows  = []

    for sample in valid_samples:
        sid      = sample["id"]
        prompt   = sample["prompt"]
        img_path = images_root / sample["image_path"]
        pil_img  = Image.open(img_path).convert("RGB")

        objects = extract_objects(sample, source_map)
        if objects is None:
            if args.infer_objects_from_prompt:
                objects = [{"name": prompt, "attribute": ""}]
            else:
                print(f"  [skip] {sid}: no object metadata (use --source-jsonl or --infer-objects-from-prompt)")
                continue

        print(f"── {sid}")
        print(f"   prompt: {prompt}")

        # yes/no baseline
        yesno_sec = None
        if args.compare_yesno:
            base_id = sid.rsplit("_seed", 1)[0] if "_seed" in sid else sid
            if base_id in source_map:
                from adaptive_curriculum.data.schemas import BucketItem
                src_item = BucketItem.from_dict(source_map[base_id])
                t0 = time.time()
                reward_model.score_image(pil_img, src_item, mode="pseudo_soft_grpo_target_heavy")
                yesno_sec = round(time.time() - t0, 3)
                print(f"   yesno: {yesno_sec:.2f}s")

        # localization per mode
        for mode in args.run_modes:
            phrases      = get_phrases(objects, mode)
            prompt_text  = build_localization_prompt(prompt, phrases)
            inputs       = reward_model._build_inputs(pil_img, prompt_text)

            t0      = time.time()
            raw     = reward_model._generate_text(inputs, max_new_tokens=512)
            elapsed = round(time.time() - t0, 3)

            raw_outputs.append({"id": sid, "mode": mode, "raw": raw, "runtime_sec": elapsed})

            parsed, repair_used = parse_localization_response(raw, len(phrases))
            parse_ok = parsed is not None

            obj_results = []
            if parsed:
                W, H = pil_img.size
                for obj, query in zip(parsed, phrases):
                    bn = obj["bbox_norm"]
                    valid, vnote = validate_bbox(bn)
                    bp = ([int(bn[0]*W), int(bn[1]*H), int(bn[2]*W), int(bn[3]*H)]
                          if bn else None)
                    mask_raw = bbox_to_token_mask(bn, args.grid_size, dilate=0)
                    mask_dil = bbox_to_token_mask(bn, args.grid_size, dilate=args.mask_dilate)
                    obj_results.append({
                        "query":                    query,
                        "label":                    obj["label"],
                        "visible":                  obj["visible"],
                        "confidence":               round(obj["confidence"], 3),
                        "bbox_norm":                [round(v, 4) for v in bn] if bn else None,
                        "bbox_px":                  bp,
                        "bbox_valid":               valid,
                        "bbox_validity_note":       vnote,
                        "token_mask_16x16":         mask_raw,
                        "token_mask_16x16_dilated": mask_dil,
                        "num_tokens":               num_tokens(mask_raw),
                        "num_tokens_dilated":       num_tokens(mask_dil),
                    })

            parsed_rows.append({
                "id":              sid,
                "prompt":          prompt,
                "image_path":      sample["image_path"],
                "qwen_mode":       mode,
                "runtime_sec":     elapsed,
                "objects":         obj_results,
                "parse_success":   parse_ok,
                "json_repair_used": repair_used,
            })
            timing_rows.append({
                "id":           sid,
                "mode":         mode,
                "runtime_sec":  elapsed,
                "yesno_sec":    yesno_sec,
                "parse_success": parse_ok,
                "num_visible":  sum(1 for o in obj_results if o.get("visible")),
            })

            summary = " | ".join(
                f"{o['query']}: {'vis' if o['visible'] else 'NOT_VIS'} "
                f"conf={o['confidence']:.2f} toks={o['num_tokens']}"
                for o in obj_results
            ) if obj_results else "parse_failed"
            print(f"   [{mode}] {elapsed:.2f}s  {summary}")

            if args.save_overlays and parsed:
                draw_overlay(pil_img, obj_results, phrases, args.grid_size).save(
                    overlay_dir / f"{sid}_{mode}_overlay.png")
                draw_grid_mask(obj_results, args.grid_size, args.mask_dilate).save(
                    overlay_dir / f"{sid}_{mode}_grid{args.grid_size}.png")

        print()

    # ── write outputs ─────────────────────────────────────────────────────────
    with open(out_dir / "raw_qwen_outputs.jsonl", "w", encoding="utf-8") as f:
        for r in raw_outputs:
            f.write(json.dumps(r) + "\n")

    with open(out_dir / "parsed_boxes.jsonl", "w", encoding="utf-8") as f:
        for r in parsed_rows:
            f.write(json.dumps(r) + "\n")

    # ── timing + sanity summary ───────────────────────────────────────────────
    yesno_times = [r["yesno_sec"] for r in timing_rows if r["yesno_sec"] is not None]
    yesno_mean  = sum(yesno_times) / len(yesno_times) if yesno_times else None

    timing_summary = {
        "num_samples":    len(valid_samples),
        "yesno_mean_sec": round(yesno_mean, 3) if yesno_mean else None,
    }

    sanity = {}
    for mode in args.run_modes:
        mode_timing = [r for r in timing_rows if r["mode"] == mode]
        mode_parsed = [r for r in parsed_rows if r["qwen_mode"] == mode]
        all_objs     = [o for r in mode_parsed for o in r["objects"]]
        vis_objs     = [o for o in all_objs if o.get("visible")]
        valid_bboxes = [o for o in vis_objs if o.get("bbox_valid")]
        areas = [(o["bbox_norm"][2]-o["bbox_norm"][0])*(o["bbox_norm"][3]-o["bbox_norm"][1])
                 for o in vis_objs if o.get("bbox_norm")]
        toks  = [o["num_tokens"] for o in valid_bboxes]
        confs = [o["confidence"] for o in vis_objs]
        n_ok  = sum(1 for r in mode_timing if r["parse_success"])

        sanity[mode] = {
            "parse_success_rate": round(n_ok / len(mode_timing), 3) if mode_timing else 0,
            "visible_rate":       round(len(vis_objs) / len(all_objs), 3) if all_objs else 0,
            "valid_bbox_rate":    round(len(valid_bboxes) / len(vis_objs), 3) if vis_objs else 0,
            "mean_confidence":    round(sum(confs) / len(confs), 3) if confs else 0,
            "mean_box_area":      round(sum(areas) / len(areas), 3) if areas else 0,
            "mean_num_tokens":    round(sum(toks) / len(toks), 1) if toks else 0,
            "fraction_too_large": round(sum(1 for a in areas if a > 0.75) / len(areas), 3) if areas else 0,
            "fraction_too_small": round(sum(1 for a in areas if a < 0.01) / len(areas), 3) if areas else 0,
        }

        times = [r["runtime_sec"] for r in mode_timing]
        mean_sec = round(sum(times) / len(times), 3) if times else 0
        timing_summary[f"{mode}_mean_sec"]   = mean_sec
        timing_summary[f"{mode}_sanity"]     = sanity[mode]
        if yesno_mean:
            timing_summary[f"{mode}_overhead_vs_yesno"] = round(mean_sec / yesno_mean, 2)

    with open(out_dir / "timing_summary.json", "w") as f:
        json.dump(timing_summary, f, indent=2)

    # ── summary.md ────────────────────────────────────────────────────────────
    md = [
        "# Qwen Localization Smoke Test",
        "",
        f"**Samples**: {len(valid_samples)}  "
        f"**Grid**: {args.grid_size}×{args.grid_size}  "
        f"**Dilate**: {args.mask_dilate}",
        "",
        "## Timing",
        "",
    ]
    if yesno_mean:
        md.append(f"- yes/no scorer: **{yesno_mean:.2f}s**/image")
    for mode in args.run_modes:
        mean_sec = timing_summary.get(f"{mode}_mean_sec", "?")
        overhead = timing_summary.get(f"{mode}_overhead_vs_yesno", "N/A")
        md.append(f"- `{mode}`: **{mean_sec}s**/image  (overhead: {overhead}×)")

    md += ["", "## Sanity Metrics", ""]
    for mode in args.run_modes:
        s = sanity.get(mode, {})
        md += [f"### `{mode}`", "| metric | value |", "|--------|-------|"]
        for k, v in s.items():
            md.append(f"| {k} | {v} |")
        md.append("")

    md += ["## Decision", ""]
    for mode in args.run_modes:
        s        = sanity.get(mode, {})
        overhead = timing_summary.get(f"{mode}_overhead_vs_yesno", 999)
        feasible = (
            s.get("parse_success_rate", 0) >= 0.85
            and s.get("valid_bbox_rate", 0) >= 0.75
            and (not isinstance(overhead, float) or overhead <= 2.5)
        )
        md.append(f"- `{mode}`: **{'FEASIBLE ✓' if feasible else 'NOT FEASIBLE ✗'}**")
        if not feasible:
            reasons = []
            if s.get("parse_success_rate", 0) < 0.85:
                reasons.append(f"parse_success={s.get('parse_success_rate'):.0%}")
            if s.get("valid_bbox_rate", 0) < 0.75:
                reasons.append(f"valid_bbox_rate={s.get('valid_bbox_rate'):.0%}")
            if isinstance(overhead, float) and overhead > 2.5:
                reasons.append(f"overhead={overhead}×")
            if reasons:
                md.append(f"  - failing: {', '.join(reasons)}")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n[smoke] done → {out_dir}")
    print(json.dumps(timing_summary, indent=2))


if __name__ == "__main__":
    main()
