"""
Generate the final CoT-planning comparison table.

Loads eval results from multiple experiments and renders a markdown table with:
  - method name
  - whether the model weights changed
  - prompt used at inference
  - training source
  - hard_target mean, attribute component, anti-swap component

Usage:
  python scripts_cot/make_cot_final_comparison.py \
    --base-eval                 outputs/<grpo_run>/fixed_eval_attribute.json \
    --direct-grpo-eval          outputs/<grpo_run>/fixed_eval_attribute.json \
    --structured-prompt-eval    outputs_cot_planning/structured_prompt_eval_attribute_<t>/summary.json \
    --cot-sft-eval              outputs_cot_planning/cot_rejection_sft_runs/<run>/fixed_eval_attribute.json \
    --cot-sft-grpo-eval         outputs_cot_planning/cot_sft_then_grpo/<run>/fixed_eval_attribute.json \
    [--best-of-g-eval           outputs/<bog_run>/best_of_g_summary.json] \
    --out outputs_cot_planning/final_cot_comparison.md
"""
import argparse
import json
import math
from pathlib import Path


def _fmt(val, digits=4):
    if isinstance(val, float) and math.isnan(val):
        return "—"
    return f"{val:.{digits}f}"


def _load_fixed_eval(path: str, ckpt_key: str = "best"):
    """Load from eval_checkpoints_bucket.py output (has 'results' key)."""
    with open(path) as f:
        data = json.load(f)
    if "results" not in data:
        raise ValueError(f"Expected 'results' key in {path}")
    results = data["results"]
    # try preferred keys in order
    for key in [ckpt_key, "best", "final", "base"]:
        if key in results:
            r = results[key]
            return r.get("mean", float("nan")), r.get("stderr", float("nan")), {}
    raise KeyError(f"None of [{ckpt_key}, best, final, base] found in {path}. Keys: {list(results)}")


def _load_structured_summary(path: str):
    """Load from eval_structured_prompts.py summary.json."""
    with open(path) as f:
        data = json.load(f)
    mean   = data.get("structured_mean_hard_target", float("nan"))
    stderr = 0.0
    qt = {
        "attribute": data.get("structured_attribute_component", float("nan")),
        "anti_swap": data.get("structured_anti_swap_component",
                     data.get("structured_anti_component", float("nan"))),
    }
    return mean, stderr, qt


def _load_best_of_g(path: str):
    """Load from best_of_g_summary.json."""
    with open(path) as f:
        data = json.load(f)
    mean   = data.get("best_grpo", {}).get("mean_hard_target",
             data.get("mean_hard_target", float("nan")))
    stderr = data.get("best_grpo", {}).get("se_hard_target", 0.0)
    return mean, stderr, {}


def _load_probe_history(path: str):
    """Load best probe mean from probe_eval_history.jsonl."""
    best_mean, best_se = -1.0, 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            m = rec.get("mean_reward", -1.0)
            if m > best_mean:
                best_mean = m
                best_se   = rec.get("se_reward", 0.0)
    return best_mean, best_se, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval",              default=None)
    parser.add_argument("--direct-grpo-eval",       default=None)
    parser.add_argument("--structured-prompt-eval", default=None)
    parser.add_argument("--best-of-g-eval",         default=None)
    parser.add_argument("--cot-sft-eval",           default=None)
    parser.add_argument("--cot-sft-grpo-eval",      default=None)
    parser.add_argument("--out",                    required=True)
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Row definitions: (label, model_changed, inference_prompt, training_src, loader_fn)
    # -------------------------------------------------------------------
    spec = [
        ("Base raw prompt",        "No",  "raw",        "none",
         lambda: _load_fixed_eval(args.base_eval, "base") if args.base_eval else None),

        ("Structured prompt only", "No",  "structured", "none",
         lambda: _load_structured_summary(args.structured_prompt_eval)
         if args.structured_prompt_eval else None),

        ("Direct GRPO stable",     "Yes", "raw",        "GRPO",
         lambda: _load_fixed_eval(args.direct_grpo_eval, "best")
         if args.direct_grpo_eval else None),

        ("Best-of-G rerank",       "No",  "raw",        "inference rerank",
         lambda: _load_best_of_g(args.best_of_g_eval)
         if args.best_of_g_eval else None),

        ("CoT rejection-SFT",      "Yes", "raw",        "structured gen → reward select → SFT",
         lambda: _load_fixed_eval(args.cot_sft_eval, "best")
         if args.cot_sft_eval else None),

        ("CoT-SFT + GRPO",         "Yes", "raw",        "SFT then GRPO",
         lambda: _load_fixed_eval(args.cot_sft_grpo_eval, "best")
         if args.cot_sft_grpo_eval else None),
    ]

    rows = []
    base_mean = None

    for label, model_changed, inf_prompt, train_src, loader in spec:
        try:
            result = loader()
        except Exception as e:
            print(f"  WARNING [{label}]: {e}")
            result = None

        if result is None:
            rows.append({
                "label": label, "model_changed": model_changed,
                "inf_prompt": inf_prompt, "train_src": train_src,
                "mean": float("nan"), "stderr": float("nan"),
                "qt": {}, "delta": float("nan"), "notes": "data missing",
            })
            continue

        mean, stderr, qt = result
        if base_mean is None:
            base_mean = mean
        delta = mean - base_mean if not math.isnan(mean) and base_mean is not None else float("nan")

        rows.append({
            "label": label, "model_changed": model_changed,
            "inf_prompt": inf_prompt, "train_src": train_src,
            "mean": mean, "stderr": stderr,
            "qt": qt, "delta": delta, "notes": "",
        })
        print(f"  {label:28s}  mean={_fmt(mean)}  se={_fmt(stderr, 3)}  delta={_fmt(delta, 4)}")

    # -------------------------------------------------------------------
    # Markdown table
    # -------------------------------------------------------------------
    md_lines = [
        "# Attribute binding — CoT-planning final comparison\n",
        "| Method | Model changed? | Inference prompt | Training source | Hard target ↑ | SE | Δ vs base |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        m_str     = _fmt(r["mean"])
        se_str    = _fmt(r["stderr"], 3)
        d_str     = (f"{r['delta']:+.4f}" if not math.isnan(r.get("delta", float("nan"))) else "—")
        note      = r.get("notes", "")
        suffix    = f"  <!-- {note} -->" if note else ""
        md_lines.append(
            f"| {r['label']} | {r['model_changed']} | {r['inf_prompt']} "
            f"| {r['train_src']} | {m_str} | {se_str} | {d_str} |{suffix}"
        )

    md = "\n".join(md_lines) + "\n"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\n[cot_table] Saved → {out_path}")
    print(md)


if __name__ == "__main__":
    main()
