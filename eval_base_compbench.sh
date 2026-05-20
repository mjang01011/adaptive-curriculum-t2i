#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=base_compbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/base_compbench_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export TOKENIZERS_PARALLELISM=false

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models
COMPBENCH_DIR=/viscam/u/jj277/adaptive-curriculum-t2i/T2I-CompBench

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

# ── Step 1: clone T2I-CompBench++ if not present ──────────────────────────────
if [ ! -d "$COMPBENCH_DIR" ]; then
    echo "[setup] Cloning T2I-CompBench++ ..."
    git clone https://github.com/Karine-Huang/T2I-CompBench $COMPBENCH_DIR
fi

PROMPTS_DIR=$COMPBENCH_DIR/prompts
if [ ! -d "$PROMPTS_DIR" ]; then
    echo "ERROR: $PROMPTS_DIR not found after clone. Check repo structure."
    exit 1
fi

echo "[setup] Using prompts from: $PROMPTS_DIR"
echo "[setup] Prompt files:"
ls $PROMPTS_DIR/*.json 2>/dev/null || echo "  (no .json files found — check repo)"

# ── Step 2: generate images with base LlamaGen (no LoRA) ──────────────────────
OUT_DIR=/viscam/u/jj277/adaptive-curriculum-t2i/outputs/compbench_eval
mkdir -p $OUT_DIR

echo ""
echo "[generate] Running base LlamaGen on T2I-CompBench++ prompts ..."
echo "[generate] Output → $OUT_DIR/base/"

python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root   $LLAMAGEN \
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path     $PRETRAINED/t5-ckpt \
    --prompts     $PROMPTS_DIR \
    --out-dir     $OUT_DIR \
    --model-name  base \
    --num-samples 4 \
    --cfg-scale   2.0 \
    --batch-size  8 \
    --seed        42

echo ""
echo "[generate] Done. Images in $OUT_DIR/base/"
echo ""

# ── Step 3: run T2I-CompBench++ evaluation ────────────────────────────────────
# The benchmark evaluates each category with a different metric:
#   color/shape/texture attributes  → BLIP-VQA
#   spatial relations               → UniDet
#   non-spatial relations           → CLIP-Score
#   complex / counting              → 3-in-1 / BLIP-VQA
#
# Their eval scripts expect images at: <img_folder>/<idx>.png
# Our script saves as: <out_dir>/base/<category>/<id>_s<k>.png
# The rename step below produces the 0.png, 1.png, ... layout they expect.

echo "[eval] Renaming images to T2I-CompBench++ expected format (0.png, 1.png, ...) ..."

RENAMED_DIR=$OUT_DIR/base_renamed
python - <<'PYEOF'
import json, shutil
from pathlib import Path

src = Path("/viscam/u/jj277/adaptive-curriculum-t2i/outputs/compbench_eval/base")
dst = Path("/viscam/u/jj277/adaptive-curriculum-t2i/outputs/compbench_eval/base_renamed")
manifest = json.load(open(src / "manifest.json"))

for cat_dir in sorted(src.iterdir()):
    if not cat_dir.is_dir() or cat_dir.name == "base_renamed":
        continue
    # collect items for this category in id-sorted order
    cat_items = sorted(
        [(pid, info) for pid, info in manifest.items() if info["category"] == cat_dir.name],
        key=lambda x: x[0]
    )
    out_cat = dst / cat_dir.name
    out_cat.mkdir(parents=True, exist_ok=True)
    for idx, (pid, info) in enumerate(cat_items):
        # use sample 0 only (T2I-CompBench++ standard single-image eval)
        src_img = Path(info["image_paths"][0])
        dst_img = out_cat / f"{idx}.png"
        shutil.copy2(str(src_img), str(dst_img))
    print(f"  {cat_dir.name}: {len(cat_items)} images → {out_cat}")

print(f"\nRenamed images ready at: {dst}")
PYEOF

echo ""
echo "[eval] Running T2I-CompBench++ metrics ..."
echo "       See $COMPBENCH_DIR/README.md for full instructions."
echo ""

# Install benchmark dependencies if needed
pip install -q open_clip_torch transformers accelerate

# Run their evaluation script for each category
# T2I-CompBench++ uses: python evaluate.py --model_path ... --img_folder ... --outpath ...
# Check $COMPBENCH_DIR for exact script names — they vary by metric/category.

EVAL_SCRIPT=$COMPBENCH_DIR/evaluation/evaluate.py
if [ ! -f "$EVAL_SCRIPT" ]; then
    # Some versions use a different path
    EVAL_SCRIPT=$(find $COMPBENCH_DIR -name "evaluate*.py" | head -1)
fi

if [ -z "$EVAL_SCRIPT" ]; then
    echo "[eval] Could not find T2I-CompBench++ evaluate script."
    echo "       Run their evaluation manually:"
    echo "         cd $COMPBENCH_DIR"
    echo "         python evaluate.py --img_folder $RENAMED_DIR/<category> --outpath $OUT_DIR/scores"
    exit 0
fi

SCORES_DIR=$OUT_DIR/scores/base
mkdir -p $SCORES_DIR

for cat_dir in $RENAMED_DIR/*/; do
    category=$(basename $cat_dir)
    echo "  evaluating $category ..."
    python $EVAL_SCRIPT \
        --img_folder $cat_dir \
        --outpath    $SCORES_DIR/${category}_scores.json \
        2>&1 | tail -5
done

echo ""
echo "[eval] Scores written to $SCORES_DIR"
echo ""
echo "Summary:"
python - <<'PYEOF'
import json
from pathlib import Path

scores_dir = Path("/viscam/u/jj277/adaptive-curriculum-t2i/outputs/compbench_eval/scores/base")
if not scores_dir.exists():
    print("  (scores directory not found)")
else:
    for f in sorted(scores_dir.glob("*.json")):
        try:
            data = json.load(open(f))
            score = data.get("score", data.get("mean", data.get("average", "?")))
            print(f"  {f.stem:40s}  {score}")
        except Exception as e:
            print(f"  {f.stem}: could not parse ({e})")
PYEOF
