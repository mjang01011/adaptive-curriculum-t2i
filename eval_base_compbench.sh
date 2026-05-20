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
OUT_DIR=/viscam/u/jj277/adaptive-curriculum-t2i/outputs/compbench_eval

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

# ── Step 1: clone T2I-CompBench++ if not present ──────────────────────────────
if [ ! -d "$COMPBENCH_DIR" ]; then
    echo "[setup] Cloning T2I-CompBench++ ..."
    git clone https://github.com/Karine-Huang/T2I-CompBench $COMPBENCH_DIR
fi

PROMPTS_DIR=$COMPBENCH_DIR/examples/dataset
if [ ! -d "$PROMPTS_DIR" ]; then
    echo "ERROR: $PROMPTS_DIR not found. Check repo structure."
    exit 1
fi

echo "[setup] Using prompts from: $PROMPTS_DIR"
mkdir -p $OUT_DIR

# ── Step 2: generate images with base LlamaGen (no LoRA) ──────────────────────
# Note: --model-name is for LoRA models only; "base" is always generated automatically.
# Do NOT pass --model-name here.
echo ""
echo "[generate] Running base LlamaGen on T2I-CompBench++ prompts ..."

python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root   $LLAMAGEN \
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path     $PRETRAINED/t5-ckpt \
    --prompts     $PROMPTS_DIR \
    --out-dir     $OUT_DIR \
    --num-samples 4 \
    --cfg-scale   2.0 \
    --batch-size  8 \
    --seed        42

if [ $? -ne 0 ]; then
    echo "ERROR: generation failed. Check log above."
    exit 1
fi

echo ""
echo "[generate] Done. Images in $OUT_DIR/base/"

# ── Step 3: rename to {idx}.png layout expected by T2I-CompBench++ eval scripts ─
RENAMED_DIR=$OUT_DIR/base_renamed
echo ""
echo "[rename] Reformatting image names → $RENAMED_DIR ..."

python - "$OUT_DIR/base" "$RENAMED_DIR" <<'PYEOF'
import json, shutil, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
manifest_path = src / "manifest.json"

if not manifest_path.exists():
    print(f"ERROR: {manifest_path} not found. Generation may have failed.")
    sys.exit(1)

manifest = json.load(open(manifest_path))

for cat_dir in sorted(src.iterdir()):
    if not cat_dir.is_dir():
        continue
    cat_items = sorted(
        [(pid, info) for pid, info in manifest.items() if info["category"] == cat_dir.name],
        key=lambda x: x[0],
    )
    if not cat_items:
        continue
    out_cat = dst / cat_dir.name
    out_cat.mkdir(parents=True, exist_ok=True)
    for idx, (pid, info) in enumerate(cat_items):
        src_img = Path(info["image_paths"][0])
        dst_img = out_cat / f"{idx}.png"
        shutil.copy2(str(src_img), str(dst_img))
    print(f"  {cat_dir.name}: {len(cat_items)} images → {out_cat}")

print(f"\nRenamed images ready at: {dst}")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: rename step failed."
    exit 1
fi

# ── Step 4: T2I-CompBench++ evaluation ────────────────────────────────────────
# Their repo has separate eval scripts per metric type:
#   BLIPvqa_eval/  → color, shape, texture, non_spatial, numeracy, complex
#   UniDet_eval/   → spatial, 3d_spatial
#
# The scripts vary by version; print instructions and attempt auto-detection.

echo ""
echo "[eval] T2I-CompBench++ evaluation"
echo "       Images ready at: $RENAMED_DIR"
echo ""
echo "  Manual eval commands:"
echo "  -- BLIP-VQA (color / shape / texture / non_spatial / numeracy / complex) --"
echo "  cd $COMPBENCH_DIR/BLIPvqa_eval"
echo "  python BLIP_vqa.py --img_folder $RENAMED_DIR/<category> --out_dir $OUT_DIR/scores/base"
echo ""
echo "  -- UniDet (spatial / 3d_spatial) --"
echo "  cd $COMPBENCH_DIR/UniDet_eval"
echo "  python UniDet_eval.py --img_folder $RENAMED_DIR/<category> --out_dir $OUT_DIR/scores/base"
echo ""

# attempt to run BLIP-VQA automatically if script exists
BLIP_SCRIPT=$COMPBENCH_DIR/BLIPvqa_eval/BLIP_vqa.py
UNIDET_SCRIPT=$COMPBENCH_DIR/UniDet_eval/UniDet_eval.py
SCORES_DIR=$OUT_DIR/scores/base
mkdir -p $SCORES_DIR

BLIP_CATS="color shape texture non_spatial numeracy complex"
UNIDET_CATS="spatial 3d_spatial"

if [ -f "$BLIP_SCRIPT" ]; then
    echo "[eval] Running BLIP-VQA ..."
    pip install -q transformers
    for cat in $BLIP_CATS; do
        if [ -d "$RENAMED_DIR/$cat" ]; then
            echo "  $cat ..."
            python $BLIP_SCRIPT \
                --img_folder $RENAMED_DIR/$cat \
                --out_dir    $SCORES_DIR \
                2>&1 | tail -3
        fi
    done
else
    echo "[eval] BLIP-VQA script not found at expected path $BLIP_SCRIPT"
    echo "       Run manually using the commands above."
fi

if [ -f "$UNIDET_SCRIPT" ]; then
    echo "[eval] Running UniDet ..."
    for cat in $UNIDET_CATS; do
        if [ -d "$RENAMED_DIR/$cat" ]; then
            echo "  $cat ..."
            python $UNIDET_SCRIPT \
                --img_folder $RENAMED_DIR/$cat \
                --out_dir    $SCORES_DIR \
                2>&1 | tail -3
        fi
    done
else
    echo "[eval] UniDet script not found at expected path $UNIDET_SCRIPT"
    echo "       Run manually using the commands above."
fi

echo ""
echo "[done] All steps complete."
echo "       Generated images : $OUT_DIR/base/"
echo "       Renamed (eval fmt): $RENAMED_DIR/"
echo "       Scores            : $SCORES_DIR/"
