#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=compbench_1s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/compbench_1s_%j.out

set -euo pipefail

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export TOKENIZERS_PARALLELISM=false

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models
COMP=/viscam/u/jj277/T2I-CompBench

# Required:
#   MODEL_CKPT=/path/to/t2i_XL_stage1_256.pt sbatch scripts_compbench/run_vanilla_compbench_1sample.sh
#
# Optional LoRA:
#   MODEL_CKPT=...  LORA_CKPT=outputs/<run>/checkpoints/best.pt  RUN_NAME=llamagen_grpo_compbench_1sample
#   sbatch scripts_compbench/run_vanilla_compbench_1sample.sh

MODEL_CKPT=${MODEL_CKPT:?MODEL_CKPT must be set}
LORA_CKPT=${LORA_CKPT:-""}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}

# derive a default run name from the checkpoint filename
_CKPT_STEM=$(basename "${LORA_CKPT:-$MODEL_CKPT}" .pt)
RUN_NAME=${RUN_NAME:-llamagen_${_CKPT_STEM}_compbench_1sample_${JOB_ID}_${TIMESTAMP}}
ROOT=/viscam/u/jj277/adaptive-curriculum-t2i/outputs_compbench_vanilla/${RUN_NAME}

mkdir -p "$ROOT"
mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

echo "[compbench_1s] MODEL_CKPT=${MODEL_CKPT}"
echo "[compbench_1s] LORA_CKPT=${LORA_CKPT:-none}"
echo "[compbench_1s] ROOT=${ROOT}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:${PYTHONPATH:-}

declare -A PROMPT_FILES
PROMPT_FILES[color]="${COMP}/examples/dataset/color_val.txt"
PROMPT_FILES[shape]="${COMP}/examples/dataset/shape_val.txt"
PROMPT_FILES[texture]="${COMP}/examples/dataset/texture_val.txt"
PROMPT_FILES[spatial]="${COMP}/examples/dataset/spatial_val.txt"
PROMPT_FILES[non_spatial]="${COMP}/examples/dataset/non_spatial_val.txt"
PROMPT_FILES[complex]="${COMP}/examples/dataset/complex_val.txt"

CATEGORIES=(color shape texture spatial non_spatial complex)

# --- Step 1: generate images -------------------------------------------
for CAT in "${CATEGORIES[@]}"; do
    echo "[generate] ${CAT}"
    OUT="${ROOT}/${CAT}/samples"
    mkdir -p "$OUT"

    LORA_ARG=""
    if [ -n "$LORA_CKPT" ]; then
        LORA_ARG="--lora-checkpoint ${PROJECT}/${LORA_CKPT}"
    fi

    python3 ${PROJECT}/scripts_compbench/generate_llamagen_compbench_1sample.py \
        --prompt-file   "${PROMPT_FILES[$CAT]}" \
        --category      "$CAT" \
        --repo-root     "$LLAMAGEN" \
        --gpt-ckpt      "$MODEL_CKPT" \
        --vq-ckpt       "$PRETRAINED/vq_ds16_t2i.pt" \
        --t5-path       "$PRETRAINED/t5-ckpt" \
        --output-dir    "$OUT" \
        --num-samples-per-prompt 1 \
        --seed          0 \
        --cfg-scale     2.0 \
        --image-size    256 \
        --batch-size    8 \
        $LORA_ARG
done

# --- Step 2: evaluate each category ------------------------------------
for CAT in "${CATEGORIES[@]}"; do
    echo "[eval] ${CAT}"

    cd "$COMP"
    rm -rf examples/samples
    mkdir -p examples/samples
    cp "${ROOT}/${CAT}/samples/"*.png examples/samples/

    if [[ "$CAT" == "color" || "$CAT" == "shape" || "$CAT" == "texture" ]]; then
        cd "${COMP}/BLIPvqa_eval"
        python3 BLIP_vqa.py --out_dir="../examples/"
        mkdir -p "${ROOT}/${CAT}"
        cp "${COMP}/examples/annotation_blip/vqa_result.json" "${ROOT}/${CAT}/vqa_result.json"

    elif [[ "$CAT" == "spatial" ]]; then
        cd "${COMP}/UniDet_eval"
        python3 2D_spatial_eval.py
        mkdir -p "${ROOT}/${CAT}"
        cp "${COMP}/examples/labels/annotation_obj_detection_2d/vqa_result.json" "${ROOT}/${CAT}/vqa_result.json"

    elif [[ "$CAT" == "non_spatial" ]]; then
        cd "$COMP"
        python3 CLIPScore_eval/CLIP_similarity.py --outpath examples/
        mkdir -p "${ROOT}/${CAT}"
        cp "${COMP}/examples/annotation_clip/vqa_result.json" "${ROOT}/${CAT}/vqa_result.json"

    elif [[ "$CAT" == "complex" ]]; then
        cd "${COMP}/3_in_1_eval"
        python3 3_in_1.py --outpath ../examples/
        mkdir -p "${ROOT}/${CAT}"
        cp "${COMP}/examples/annotation_3_in_1/vqa_result.json" "${ROOT}/${CAT}/vqa_result.json"
    fi

    cd "$PROJECT"
    echo "[eval] ${CAT} done → ${ROOT}/${CAT}/vqa_result.json"
done

# --- Step 3: summarize -------------------------------------------------
python3 ${PROJECT}/scripts_compbench/summarize_compbench_results.py \
    --run-dir    "$ROOT" \
    --categories color shape texture spatial non_spatial complex \
    --out        "$ROOT/compbench_1sample_summary.json"

echo "[compbench_1s] Done. Results at: ${ROOT}/compbench_1sample_summary.json"
