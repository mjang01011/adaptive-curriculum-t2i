#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=lc_true
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/lc_true_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

EXPERIMENT=${EXPERIMENT:-lc_true_v1}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
RUN_NAME="${EXPERIMENT}_${JOB_ID}_${TIMESTAMP}"
OUTPUT_DIR="/viscam/u/jj277/adaptive-curriculum-t2i/outputs/${RUN_NAME}"

CLEAN_JSONL=${CLEAN_JSONL:-/viscam/u/jj277/adaptive-curriculum-t2i/data/gpic_merged/dataset.jsonl}
VAL_JSONL=${VAL_JSONL:-/viscam/u/jj277/adaptive-curriculum-t2i/data/attribute_binding/attribute_binding_val_20.jsonl}

echo "[lc_true] EXPERIMENT=${EXPERIMENT}"
echo "[lc_true] CLEAN_JSONL=${CLEAN_JSONL}"
echo "[lc_true] JOB_ID=${JOB_ID}"
echo "[lc_true] OUTPUT_DIR=${OUTPUT_DIR}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

python3 SFT/train_lc_true.py \
    --clean-jsonl  $CLEAN_JSONL \
    --val-jsonl    $VAL_JSONL \
    --output-dir   $OUTPUT_DIR \
    --repo-root    $LLAMAGEN \
    --gpt-ckpt     $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt      $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path      $PRETRAINED/t5-ckpt \
    --use-raw-caption \
    --num-epochs   100 \
    --batch-size   16 \
    --lr           2e-6 \
    --lambda-ce    1.0 \
    --lambda-contrast 0.05 \
    --tau-contrast  0.2 \
    --target-ratio  0.05 \
    --max-gamma     0.01 \
    --grad-clip     0.5 \
    --eval-every   1000 \
    --save-every   10000 \
    --dl-workers   2 \
    --min-rows     1 \
    --run-name     $RUN_NAME \
    --wandb
