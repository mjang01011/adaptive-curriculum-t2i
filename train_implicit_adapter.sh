#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=implicit_adapter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/implicit_adapter_%j.out

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

EXPERIMENT=${EXPERIMENT:-implicit_adapter_v3}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
RUN_NAME="${EXPERIMENT}_${JOB_ID}_${TIMESTAMP}"
OUTPUT_DIR="/viscam/u/jj277/adaptive-curriculum-t2i/outputs/${RUN_NAME}"

TRAIN_JSONL=${TRAIN_JSONL:-/viscam/u/jj277/adaptive-curriculum-t2i/data/gpic_slots_v2_clean/dataset.jsonl}
VAL_JSONL=${VAL_JSONL:-/viscam/u/jj277/adaptive-curriculum-t2i/data/attribute_binding/attribute_binding_val_20.jsonl}

echo "[implicit_adapter] EXPERIMENT=${EXPERIMENT}"
echo "[implicit_adapter] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[implicit_adapter] JOB_ID=${JOB_ID}"
echo "[implicit_adapter] OUTPUT_DIR=${OUTPUT_DIR}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

python3 SFT/train_implicit_adapter.py \
    --train-jsonl  $TRAIN_JSONL \
    --val-jsonl    $VAL_JSONL \
    --output-dir   $OUTPUT_DIR \
    --repo-root    $LLAMAGEN \
    --gpt-ckpt     $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt      $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path      $PRETRAINED/t5-ckpt \
    --freeze-llamagen \
    --use-raw-caption \
    --num-epochs   60 \
    --batch-size   8 \
    --lr           1e-5 \
    --lambda-contrast    0.3 \
    --lambda-delta-ratio 10.0 \
    --delta-ratio-target 0.10 \
    --max-gamma          0.01 \
    --grad-clip          0.5 \
    --tau-contrast 0.1 \
    --eval-every   200 \
    --save-every   200 \
    --dl-workers   2 \
    --min-rows     1 \
    --run-name     $RUN_NAME \
    --wandb
