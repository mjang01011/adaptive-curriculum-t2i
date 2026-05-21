#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=eval_ckpts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/eval_ckpts_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=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

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

# Override with env vars:
#   RUN_DIR=outputs/<run>  BUCKET=attribute_binding  sbatch run_eval_checkpoints.sh
RUN_DIR=${RUN_DIR:-outputs/attribute_only_overfit_v3_stable_fixed_bucket_15516148_20260521_020359}
BUCKET=${BUCKET:-attribute_binding}
CHECKPOINTS=${CHECKPOINTS:-"base step_000005 step_000010 step_000015 step_000020 step_000025 step_000030 best"}
NUM_VAL_PROMPTS=${NUM_VAL_PROMPTS:-20}
NUM_SAMPLES=${NUM_SAMPLES:-8}
CFG_SCALE=${CFG_SCALE:-2.0}
REWARD_MODE=${REWARD_MODE:-hard_target}
FIXED_SEEDS=${FIXED_SEEDS:-"0 1 2 3 4 5 6 7"}

OUT_FILE="${RUN_DIR}/fixed_eval_${BUCKET}.json"

echo "[eval_ckpts] RUN_DIR=${RUN_DIR}"
echo "[eval_ckpts] BUCKET=${BUCKET}"
echo "[eval_ckpts] CHECKPOINTS=${CHECKPOINTS}"
echo "[eval_ckpts] OUT=${OUT_FILE}"

python3 scripts/eval_checkpoints_bucket.py \
    --run-dir   /viscam/u/jj277/adaptive-curriculum-t2i/${RUN_DIR} \
    --data-root /viscam/u/jj277/adaptive-curriculum-t2i/data \
    --bucket    ${BUCKET} \
    --checkpoints ${CHECKPOINTS} \
    --num-val-prompts        ${NUM_VAL_PROMPTS} \
    --num-samples-per-prompt ${NUM_SAMPLES} \
    --cfg-scale   ${CFG_SCALE} \
    --reward-mode ${REWARD_MODE} \
    --fixed-seeds ${FIXED_SEEDS} \
    --out /viscam/u/jj277/adaptive-curriculum-t2i/${OUT_FILE}
