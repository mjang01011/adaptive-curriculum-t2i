#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=best_of_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/best_of_g_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=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

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

# Override with env vars:
#   BUCKET=spatial_relations_anchored sbatch run_best_of_g_diagnostic.sh
#   REWARD_MODE=qwen_logit_grpo_target_heavy BUCKET=attribute_binding sbatch run_best_of_g_diagnostic.sh
BUCKET=${BUCKET:-attribute_binding}
REWARD_MODE=${REWARD_MODE:-pseudo_soft_grpo_target_heavy}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR="${PROJECT}/outputs/best_of_g_${BUCKET}_${REWARD_MODE}_${JOB_ID}_${TIMESTAMP}"

echo "[best_of_g] BUCKET=${BUCKET}"
echo "[best_of_g] REWARD_MODE=${REWARD_MODE}"
echo "[best_of_g] OUTPUT_DIR=${OUTPUT_DIR}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

/viscam/u/jj277/svl/bin/python scripts/best_of_g_reward_ranker.py \
    --bucket        $BUCKET \
    --data-root     ${PROJECT}/data \
    --repo-root     $LLAMAGEN \
    --gpt-ckpt      $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt       $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path       $PRETRAINED/t5-ckpt \
    --t5-cache-dir  ${PROJECT}/data/t5_cache \
    --reward-mode   $REWARD_MODE \
    --num-prompts   20 \
    --num-generations 6 \
    --seeds         0 1 2 3 4 5 \
    --output-dir    $OUTPUT_DIR

echo "[best_of_g] Done. Results at: ${OUTPUT_DIR}"
