#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=cot_gen_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/cot_gen_sft_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=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

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

# Override: BUCKET=attribute_binding REWARD_MODE=pseudo_soft_grpo_target_heavy
#           NUM_GENERATIONS=6  sbatch run_generate_cot_rejection_sft.sh
BUCKET=${BUCKET:-attribute_binding}
REWARD_MODE=${REWARD_MODE:-pseudo_soft_grpo_target_heavy}
NUM_GENERATIONS=${NUM_GENERATIONS:-6}
EXPERIMENT=${EXPERIMENT:-cot_rejection_sft_${BUCKET}_data}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR="/viscam/u/jj277/adaptive-curriculum-t2i/outputs_cot_planning/cot_rejection_sft/${BUCKET}_g${NUM_GENERATIONS}_${JOB_ID}_${TIMESTAMP}"

echo "[cot_gen_sft] BUCKET=${BUCKET}"
echo "[cot_gen_sft] REWARD_MODE=${REWARD_MODE}"
echo "[cot_gen_sft] NUM_GENERATIONS=${NUM_GENERATIONS}"
echo "[cot_gen_sft] OUTPUT_DIR=${OUTPUT_DIR}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH
mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

# Ensure structured prompts exist
if [ ! -d "/viscam/u/jj277/adaptive-curriculum-t2i/data_cot_structured" ]; then
    echo "[cot_gen_sft] Creating structured prompts first..."
    python3 scripts_cot/create_structured_prompts.py \
        --data-root   /viscam/u/jj277/adaptive-curriculum-t2i/data \
        --output-root /viscam/u/jj277/adaptive-curriculum-t2i/data_cot_structured
fi

export EXPERIMENT=${EXPERIMENT}
python3 scripts_cot/generate_cot_rejection_sft_data.py \
    --base-config          $PROJECT/adaptive_curriculum/configs/experiment.yaml \
    --data-root            /viscam/u/jj277/adaptive-curriculum-t2i/data \
    --structured-data-root /viscam/u/jj277/adaptive-curriculum-t2i/data_cot_structured \
    --repo-root            $LLAMAGEN \
    --gpt-ckpt             $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt              $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path              $PRETRAINED/t5-ckpt \
    --t5-cache-dir         /viscam/u/jj277/adaptive-curriculum-t2i/data/t5_cache \
    --bucket               ${BUCKET} \
    --split                train \
    --num-prompts          500 \
    --num-generations      ${NUM_GENERATIONS} \
    --cfg-scale            2.0 \
    --reward-mode          ${REWARD_MODE} \
    --batch-size           4 \
    --output-dir           ${OUTPUT_DIR}

echo "[cot_gen_sft] Generation done. Running selection..."

python3 scripts/select_rejection_sft_data.py \
    --input    ${OUTPUT_DIR}/samples.jsonl \
    --selection best_by_grpo_total \
    --top-k    1 \
    --min-object-presence 0.50 \
    --out      ${OUTPUT_DIR}/selected_top1.jsonl

echo "[cot_gen_sft] Selected data at: ${OUTPUT_DIR}/selected_top1.jsonl"
echo "[cot_gen_sft] Next: EXPERIMENT=attribute_cot_rejection_sft_top1 SELECTED_JSONL=${OUTPUT_DIR}/selected_top1.jsonl sbatch train_rejection_sft.sh"
