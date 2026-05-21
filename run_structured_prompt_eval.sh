#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=struct_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/struct_eval_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=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

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

# Override: BUCKET=attribute_binding NUM_SAMPLES=8 sbatch run_structured_prompt_eval.sh
BUCKET=${BUCKET:-attribute_binding}
EXPERIMENT=${EXPERIMENT:-structured_prompt_eval_${BUCKET}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUTPUT_DIR="/viscam/u/jj277/adaptive-curriculum-t2i/outputs_cot_planning/${EXPERIMENT}_${JOB_ID}_${TIMESTAMP}"

echo "[struct_eval] BUCKET=${BUCKET}"
echo "[struct_eval] OUTPUT_DIR=${OUTPUT_DIR}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH
mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

# Step 1: create structured prompts if not already done
if [ ! -d "/viscam/u/jj277/adaptive-curriculum-t2i/data_cot_structured" ]; then
    echo "[struct_eval] Creating structured prompts..."
    python3 scripts_cot/create_structured_prompts.py \
        --data-root    /viscam/u/jj277/adaptive-curriculum-t2i/data \
        --output-root  /viscam/u/jj277/adaptive-curriculum-t2i/data_cot_structured
fi

# Step 2: run eval
export EXPERIMENT=${EXPERIMENT}
python3 scripts_cot/eval_structured_prompts.py \
    --base-config           $PROJECT/adaptive_curriculum/configs/experiment.yaml \
    --data-root             /viscam/u/jj277/adaptive-curriculum-t2i/data \
    --structured-data-root  /viscam/u/jj277/adaptive-curriculum-t2i/data_cot_structured \
    --repo-root             $LLAMAGEN \
    --gpt-ckpt              $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt               $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path               $PRETRAINED/t5-ckpt \
    --t5-cache-dir          /viscam/u/jj277/adaptive-curriculum-t2i/data/t5_cache \
    --bucket                ${BUCKET} \
    --split                 val \
    --num-prompts           20 \
    --num-samples-per-prompt 8 \
    --cfg-scale             2.0 \
    --seeds                 0 1 2 3 4 5 6 7 \
    --reward-mode           hard_target \
    --output-dir            ${OUTPUT_DIR}
