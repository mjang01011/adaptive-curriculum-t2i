#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=curriculum_t2i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/curriculum_%j.out

# viscam home dir quota fix
export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
# Set WANDB_API_KEY in your environment or ~/.bashrc on viscam — do NOT hardcode here
export WANDB_API_KEY=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


source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

EXPERIMENT=${EXPERIMENT:-pilot_fast}   # override with: EXPERIMENT=full_ucb sbatch train_curriculum.sh
STRATEGY=${STRATEGY:-ucb}              # override with: STRATEGY=pooled_random or STRATEGY=round_robin
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
RUN_NAME="${EXPERIMENT}_${STRATEGY}_${JOB_ID}_${TIMESTAMP}"
OUTPUT_DIR="/viscam/u/jj277/adaptive-curriculum-t2i/outputs/${RUN_NAME}"

export OUTPUT_DIR
echo "[run] EXPERIMENT=${EXPERIMENT}"
echo "[run] STRATEGY=${STRATEGY}"
echo "[run] JOB_ID=${JOB_ID}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

python -m adaptive_curriculum.train.run_experiment \
    --config $PROJECT/adaptive_curriculum/configs/experiment.yaml \
    --experiment $PROJECT/adaptive_curriculum/configs/experiments/${EXPERIMENT}.yaml \
    --strategy $STRATEGY \
    --repo-root      $LLAMAGEN \
    --data-root      /viscam/u/jj277/adaptive-curriculum-t2i/data \
    --gpt-ckpt       $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt        $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path        $PRETRAINED/t5-ckpt \
    --t5-cache-dir   /viscam/u/jj277/adaptive-curriculum-t2i/data/t5_cache \
    --output-dir     $OUTPUT_DIR \
    --run-name       $RUN_NAME \
    --wandb
