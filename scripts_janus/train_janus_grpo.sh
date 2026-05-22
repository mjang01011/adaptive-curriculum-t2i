#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=janus_grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/viscam/u/jj277/janus_project/logs/janus_grpo_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.cache/huggingface
export TORCH_HOME=/viscam/u/jj277/.cache/torch
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=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

EXPERIMENT=${EXPERIMENT:-janus_attribute_only_grpo_stable}
CONFIG=${CONFIG:-configs_janus/janus_attribute_only_grpo_stable.yaml}

JOB_ID=${SLURM_JOB_ID:-local}
echo "[run] EXPERIMENT=${EXPERIMENT}"
echo "[run] CONFIG=${CONFIG}"
echo "[run] JOB_ID=${JOB_ID}"

source /viscam/u/jj277/envs/januspro_venv/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
JANUS_PROJECT=/viscam/u/jj277/janus_project

mkdir -p ${JANUS_PROJECT}/logs

export PYTHONPATH=${PROJECT}:${JANUS_PROJECT}:${PYTHONPATH}

cd ${PROJECT}

python3 ${PROJECT}/scripts_janus/train_janus_grpo.py \
    --config ${PROJECT}/${CONFIG}
