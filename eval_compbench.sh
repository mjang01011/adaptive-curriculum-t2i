#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=compbench_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/compbench_eval_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export WANDB_DIR=/viscam/u/jj277/.wandb
export TOKENIZERS_PARALLELISM=false

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

# ── which checkpoints to evaluate ─────────────────────────────────────────────
# Set these before sbatching, e.g.:
#   UCB_CKPT=/viscam/.../outputs/ucb_run/checkpoints/best.pt
#   POOLED_CKPT=/viscam/.../outputs/pooled_run/checkpoints/best.pt
#   RR_CKPT=/viscam/.../outputs/rr_run/checkpoints/best.pt
#
# Leave empty to skip that model. Base model is always generated.
UCB_CKPT=${UCB_CKPT:-""}
POOLED_CKPT=${POOLED_CKPT:-""}
RR_CKPT=${RR_CKPT:-""}

# ── prompts ────────────────────────────────────────────────────────────────────
# Option A: T2I-CompBench++ prompts JSON (download from the benchmark repo)
PROMPTS_FILE=${PROMPTS_FILE:-"/viscam/u/jj277/adaptive-curriculum-t2i/data/t2icompbench_prompts.json"}
# Option B: use our internal val sets (set USE_VAL_SETS=1 to activate)
USE_VAL_SETS=${USE_VAL_SETS:-0}

OUT_DIR=/viscam/u/jj277/adaptive-curriculum-t2i/outputs/compbench_eval
NUM_SAMPLES=${NUM_SAMPLES:-4}   # T2I-CompBench++ standard is 4 images/prompt

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH
mkdir -p $OUT_DIR

# ── build checkpoint args ──────────────────────────────────────────────────────
LORA_CKPTS=""
MODEL_NAMES=""

if [ -n "$UCB_CKPT" ]; then
    LORA_CKPTS="$LORA_CKPTS $UCB_CKPT"
    MODEL_NAMES="$MODEL_NAMES ucb_lora"
fi
if [ -n "$POOLED_CKPT" ]; then
    LORA_CKPTS="$LORA_CKPTS $POOLED_CKPT"
    MODEL_NAMES="$MODEL_NAMES pooled_lora"
fi
if [ -n "$RR_CKPT" ]; then
    LORA_CKPTS="$LORA_CKPTS $RR_CKPT"
    MODEL_NAMES="$MODEL_NAMES round_robin_lora"
fi

# ── prompts flag ───────────────────────────────────────────────────────────────
if [ "$USE_VAL_SETS" = "1" ]; then
    PROMPTS_ARGS="--use-val-sets --data-root /viscam/u/jj277/adaptive-curriculum-t2i/data"
else
    PROMPTS_ARGS="--prompts $PROMPTS_FILE"
fi

# ── run generation ─────────────────────────────────────────────────────────────
python -m adaptive_curriculum.eval.generate_for_compbench \
    --repo-root   $LLAMAGEN \
    --gpt-ckpt    $PRETRAINED/t2i_XL_stage1_256.pt \
    --vq-ckpt     $PRETRAINED/vq_ds16_t2i.pt \
    --t5-path     $PRETRAINED/t5-ckpt \
    --t5-cache-dir /viscam/u/jj277/adaptive-curriculum-t2i/data/t5_cache \
    ${LORA_CKPTS:+--lora-checkpoint $LORA_CKPTS} \
    ${MODEL_NAMES:+--model-name $MODEL_NAMES} \
    $PROMPTS_ARGS \
    --out-dir     $OUT_DIR \
    --num-samples $NUM_SAMPLES \
    --cfg-scale   2.0 \
    --batch-size  8 \
    --seed        42
