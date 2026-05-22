#!/bin/bash
# ── SLURM job for verifier-guided GRPO (3 variants) ────────────────────────
# Usage:
#   sbatch scripts_verifier/train_verifier_grpo.sh vanilla
#   sbatch scripts_verifier/train_verifier_grpo.sh winner
#   sbatch scripts_verifier/train_verifier_grpo.sh gcpo_lite
#
# Default: vanilla

#SBATCH --job-name=verifier_grpo
#SBATCH --output=slurm_logs/verifier_grpo_%j.out
#SBATCH --error=slurm_logs/verifier_grpo_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

set -e

VARIANT="${1:-vanilla}"
PROJ=/viscam/u/jj277/adaptive-curriculum-t2i
export HOME=/viscam/u/jj277
export HF_HOME=$PROJ/hf_cache
export TRANSFORMERS_CACHE=$PROJ/hf_cache/transformers

mkdir -p "$PROJ/slurm_logs"
mkdir -p "$PROJ/outputs_verifier/grpo_runs"

# ── select config ─────────────────────────────────────────────────────────────
case "$VARIANT" in
    vanilla)
        CONFIG="$PROJ/configs_verifier/synthetic_shapes_vanilla_grpo.yaml"
        RUN_NAME="synthetic_shapes_vanilla_grpo"
        ;;
    winner)
        CONFIG="$PROJ/configs_verifier/synthetic_shapes_winner_grpo.yaml"
        RUN_NAME="synthetic_shapes_winner_grpo"
        ;;
    gcpo_lite)
        CONFIG="$PROJ/configs_verifier/synthetic_shapes_winner_gcpo_lite.yaml"
        RUN_NAME="synthetic_shapes_winner_gcpo_lite"
        ;;
    *)
        echo "Unknown variant: $VARIANT  (choices: vanilla, winner, gcpo_lite)"
        exit 1
        ;;
esac

echo "=== verifier GRPO: variant=$VARIANT ==="
echo "    config:   $CONFIG"
echo "    run_name: $RUN_NAME"
echo "    slurm_id: $SLURM_JOB_ID"

# ── activate env ──────────────────────────────────────────────────────────────
# LlamaGen training uses the svl conda env (Python 3.13, PyTorch, omegaconf)
source /viscam/u/jj277/miniconda3/etc/profile.d/conda.sh
conda activate svl

# ── set PYTHONPATH ────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJ:$PROJ/LlamaGen:${PYTHONPATH:-}"

# ── run ───────────────────────────────────────────────────────────────────────
cd "$PROJ"

python scripts_verifier/train_verifier_grpo.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME"

echo "=== done: variant=$VARIANT ==="
