#!/bin/bash
# Generate Janus-Pro-1B images for T2I-CompBench (1 image per prompt).
# Run from an interactive job or sbatch.
#
# Usage:
#   LIMIT=20 bash scripts_janus/run_januspro_compbench_generate_1sample.sh   # debug
#   LIMIT=-1 bash scripts_janus/run_januspro_compbench_generate_1sample.sh   # full
set -euo pipefail

ROOT=/viscam/u/jj277/janus_project
COMP=${ROOT}/T2I-CompBench
SCRIPTS=${ROOT}/scripts_janus
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${SLURM_JOB_ID:-local}
OUT=${ROOT}/outputs_janus_compbench/januspro1b_base_1sample_${JOB_ID}_${TIMESTAMP}
LIMIT=${LIMIT:--1}

mkdir -p "$OUT"
echo "[janus_gen] OUT=${OUT}  LIMIT=${LIMIT}"

declare -A FILES
FILES[color]="${COMP}/examples/dataset/color_val.txt"
FILES[shape]="${COMP}/examples/dataset/shape_val.txt"
FILES[texture]="${COMP}/examples/dataset/texture_val.txt"
FILES[spatial]="${COMP}/examples/dataset/spatial_val.txt"
FILES[non_spatial]="${COMP}/examples/dataset/non_spatial_val.txt"
FILES[complex]="${COMP}/examples/dataset/complex_val.txt"

for CAT in color shape texture spatial non_spatial complex; do
    echo "[janus_gen] category=${CAT}"
    python3 ${SCRIPTS}/generate_janus_compbench_1sample.py \
        --prompt-file  "${FILES[$CAT]}" \
        --category     "$CAT" \
        --model-path   deepseek-ai/Janus-Pro-1B \
        --output-dir   "${OUT}/${CAT}/samples" \
        --seed         0 \
        --cfg-weight   5.0 \
        --temperature  1.0 \
        --limit        ${LIMIT}
done

echo "[janus_gen] Generation done. Run dir: ${OUT}"
echo "[janus_gen] Next: RUN_DIR=${OUT} bash ${SCRIPTS}/eval_compbench_generated_samples.sh"
