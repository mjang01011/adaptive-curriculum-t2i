#!/bin/bash
#SBATCH --account=viscam
#SBATCH --exclude=viscam1,viscam2,viscam5,viscam9,viscam14,viscam15,viscam-hgx-1,viscam-hgx-2
#SBATCH --job-name=extract_t5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/viscam/u/jj277/adaptive-curriculum-t2i/logs/extract_t5_%j.out

export HOME=/viscam/u/jj277
export HF_HOME=/viscam/u/jj277/.hf_cache
export TOKENIZERS_PARALLELISM=false

source /viscam/u/jj277/svl/bin/activate

PROJECT=/viscam/u/jj277/adaptive-curriculum-t2i
LLAMAGEN=/viscam/u/jj277/adaptive-curriculum-t2i/LlamaGen
PRETRAINED=/viscam/u/jj277/svl/B3S/baselines/LlamaGen/pretrained_models

# Override with: BUCKETS="counting complex_composition" sbatch run_extract_t5.sh
BUCKETS=${BUCKETS:-spatial_relations_anchored}

cd $PROJECT
export PYTHONPATH=$PROJECT:$LLAMAGEN:$PYTHONPATH

mkdir -p /viscam/u/jj277/adaptive-curriculum-t2i/logs

echo "[extract_t5] Extracting T5 embeddings for buckets: ${BUCKETS}"

python -m adaptive_curriculum.data.extract_t5_embeddings \
    --data-root  ${PROJECT}/data \
    --out-dir    ${PROJECT}/data/t5_cache \
    --repo-root  $LLAMAGEN \
    --t5-path    $PRETRAINED/t5-ckpt \
    --batch-size 32 \
    --buckets    $BUCKETS

echo "[extract_t5] Done."
