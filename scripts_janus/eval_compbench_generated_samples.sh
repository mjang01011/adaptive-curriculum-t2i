#!/bin/bash
# Evaluate generated images with T2I-CompBench evaluators.
# Works for both Janus and LlamaGen outputs — just needs a run dir
# with <category>/samples/*.png and a category layout.
#
# Usage:
#   RUN_DIR=/viscam/u/jj277/janus_project/outputs_janus_compbench/<run> \
#   bash scripts_janus/eval_compbench_generated_samples.sh
set -euo pipefail

RUN_DIR=${RUN_DIR:?RUN_DIR must be set}
COMP=/viscam/u/jj277/janus_project/T2I-CompBench
SCRIPTS=/viscam/u/jj277/janus_project/scripts_janus

echo "[compbench_eval] RUN_DIR=${RUN_DIR}"

# --- color / shape / texture : BLIP-VQA ---
for CAT in color shape texture; do
    echo "[eval] ${CAT} (BLIP-VQA)"
    rm -rf "${COMP}/examples/samples"
    mkdir -p "${COMP}/examples/samples"
    cp "${RUN_DIR}/${CAT}/samples/"*.png "${COMP}/examples/samples/"
    python3 "${COMP}/BLIPvqa_eval/BLIP_vqa.py" --out_dir="${COMP}/examples/"
    mkdir -p "${RUN_DIR}/${CAT}"
    cp "${COMP}/examples/annotation_blip/vqa_result.json" "${RUN_DIR}/${CAT}/vqa_result.json"
    echo "[eval] ${CAT} done"
done

# --- spatial : UniDet ---
echo "[eval] spatial (UniDet)"
rm -rf "${COMP}/examples/samples"
mkdir -p "${COMP}/examples/samples"
cp "${RUN_DIR}/spatial/samples/"*.png "${COMP}/examples/samples/"
python3 "${COMP}/UniDet_eval/2D_spatial_eval.py"
mkdir -p "${RUN_DIR}/spatial"
cp "${COMP}/examples/labels/annotation_obj_detection_2d/vqa_result.json" "${RUN_DIR}/spatial/vqa_result.json"
echo "[eval] spatial done"

# --- non_spatial : CLIPScore ---
echo "[eval] non_spatial (CLIPScore)"
rm -rf "${COMP}/examples/samples"
mkdir -p "${COMP}/examples/samples"
cp "${RUN_DIR}/non_spatial/samples/"*.png "${COMP}/examples/samples/"
python3 "${COMP}/CLIPScore_eval/CLIP_similarity.py" --outpath "${COMP}/examples/"
mkdir -p "${RUN_DIR}/non_spatial"
cp "${COMP}/examples/annotation_clip/vqa_result.json" "${RUN_DIR}/non_spatial/vqa_result.json"
echo "[eval] non_spatial done"

# --- complex : 3-in-1 ---
echo "[eval] complex (3-in-1)"
rm -rf "${COMP}/examples/samples"
mkdir -p "${COMP}/examples/samples"
cp "${RUN_DIR}/complex/samples/"*.png "${COMP}/examples/samples/"
python3 "${COMP}/3_in_1_eval/3_in_1.py" --outpath "${COMP}/examples/"
mkdir -p "${RUN_DIR}/complex"
cp "${COMP}/examples/annotation_3_in_1/vqa_result.json" "${RUN_DIR}/complex/vqa_result.json"
echo "[eval] complex done"

# --- summarize ---
echo "[summarize]"
python3 "${SCRIPTS}/summarize_compbench_results.py" \
    --run-dir    "$RUN_DIR" \
    --categories color shape texture spatial non_spatial complex \
    --out        "$RUN_DIR/compbench_1sample_summary.json"

echo "[compbench_eval] Done. Summary: ${RUN_DIR}/compbench_1sample_summary.json"
