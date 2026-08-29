#!/usr/bin/env bash
set -euo pipefail

BIZ="${BIZ:-/mmu-vcg/zb08/zixuan/BIZ}"
REPO="${REPO:-$BIZ/repos/BIZGenAgent}"
ENV_DIR="${SENSENOVA_ENV:-/mmu-vcg/zb08/zixuan/envs/sensenova_u15}"
CODE_DIR="${SENSENOVA_CODE_DIR:-$BIZ/SenseNova-U1}"
MODEL_DIR="${BIZ_EDIT_MODEL_PATH:-/mmu-vcg/zb08/zixuan/models/SenseNova-U1.5-8B-MoT}"
SAMPLE_DIR="${BIZ_SAMPLE_DIR:-$BIZ/results/sample20_nanobanana2_seed42}"
RESULT_NAME="${BIZ_RESULT_NAME:-agent1_v4_sensenova_nanobanana2_20_8gpu_r1}"
EVAL_ROOT="${BIZ_EVAL_ROOT:-/mmu-vcg/zb08/wps4.28/7-25-BizGen/BizGenEval}"
CREDENTIAL="${BIZ_CREDENTIAL:-/mmu-vcg/zb08/llm-6669-1b56d4a3712d.json}"

DATA_PATH="${BIZ_DATA_PATH:-$SAMPLE_DIR/sample20.jsonl}"
IMAGE_DIR="${BIZ_IMAGE_DIR:-$SAMPLE_DIR/originals}"
MARK_DIR="${BIZ_MARK_DIR:-$SAMPLE_DIR/mark_original}"

required=(
  "$ENV_DIR/bin/python"
  "$CODE_DIR/examples/editing/inference.py"
  "$BIZ/tools/run_agent1_v4.py"
  "$DATA_PATH"
  "$CREDENTIAL"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] required path missing: $path" >&2
    exit 2
  fi
done
for dir in "$IMAGE_DIR" "$MARK_DIR"; do
  if [[ ! -d "$dir" ]]; then
    echo "[ERROR] required directory missing: $dir" >&2
    exit 2
  fi
done
for shard in 01 02 03 04 05 06 07 08; do
  file="$MODEL_DIR/model-000${shard}-of-00008.safetensors"
  if [[ ! -s "$file" ]]; then
    echo "[ERROR] model shard missing or empty: $file" >&2
    exit 2
  fi
done

export BIZ_EDIT_BACKEND=sensenova
export BIZ_EDIT_MODEL_PATH="$MODEL_DIR"
export SENSENOVA_CODE_DIR="$CODE_DIR"
export SENSENOVA_VRAM_MODE="${SENSENOVA_VRAM_MODE:-full}"
export SENSENOVA_IMG_CFG_SCALE="${SENSENOVA_IMG_CFG_SCALE:-1.0}"
export SENSENOVA_CFG_NORM="${SENSENOVA_CFG_NORM:-none}"
export SENSENOVA_TIMESTEP_SHIFT="${SENSENOVA_TIMESTEP_SHIFT:-3.0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BIZ/tools${PYTHONPATH:+:$PYTHONPATH}"

echo "[START] backend=$BIZ_EDIT_BACKEND"
echo "[START] model=$MODEL_DIR"
echo "[START] GPUs=0,1,2,3,4,5,6,7 (one worker/model replica per GPU)"
echo "[START] result=$BIZ/results/$RESULT_NAME"

exec "$ENV_DIR/bin/python" -u "$BIZ/tools/run_agent1_v4.py" \
  --agent1-dir "$REPO/agent1" \
  --tools-dir "$BIZ/tools" \
  --data-path "$DATA_PATH" \
  --image-dir "$IMAGE_DIR" \
  --baseline-mark-dir "$MARK_DIR" \
  --eval-root "$EVAL_ROOT" \
  --credential "$CREDENTIAL" \
  --gpus 0,1,2,3,4,5,6,7 \
  --result-name "$RESULT_NAME" \
  --force-score
