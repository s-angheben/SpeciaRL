#!/bin/bash
# Reference launch script used on the Leonardo HPC cluster (SLURM + Apptainer).
# Documentation of what we ran, not a portable entry point.
# Set REPO_ROOT, MODELS_ROOT and the SLURM --account before use,
# and adapt the array layout to your scheduler.
#
#SBATCH --job-name=db_array
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gres gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 05:00:00
#SBATCH -o logs/%A_%a_%x.out
#SBATCH -e logs/%A_%a_%x.err
#SBATCH --array=0-100

: "${REPO_ROOT:?set REPO_ROOT to the absolute path of this repository}"
: "${MODELS_ROOT:?set MODELS_ROOT to the absolute path of the directory with model weights}"

# Calculate dynamic ports based on SLURM_ARRAY_TASK_ID to avoid conflicts
export REDIS_PORT=$((6379 + ${SLURM_ARRAY_TASK_ID}))
export MONGODB_PORT=$((27017 + ${SLURM_ARRAY_TASK_ID}))
export VLLM_PORT=$((8000 + ${SLURM_ARRAY_TASK_ID}))
export LLM_CACHE_PORT=$((9000 + ${SLURM_ARRAY_TASK_ID}))
export LLM_CACHE_URL="http://localhost:${LLM_CACHE_PORT}"

echo "Dynamic ports for task ${SLURM_ARRAY_TASK_ID}: Redis=${REDIS_PORT}, MongoDB=${MONGODB_PORT}, vLLM=${VLLM_PORT}, LLM Cache=${LLM_CACHE_PORT}"

# Generate dynamic LLM cache config from template
DYNAMIC_CONFIG_PATH="/tmp/config.apptainer_dynamic_${SLURM_ARRAY_TASK_ID}.yml"
sed -e "s/{{REDIS_PORT}}/${REDIS_PORT}/g" \
    -e "s/{{MONGODB_PORT}}/${MONGODB_PORT}/g" \
    -e "s/{{VLLM_PORT}}/${VLLM_PORT}/g" \
    ${REPO_ROOT}/scripts/eval/config.apptainer_template.yml > "${DYNAMIC_CONFIG_PATH}"

# Set this to the directory containing the evaluation configs you want to run.
# See vlm_openworld_evaluator/configs/ for available options.
CONFIG_DIR="${REPO_ROOT}/vlm_openworld_evaluator/configs/main_evaluation/normal/base_master"


config_files=($(find "$CONFIG_DIR" -maxdepth 1 -name "*.yaml"))

num_configs=${#config_files[@]}
array_limit=$((num_configs - 1))

# 4. Set the SLURM array size dynamically.
#SBATCH --array=0-$array_limit

CURRENT_CONFIG_PATH="${config_files[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$CURRENT_CONFIG_PATH" ]; then
    echo "Error: No config file found for array task ID $SLURM_ARRAY_TASK_ID."
    exit 1
fi

echo "--- Starting Job Array Task ${SLURM_ARRAY_TASK_ID} ---"
echo "Using configuration file: ${CURRENT_CONFIG_PATH}"

export RUN_NAME="evaluation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOG_DIR="${REPO_ROOT}/logs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

CONFIG_FILENAME=$(basename "$CURRENT_CONFIG_PATH")
# Extract config directory relative to vlm_openworld_evaluator
CONFIG_RELATIVE_DIR=$(echo "$CONFIG_DIR" | sed 's|.*/vlm_openworld_evaluator/||')

# Determine model based on config content
if grep -q "Qwen3-30B-A3B-Instruct-2507-FP8" "$CURRENT_CONFIG_PATH"; then
    export MODEL_PATH="/workspace/llm_caching_service/models/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8"
elif grep -q "Llama-4-Maverick-17B-128E-Instruct-FP8" "$CURRENT_CONFIG_PATH"; then
    export MODEL_PATH="/workspace/llm_caching_service/models/Llama-4-Maverick-17B-128E-Instruct-FP8"
else
    # Default fallback to Qwen
    export MODEL_PATH="/workspace/llm_caching_service/models/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8"
fi

echo "Selected model: $MODEL_PATH"
export EVALUATOR_CMD="python /workspace/vlm_openworld_evaluator/run_pipeline_vllm.py --config /workspace/vlm_openworld_evaluator/${CONFIG_RELATIVE_DIR}/${CONFIG_FILENAME} --stages verify"

singularity run --nv \
  --bind ${REPO_ROOT}/vlm_openworld_evaluator:/workspace/vlm_openworld_evaluator \
  --bind ${REPO_ROOT}/llm_caching_service:/workspace/llm_caching_service \
  --bind ${MODELS_ROOT}:/workspace/llm_caching_service/models \
  --bind ${REPO_ROOT}/scripts/eval/run_evaluation_verify.sh:/workspace/run_evaluation.sh \
  --bind ${REPO_ROOT}/scripts/eval/redis_indep.conf:/workspace/llm_caching_service/config/redis_indep.conf \
  --bind ${REPO_ROOT}/scripts/eval/wait-for-services.sh:/workspace/llm_caching_service/wait-for-services.sh \
  --bind "${LOG_DIR}":/logs \
  --bind ${REPO_ROOT}/verl:/workspace/verl \
  --bind "${DYNAMIC_CONFIG_PATH}":/workspace/config.apptainer_dynamic.yml \
  --env ROCR_VISIBLE_DEVICES="" \
  --env HIP_VISIBLE_DEVICES="" \
  --env "EVALUATOR_CMD=$EVALUATOR_CMD" \
  --env "MODEL_PATH=$MODEL_PATH" \
  --env "REDIS_PORT=$REDIS_PORT" \
  --env "MONGODB_PORT=$MONGODB_PORT" \
  --env "VLLM_PORT=$VLLM_PORT" \
  --env "LLM_CACHE_PORT=$LLM_CACHE_PORT" \
  --env "LLM_CACHE_URL=$LLM_CACHE_URL" \
  ${REPO_ROOT}/containers/image.sif /workspace/run_evaluation.sh

echo "--- Finished Job Array Task ${SLURM_ARRAY_TASK_ID} ---"

