#!/bin/bash
# Reference launch script used on the Leonardo HPC cluster (SLURM + Apptainer).
# Documentation of what we ran, not a portable entry point.
# Set REPO_ROOT, MODELS_ROOT and the SLURM --account before use,
# and adapt the array layout to your scheduler.
#
#SBATCH --job-name=db_array
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 20:00:00
#SBATCH -o logs/%A_%a_%x.out
#SBATCH -e logs/%A_%a_%x.err
#SBATCH --array=0-30

: "${REPO_ROOT:?set REPO_ROOT to the absolute path of this repository}"
: "${MODELS_ROOT:?set MODELS_ROOT to the absolute path of the directory with model weights}"

# Set this to the directory containing the prediction configs you want to run.
# See vlm_openworld_evaluator/configs/ for available options.
CONFIG_DIR="${REPO_ROOT}/vlm_openworld_evaluator/configs/main_evaluation/bon64/base_bon64_master"

# Find only test_general.yml or train_general_bon64.yml files
config_files=($(find "$CONFIG_DIR" \( -name "*.yaml" -o -name "train_general_bon64.yml" \)))

num_configs=${#config_files[@]}
array_limit=$((num_configs - 1))

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

# Extract the relative path from vlm_openworld_evaluator onwards
CONFIG_RELATIVE_PATH=$(echo "$CURRENT_CONFIG_PATH" | sed 's|.*/vlm_openworld_evaluator/||')

echo "Selected model: $MODEL_PATH"
export EVALUATOR_CMD="cd /workspace/vlm_openworld_evaluator; python /workspace/vlm_openworld_evaluator/run_pipeline_vllm.py --config /workspace/vlm_openworld_evaluator/${CONFIG_RELATIVE_PATH} --stages predict"

singularity run --nv \
  --bind ${REPO_ROOT}/vlm_openworld_evaluator:/workspace/vlm_openworld_evaluator \
  --bind ${REPO_ROOT}/llm_caching_service:/workspace/llm_caching_service \
  --bind ${MODELS_ROOT}:/workspace/llm_caching_service/models \
  --bind ${REPO_ROOT}/scripts/eval/run_evaluation.sh:/workspace/run_evaluation.sh \
  --bind ${REPO_ROOT}/scripts/eval/config.apptainer.yml:/workspace/llm_caching_service/config.apptainer.yml \
  --bind ${REPO_ROOT}/scripts/eval/redis_indep.conf:/workspace/llm_caching_service/config/redis_indep.conf \
  --bind ${REPO_ROOT}/scripts/eval/wait-for-services.sh:/workspace/llm_caching_service/wait-for-services.sh \
  --bind "${LOG_DIR}":/logs \
  --bind ${REPO_ROOT}/verl:/workspace/verl \
  --env ROCR_VISIBLE_DEVICES="" \
  --env HIP_VISIBLE_DEVICES="" \
  --env "EVALUATOR_CMD=$EVALUATOR_CMD" \
  --env "MODEL_PATH=$MODEL_PATH" \
  ${REPO_ROOT}/containers/image.sif bash -c "$EVALUATOR_CMD"

echo "--- Finished Job Array Task ${SLURM_ARRAY_TASK_ID} ---"
