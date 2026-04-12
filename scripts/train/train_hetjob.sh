#!/bin/bash
# Reference launch script used on the Leonardo HPC cluster (SLURM + Apptainer).
# Documentation of what we ran, not a portable entry point.
# Set REPO_ROOT, MODELS_ROOT and the SLURM --account before use,
# and adapt the hetjob layout to your scheduler.
#
# Heterogeneous job: Component 0 = Services (1 node, 16 CPU, 2 GPU)
#SBATCH --job-name=train_hetjob
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -t 10:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH hetjob
# Component 1: Training (1 nodes, 32 CPU, 4 GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4

# REPO_ROOT: absolute path to the repository on the cluster filesystem.
# MODELS_ROOT: absolute path to a directory containing the pre-downloaded model weights.
: "${REPO_ROOT:?set REPO_ROOT to the absolute path of this repository}"
: "${MODELS_ROOT:?set MODELS_ROOT to the absolute path of the directory with model weights}"

echo "--- Starting Heterogeneous Training Job ---"

server_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)
training_nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1))
training_node1=${training_nodes[0]}

echo "Services node (16 CPU, 2 GPU): $server_node"
echo "Training node 1 (32 CPU, 4 GPU): $training_node1"

# Set this to the config you want to train; see verl/configs/ for available options.
export VERL_CONFIG="configs/main_cub/qwen2_5_vl-7b_lora_cub200_grpo_reward3.sh"

export LOG_DIR="${REPO_ROOT}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Log files will be stored in: ${LOG_DIR}"

export TRAINER_CMD="bash ${VERL_CONFIG}"

echo "Starting services container on services component (het-group=0)"
srun --het-group=0 singularity run --nv \
  --bind ${REPO_ROOT}/vlm_openworld_evaluator:/workspace/vlm_openworld_evaluator \
  --bind ${REPO_ROOT}/llm_caching_service:/workspace/llm_caching_service \
  --bind ${MODELS_ROOT}:/workspace/llm_caching_service/models \
  --bind ${REPO_ROOT}/scripts/train/setup_and_run_services.sh:/workspace/run_services.sh \
  --bind ${REPO_ROOT}/scripts/train/expand_config.sh:/workspace/expand_config.sh \
  --bind ${REPO_ROOT}/scripts/train/config.env.yml:/workspace/config.env.yml \
  --bind ${REPO_ROOT}/scripts/train/redis_multinode.conf:/workspace/llm_caching_service/config/redis_multinode.conf \
  --bind ${REPO_ROOT}/scripts/train/config.multinode.yml:/workspace/llm_caching_service/config.multinode.yml \
  --bind "${LOG_DIR}":/logs \
  --bind ${REPO_ROOT}/verl:/workspace/verl \
  --env ROCR_VISIBLE_DEVICES="" \
  --env HIP_VISIBLE_DEVICES="" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "SERVER_NODE=${server_node}" \
  ${REPO_ROOT}/containers/image.sif /workspace/run_services.sh &

echo "Waiting for services to initialize..."
sleep 20

echo "Starting distributed training on training component (het-group=1)"
srun --het-group=1 singularity run --nv \
  --bind ${REPO_ROOT}/vlm_openworld_evaluator:/workspace/vlm_openworld_evaluator \
  --bind ${REPO_ROOT}/llm_caching_service:/workspace/llm_caching_service \
  --bind ${MODELS_ROOT}:/workspace/llm_caching_service/models \
  --bind ${REPO_ROOT}/scripts/train/run_training_multinode.sh:/workspace/run_training.sh \
  --bind ${REPO_ROOT}/scripts/train/config.multinode.yml:/workspace/llm_caching_service/config.multinode.yml \
  --bind "${LOG_DIR}":/logs \
  --bind ${REPO_ROOT}/verl:/workspace/verl \
  --env ROCR_VISIBLE_DEVICES="" \
  --env HIP_VISIBLE_DEVICES="" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "TRAINER_CMD=${TRAINER_CMD}" \
  --env "VERL_CONFIG=${VERL_CONFIG}" \
  --env "SERVER_NODE=${server_node}" \
  ${REPO_ROOT}/containers/image.sif /workspace/run_training.sh

echo "--- Finished Heterogeneous Training Job ---"
