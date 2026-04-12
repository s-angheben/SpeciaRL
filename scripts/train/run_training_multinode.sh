#!/bin/bash
# /workspace/run_training.sh

# --- Log File Setup ---
LOG_FILE="/logs/training_log.txt"
echo "--- Training Execution Log ---" > "$LOG_FILE"
echo "Start Time: $(date)" >> "$LOG_FILE"
echo "Training Node: $(hostname)" >> "$LOG_FILE"
echo "Server Node: $SERVER_NODE" >> "$LOG_FILE"
echo "----------------------------------" >> "$LOG_FILE"

# Function to log messages to both console and log file
log_message() {
    echo "$1"
    echo "$(date): $1" >> "$LOG_FILE"
}

# --- Phase 1: Setup and Validation ---

log_message "--- Multi-node training script started: Phase 1 - Setup ---"

if [ -z "$TRAINER_CMD" ]; then
    log_message "Error: TRAINER_CMD environment variable is not set."
    exit 1
fi

if [ -z "$VERL_CONFIG" ]; then
    log_message "Error: VERL_CONFIG environment variable is not set."
    exit 1
fi

if [ -z "$SERVER_NODE" ]; then
    log_message "Error: SERVER_NODE environment variable is not set."
    exit 1
fi

log_message "Training command to execute: $TRAINER_CMD"
log_message "VERL config: $VERL_CONFIG"
log_message "Server node: $SERVER_NODE"
log_message "SLURM Proc ID: ${SLURM_PROCID:-0}"
log_message "SLURM N Procs: ${SLURM_NPROCS:-1}"

# Configuration will be handled by the consolidated services script
log_message ">>> Configuration will be managed via environment variables"

# --- Phase 2: Wait for Services to be Ready ---

log_message "--- Phase 2: Waiting for services on server node ---"

log_message ">>> Checking vLLM server availability at $SERVER_NODE:8001..."
until curl -s "http://$SERVER_NODE:8001/v1/models" > /dev/null; do
    log_message "Waiting for vLLM server on $SERVER_NODE..."
    sleep 10
done
log_message "vLLM server is ready."

log_message ">>> Checking Redis availability at $SERVER_NODE:6379..."
until redis-cli -h "$SERVER_NODE" -p 6379 ping > /dev/null 2>&1; do
    log_message "Waiting for Redis on $SERVER_NODE..."
    sleep 5
done
log_message "Redis is ready."

log_message ">>> Checking MongoDB availability at $SERVER_NODE:27017..."
until mongosh --host "$SERVER_NODE" --eval "db.adminCommand('ping')" &> /dev/null; do
    log_message "Waiting for MongoDB on $SERVER_NODE..."
    sleep 5
done
log_message "MongoDB is ready."

log_message ">>> Checking LLM Cache Service availability at $SERVER_NODE:8000..."
until curl -s "http://$SERVER_NODE:8000/health" > /dev/null; do
    log_message "Waiting for LLM Cache Service on $SERVER_NODE..."
    sleep 5
done
log_message "LLM Cache Service is ready."

# --- Phase 3: Execute Training ---

log_message "======================================================================="
log_message "--- Phase 3: Starting Training Execution (4 GPUs) ---"
log_message "======================================================================="

RUN_NAME="training_${SLURM_JOB_ID}"
LOG_DIR_TRAINING="/logs/${RUN_NAME}"
mkdir -p "$LOG_DIR_TRAINING"

# Create VERL-specific log directory based on config name with timestamp
CONFIG_NAME_NO_EXT=$(basename "$VERL_CONFIG" .sh)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VERL_LOG_DIR="/workspace/verl/logs/${CONFIG_NAME_NO_EXT}/${TIMESTAMP}"
mkdir -p "$VERL_LOG_DIR"

log_message "Executing training command: $TRAINER_CMD"
log_message "Training logs will be stored in: ${LOG_DIR_TRAINING}"
log_message "VERL logs will also be saved in: ${VERL_LOG_DIR}"

# Execute the training command with proper GPU assignment for trainer (all 4 GPUs)
# Use SLURM environment variables for distributed training
export NODE_RANK=${SLURM_PROCID:-0}
export WORLD_SIZE=${SLURM_NPROCS:-1}
export MASTER_ADDR=${SERVER_NODE}
export MASTER_PORT=29500
export VERIFICATION_API_BASE_URL="http://$SERVER_NODE:8000/api/v1/"

# Use environment-based configuration - CONFIG_PATH will be set by services script
(cd /workspace/verl && CUDA_VISIBLE_DEVICES="0,1,2,3" REWARD_LOG_FILE="${VERL_LOG_DIR}/reward_function_node${NODE_RANK}.log" CONFIG_PATH="${CONFIG_PATH:-/workspace/llm_caching_service/config.runtime.yml}" NODE_RANK="$NODE_RANK" WORLD_SIZE="$WORLD_SIZE" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" /bin/bash -c "${TRAINER_CMD}") > "${LOG_DIR_TRAINING}/trainer_node${NODE_RANK}.log" 2> "${LOG_DIR_TRAINING}/trainer_node${NODE_RANK}.err"

# Copy logs to VERL directory as well
cp "${LOG_DIR_TRAINING}/trainer_node${NODE_RANK}.log" "${VERL_LOG_DIR}/trainer_node${NODE_RANK}.log"
cp "${LOG_DIR_TRAINING}/trainer_node${NODE_RANK}.err" "${VERL_LOG_DIR}/trainer_node${NODE_RANK}.err"
cp "${VERL_LOG_DIR}/reward_function_node${NODE_RANK}.log" "${LOG_DIR_TRAINING}/reward_function_node${NODE_RANK}.log" 2>/dev/null || true

training_exit_code=$?

log_message "--- Training command completed with exit code: $training_exit_code ---"

# Append training logs to main log
echo -e "\n\n#######################################################################" >> "$LOG_FILE"
echo "### TRAINING EXECUTION LOGS (NODE ${NODE_RANK}) ###" >> "$LOG_FILE"
echo "#######################################################################" >> "$LOG_FILE"
echo -e "\n--- STDOUT ---\n" >> "$LOG_FILE"
cat "${LOG_DIR_TRAINING}/trainer_node${NODE_RANK}.log" >> "$LOG_FILE"
echo -e "\n--- STDERR ---\n" >> "$LOG_FILE"
cat "${LOG_DIR_TRAINING}/trainer_node${NODE_RANK}.err" >> "$LOG_FILE"

log_message "======================================================================="
log_message "--- Training script finished. ---"
log_message "======================================================================="
echo "----------------------------------" >> "$LOG_FILE"
echo "End Time: $(date)" >> "$LOG_FILE"

exit $training_exit_code