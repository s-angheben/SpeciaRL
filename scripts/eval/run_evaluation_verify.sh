#!/bin/bash
# /workspace/run_evaluation.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
LOG_DIR="/logs"
REDIS_LOG="${LOG_DIR}/redis.log"
MONGODB_LOG="${LOG_DIR}/mongodb.log"
VLLM_LOG="${LOG_DIR}/vllm.log"
LLM_CACHE_LOG="${LOG_DIR}/llm-cache-service.log"
MONGODB_INIT_LOG="${LOG_DIR}/mongodb-init.log"
EVALUATOR_LOG="${LOG_DIR}/evaluator.log"
EVALUATOR_ERR_LOG="${LOG_DIR}/evaluator.err"

# --- Cleanup Function ---
cleanup() {
    echo ">>> Shutting down background processes..."
    # Send SIGTERM to all processes in the process group
    kill 0
    wait
    echo ">>> Cleanup complete."
}
# Use trap to call cleanup on EXIT, TERM, or INT signals for robustness
trap cleanup EXIT TERM INT

# --- Service Startup ---
echo ">>> Starting Redis on port $REDIS_PORT..."
(CUDA_VISIBLE_DEVICES="" redis-server /workspace/llm_caching_service/config/redis_indep.conf --port $REDIS_PORT --daemonize no) > "$REDIS_LOG" 2>&1 &

TEMP_MONGODB_DIR="/tmp/mongodb"
mkdir -p "$TEMP_MONGODB_DIR"

echo ">>> Starting MongoDB on port $MONGODB_PORT..."
CUDA_VISIBLE_DEVICES="" mongod \
    --dbpath "$TEMP_MONGODB_DIR" \
    --bind_ip 127.0.0.1 \
    --port $MONGODB_PORT \
    --logpath "$MONGODB_LOG" \
    --fork

echo ">>> Waiting for MongoDB to be ready..."
# Wait for mongod to accept connections
until mongosh --port $MONGODB_PORT --eval "db.adminCommand('ping')" &> /dev/null; do
    echo "Waiting for MongoDB..."
    sleep 2
done
echo "MongoDB is ready."

echo ">>> Initializing MongoDB user..."
(sleep 5 && mongosh --port $MONGODB_PORT --eval 'try { db = db.getSiblingDB("admin"); db.createUser({user: "admin", pwd: "password", roles: [{role: "root", db: "admin"}]}); print("MongoDB admin user created"); } catch (e) { print("MongoDB admin user already exists or error:", e); }') > "$MONGODB_INIT_LOG" 2>&1 &

echo ">>> Starting vLLM server on port $VLLM_PORT..."
(cd /workspace/llm_caching_service && CUDA_VISIBLE_DEVICES="0,1" python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port $VLLM_PORT --model "${MODEL_PATH}" --tensor-parallel-size 2 --dtype auto --gpu-memory-utilization 0.8) > "$VLLM_LOG" 2>&1 &

echo ">>> Waiting for vLLM server to be ready..."
until curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null; do
    echo "Waiting for vLLM server..."
    sleep 5
done
echo "vLLM server is ready."

echo ">>> Starting LLM Cache Service on port $LLM_CACHE_PORT..."
(cd /workspace/llm_caching_service && PYTHONPATH="/workspace/llm_caching_service" CONFIG_PATH="/workspace/config.apptainer_dynamic.yml" LOG_LEVEL="INFO" REDIS_URL="redis://localhost:$REDIS_PORT" MONGODB_URL="mongodb://admin:${MONGODB_PASSWORD}@localhost:$MONGODB_PORT/llm_caching?authSource=admin" MONGODB_ENABLED="true" VLLM_URL="http://localhost:$VLLM_PORT/" PORT="$LLM_CACHE_PORT" /bin/bash /workspace/llm_caching_service/wait-for-services.sh) > "$LLM_CACHE_LOG" 2>&1 &

# --- Run Evaluation ---
echo ">>> Starting evaluation pipeline..."
(cd /workspace/vlm_openworld_evaluator && /bin/bash -c "${EVALUATOR_CMD}") > "$EVALUATOR_LOG" 2> "$EVALUATOR_ERR_LOG"

echo ">>> Evaluation finished."
# The trap will handle cleanup, but we exit cleanly.
exit 0