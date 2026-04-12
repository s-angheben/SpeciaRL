#!/bin/bash
# Consolidated service setup and startup script
# Replaces run_services_multinode.sh + wait-for-services_node.sh

set -e

# --- Log File Setup ---
LOG_FILE="/logs/services_log.txt"
echo "--- Consolidated Services Log ---" > "$LOG_FILE"
echo "Start Time: $(date)" >> "$LOG_FILE"
echo "----------------------------------" >> "$LOG_FILE"

# Function to log messages to both console and log file
log_message() {
    echo "$1"
    echo "$(date): $1" >> "$LOG_FILE"
}

# --- Phase 1: Environment Setup ---
log_message "--- Phase 1: Environment Setup ---"

# Get the actual hostname of this node
ACTUAL_HOSTNAME=$(hostname)
export ACTUAL_HOSTNAME

log_message "Hostname: $ACTUAL_HOSTNAME"

# Set all service endpoints via environment variables
export REDIS_URL="redis://${ACTUAL_HOSTNAME}:6379"
export MONGODB_URL="mongodb://admin:${MONGODB_PASSWORD}@${ACTUAL_HOSTNAME}:27017/llm_caching?authSource=admin"
export VLLM_BASE_URL="http://${ACTUAL_HOSTNAME}:8001/"

# Service configuration
export GUNICORN_WORKERS=${GUNICORN_WORKERS:-1}
export GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-300}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export MONGODB_ENABLED=${MONGODB_ENABLED:-true}

log_message "Environment configured:"
log_message "  REDIS_URL: $REDIS_URL"
log_message "  MONGODB_URL: $MONGODB_URL"
log_message "  VLLM_BASE_URL: $VLLM_BASE_URL"

# --- Phase 2: Configuration Setup ---
log_message "--- Phase 2: Configuration Setup ---"

# Set default configuration values
export DEFAULT_LLM_MODEL=${DEFAULT_LLM_MODEL:-"Qwen_Qwen3-30B-A3B-Instruct-2507-FP8_nothinking_single"}
export DEFAULT_VERIFIER_PROMPT=${DEFAULT_VERIFIER_PROMPT:-"ver_base_json"}

# Model detection and path setting
CONTAINER_CONFIG_PATH="/workspace/llm_caching_service/config.multinode.yml"
CONFIG_TEMPLATE_PATH="/workspace/config.env.yml"

MODEL_KEY=""
MODEL_PATH=""
if [ -f "$CONTAINER_CONFIG_PATH" ]; then
    if grep -q "Qwen3-30B-A3B-Instruct-2507-FP8" "$CONTAINER_CONFIG_PATH"; then
        MODEL_KEY="Qwen"
        MODEL_PATH="/workspace/llm_caching_service/models/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8"
    elif grep -q "Llama-4-Maverick-17B-128E-Instruct-FP8" "$CONTAINER_CONFIG_PATH"; then
        MODEL_KEY="Llama"
        MODEL_PATH="/workspace/llm_caching_service/models/Llama-4-Maverick-17B-128E-Instruct-FP8"
    elif grep -q "Meta-Llama-3-70B-Instruct" "$CONTAINER_CONFIG_PATH"; then
        MODEL_KEY="Meta-Llama-3-70B-Instruct"
        MODEL_PATH="/workspace/llm_caching_service/models/Meta-Llama-3-70B-Instruct"
    elif grep -q "Meta-Llama-3-8B-Instruct" "$CONTAINER_CONFIG_PATH"; then
        MODEL_KEY="Meta-Llama-3-8B-Instruct"
        MODEL_PATH="/workspace/llm_caching_service/models/Meta-Llama-3-8B-Instruct"
    fi
fi

# Default fallback if no config file or model found
if [ -z "$MODEL_PATH" ]; then
    MODEL_KEY="Qwen"
    MODEL_PATH="/workspace/llm_caching_service/models/Qwen_Qwen3-30B-A3B-Instruct-2507-FP8"
fi

export MODEL_PATH
log_message "Detected model: ${MODEL_KEY} -> ${MODEL_PATH}"

# Configuration will be generated after vLLM model discovery
log_message "Configuration will be generated after vLLM model discovery"
export CONFIG_PATH="$CONTAINER_CONFIG_PATH"

# --- Phase 3: Start Database Services ---
log_message "--- Phase 3: Starting Database Services ---"

log_message ">>> Starting Redis..."
(CUDA_VISIBLE_DEVICES="" redis-server /workspace/llm_caching_service/config/redis_multinode.conf --daemonize no) > /logs/redis.log 2>&1 &

TEMP_MONGODB_DIR="/tmp/mongodb"
mkdir -p "$TEMP_MONGODB_DIR"

log_message ">>> Starting MongoDB..."
(CUDA_VISIBLE_DEVICES="" mongod --dbpath "$TEMP_MONGODB_DIR" --bind_ip "$ACTUAL_HOSTNAME" --logpath /logs/mongodb.log --fork)

# --- Phase 4: Wait for Database Services ---
log_message "--- Phase 4: Waiting for Database Services ---"

log_message ">>> Waiting for Redis..."
timeout 60 bash -c "while ! nc -z ${ACTUAL_HOSTNAME} 6379; do sleep 1; done"
log_message "Redis is ready."

log_message ">>> Waiting for MongoDB..."
timeout 60 bash -c "while ! nc -z ${ACTUAL_HOSTNAME} 27017; do sleep 1; done"
until mongosh --host "$ACTUAL_HOSTNAME" --eval "db.adminCommand('ping')" &> /dev/null; do
    log_message "Waiting for MongoDB ping response..."
    sleep 2
done
log_message "MongoDB is ready."

log_message ">>> Initializing MongoDB user..."
(sleep 5 && mongosh --host "$ACTUAL_HOSTNAME" --eval 'try { db = db.getSiblingDB("admin"); db.createUser({user: "admin", pwd: "password", roles: [{role: "root", db: "admin"}]}); print("MongoDB admin user created"); } catch (e) { print("MongoDB admin user already exists or error:", e); }') > /logs/mongodb-init.log 2>&1 &

# --- Phase 5: Start Model Services ---
log_message "--- Phase 5: Starting Model Services ---"

log_message ">>> Starting vLLM server (2 GPUs)..."
(cd /workspace/llm_caching_service && CUDA_VISIBLE_DEVICES="0,1" python3 -m vllm.entrypoints.openai.api_server --host "$ACTUAL_HOSTNAME" --reasoning-parser qwen3 --port 8001 --model "${MODEL_PATH}" --tensor-parallel-size 2 --dtype auto --gpu-memory-utilization 0.9) > "/logs/vllm.log" 2>&1 &
vllm_pid=$!

log_message ">>> Waiting for vLLM server..."
timeout 300 bash -c "
  while ! nc -z ${ACTUAL_HOSTNAME} 8001; do
    echo 'vLLM port not open yet, sleeping...'
    sleep 2
  done
  echo 'vLLM port is open, checking /health endpoint...'
  while [[ \"\$(curl -s -o /dev/null -w ''%{http_code}'' http://${ACTUAL_HOSTNAME}:8001/health)\" != \"200\" ]]; do
    echo 'vLLM /health endpoint not ready, sleeping...'
    sleep 2
  done
"
log_message "vLLM server is ready."

# --- Phase 5.5: Discover Loaded Model Path ---
log_message ">>> Discovering loaded model path in vLLM..."

LOADED_MODEL_PATH=""
RETRY_COUNT=0
MAX_RETRIES=10

until [ $RETRY_COUNT -ge $MAX_RETRIES ]; do
    LOADED_MODEL_PATH=$(curl -s "http://$ACTUAL_HOSTNAME:8001/v1/models" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('data') and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        print('')
except:
    print('')
" 2>/dev/null)
    
    if [ -n "$LOADED_MODEL_PATH" ]; then
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    log_message "Waiting for vLLM model info... attempt $RETRY_COUNT/$MAX_RETRIES"
    sleep 2
done

if [ -z "$LOADED_MODEL_PATH" ]; then
    log_message "Warning: Could not determine loaded model path from vLLM"
    log_message "All model configurations will be processed normally"
else
    export LOADED_MODEL_PATH
    log_message "Loaded model path detected: $LOADED_MODEL_PATH"
    log_message "Only matching model configurations will be enabled"
fi

# --- Phase 5.6: Generate Configuration with Loaded Model Path ---
log_message ">>> Generating configuration with loaded model path..."

# Generate configuration file from environment template with loaded model path
if [ -f "$CONFIG_TEMPLATE_PATH" ]; then
    log_message ">>> Expanding environment configuration template..."
    log_message "Template path: $CONFIG_TEMPLATE_PATH"
    log_message "Output path: /workspace/llm_caching_service/config.runtime.yml"
    log_message "LOADED_MODEL_PATH: $LOADED_MODEL_PATH"
    
    # Run config expansion and capture debug output to log
    /workspace/expand_config.sh "$CONFIG_TEMPLATE_PATH" "/workspace/llm_caching_service/config.runtime.yml" 2>> "$LOG_FILE"
    expansion_result=$?
    
    if [ $expansion_result -eq 0 ]; then
        export CONFIG_PATH="/workspace/llm_caching_service/config.runtime.yml"
        log_message "Using environment-expanded config: $CONFIG_PATH"
        log_message ">>> Generated config sample (first 10 lines):"
        head -10 "$CONFIG_PATH" | while IFS= read -r line; do
            log_message "  $line"
        done
    else
        log_message "Error: Config expansion failed, falling back to original config"
        export CONFIG_PATH="$CONTAINER_CONFIG_PATH"
    fi
else
    log_message "Warning: Template not found at $CONFIG_TEMPLATE_PATH"
    log_message "Using existing config: $CONTAINER_CONFIG_PATH"
    export CONFIG_PATH="$CONTAINER_CONFIG_PATH"
fi

# --- Phase 6: Start LLM Cache Service ---
log_message "--- Phase 6: Starting LLM Cache Service ---"

log_message ">>> Starting LLM Cache Service..."
(cd /workspace/llm_caching_service && 
 PYTHONPATH="/workspace/llm_caching_service" \
 CONFIG_PATH="$CONFIG_PATH" \
 gunicorn \
    -w "${GUNICORN_WORKERS}" \
    -k uvicorn.workers.UvicornWorker \
    --timeout "${GUNICORN_TIMEOUT}" \
    src.main:app \
    --bind 0.0.0.0:8000 \
    --log-level info) > "/logs/llm-cache-service.log" 2>&1 &
cache_pid=$!

# Wait a moment for the service to start
sleep 5

# --- Phase 7: Final Health Checks ---
log_message "--- Phase 7: Final Health Checks ---"

log_message ">>> Verifying all services are healthy..."

# Check vLLM
if curl -s "http://$ACTUAL_HOSTNAME:8001/v1/models" > /dev/null; then
    log_message "✓ vLLM server is healthy"
else
    log_message "✗ vLLM server health check failed"
    exit 1
fi

# Check Redis
if redis-cli -h "$ACTUAL_HOSTNAME" -p 6379 ping > /dev/null 2>&1; then
    log_message "✓ Redis is healthy"
else
    log_message "✗ Redis health check failed"
    exit 1
fi

# Check MongoDB
if mongosh --host "$ACTUAL_HOSTNAME" --eval "db.adminCommand('ping')" &> /dev/null; then
    log_message "✓ MongoDB is healthy"
else
    log_message "✗ MongoDB health check failed"
    exit 1
fi

# Check LLM Cache Service
if curl -s "http://localhost:8000/health" > /dev/null; then
    log_message "✓ LLM Cache Service is healthy"
else
    log_message "⚠ LLM Cache Service health check failed (may still be starting)"
fi

log_message "======================================================================="
log_message ">>> All services are ready and healthy!"
log_message ">>> Services will continue running until training completes..."
log_message "======================================================================="

# Keep services running - they will be killed when the job ends
wait $vllm_pid $cache_pid

log_message "--- Services script finished ---"