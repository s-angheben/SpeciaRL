#!/bin/bash
# Wait for Redis, MongoDB and vLLM to be ready, snapshot their state into
# /workspace/llm_caching_service/backup, then exec the llm-cache-service.
set -e

GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-300}
GUNICORN_WORKERS=${GUNICORN_WORKERS:-1}

REDIS_HOST=${REDIS_HOST:-localhost}
REDIS_PORT=${REDIS_PORT:-6379}

MONGO_HOST=${MONGO_HOST:-localhost}
MONGO_PORT=${MONGO_PORT:-27017}

VLLM_HOST=${VLLM_HOST:-localhost}
VLLM_PORT=${VLLM_PORT:-8001}

echo "--- Waiting for Redis (${REDIS_HOST}:${REDIS_PORT}) ---"
timeout 60 bash -c "while ! nc -z ${REDIS_HOST} ${REDIS_PORT}; do sleep 1; done"
echo "Redis is up."

echo "--- Waiting for MongoDB (${MONGO_HOST}:${MONGO_PORT}) ---"
timeout 60 bash -c "while ! nc -z ${MONGO_HOST} ${MONGO_PORT}; do sleep 1; done"
echo "MongoDB is up."

echo "--- Waiting for vLLM server (${VLLM_HOST}:${VLLM_PORT}) ---"
timeout 300 bash -c "
  while ! nc -z ${VLLM_HOST} ${VLLM_PORT}; do
    echo 'vLLM port not open yet, sleeping...'
    sleep 2
  done
  echo 'vLLM port is open, now checking /health endpoint...'
  while [[ \"\$(curl -s -o /dev/null -w ''%{http_code}'' http://${VLLM_HOST}:${VLLM_PORT}/health)\" != \"200\" ]]; do
    echo 'vLLM /health endpoint not ready yet, sleeping...'
    sleep 2
  done
"
echo "vLLM server is up and healthy."
mkdir -p /workspace/llm_caching_service/backup

echo "--- Saving Redis data to /workspace/llm_caching_service/backup/dump.rdb ---"
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" SAVE
cp /data/dump.rdb /workspace/llm_caching_service/backup/dump.rdb || echo "Warning: Redis dump not found at /data/dump.rdb"

echo "--- Dumping MongoDB to /workspace/llm_caching_service/backup/mongodb_dump ---"
mongodump --host "${MONGO_HOST}" --port "${MONGO_PORT}" --out /workspace/llm_caching_service/backup/mongodb_dump || echo "Warning: mongodump failed, continuing..."

echo "--- All services are ready. Starting llm-cache-service... ---"
echo "Gunicorn settings:"
echo "  Workers: ${GUNICORN_WORKERS}"
echo "  Timeout: ${GUNICORN_TIMEOUT} seconds"

exec gunicorn \
    -w "${GUNICORN_WORKERS}" \
    -k uvicorn.workers.UvicornWorker \
    --timeout "${GUNICORN_TIMEOUT}" \
    src.main:app \
    --bind 0.0.0.0:8000 \
    --log-level info
