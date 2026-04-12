Bootstrap: docker
From: verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2

%environment

%post
    # Install prerequisite system packages
    apt-get update && apt-get install -y \
        supervisor procps netcat pigz \
        gnupg curl ca-certificates lsb-release software-properties-common

    # --- Add Official Redis Repository (to get a newer version) ---
    # Download the Redis GPG key
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
    # Add the repository to the sources list
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb/ $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/redis.list

    # --- Add Official MongoDB 7.0 Repository (upgraded from 6.0) ---
    # Download the MongoDB 7.0 GPG key
    curl -fsSL https://pgp.mongodb.com/server-7.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
    # Add the 7.0 repository to the sources list
    echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/7.0 multiverse" \
      > /etc/apt/sources.list.d/mongodb-org-7.0.list

    # --- Install the packages from the new repositories ---
    # Update package lists again to recognize the new repos, then install
    apt-get update && apt-get install -y \
        redis-server \
        mongodb-org \
        && rm -rf /var/lib/apt/lists/*

    # Install Python dependencies
    pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        gunicorn \
        requests \
        numpy \
        scipy \
        pandas \
        pyarrow \
        Pillow \
        tqdm \
        datasets \
        matplotlib \
        openai \
        anthropic \
        google-genai \
        pydantic>=2.0.0 \
        PyYAML \
        orjson \
        uvloop \
        msgpack \
        redis \
        motor \
        structlog \
        tenacity

    mkdir /workspace/verl
    mkdir /workspace/vlm_openworld_evaluator
    mkdir /workspace/llm_caching_service

%runscript
    cd /workspace/verl
    exec "$@"

%help
    
    Usage:
    apptainer run --nv image.sif python3 -m verl.trainer.main_ppo [args]
    
    Build:
    apptainer build image.sif Apptainer.def
