"""
Application lifecycle management for the LLM Caching Service.
Handles startup and shutdown of client pools, connections, and resources.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI

from src.core.clients import create_client_instance, LLMClientABC
from src.core.clients.redis_client import redis_client
from src.core.mongodb_client import mongodb_client
from src.core.config import settings
from src.core.logging import get_logger, LogEvents
from src.services.mongodb_worker_manager import mongodb_worker_manager

logger = get_logger(__name__)


async def initialize_client_pools(app: FastAPI):
    logger.info(LogEvents.APP_START, stage="client_pools")
    
    app.state.llm_client_pools: Dict[str, List[LLMClientABC]] = {}
    
    for model_config in settings.MODELS_CONFIG:
        model_name = model_config.name
        try:
            logger.info(LogEvents.CLIENT_INIT, 
                       model=model_name, 
                       client_type=model_config.type)
            
            client = create_client_instance(model_config)
            await client.initialize()
            
            if model_name not in app.state.llm_client_pools:
                app.state.llm_client_pools[model_name] = []
            
            app.state.llm_client_pools[model_name].append(client)
            pool_size = len(app.state.llm_client_pools[model_name])
            
            logger.info(LogEvents.CLIENT_READY, 
                       model=model_name, 
                       pool_size=pool_size,
                       client_type=model_config.type)

        except Exception as e:
            logger.error(LogEvents.CLIENT_FAILED, 
                        model=model_name, 
                        error=str(e),
                        error_type=type(e).__name__)


def initialize_round_robin_state(app: FastAPI):
    app.state.round_robin_counters = {
        model_name: 0 for model_name in app.state.llm_client_pools
    }
    
    app.state.pool_locks = {
        model_name: asyncio.Lock() for model_name in app.state.llm_client_pools
    }


def validate_configuration(app: FastAPI):
    pool_info = {
        name: len(pool) for name, pool in app.state.llm_client_pools.items()
    }
    logger.info("Client pools initialized", pools=pool_info)

    if settings.DEFAULT_LLM_MODEL not in app.state.llm_client_pools:
        logger.warning("Default model not available", 
                      default_model=settings.DEFAULT_LLM_MODEL,
                      available_models=list(pool_info.keys()))


async def shutdown_resources(app: FastAPI):
    logger.info(LogEvents.APP_SHUTDOWN, stage="starting")

    try:
        await mongodb_worker_manager.stop_workers()

        await mongodb_client.disconnect()
        logger.info("MongoDB connection closed")

        await redis_client.close()
        logger.info("Redis connection closed")

        for model_name, client_pool in app.state.llm_client_pools.items():
            for client in client_pool:
                await client.close_client()
            logger.info("Client pool closed", model=model_name, pool_size=len(client_pool))
        
        logger.info(LogEvents.APP_SHUTDOWN, stage="complete")
    except Exception as e:
        logger.error("Error during shutdown", 
                    error=str(e), 
                    error_type=type(e).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown of all application resources.
    """
    logger.info(LogEvents.APP_START)
    try:
        await initialize_client_pools(app)
        initialize_round_robin_state(app)
        validate_configuration(app)

        await mongodb_worker_manager.start_workers()
        
        logger.info(LogEvents.APP_READY)
    except Exception as e:
        logger.error("Application startup failed", 
                    error=str(e), 
                    error_type=type(e).__name__)
        raise
    
    yield

    await shutdown_resources(app)