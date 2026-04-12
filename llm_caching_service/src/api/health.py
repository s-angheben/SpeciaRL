"""
Health check endpoints for the LLM Caching Service.
Provides system status and client pool health information.
"""

import asyncio
from fastapi import APIRouter, Request

from src.core.config import settings

health_router = APIRouter()


@health_router.get("/health", tags=["Health"])
async def health_check(request: Request):
    """
    Comprehensive health check endpoint.
    Returns status of all client pools and their health.
    """
    healthy_instances = {}
    
    # Check health of all clients in all pools and update metrics
    for name, pool in request.app.state.llm_client_pools.items():
        health_checks = await asyncio.gather(
            *[client.check_health() for client in pool]
        )
        healthy_count = sum(1 for is_healthy in health_checks if is_healthy)
        healthy_instances[name] = healthy_count
        

    # Build comprehensive status response
    loaded_model_pools = {
        name: {
            "configured_instances": len(pool),
            "healthy_instances": healthy_instances.get(name, 0),
        }
        for name, pool in request.app.state.llm_client_pools.items()
    }

    return {
        "status": "ok",
        "loaded_model_pools": loaded_model_pools,
        "default_model": settings.DEFAULT_LLM_MODEL,
    }