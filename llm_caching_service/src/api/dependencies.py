import asyncio
from fastapi import Request, Depends
from src.schemas import ClassificationRequest
from src.core.clients.base import LLMClientABC
from src.core.config import settings
from src.core.exceptions import ModelNotFoundError
from src.core.logging import get_logger

logger = get_logger(__name__)

async def get_llm_client_for_model(request: Request, model_name: str) -> LLMClientABC:
    
    client_pool = request.app.state.llm_client_pools.get(model_name)
    lock = request.app.state.pool_locks.get(model_name)

    if not client_pool:
        available_models = list(request.app.state.llm_client_pools.keys())
        raise ModelNotFoundError(model_name, available_models)
    
    # Check if this is a Gemini client (fast path - no lock, no health check)
    first_client = client_pool[0]
    if first_client.get_client_type() == "gemini":
        #print(f"Routing request for '{model_name}' to Gemini API client (no lock)")
        return first_client
    
    # vLLM path - use existing round-robin with lock and health checks
    if not lock:
        available_models = list(request.app.state.llm_client_pools.keys())
        raise ModelNotFoundError(model_name, available_models)
    
    async with lock:
        # Loop up to the number of clients in the pool to find a healthy one
        for _ in range(len(client_pool)):
            current_index = request.app.state.round_robin_counters.get(model_name, 0)
            
            client_to_test = client_pool[current_index]
            
            # Increment counter for the next request immediately
            next_index = (current_index + 1) % len(client_pool)
            request.app.state.round_robin_counters[model_name] = next_index

            # Perform the health check
            if await client_to_test.check_health():
                logger.debug("Routing to healthy vLLM instance",
                           model_name=model_name,
                           api_base=client_to_test.api_base,
                           client_index=current_index)
                return client_to_test
            else:
                logger.warning("Skipping unhealthy client",
                             model_name=model_name,
                             client_index=current_index)

    # If we exit the loop, no healthy clients were found
    available_models = list(request.app.state.llm_client_pools.keys())
    raise ModelNotFoundError(model_name, available_models)
