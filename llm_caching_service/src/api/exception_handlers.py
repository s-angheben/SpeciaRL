"""
Production-level exception handlers for the FastAPI application.
Centralizes all error handling logic with user-friendly responses.
"""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.core.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    LLMResponseParseError,
    CachePollTimeoutError,
    InvalidPromptError,
    ModelNotFoundError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


async def rate_limit_error_handler(request: Request, exc: RateLimitError):
    """Handle rate limit errors from downstream APIs"""
    logger.warning("Rate limit exceeded", 
                  error=str(exc),
                  path=request.url.path,
                  method=request.method)
    
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded for downstream API. Please try again later. Original error: {exc}"
        },
    )


async def service_unavailable_error_handler(request: Request, exc: ServiceUnavailableError):
    """Handle service unavailable errors"""
    return JSONResponse(
        status_code=503,
        content={
            "detail": f"A downstream service is unavailable. Please try again later. Original error: {exc}"
        },
    )


async def llm_response_parse_error_handler(request: Request, exc: LLMResponseParseError):
    """Handle LLM response parsing errors"""
    return JSONResponse(
        status_code=502,
        content={
            "detail": f"Failed to parse response from the downstream LLM. Original error: {exc}"
        },
    )


async def cache_poll_timeout_error_handler(request: Request, exc: CachePollTimeoutError):
    """Handle cache polling timeout errors"""
    return JSONResponse(
        status_code=504,
        content={
            "detail": f"Timed out waiting for another worker to process an item. Please try again. Original error: {exc}"
        },
    )


async def invalid_prompt_error_handler(request: Request, exc: InvalidPromptError):
    """Handle invalid verifier prompt errors"""
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "invalid_prompt": exc.prompt_name,
            "available_prompts": exc.available_prompts,
        },
    )


async def model_not_found_error_handler(request: Request, exc: ModelNotFoundError):
    """Handle invalid model errors"""
    logger.warning("Invalid model requested", 
                  model=exc.model_name,
                  available_models=exc.available_models,
                  path=request.url.path)
    
    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc),
            "invalid_model": exc.model_name,
            "available_models": exc.available_models,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Transform Pydantic validation errors into user-friendly messages"""
    errors = {}
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"] if loc != "body")
        errors[field_path] = error["msg"]

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": errors,
            "example": {
                "ground_truths": ["cat", "dog"],
                "predictions": ["feline", "canine"],
                "model": "gemini-2.5-flash-lite-preview-06-17",
                "verifier_prompt": "ver_base_ndjson",
            },
            "help": "See GET /api/v1/schema for full request specification",
        },
    )


async def value_error_handler(request: Request, exc: ValueError):
    """Handle general ValueError exceptions"""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


async def not_found_handler(request: Request, exc):
    """Handle 404 not found errors"""
    return JSONResponse(
        status_code=404,
        content={
            "detail": f"Endpoint not found: {request.method} {request.url.path}",
            "available_endpoints": [
                "GET /health",
                "POST /api/v1/classify",
                "GET /api/v1/prompts",
                "GET /api/v1/models",
                "GET /api/v1/schema",
            ],
        },
    )