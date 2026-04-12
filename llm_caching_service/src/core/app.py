"""
FastAPI application factory and configuration.
Creates and configures the FastAPI application with all routes and handlers.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from src.api.router import api_router
from src.api.health import health_router
from src.api import exception_handlers
from src.core.lifecycle import lifespan
from src.core.config import settings
from src.core.logging import setup_logging, get_logger
from src.core.middleware import RequestCorrelationMiddleware
from src.core.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    LLMResponseParseError,
    CachePollTimeoutError,
    InvalidPromptError,
    ModelNotFoundError,
)


def create_application() -> FastAPI:
    """
    Application factory that creates and configures the FastAPI app.
    
    Returns:
        FastAPI: Fully configured application instance
    """
    use_json = settings.LOG_FORMAT.lower() == "json"
    setup_logging(
        log_level=settings.LOG_LEVEL,
        use_json=use_json,
        include_stdlib=settings.LOG_INCLUDE_STDLIB,
        file_logging_enabled=settings.LOG_FILE_ENABLED,
        error_file_path=settings.LOG_ERROR_FILE,
        max_file_size=settings.LOG_MAX_SIZE,
        rotate_count=settings.LOG_ROTATE_COUNT
    )
    
    logger = get_logger(__name__)
    logger.info("Creating FastAPI application", 
               log_level=settings.LOG_LEVEL, 
               log_format=settings.LOG_FORMAT)
    
    app = FastAPI(
        title="LLM Caching Service",
        description="A multi-model, load-balanced, and concurrency-safe caching service.",
        version="1.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestCorrelationMiddleware)

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(health_router)

    _register_exception_handlers(app)

    return app


def _register_exception_handlers(app: FastAPI):
    app.add_exception_handler(RateLimitError, exception_handlers.rate_limit_error_handler)
    app.add_exception_handler(ServiceUnavailableError, exception_handlers.service_unavailable_error_handler)
    app.add_exception_handler(LLMResponseParseError, exception_handlers.llm_response_parse_error_handler)
    app.add_exception_handler(CachePollTimeoutError, exception_handlers.cache_poll_timeout_error_handler)
    app.add_exception_handler(InvalidPromptError, exception_handlers.invalid_prompt_error_handler)
    app.add_exception_handler(ModelNotFoundError, exception_handlers.model_not_found_error_handler)

    app.add_exception_handler(RequestValidationError, exception_handlers.validation_exception_handler)
    app.add_exception_handler(ValueError, exception_handlers.value_error_handler)
    app.add_exception_handler(404, exception_handlers.not_found_handler)