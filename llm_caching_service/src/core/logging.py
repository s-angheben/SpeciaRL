"""
Production-level structured logging configuration.
Provides JSON logging for production and pretty console logging for development.
"""

import logging
import logging.handlers
import os
import sys
from typing import Any, Dict

import structlog
from structlog.types import Processor


def setup_logging(
    log_level: str = "INFO",
    use_json: bool = True,
    include_stdlib: bool = True,
    file_logging_enabled: bool = False,
    error_file_path: str = "logs/errors.log",
    max_file_size: str = "50MB",
    rotate_count: int = 3
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_json: Use JSON formatting (True for production, False for development)
        include_stdlib: Include standard library logging integration
        file_logging_enabled: Enable file logging for errors
        error_file_path: Path to error log file
        max_file_size: Maximum file size before rotation (e.g., "50MB")
        rotate_count: Number of backup files to keep
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup file logging if enabled
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if file_logging_enabled:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(error_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Parse max file size (convert "50MB" to bytes)
        max_bytes = _parse_file_size(max_file_size)
        
        # Add rotating file handler for errors
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=max_bytes,
            backupCount=rotate_count
        )
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        handlers=handlers,
        level=numeric_level,
    )
    
    # Define processors based on environment
    processors: list[Processor] = [
        # Add request ID to all logs if available
        structlog.contextvars.merge_contextvars,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add log level
        structlog.stdlib.add_log_level,
        # Add logger name
        structlog.stdlib.add_logger_name,
    ]
    
    if use_json:
        # Production: JSON formatting for log aggregation
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Development: Pretty console output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Reduce noise from third-party libraries
    if include_stdlib:
        _configure_third_party_logging()


def _configure_third_party_logging() -> None:
    """Configure logging levels for third-party libraries to reduce noise"""
    # Reduce uvicorn access log noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Reduce HTTP client noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Reduce redis client noise
    logging.getLogger("redis").setLevel(logging.WARNING)
    
    # Reduce prometheus noise
    logging.getLogger("prometheus_client").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def bind_request_context(request_id: str, **kwargs: Any) -> None:
    """
    Bind request-specific context to all subsequent logs.
    
    Args:
        request_id: Unique request identifier
        **kwargs: Additional context to bind
    """
    context = {"request_id": request_id, **kwargs}
    structlog.contextvars.bind_contextvars(**context)


def clear_request_context() -> None:
    """Clear request-specific context from logs"""
    structlog.contextvars.clear_contextvars()


def _parse_file_size(size_str: str) -> int:
    """
    Parse file size string to bytes.
    
    Args:
        size_str: Size string like "50MB", "1GB", "100KB"
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # Assume bytes if no unit
        return int(size_str)


# Global logger instances for common use
logger = get_logger(__name__)


# Common log event templates for consistency
class LogEvents:
    """Standard log event templates for consistent logging"""
    
    # Application lifecycle
    APP_START = "Application starting"
    APP_READY = "Application ready"
    APP_SHUTDOWN = "Application shutdown"
    
    # Client management
    CLIENT_INIT = "Initializing client"
    CLIENT_READY = "Client initialized successfully"
    CLIENT_FAILED = "Client initialization failed"
    CLIENT_HEALTH_CHECK = "Client health check"
    
    # Request processing
    REQUEST_START = "Processing request"
    REQUEST_COMPLETE = "Request completed"
    REQUEST_ERROR = "Request failed"
    
    # Cache operations
    CACHE_HIT = "Cache hit"
    CACHE_MISS = "Cache miss"
    CACHE_ERROR = "Cache operation failed"
    
    # LLM operations
    LLM_REQUEST = "LLM API request"
    LLM_RESPONSE = "LLM API response"
    LLM_ERROR = "LLM API error"
    
    # Configuration
    CONFIG_LOADED = "Configuration loaded"
    CONFIG_ERROR = "Configuration error"