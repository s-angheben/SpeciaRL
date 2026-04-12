"""
Request correlation middleware for tracing requests across the application.
Adds unique request IDs and binds them to logging context.
"""

import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import bind_request_context, clear_request_context, get_logger

logger = get_logger(__name__)


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request correlation IDs to all requests.
    Automatically binds the request ID to the logging context.
    """
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())

        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None
        )

        try:
            response = await call_next(request)

            response.headers[self.header_name] = request_id
            
            return response

        except AttributeError as e:
            # Log the request body for debugging
            body = await request.body()
            logger.error("AttributeError during request processing", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        request_body=body.decode('utf-8', errors='ignore'))
            raise
            
        except Exception as e:
            logger.error("Request processing failed", 
                        error=str(e), 
                        error_type=type(e).__name__)
            raise
        finally:
            clear_request_context()