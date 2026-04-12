"""
LLM Caching Service - Main Application Entry Point

A production-ready FastAPI service for caching and classifying 
LLM predictions with multi-model support and load balancing.
"""

try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

from src.core.app import create_application

app = create_application()