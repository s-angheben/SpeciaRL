"""
API routes for run management.
Provides endpoints for monitoring run status and retrieving results.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.schemas.runs import RunStatusResponse, RunResultsResponse
from src.services.run_service import run_service
from src.core.logging import get_logger
from src.core.config import settings

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/runs/{run_name}/status",
    response_model=RunStatusResponse,
    tags=["Run Management"],
    summary="Get run processing status"
)
async def get_run_status(run_name: str):
    """
    Get the current processing status of a classification run.
    
    Returns information about:
    - Current status (processing/completed/idle)
    - Number of pending events in queue
    - Total events processed
    - Last activity timestamp
    """
    try:
        status_data = await run_service.get_run_status(run_name)
        return RunStatusResponse(**status_data)
        
    except Exception as e:
        logger.error("Failed to get run status", run_name=run_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve run status: {str(e)}"
        )


@router.get(
    "/runs/{run_name}/results",
    response_model=RunResultsResponse,
    tags=["Run Management"],
    summary="Get run classification results"
)
async def get_run_results(
    run_name: str,
    limit: int = Query(default=10000, ge=1, le=50000, description="Maximum number of results per page"),
    offset: int = Query(default=0, ge=0, description="Number of results to skip"),
    include_stats: bool = Query(default=False, description="Include aggregated statistics"),
    fields: Optional[str] = Query(default=None, description="Comma-separated fields to include (e.g., 'basic' for essential fields only)")
):
    """
    Get classification results for a run with pagination and optional statistics.
    
    Returns:
    - Paginated classification log entries
    - Total count and pagination info
    - Optional aggregated statistics including:
      - Classification distribution (equivalent/different/error counts)
      - Success rate
      - Models used
      - Time range
    """
    try:
        if not settings.MONGODB_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="MongoDB logging is disabled. Cannot retrieve run results."
            )

        fields_list = None
        if fields:
            if fields == "basic":
                fields_list = ["basic"]
            else:
                fields_list = [f.strip() for f in fields.split(",")]
        
        results_data = await run_service.get_run_results(
            run_name=run_name,
            limit=limit,
            offset=offset,
            include_stats=include_stats,
            fields=fields_list
        )
        
        return RunResultsResponse(**results_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get run results", run_name=run_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve run results: {str(e)}"
        )