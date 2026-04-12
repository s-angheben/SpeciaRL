"""
Service layer for run management operations.
Handles status checking and data retrieval for classification runs.
"""

import orjson
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo

from src.core.clients.redis_client import redis_client
from src.core.mongodb_client import mongodb_client
from src.core.config import settings
from src.core.logging import get_logger
from src.schemas.runs import RunStatus, ClassificationStats
from src.schemas.logging import ClassificationLogEntry

logger = get_logger(__name__)

ROME_TZ = ZoneInfo("Europe/Rome")

CLASSIFICATION_LOG_QUEUE = "classification_log_queue"


class RunService:
    """Service for managing classification runs."""
    
    async def get_run_status(self, run_name: str) -> Dict[str, Any]:
        """
        Get the status of a classification run.
        
        Args:
            run_name: Name of the run to check
            
        Returns:
            Dict containing run status information
        """
        try:
            queue_pending_count = await self._count_pending_events_for_run(run_name)

            total_processed, last_activity = await self._get_run_processing_stats(run_name)

            if queue_pending_count > 0:
                status = RunStatus.PROCESSING
            elif total_processed > 0:
                status = RunStatus.COMPLETED
            else:
                status = RunStatus.IDLE
            
            return {
                "run_name": run_name,
                "status": status,
                "queue_pending_count": queue_pending_count,
                "total_processed": total_processed,
                "last_activity": last_activity
            }
            
        except Exception as e:
            logger.error("Failed to get run status", run_name=run_name, error=str(e))
            raise
    
    async def get_run_results(
        self, 
        run_name: str, 
        limit: int = 10000, 
        offset: int = 0,
        include_stats: bool = False,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get classification results for a run with pagination.
        
        Args:
            run_name: Name of the run
            limit: Maximum number of results to return (default: 10000, max: 50000)
            offset: Number of results to skip
            include_stats: Whether to include aggregated statistics (default: False for performance)
            fields: List of fields to include (None = all fields, ['basic'] = essential fields only)
            
        Returns:
            Dict containing paginated results and optional statistics
        """
        try:
            if not mongodb_client.is_connected:
                raise RuntimeError("MongoDB not connected")
            
            collection = mongodb_client.collection

            limit = min(limit, 50000)

            projection = None
            if fields is not None:
                if fields == ['basic']:
                    # Essential fields only for performance
                    projection = {
                        "ground_truth": 1,
                        "prediction": 1, 
                        "classification_result": 1,
                        "count": 1,
                        "first_seen": 1,
                        "last_seen": 1,
                        "model": 1,
                        "run_name": 1
                    }
                else:
                    projection = {field: 1 for field in fields}
                    projection["_id"] = 1  # Always include _id

            total_count = await collection.count_documents({"run_name": run_name})

            cursor = collection.find(
                {"run_name": run_name}, 
                projection
            ).sort([("last_seen", -1)]).skip(offset).limit(limit)
            results = await cursor.to_list(length=limit)

            classification_entries = []
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                classification_entries.append(ClassificationLogEntry(**doc))

            stats = None
            if include_stats and total_count > 0:
                stats = await self._get_run_statistics(run_name)
            
            return {
                "run_name": run_name,
                "total_count": total_count,
                "returned_count": len(results),
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(results) < total_count,
                "stats": stats,
                "results": classification_entries
            }
            
        except Exception as e:
            logger.error("Failed to get run results", run_name=run_name, error=str(e))
            raise
    
    async def _count_pending_events_for_run(self, run_name: str) -> int:
        """Count pending events in Redis queue for a specific run."""
        try:
            queue_length = await redis_client.llen(CLASSIFICATION_LOG_QUEUE)
            if queue_length == 0:
                return 0
            
            # Sample queue items to count events for this run
            # Note: This is a sampling approach for performance
            sample_size = min(100, queue_length)
            items = await redis_client.lrange(CLASSIFICATION_LOG_QUEUE, 0, sample_size - 1)
            
            run_events_in_sample = 0
            for item in items:
                try:
                    batch_data = orjson.loads(item)
                    if isinstance(batch_data, list):
                        for event in batch_data:
                            if event.get("run_name") == run_name:
                                run_events_in_sample += 1
                except orjson.JSONDecodeError:
                    continue
            
            if sample_size < queue_length:
                estimated_count = int((run_events_in_sample / sample_size) * queue_length)
                return estimated_count
            else:
                return run_events_in_sample
                
        except Exception as e:
            logger.warning("Failed to count pending events", run_name=run_name, error=str(e))
            return 0
    
    async def _get_run_processing_stats(self, run_name: str) -> Tuple[int, Optional[datetime]]:
        """Get total processed count and last activity for a run."""
        try:
            if not mongodb_client.is_connected:
                return 0, None
            
            collection = mongodb_client.collection
            
            # Get total processed events (sum of count field)
            pipeline = [
                {"$match": {"run_name": run_name}},
                {"$group": {"_id": None, "total": {"$sum": "$count"}}}
            ]
            
            result = await collection.aggregate(pipeline).to_list(length=1)
            total_processed = result[0]["total"] if result else 0

            last_doc = await collection.find_one(
                {"run_name": run_name},
                sort=[("last_seen", -1)]
            )
            
            last_activity = last_doc["last_seen"] if last_doc else None
            
            return total_processed, last_activity
            
        except Exception as e:
            logger.warning("Failed to get processing stats", run_name=run_name, error=str(e))
            return 0, None
    
    async def _get_run_statistics(self, run_name: str) -> Optional[ClassificationStats]:
        """Get aggregated statistics for a run."""
        try:
            if not mongodb_client.is_connected:
                return None
            
            collection = mongodb_client.collection
            
            # Aggregation pipeline for classification distribution
            pipeline = [
                {"$match": {"run_name": run_name}},
                {"$group": {
                    "_id": "$classification_result",
                    "count": {"$sum": "$count"},
                    "docs": {"$sum": 1}
                }}
            ]
            
            distribution_result = await collection.aggregate(pipeline).to_list(length=None)
            classification_distribution = {
                item["_id"]: item["count"] for item in distribution_result
            }

            total_classifications = sum(classification_distribution.values())
            error_count = classification_distribution.get("error", 0)
            success_rate = ((total_classifications - error_count) / total_classifications * 100) if total_classifications > 0 else 0

            models_pipeline = [
                {"$match": {"run_name": run_name}},
                {"$group": {"_id": "$model"}},
                {"$sort": {"_id": 1}}
            ]
            models_result = await collection.aggregate(models_pipeline).to_list(length=None)
            models_used = [item["_id"] for item in models_result]

            time_pipeline = [
                {"$match": {"run_name": run_name}},
                {"$group": {
                    "_id": None,
                    "first_seen": {"$min": "$first_seen"},
                    "last_seen": {"$max": "$last_seen"}
                }}
            ]
            time_result = await collection.aggregate(time_pipeline).to_list(length=1)
            time_range = {
                "first_seen": time_result[0]["first_seen"] if time_result else None,
                "last_seen": time_result[0]["last_seen"] if time_result else None
            }
            
            return ClassificationStats(
                classification_distribution=classification_distribution,
                total_classifications=total_classifications,
                success_rate=round(success_rate, 2),
                models_used=models_used,
                time_range=time_range
            )
            
        except Exception as e:
            logger.warning("Failed to get run statistics", run_name=run_name, error=str(e))
            return None


run_service = RunService()