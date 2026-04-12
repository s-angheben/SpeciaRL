"""
Async classification logging service.
Publishes classification events to Redis queue for processing by MongoDB workers.
"""

import orjson
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from zoneinfo import ZoneInfo
import structlog
from src.core.clients.redis_client import redis_client
from src.core.config import settings
from src.core.logging import get_logger
from src.schemas.logging import ClassificationLogEvent

logger = get_logger(__name__)

ROME_TZ = ZoneInfo("Europe/Rome")

CLASSIFICATION_LOG_QUEUE = "classification_log_queue"
QUEUE_BATCH_SIZE = 50
QUEUE_TIMEOUT = 5.0


class ClassificationLogger:
    """
    Async logger for classification events.
    Publishes events to Redis queue for processing by MongoDB workers.
    """
    
    def __init__(self):
        self._enabled = settings.MONGODB_ENABLED
        if not self._enabled:
            logger.info("MongoDB logging disabled in configuration")
    
    async def log_classifications(
        self,
        ground_truths: List[str],
        predictions: List[str],
        classifications: List[str],
        run_name: str,
        model: str,
        prompt: str,
        extra_infos: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> None:
        """
        Log classification results asynchronously.
        
        Args:
            ground_truths: List of ground truth labels
            predictions: List of predicted labels
            classifications: List of classification results
            run_name: Name of the run/experiment
            model: Model used for classification
            prompt: Prompt template used
            extra_infos: Optional list of metadata dicts for each ground_truth/prediction pair
        """
        if not self._enabled:
            return
        
        try:
            current_time = datetime.now(ROME_TZ)

            request_id = structlog.contextvars.get_contextvars().get('request_id')

            events = []
            for i, (gt, pred, result) in enumerate(zip(ground_truths, predictions, classifications)):
                extra_info = None
                if extra_infos is not None and i < len(extra_infos):
                    extra_info = extra_infos[i]
                
                event = ClassificationLogEvent(
                    ground_truth=gt,
                    prediction=pred,
                    run_name=run_name,
                    model=model,
                    prompt=prompt,
                    classification_result=result,
                    timestamp=current_time,
                    request_id=request_id,
                    extra_info=extra_info
                )
                events.append(event)

            await self._publish_events(events)
            
            logger.debug("Classification events queued for logging",
                        count=len(events),
                        run_name=run_name,
                        model=model)
            
        except Exception as e:
            logger.error("Failed to queue classification events for logging",
                        error=str(e),
                        run_name=run_name,
                        model=model)
    
    async def _publish_events(self, events: List[ClassificationLogEvent]) -> None:
        if not events:
            return

        for i in range(0, len(events), QUEUE_BATCH_SIZE):
            batch = events[i:i + QUEUE_BATCH_SIZE]

            batch_data = [event.model_dump() for event in batch]
            batch_json = orjson.dumps(batch_data).decode('utf-8')

            await redis_client.lpush(CLASSIFICATION_LOG_QUEUE, batch_json)
    
    async def log_single_classification(
        self,
        ground_truth: str,
        prediction: str,
        classification: str,
        run_name: str,
        model: str,
        prompt: str,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a single classification result.
        
        Args:
            ground_truth: Ground truth label
            prediction: Predicted label
            classification: Classification result
            run_name: Name of the run/experiment
            model: Model used for classification
            prompt: Prompt template used
            extra_info: Optional metadata for this classification
        """
        await self.log_classifications(
            ground_truths=[ground_truth],
            predictions=[prediction],
            classifications=[classification],
            run_name=run_name,
            model=model,
            prompt=prompt,
            extra_infos=[extra_info] if extra_info is not None else None
        )
    
    async def get_queue_size(self) -> int:
        if not self._enabled:
            return 0

        try:
            return await redis_client.llen(CLASSIFICATION_LOG_QUEUE)
        except Exception as e:
            logger.error("Failed to get queue size", error=str(e))
            return 0

    @property
    def is_enabled(self) -> bool:
        return self._enabled


classification_logger = ClassificationLogger()