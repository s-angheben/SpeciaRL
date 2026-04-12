"""
MongoDB writer worker for processing classification log events.
Consumes events from Redis queue and writes to MongoDB using atomic operations.
"""

import asyncio
import orjson
from datetime import datetime
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo
from pymongo.errors import DuplicateKeyError
from src.core.mongodb_client import mongodb_client
from src.core.clients.redis_client import redis_client
from src.core.config import settings
from src.core.logging import get_logger
from src.schemas.logging import ClassificationLogEvent, OccurrenceEntry

logger = get_logger(__name__)

ROME_TZ = ZoneInfo("Europe/Rome")

CLASSIFICATION_LOG_QUEUE = "classification_log_queue"


class MongoDBWriter:
    """
    MongoDB writer worker that processes classification log events.
    Uses atomic operations to safely handle concurrent writes.
    """
    
    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id
        self._running = False
        self._shutdown_event = asyncio.Event()
        
    async def start(self):
        if not settings.MONGODB_ENABLED:
            logger.info("MongoDB logging disabled, worker not starting")
            return

        logger.info(f"Starting MongoDB writer worker {self.worker_id}")

        await mongodb_client.connect()
        
        self._running = True
        
        try:
            await self._run_worker_loop()
        except Exception as e:
            logger.error(f"Worker {self.worker_id} loop failed", error=str(e))
            raise
        finally:
            await self._cleanup()
    
    async def stop(self):
        logger.info(f"Stopping MongoDB writer worker {self.worker_id}")
        self._running = False
        self._shutdown_event.set()

    async def _run_worker_loop(self):
        logger.info(f"MongoDB writer worker {self.worker_id} loop started")

        while self._running:
            try:
                if self._shutdown_event.is_set():
                    break

                batch = await self._get_event_batch()

                if not batch:
                    await asyncio.sleep(0.1)
                    continue

                await self._process_event_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in worker {self.worker_id} loop", error=str(e))
                await asyncio.sleep(1.0)
    
    async def _get_event_batch(self) -> List[ClassificationLogEvent]:
        try:
            batch_data = await redis_client.brpop(CLASSIFICATION_LOG_QUEUE, timeout=5.0)

            if not batch_data:
                return []

            _, batch_json = batch_data
            batch_raw = orjson.loads(batch_json)

            events = []
            for event_data in batch_raw:
                if isinstance(event_data.get('timestamp'), str):
                    event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                
                event = ClassificationLogEvent(**event_data)
                events.append(event)
            
            return events
            
        except orjson.JSONDecodeError as e:
            logger.error("Failed to parse event batch JSON", error=str(e))
            return []
        except Exception as e:
            logger.error("Failed to get event batch", error=str(e))
            return []
    
    async def _process_event_batch(self, events: List[ClassificationLogEvent]):
        if not events:
            return

        logger.debug(f"Processing batch of {len(events)} events")

        grouped_events = self._group_events_by_key(events)

        for key, event_group in grouped_events.items():
            await self._process_event_group(key, event_group)

        logger.debug(f"Worker {self.worker_id} processed batch of {len(events)} events into {len(grouped_events)} documents")

    def _group_events_by_key(self, events: List[ClassificationLogEvent]) -> Dict[str, List[ClassificationLogEvent]]:
        grouped = {}

        for event in events:
            key = f"{event.ground_truth}:{event.prediction}:{event.run_name}:{event.model}:{event.prompt}"
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)
        
        return grouped
    
    async def _process_event_group(self, key: str, events: List[ClassificationLogEvent]):
        if not events:
            return

        template_event = events[0]

        occurrences = [
            OccurrenceEntry(timestamp=event.timestamp, extra_info=event.extra_info).model_dump()
            for event in events
        ]

        update_doc = {
            "$inc": {"count": len(events)},
            "$set": {"last_seen": template_event.timestamp},
            "$push": {
                "occurrences": {
                    "$each": occurrences,
                    "$slice": -settings.MONGODB_MAX_OCCURRENCES  # Keep only last N occurrences
                }
            },
            "$setOnInsert": {
                "ground_truth": template_event.ground_truth,
                "prediction": template_event.prediction,
                "run_name": template_event.run_name,
                "model": template_event.model,
                "prompt": template_event.prompt,
                "classification_result": template_event.classification_result,
                "first_seen": template_event.timestamp
            }
        }
        
        filter_doc = {
            "ground_truth": template_event.ground_truth,
            "prediction": template_event.prediction,
            "run_name": template_event.run_name,
            "model": template_event.model,
            "prompt": template_event.prompt
        }
        
        retry_count = 0
        while retry_count < 3:
            try:
                result = await mongodb_client.collection.update_one(
                    filter_doc,
                    update_doc,
                    upsert=True
                )
                
                if result.upserted_id:
                    logger.debug("Created new document", document_id=str(result.upserted_id))
                else:
                    logger.debug("Updated existing document", matched_count=result.matched_count)
                
                break
                
            except DuplicateKeyError:
                # This can happen in rare cases with concurrent upserts
                retry_count += 1
                if retry_count < 3:
                    logger.warning(f"Duplicate key error, retrying ({retry_count}/3)")
                    await asyncio.sleep(1.0 * retry_count)
                else:
                    logger.error("Max retries exceeded for duplicate key error")
                    break
            except Exception as e:
                logger.error("Failed to update document", error=str(e), retry_count=retry_count)
                retry_count += 1
                if retry_count < 3:
                    await asyncio.sleep(1.0 * retry_count)
                else:
                    break
    
    async def _cleanup(self):
        logger.info(f"Cleaning up MongoDB writer worker {self.worker_id}")
        # Don't disconnect here - let the main app handle MongoDB connection