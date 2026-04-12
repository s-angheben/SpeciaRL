"""
MongoDB worker manager for integrated classification logging.
Manages worker tasks that run within the main application process.
"""

import asyncio
from typing import List
from src.core.config import settings
from src.core.logging import get_logger
from src.workers.mongodb_writer import MongoDBWriter

logger = get_logger(__name__)


class MongoDBWorkerManager:
    """Manages MongoDB writer workers as async tasks within the main application."""
    
    def __init__(self):
        self.worker_tasks: List[asyncio.Task] = []
        self.workers: List[MongoDBWriter] = []
        self._running = False
    
    async def start_workers(self):
        """Start MongoDB writer workers as async tasks."""
        if not settings.MONGODB_ENABLED:
            logger.info("MongoDB logging disabled, no workers started")
            return
        
        logger.info(f"Starting {settings.MONGODB_WORKER_COUNT} MongoDB workers")
        
        for i in range(settings.MONGODB_WORKER_COUNT):
            worker = MongoDBWriter(worker_id=i)
            self.workers.append(worker)
            
            task = asyncio.create_task(worker.start())
            self.worker_tasks.append(task)
        
        self._running = True
        logger.info(f"Started {len(self.worker_tasks)} MongoDB workers")
    
    async def stop_workers(self):
        """Stop all MongoDB writer workers gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping MongoDB workers")
        
        # Signal workers to stop
        for worker in self.workers:
            await worker.stop()
        
        # Cancel all tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        self.workers.clear()
        self._running = False
        
        logger.info("MongoDB workers stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if workers are running."""
        return self._running
    
    def get_worker_count(self) -> int:
        """Get the number of active workers."""
        return len(self.worker_tasks)


# Global worker manager instance
mongodb_worker_manager = MongoDBWorkerManager()