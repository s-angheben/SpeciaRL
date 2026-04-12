"""
MongoDB client for classification logging.
Provides async connection and operations for logging classification data.
"""

import asyncio
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class MongoDBClient:
    def __init__(self):
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
        self._connected = False
    
    async def connect(self) -> None:
        if self._connected:
            return
            
        try:
            self._client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=50,
                minPoolSize=5
            )

            await self._client.admin.command('ping')

            self._db = self._client[settings.MONGODB_DATABASE]
            self._collection = self._db[settings.MONGODB_COLLECTION]

            await self._create_indexes()
            
            self._connected = True
            logger.info("Connected to MongoDB", 
                       database=settings.MONGODB_DATABASE,
                       collection=settings.MONGODB_COLLECTION)
            
        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise
    
    async def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        try:
            # Compound unique index for upsert operations
            await self._collection.create_index([
                ("ground_truth", 1),
                ("prediction", 1),
                ("run_name", 1),
                ("model", 1),
                ("prompt", 1)
            ], unique=True, background=True)
            
            # Index for time-based queries
            await self._collection.create_index([("last_seen", -1)], background=True)
            await self._collection.create_index([("run_name", 1), ("last_seen", -1)], background=True)
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error("Failed to create MongoDB indexes", error=str(e))
            raise
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        if not self._connected or self._collection is None:
            raise RuntimeError("MongoDB client not connected")
        return self._collection

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def health_check(self) -> bool:
        try:
            if not self._connected:
                return False
            await self._client.admin.command('ping')
            return True
        except Exception:
            return False


mongodb_client = MongoDBClient()