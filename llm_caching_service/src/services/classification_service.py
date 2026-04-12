import asyncio
import traceback
from typing import Tuple, Dict, List
from src.core.clients.base import LLMClientABC
from src.core.clients.redis_client import redis_client
from src.core.prompt import PROMPT_REGISTRY
from src.schemas import ClassificationRequest
from src.utils.hashing import create_item_cache_key
from src.core.config import settings
from src.core.exceptions import CachePollTimeoutError, InvalidPromptError, LLMResponseParseError
from src.core.logging import get_logger, LogEvents
from src.services.classification_logger import classification_logger

logger = get_logger(__name__)

LOCK_EXPIRATION_SECONDS = 240
POLL_INTERVAL_SECONDS = 0.2
POLL_TIMEOUT_SECONDS = 40.0

class ClassificationService:
    async def get_classifications(
        self, 
        request_data: ClassificationRequest, 
        model_client: LLMClientABC,
        categories: List[str]
    ) -> List[str]:
        model_name = model_client.get_name()
        model_type = model_client.get_client_type()

        if len(request_data.ground_truths) != len(request_data.predictions):
            raise ValueError("ground_truths and predictions must have the same length.")

        prompt_name = request_data.verifier_prompt or settings.DEFAULT_VERIFIER_PROMPT

        if prompt_name not in PROMPT_REGISTRY:
            available_prompts = list(PROMPT_REGISTRY.keys())
            raise InvalidPromptError(prompt_name, available_prompts)

        prompt_template = PROMPT_REGISTRY[prompt_name]

        work_map = self._build_work_map(request_data, model_name, prompt_name, model_type, model_client)
        unique_cache_keys = [data['key'] for data in work_map.values()]

        cached_values = await redis_client.mget(unique_cache_keys) if unique_cache_keys else []

        results_map = {}
        pairs_to_fetch = []
        cache_hits = 0
        cache_misses = 0

        for pair, cached_value in zip(work_map.keys(), cached_values):
            cache_key = work_map[pair]['key']
            if cached_value:
                results_map[cache_key] = cached_value
                cache_hits += 1
            else:
                pairs_to_fetch.append(pair)
                cache_misses += 1

        logger.info("Cache operation completed",
                   cache_hits=cache_hits,
                   cache_misses=cache_misses,
                   total_items=len(work_map),
                   hit_ratio=cache_hits / len(work_map) if work_map else 0)

        if not pairs_to_fetch:
            final_results = self._build_final_results(request_data, work_map, results_map, categories)

            classification_tuples = [
                (pred, gt, result)
                for pred, gt, result in zip(request_data.predictions, request_data.ground_truths, final_results)
            ]
            logger.info("Classification results",
                       classification_tuples=classification_tuples)

            await self._log_classifications(request_data, final_results, model_name, prompt_name)

            return final_results

        my_work_pairs, others_work_pairs = await self._acquire_locks_for_batch(pairs_to_fetch, work_map)

        my_work_task = self._execute_my_work(my_work_pairs, work_map, model_client, prompt_template, model_name, prompt_name, categories)
        others_work_task = self._poll_for_others_work(others_work_pairs, work_map)

        task_results = await asyncio.gather(my_work_task, others_work_task, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                raise result
            results_map.update(result)

        final_results = self._build_final_results(request_data, work_map, results_map, categories)

        classification_tuples = [
            (pred, gt, result)
            for pred, gt, result in zip(request_data.predictions, request_data.ground_truths, final_results)
        ]
        logger.info("Classification results",
                   classification_tuples=classification_tuples)

        await self._log_classifications(request_data, final_results, model_name, prompt_name)

        return final_results

    async def _acquire_locks_for_batch(self, pairs_to_fetch: List[Tuple[str, str]], work_map: Dict) -> Tuple[List, List]:
        my_work = []
        others_work = []
        
        pipe = redis_client.pipeline()
        for pair in pairs_to_fetch:
            lock_key = f"lock:{work_map[pair]['key']}"
            pipe.set(lock_key, "1", nx=True, ex=LOCK_EXPIRATION_SECONDS)
        
        lock_results = await pipe.execute()

        for pair, lock_acquired in zip(pairs_to_fetch, lock_results):
            if lock_acquired:
                my_work.append(pair)
            else:
                others_work.append(pair)
        
        return my_work, others_work

    async def _execute_my_work(self, pairs: List[Tuple[str, str]], work_map: Dict, model_client: LLMClientABC, prompt_template, model_name: str, prompt_name: str, categories: List[str]) -> Dict[str, str]:
        if not pairs:
            return {}

        lock_keys = [f"lock:{work_map[pair]['key']}" for pair in pairs]
        
        try:
            batch_size = settings.MAX_BATCH_PROMPT_SIZE
            pair_chunks = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
            
            tasks = []
            for chunk in pair_chunks:
                task = model_client.get_classifications(
                    ground_truths=[gt for gt, pred in chunk],
                    predictions=[pred for gt, pred in chunk],
                    prompt_template=prompt_template,
                    categories=categories
                )
                tasks.append(task)
            
            # Apply concurrency limiting for Gemini clients to prevent 503 errors
            if model_client.get_client_type() == "gemini":
                # Use semaphore to limit concurrent Gemini API calls
                semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GEMINI)
                
                async def limited_task(task):
                    async with semaphore:
                        return await task
                
                limited_tasks = [limited_task(task) for task in tasks]
                chunked_results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            else:
                # Unlimited concurrency for vLLM clients
                chunked_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            llm_results = {}
            for i, result in enumerate(chunked_results):
                if isinstance(result, Exception):
                    chunk = pair_chunks[i]
                    logger.error("Chunk processing failed",
                               chunk_index=i,
                               chunk_size=len(chunk),
                               error_type=type(result).__name__,
                               error_message=str(result))
                    for pair in chunk:
                        raise LLMResponseParseError(f"Chunk processing failed: {str(result)}")
                else:
                    llm_results.update(result)

            logger.debug("Caching results to Redis",
                       result_count=len(llm_results))
            pipe = redis_client.pipeline()
            results = {}
            for pair in pairs:
                cache_key = work_map[pair]['key']
                if pair not in llm_results:
                    raise LLMResponseParseError(f"Missing classification result for ground_truth='{pair[0]}', prediction='{pair[1]}'")
                classification = llm_results[pair]
                results[cache_key] = classification
                pipe.set(cache_key, classification)
            
            await pipe.execute()
            success_count = len(results)
            logger.info("Results cached to Redis",
                      total_results=len(results),
                      successful_cache_operations=success_count)
            return results
        except Exception as e:
            logger.error("Fatal error in _execute_my_work",
                       exception_type=type(e).__name__,
                       exception_message=str(e),
                       traceback=traceback.format_exc(),
                       pair_count=len(pairs) if pairs else 0)
            raise
        finally:
            if lock_keys:
                await redis_client.delete(*lock_keys)
    
    async def _poll_for_others_work(self, pairs: List[Tuple[str, str]], work_map: Dict) -> Dict[str, str]:
        if not pairs:
            return {}
        keys_to_poll = {work_map[pair]['key'] for pair in pairs}
        results = {}
        end_time = asyncio.get_running_loop().time() + POLL_TIMEOUT_SECONDS
        
        while keys_to_poll and asyncio.get_running_loop().time() < end_time:
            cached_values = await redis_client.mget(list(keys_to_poll))
            found_keys = set()
            for key, value in zip(list(keys_to_poll), cached_values):
                if value is not None:
                    results[key] = value
                    found_keys.add(key)
            keys_to_poll -= found_keys
            if keys_to_poll:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if keys_to_poll:
            raise CachePollTimeoutError(f"Timed out waiting for {len(keys_to_poll)} items from other workers.")
            
        return results

    def _build_work_map(self, request_data, model_name, prompt_name, model_type, model_client) -> Dict:
        work_map = {}
        thinking_enabled = model_client.get_thinking_enabled()
        for i, (gt, pred) in enumerate(zip(request_data.ground_truths, request_data.predictions)):
            pair = (gt, pred)
            if pair not in work_map:
                work_map[pair] = {
                    'key': create_item_cache_key(gt, pred, model_name, prompt_name, model_type, thinking_enabled),
                    'indices': []
                }
            work_map[pair]['indices'].append(i)
        return work_map

    def _build_final_results(self, request_data, work_map, results_map, categories: List[str]) -> List[str]:
        final_results = [None] * len(request_data.predictions)
        for pair, data in work_map.items():
            cache_key = data['key']
            if cache_key not in results_map:
                raise LLMResponseParseError("Missing classification result from cache")
            classification = results_map[cache_key]
            for index in data['indices']:
                final_results[index] = classification
        return [self._validate_classification(val, categories) for val in final_results]

    def _validate_classification(self, classification: str, categories: List[str]) -> str:
        if classification in categories:
            return classification
        raise ValueError(f"Invalid classification '{classification}'. Valid categories: {categories}")
    
    async def _log_classifications(
        self, 
        request_data: ClassificationRequest, 
        final_results: List[str], 
        model_name: str, 
        prompt_name: str
    ) -> None:
        try:
            await classification_logger.log_classifications(
                ground_truths=request_data.ground_truths,
                predictions=request_data.predictions,
                classifications=final_results,
                run_name=request_data.run_name,
                model=model_name,
                prompt=prompt_name,
                extra_infos=request_data.extra_info
            )
        except Exception as e:
            logger.error("Failed to log classification results", error=str(e))

classification_service = ClassificationService()