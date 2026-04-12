# src/verification/results_saver.py
import time
import logging
import orjson
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urljoin

if TYPE_CHECKING:
    from .api_client import VerificationAPIClient

logger = logging.getLogger(__name__)


class VerifierResultsSaver:
    """Handles waiting for API completion and downloading all verification results."""
    
    def __init__(self, api_client: "VerificationAPIClient", results_dir: Path, experiment_group: str, experiment_name: str):
        self.api_client = api_client
        self.results_dir = results_dir / experiment_group / experiment_name
    
    def wait_and_save_results(self, verification_hash: str) -> Path:
        """Main method: wait for API completion and save all results.
        
        Returns:
            Path: Path to the saved API results file
        """
        logger.info(f"Starting results saving process for verification_hash: {verification_hash}")
        
        self._wait_for_completion(verification_hash)
        results_path = self._download_and_save_results(verification_hash)
        
        logger.info(f"Results saving process completed for verification_hash: {verification_hash}")
        return results_path
    
    def _wait_for_completion(self, verification_hash: str):
        """
        Poll API status until processing is completed and the pending queue is empty.
        Includes a final grace period for the last DB write to commit.
        """
        logger.info(f"Waiting for API processing completion for verification_hash: {verification_hash}")
        
        # Set a timeout to prevent waiting forever (e.g., 10 minutes)
        timeout_seconds = 600 
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                status_data = self.api_client.get_run_status(verification_hash)
                current_status = status_data.get('status', 'unknown')
                pending_events = status_data.get('queue_pending_count', 0)
                processed_events = status_data.get('total_processed', 0)

                # The run is truly complete only when the status is 'completed'
                # AND the pending event queue for that run is empty.
                if current_status == 'completed' and pending_events == 0:
                    logger.info(f"API status is 'completed' with 0 pending events. Processed: {processed_events}.")
                    # Add a short, final wait to ensure the last worker has finished its async DB write.
                    grace_period_seconds = 2
                    logger.info(f"Adding a {grace_period_seconds}s grace period for final DB writes to commit...")
                    time.sleep(grace_period_seconds)
                    logger.info("Grace period finished. Ready to download results.")
                    return # Exit the function successfully

                elif current_status == 'idle' and processed_events > 0:
                    # This is another valid completion state
                    logger.info(f"API status is 'idle' but {processed_events} events were processed. Assuming completion.")
                    return

                else:
                    logger.info(f"API status: {current_status}, processed: {processed_events}, pending: {pending_events}. Waiting...")
                    time.sleep(5) # Poll every 5 seconds
                
            except Exception as e:
                logger.warning(f"Failed to check API status: {e}. Retrying in 10s...")
                time.sleep(10)

        # If the loop finishes without returning, it has timed out.
        logger.error(f"Timed out after {timeout_seconds}s waiting for verification run to complete.")
        raise TimeoutError(f"Verification run '{verification_hash}' did not complete within the timeout period.")
    
    def _download_and_save_results(self, verification_hash: str) -> Path:
        """Download all results and save to JSON file.
        
        Returns:
            Path: Path to the saved API results file
        """
        logger.info(f"Downloading all results for verification_hash: {verification_hash}")
        
        try:
            all_results_data = self.get_all_run_results(verification_hash)
            
            # Ensure directory exists (same as BaseReporter)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            results_path = self.results_dir / f"api_results_{verification_hash}.json"
            
            # Save complete results as JSON
            with results_path.open('w', encoding='utf-8') as f:
                f.write(orjson.dumps(all_results_data, option=orjson.OPT_INDENT_2).decode('utf-8'))
            
            total_results = len(all_results_data.get('results', []))
            logger.info(f"Successfully saved {total_results} API results to {results_path}")
            
            return results_path
            
        except Exception as e:
            logger.error(f"Failed to download and save results: {e}")
            raise
    
    def get_all_run_results(self, verification_hash: str) -> dict:
        """Get ALL classification results for a run with automatic pagination."""
        logger.info(f"Fetching all results for verification_hash: {verification_hash}")
        
        # Get first batch to determine total
        first_response = self._get_batch(verification_hash, limit=50000, skip=0)
        total_count = first_response["total_count"]
        all_results = first_response["results"]
        
        logger.info(f"Total results to fetch: {total_count}, first batch: {len(all_results)}")

        # Calculate remaining batches needed
        limit = 50000
        remaining = total_count - len(all_results)

        # Fetch remaining batches
        for skip in range(limit, total_count, limit):
            batch_size = min(limit, remaining)
            logger.info(f"Fetching batch: skip={skip}, limit={batch_size}")
            
            batch = self._get_batch(verification_hash, limit=batch_size, skip=skip)
            all_results.extend(batch["results"])
            remaining -= len(batch["results"])
            
            logger.info(f"Progress: {len(all_results)}/{total_count} results fetched")

        logger.info(f"Successfully fetched all {len(all_results)} results")
        
        return {
            "results": all_results,
            "total_count": len(all_results),
            "stats": first_response.get("stats"),
            "verification_hash": verification_hash
        }

    def _get_batch(self, verification_hash: str, limit: int, skip: int) -> dict:
        """Get a single batch of results."""
        params = {"limit": limit, "offset": skip, "include_stats": True}
        
        endpoint_path = f"runs/{verification_hash}/results"
        api_url = urljoin(self.api_client.base_url, endpoint_path)
        
        response = self.api_client.session.get(
            api_url, 
            params=params, 
            timeout=self.api_client.timeout
        )
        response.raise_for_status()
        return orjson.loads(response.content)