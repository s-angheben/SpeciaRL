import requests
import orjson
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class VerificationAPIClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/') + '/'
        self.timeout = timeout
        self.session = requests.Session()

    def verify_batch(
        self,
        endpoint: str,
        ground_truths: List[str],
        predictions: List[str],
        model: Optional[str] = None,
        verifier_prompt: Optional[str] = None,
        run_name: Optional[str] = None,
        extra_info: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        if len(ground_truths) != len(predictions):
            raise ValueError("ground_truths and predictions must have the same length")

        request_data = {
            "ground_truths": ground_truths,
            "predictions": predictions,
            "model": model,
            "verifier_prompt": verifier_prompt,
            "run_name": run_name,
            "extra_info": extra_info,
        }

        try:
            api_url = urljoin(self.base_url, endpoint)
            response = self.session.post(
                api_url,
                json=request_data,
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"Response content: {response.text}")

            response.raise_for_status()
            return orjson.loads(response.content)

        except requests.RequestException as e:
            logger.error(f"API request to {api_url} failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse API response from {api_url}: {e}")
            raise ValueError(f"Invalid API response: {e}")

    def get_run_status(self, run_name: str) -> dict:
        try:
            endpoint = f"runs/{run_name}/status"
            api_url = urljoin(self.base_url, endpoint)
            response = self.session.get(api_url, timeout=self.timeout)
            response.raise_for_status()
            return orjson.loads(response.content)
        except requests.RequestException as e:
            logger.error(f"Failed to get run status for {run_name}: {e}")
            raise

    def health_check(self) -> bool:
        try:
            api_url = urljoin(self.base_url, "health")
            response = self.session.get(api_url, timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False