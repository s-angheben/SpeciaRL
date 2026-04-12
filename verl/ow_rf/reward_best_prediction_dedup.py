import os
import traceback
import regex as re
from typing import List, Dict, Any, Optional
from enum import Enum
import requests
import orjson
from pydantic import BaseModel
from datetime import datetime
import uuid
import sys
import hashlib
import time

API_BASE_URL = os.getenv("VERIFICATION_API_BASE_URL", "http://localhost:8000/api/v1/")
if not API_BASE_URL:
    raise ValueError("VERIFICATION_API_BASE_URL cannot be empty.")

API_TIMEOUT = int(os.getenv("VERIFICATION_API_TIMEOUT", "7200"))
VERBOSE = os.getenv("VERBOSE", "true").lower() in ("true", "1", "yes")

VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", None)
VERIFICATION_PROMPT = os.getenv("VERIFICATION_PROMPT", None)
VERIFICATION_RUN_NAME = os.getenv("VERIFICATION_RUN_NAME", None)

REWARD_MODE = os.getenv("REWARD_MODE", "single")

REWARD_LOG_FILE = os.getenv("REWARD_LOG_FILE", None)
reward_log_file_handle = None

batch_step_counter = 0

if REWARD_LOG_FILE:
    try:
        reward_log_file_handle = open(REWARD_LOG_FILE, 'a', buffering=1)
    except Exception as e:
        print(f"[WARNING] Could not open reward log file {REWARD_LOG_FILE}: {e}")

if REWARD_MODE not in ["single", "all"]:
    raise ValueError(f"Invalid REWARD_MODE: '{REWARD_MODE}'. Must be 'single' or 'all'")

def log_print(message, force_flush=True):
    """Print to both stdout and log file if specified"""
    print(message)
    if reward_log_file_handle:
        reward_log_file_handle.write(f"{message}\n")
        if force_flush:
            reward_log_file_handle.flush()
    if force_flush:
        sys.stdout.flush()

class VerificationErrorType(str, Enum):
    PARSING_VLM_OUTPUT = "parsing_vlm_output"
    API_NETWORK_ERROR = "api_network_error"
    API_RESPONSE_ERROR = "api_response_error"
    API_UNKNOWN_ERROR = "api_unknown_error"

class ClassificationCategory(str, Enum):
    WRONG = "wrong"
    ABSTAIN = "abstain"
    GENERIC = "generic"
    LESS_SPECIFIC = "less specific"
    SPECIFIC = "specific"
    MORE_SPECIFIC = "more specific"

class ClassificationRequest(BaseModel):
    ground_truths: List[str]
    predictions: List[str]
    model: Optional[str] = None
    verifier_prompt: Optional[str] = None
    run_name: Optional[str] = None
    extra_info: Optional[List[Optional[Dict[str, Any]]]] = None

class ClassificationResponse(BaseModel):
    classifications: List[ClassificationCategory]

class VerificationAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def verify_batch(self, endpoint: str, ground_truths: List[str], predictions: List[str],
                     model: Optional[str] = None, verifier_prompt: Optional[str] = None,
                     run_name: Optional[str] = None, extra_info: Optional[List[Optional[Dict[str, Any]]]] = None,
                     timeout: Optional[float] = 120.0, max_retries: int = 10, retry_delay: float = 10.0) -> List[ClassificationCategory]:
        request_data = ClassificationRequest(
            ground_truths=ground_truths, predictions=predictions, model=model,
            verifier_prompt=verifier_prompt, run_name=run_name, extra_info=extra_info
        )

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(f"{self.base_url}{endpoint}", json=request_data.model_dump(), timeout=timeout)
                response.raise_for_status()
                return ClassificationResponse(**orjson.loads(response.content)).classifications
            except requests.RequestException as e:
                last_exception = e
                if attempt < max_retries:
                    log_print(f"[RETRY] API request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    log_print(f"[RETRY] Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    log_print(f"[RETRY] All {max_retries + 1} attempts failed. Giving up.")
                    raise
            except Exception as e:
                raise ValueError(f"Invalid API response: {e}") from e

        if last_exception:
            raise last_exception

def normalize_text(text: str) -> str:
    """Normalize text by removing punctuation (keep letters, digits, dashes, whitespace),
    convert to lowercase, collapse whitespace."""
    if not text:
        return ""
    normalized = re.sub(r'[^\p{L}\p{N}\p{Pd}\s]', ' ', str(text))
    normalized = normalized.lower()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()

def extract_answer_from_tag(output: str, tag: str) -> str:
    if not isinstance(output, str) or not tag: return ""
    try:
        match = re.search(f"<{tag}>(.*?)</{tag}>", output, re.DOTALL)
        return match.group(1).strip() if match else ""
    except Exception:
        return ""

def truncate(text: str, max_len: int = 60) -> str:
    return (text[:max_len-3] + "...") if len(text) > max_len else text

def compute_sample_hash(data_source: str, index_orig: int, source_split: str, source_index: int) -> str:
    """Compute a deterministic hash from sample identification fields"""
    hash_input = f"{data_source}|{index_orig}|{source_split}|{source_index}"
    return hashlib.sha256(hash_input.encode()).hexdigest()

def get_classification_rank(classification: str) -> int:
    """Get rank for classification category. Lower rank = better.
    MORE_SPECIFIC and SPECIFIC are tied at rank 1."""
    rank_map = {
        "more specific": 1,
        "specific": 1,
        "less specific": 2,
        "generic": 3,
        "abstain": 4,
        "wrong": 5,
    }
    return rank_map.get(classification.lower(), 99)

def group_by_hash_and_score(parsed_items: List[Dict[str, Any]]) -> None:
    """Group items by sample_hash and assign scores based on best prediction per hash.

    Scoring rules:
    - PARSING_FAILED: always -1.0
    - REWARD_MODE='single': Best response (first with best rank) gets 1.0, others get 0.0
    - REWARD_MODE='all': All responses with best rank get 1.0, others get 0.0
    - Special case: if best is WRONG (rank 5), all valid responses get 0.0
    - Tie-breaking (single mode only): first occurrence wins

    Modifies parsed_items in place.
    """
    hash_groups = {}
    for item in parsed_items:
        sample_hash = item.get("sample_hash", "N/A")
        if sample_hash not in hash_groups:
            hash_groups[sample_hash] = []
        hash_groups[sample_hash].append(item)

    for sample_hash, items_in_group in hash_groups.items():
        parsing_failed_items = [item for item in items_in_group if item["status"] == "PARSING_FAILED"]
        error_items = [item for item in items_in_group
                       if item["status"] not in ("PARSING_FAILED", "VERIFIED")]
        valid_items = [item for item in items_in_group if item["status"] == "VERIFIED"]

        for item in parsing_failed_items:
            item["score"] = -1.0
            item["is_best"] = False

        for item in error_items:
            item["score"] = 0.0
            item["is_best"] = False

        if valid_items:
            best_rank = min(get_classification_rank(item["verifier_response"]) for item in valid_items)

            if best_rank == 5:
                for item in valid_items:
                    item["score"] = 0.0
                    item["is_best"] = False
            else:
                if REWARD_MODE == "single":
                    best_assigned = False
                    for item in valid_items:
                        item_rank = get_classification_rank(item["verifier_response"])
                        if item_rank == best_rank and not best_assigned:
                            item["score"] = 1.0
                            item["is_best"] = True
                            best_assigned = True
                        else:
                            item["score"] = 0.0
                            item["is_best"] = False
                else:
                    for item in valid_items:
                        item_rank = get_classification_rank(item["verifier_response"])
                        if item_rank == best_rank:
                            item["score"] = 1.0
                            item["is_best"] = True
                        else:
                            item["score"] = 0.0
                            item["is_best"] = False

def print_verification_table(parsed_items: List[Dict[str, Any]]) -> None:
    """Print verification results in a compact tabular format"""
    if not parsed_items or not VERBOSE:
        return

    log_print(f"\n{'='*220}")
    log_print(f"{'STEP':>4} | {'IDX':>3} | {'SCORE':>5} | {'VERIFIER':^16} | {'IS_BEST':^8} | {'GROUND TRUTH':<40} | {'PREDICTION':<40} | {'SAMPLE HASH':<64}")
    log_print(f"{'='*220}")

    for item in parsed_items:
        batch_step = item.get('batch_step', 'N/A')
        idx = item["original_index"]
        score = item["score"]
        verifier = item["verifier_response"]
        is_best = "[BEST]" if item.get("is_best", False) else ""
        gt_trunc = truncate(item['ground_truth'], 40)
        pred_trunc = truncate(item['parsed_answer'], 40) if item['status'] != 'PARSING_FAILED' else "[PARSING FAILED]"
        sample_hash = item.get('sample_hash', 'N/A')

        log_print(f"{batch_step:>4} | {idx:>3} | {score:>5.2f} | {verifier:^16} | {is_best:^8} | {gt_trunc:<40} | {pred_trunc:<40} | {sample_hash:<64}")

    log_print(f"{'='*220}\n")

def get_category_counts(parsed_items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count occurrences of each verification category and error type"""
    counts = {}

    for item in parsed_items:
        response = item["verifier_response"]
        counts[response] = counts.get(response, 0) + 1

    return counts

try:
    api_client = VerificationAPIClient(base_url=API_BASE_URL)
    log_print(f"[INFO] Reward function (best prediction dedup - mode: {REWARD_MODE}) initialized. API: {API_BASE_URL}")
    log_print(f"[INFO] API timeout: {API_TIMEOUT}s for all requests.")
    if REWARD_LOG_FILE:
        log_print(f"[INFO] Reward logging to file: {REWARD_LOG_FILE}")
except (ValueError, KeyError) as e:
    log_print(f"[CRITICAL] Failed to initialize reward function components: {e}")
    raise

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]]
) -> List[float]:
    global batch_step_counter
    batch_step_counter += 1
    batch_step = batch_step_counter

    start_time = datetime.now()
    batch_size = len(solution_strs)
    batch_id = str(uuid.uuid4())[:8]

    log_print(f"\n{'#'*80}")
    log_print(f"# BATCH {batch_id} | STEP {batch_step} | {start_time.strftime('%Y-%m-%d %H:%M:%S')} | {batch_size} items")
    log_print(f"{'#'*80}")

    if not solution_strs:
        log_print("[INFO] Empty batch received. Nothing to do.")
        log_print(f"{'#'*80}\n")
        return []

    parsed_items = []
    parsing_failures = 0
    for i, (solution, gt, extra_info) in enumerate(zip(solution_strs, ground_truths, extra_infos)):
        parsed_answer = extract_answer_from_tag(solution, "answer")

        sample_hash = "N/A"
        if extra_info:
            try:
                data_source = extra_info.get("data_source", "")
                index_orig = extra_info.get("index_orig", 0)
                source_split = extra_info.get("source_split", "")
                source_index = extra_info.get("source_index", 0)
                sample_hash = compute_sample_hash(data_source, index_orig, source_split, source_index)
            except Exception as e:
                log_print(f"[WARNING] Failed to compute hash for item {i}: {e}")

        item_info = {
            "batch_step": batch_step, "original_index": i, "ground_truth": gt, "parsed_answer": parsed_answer,
            "status": "", "verifier_response": "N/A", "score": 0.0, "sample_hash": sample_hash, "is_best": False
        }
        if parsed_answer:
            item_info["status"] = "PARSED"
        else:
            item_info["status"] = "PARSING_FAILED"
            item_info["score"] = -1.0
            item_info["verifier_response"] = "PARSING_FAILED"
            parsing_failures += 1
        parsed_items.append(item_info)

    parsing_successes = batch_size - parsing_failures
    log_print(f"[PARSING] Summary: {parsing_successes} successfully parsed, {parsing_failures} failed.")

    api_batch_items = [item for item in parsed_items if item["status"] == "PARSED"]

    if api_batch_items:
        num_to_verify = len(api_batch_items)
        timeout = float(API_TIMEOUT)
        log_print(f"[API] Verifying {num_to_verify} items...")
        try:
            gt_api = [normalize_text(item["ground_truth"]) for item in api_batch_items]
            pred_api = [normalize_text(item["parsed_answer"]) for item in api_batch_items]
            classifications = api_client.verify_batch("/classify/standard", gt_api, pred_api, VERIFICATION_MODEL, VERIFICATION_PROMPT, VERIFICATION_RUN_NAME, timeout=timeout)
            for i, classification in enumerate(classifications):
                api_batch_items[i].update({
                    "status": "VERIFIED", "verifier_response": classification.value
                })
        except requests.RequestException as e:
            log_print(f"[ERROR] API network error: {e}")
            for item in api_batch_items:
                item.update({"status": "API_NETWORK_ERROR", "verifier_response": "API_NETWORK_ERROR", "score": 0.0})
        except ValueError as e:
            log_print(f"[ERROR] API response error: {e}")
            for item in api_batch_items:
                item.update({"status": "API_RESPONSE_ERROR", "verifier_response": "API_RESPONSE_ERROR", "score": 0.0})
        except Exception as e:
            log_print(f"[ERROR] API unexpected error: {e}")
            if reward_log_file_handle:
                traceback.print_exc(file=reward_log_file_handle)
                reward_log_file_handle.flush()
            traceback.print_exc()
            for item in api_batch_items:
                item.update({"status": "API_UNKNOWN_ERROR", "verifier_response": "API_UNKNOWN_ERROR", "score": 0.0})
    else:
        log_print("[API] No items to send to verification API. Skipping.")

    group_by_hash_and_score(parsed_items)

    final_scores = [0.0] * batch_size
    total_score = 0
    for item in parsed_items:
        idx = item["original_index"]
        final_scores[idx] = item["score"]
        total_score += item["score"]

    print_verification_table(parsed_items)

    average_score = total_score / batch_size if batch_size else 0
    category_counts = get_category_counts(parsed_items)

    best_count = sum(1 for item in parsed_items if item.get("is_best", False))

    log_print(f"[BATCHSTEP] Step {batch_step}")
    log_print(f"[SUMMARY] Batch {batch_id} Results:")
    log_print(f"[SUMMARY] Total Items: {len(parsed_items)} | Average Score: {average_score:.4f} | Best Predictions: {best_count}")
    log_print(f"[SUMMARY] Category Breakdown:")

    verification_categories = []
    error_types = []

    for category, count in sorted(category_counts.items()):
        if category in ["more specific", "specific", "less specific", "generic", "abstain", "wrong"]:
            verification_categories.append(f"{category.upper()}: {count}")
        else:
            error_types.append(f"{category}: {count}")

    if verification_categories:
        log_print(f"[SUMMARY]   Verifications: {', '.join(verification_categories)}")
    if error_types:
        log_print(f"[SUMMARY]   Errors: {', '.join(error_types)}")

    end_time = datetime.now()
    log_print(f"[INFO] Batch {batch_id} completed. Duration: {(end_time - start_time).total_seconds():.2f}s")
    log_print(f"{'#'*80}\n")

    return final_scores
