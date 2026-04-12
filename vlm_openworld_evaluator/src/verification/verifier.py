import orjson
import logging
import hashlib
from pathlib import Path
from typing import List, Set, Tuple, Union, Optional
from collections import defaultdict
import requests
from tqdm import tqdm

from .config import VerificationConfig
from .api_client import VerificationAPIClient
from .classifications import VerificationStatus, VerificationErrorType
from .parsing import normalize_text, extract_answer
from .utils import batch_list, ensure_output_directory
from .prompt_configs import VERIFIER_MANIFEST, VerifierPromptConfig
from .api_schemas import StandardClassificationResponse
from src.schemas.prediction_record import PredictionRecord
from src.schemas.verification import VerificationRecord, VerificationConfigSnapshot, PreparedPrediction, StandardClassificationCategory

logger = logging.getLogger(__name__)

class Verifier:
    """Verifies VLM predictions against ground truth via a remote API service."""

    def __init__(self, config: VerificationConfig, predictions_path: Path, verification_hash: str, prediction_config=None, experiment_name: str = None, experiment_group: str = "no_group"):
        self.config = config
        self.predictions_path = predictions_path
        self.verification_hash = verification_hash
        self.prediction_config = prediction_config
        self.experiment_name = experiment_name
        self.experiment_group = experiment_group
        self.output_dir = ensure_output_directory(Path(config.output_dir) / "verification")
        self.api_client = VerificationAPIClient(config.api_base_url, config.api_timeout)

        verifier_prompt_name = self.config.verifier_prompt
        if verifier_prompt_name not in VERIFIER_MANIFEST:
            raise ValueError(
                f"Verifier prompt '{verifier_prompt_name}' not found in VERIFIER_MANIFEST. "
                f"Available prompts: {list(VERIFIER_MANIFEST.keys())}"
            )
        self.prompt_config: VerifierPromptConfig = VERIFIER_MANIFEST[verifier_prompt_name]
        
        self.config_snapshot = VerificationConfigSnapshot(
            model=self.config.model,
            verifier_prompt=self.config.verifier_prompt,
            api_base_url=self.config.api_base_url
        )

    def run(self) -> Tuple[Path, Optional[Path]]:
        verification_path = self.output_dir / f"verification_{self.verification_hash}.ndjson"

        all_predictions = self._load_predictions(self.predictions_path)
        self._validate_prediction_counts(all_predictions)

        verified_ids = self._get_verified_ids(verification_path) if self.config.resume else set()
        predictions_to_process = [p for p in all_predictions if self._create_prediction_id(p) not in verified_ids]
        
        if not predictions_to_process:
            logger.info("All predictions have already been verified.")
            return verification_path, None
            
        logger.info(f"Verifying {len(predictions_to_process)} new predictions.")
        
        prepared, failed = self._prepare_predictions(predictions_to_process)
        
        if failed:
            with verification_path.open('a', encoding='utf-8') as f:
                for record in tqdm(failed, desc="Saving preparation failures"):
                    f.write(record.model_dump_json() + '\n')
        
        self._execute_api_verification(prepared, verification_path)

        logger.info(f"Verification complete. Results saved to {verification_path}")

        api_results_path = None
        return verification_path, api_results_path

    def _prepare_predictions(self, predictions: List[PredictionRecord]) -> Tuple[List[PreparedPrediction], List[VerificationRecord]]:
        prepared_for_api = []
        failed_records = []

        for pred_record in tqdm(predictions, desc="Preparing predictions"):
            base_fail_record = {
                "prediction_id": self._create_prediction_id(pred_record),
                "original_record": pred_record,
            }
            
            ground_truth = pred_record.get_ground_truth()
            if ground_truth is None:
                error_rec = self._create_failure_record(base_fail_record, VerificationErrorType.PARSING_GROUND_TRUTH, "Ground truth not found in data_context")
                failed_records.append(error_rec)
                continue

            answer_format = pred_record.get_answer_format()
            if answer_format is None:
                error_rec = self._create_failure_record(base_fail_record, VerificationErrorType.PARSING_ANSWER_FORMAT, "Answer format not found in data_context")
                failed_records.append(error_rec)
                continue
            
            parsed_prediction = extract_answer(pred_record.vlm_prediction, answer_format)
            if not parsed_prediction:
                error_rec = self._create_failure_record(base_fail_record, VerificationErrorType.PARSING_VLM_OUTPUT, "Could not extract answer from VLM output")
                failed_records.append(error_rec)
                continue
            
            base_fail_record["parsed_prediction"] = parsed_prediction

            if self.config.max_label_words and len(parsed_prediction.split()) > self.config.max_label_words:
                detail = f"Prediction too long: {len(parsed_prediction.split())} words (max: {self.config.max_label_words})"
                error_rec = self._create_failure_record(base_fail_record, VerificationErrorType.PREDICTION_TOO_LONG, detail)
                failed_records.append(error_rec)
                continue
            
            prepared_for_api.append(PreparedPrediction(
                **base_fail_record,
                normalized_prediction=normalize_text(parsed_prediction),
                normalized_ground_truth=normalize_text(ground_truth),
            ))
            
        return prepared_for_api, failed_records
        
    def _execute_api_verification(self, prepared_batch: List[PreparedPrediction], verification_path: Path) -> None:
        batch_size = self.config.batch_size
        
        for batch in tqdm(batch_list(prepared_batch, batch_size), desc="Verifying with API", total=(len(prepared_batch) + batch_size - 1) // batch_size):
            batch_results = []
            
            try:
                ground_truths = [item.normalized_ground_truth for item in batch]
                predictions = [item.normalized_prediction for item in batch]
                extra_info = [
                    {
                        "sample_id": item.original_record.sample_id,
                        "prediction_index": item.original_record.prediction_index,
                        "data_context": item.original_record.data_context.model_dump()
                    }
                    for item in batch
                ]
                
                ResponseModel = self.prompt_config.response_model
                
                api_response_dict = self.api_client.verify_batch(
                    endpoint=self.prompt_config.endpoint,
                    ground_truths=ground_truths,
                    predictions=predictions,
                    model=self.config.model,
                    verifier_prompt=self.config.verifier_prompt,
                    run_name=self.verification_hash,
                    extra_info=extra_info
                )

                parsed_response = ResponseModel(**api_response_dict)
                if len(parsed_response.classifications) != len(batch):
                    raise ValueError(
                        "API response size mismatch: "
                        f"expected {len(batch)} classifications, got {len(parsed_response.classifications)}"
                    )
                for item, classification_enum in zip(batch, parsed_response.classifications):
                    record = self._create_success_record(item, classification_enum)
                    batch_results.append(record)

            except requests.RequestException as e:
                logger.error(f"API network error on batch: {e}. Marking batch as failed.")
                for item in batch:
                    batch_results.append(self._create_failure_record(item, VerificationErrorType.API_NETWORK_ERROR, f"Network Error: {e}", VerificationStatus.API_FAILURE))
                    exit(1)
            except ValueError as e:
                logger.error(f"API response error on batch: {e}. Marking batch as failed.")
                for item in batch:
                    batch_results.append(self._create_failure_record(item, VerificationErrorType.API_RESPONSE_ERROR, f"Response Error: {e}", VerificationStatus.API_FAILURE))
            except Exception as e:
                logger.error(f"Unknown error during API call: {e}. Marking batch as failed.")
                for item in batch:
                    batch_results.append(self._create_failure_record(item, VerificationErrorType.API_UNKNOWN_ERROR, f"Unknown Error: {e}", VerificationStatus.API_FAILURE))

            with verification_path.open('a', encoding='utf-8') as f:
                for record in batch_results:
                    f.write(record.model_dump_json() + '\n')
                f.flush()

    def _create_success_record(self, item: PreparedPrediction, classification) -> VerificationRecord:
        orig = item.original_record
        return VerificationRecord(
            prediction_id=item.prediction_id, sample_id=orig.sample_id, prediction_group_id=orig.prediction_group_id,
            prediction_index=orig.prediction_index, model_name=orig.model_name, vlm_prediction=orig.vlm_prediction,
            parsed_prediction=item.parsed_prediction, normalized_prediction=item.normalized_prediction,
            normalized_ground_truth=item.normalized_ground_truth,
            status=VerificationStatus.SUCCESS,
            classification=classification,
            verification_config=self.config_snapshot, data_context=orig.data_context
        )

    def _create_failure_record(self, item: Union[dict, PreparedPrediction], error_type: VerificationErrorType, detail: str, status: VerificationStatus = VerificationStatus.PREPARATION_FAILURE) -> VerificationRecord:
        orig = item.original_record if isinstance(item, PreparedPrediction) else item['original_record']
        parsed = (item.parsed_prediction if isinstance(item, PreparedPrediction)
                  else item.get('parsed_prediction', ''))
        
        if isinstance(item, PreparedPrediction):
            normalized_gt = item.normalized_ground_truth
        else:
            gt = orig.get_ground_truth()
            if gt is None:
                normalized_gt = ""
            else:
                normalized_gt = normalize_text(gt)

        return VerificationRecord(
            prediction_id=item.prediction_id if isinstance(item, PreparedPrediction) else item['prediction_id'],
            sample_id=orig.sample_id, prediction_group_id=orig.prediction_group_id,
            prediction_index=orig.prediction_index, model_name=orig.model_name, vlm_prediction=orig.vlm_prediction,
            parsed_prediction=parsed, normalized_prediction=normalize_text(parsed),
            normalized_ground_truth=normalized_gt,
            status=status,
            error_type=error_type,
            error_detail=detail,
            verification_config=self.config_snapshot, data_context=orig.data_context
        )

    def _validate_prediction_counts(self, predictions: List[PredictionRecord]) -> None:
        """Ensure each sample has exactly num_predictions_per_sample entries (exploration mode only)."""
        if not self.prediction_config or self.prediction_config.num_predictions_per_sample <= 1:
            return
        
        expected_count = self.prediction_config.num_predictions_per_sample
        samples = defaultdict(list)
        
        for pred in predictions:
            samples[pred.sample_id].append(pred)
        
        for sample_id, preds in samples.items():
            if len(preds) != expected_count:
                raise ValueError(f"Sample {sample_id} has {len(preds)} predictions, expected {expected_count}")
        
        logger.info(f"Validation passed: All {len(samples)} samples have exactly {expected_count} predictions each")

    def _load_predictions(self, predictions_path: Path) -> List[PredictionRecord]:
        predictions = []
        with predictions_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = orjson.loads(line.strip())
                    predictions.append(PredictionRecord(**data))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping malformed prediction record: {e}")
        logger.info(f"Loaded {len(predictions)} predictions from {predictions_path}")
        return predictions

    def _get_verified_ids(self, path: Path) -> Set[str]:
        verified_ids = set()
        if path.exists():
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = orjson.loads(line.strip())
                        verified_ids.add(record.get('prediction_id'))
                    except (ValueError, KeyError):
                        continue
        return verified_ids

    @staticmethod
    def _create_prediction_id(prediction_record: PredictionRecord) -> str:
        id_string = (
            f"{prediction_record.sample_id}"
            f"{prediction_record.prediction_group_id}"
            f"{prediction_record.prediction_index}"
        )
        return hashlib.sha256(id_string.encode('utf-8')).hexdigest()
