import orjson
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

from ..config import VerificationConfig
from src.predictions.config import PredictionConfig
from src.dataset_creator.config import BuildConfig
from src.schemas.verification import VerificationRecord
from ..prompt_configs import VerifierPromptConfig
from .utils import generate_summary_table_from_records, get_error_types_for_report

logger = logging.getLogger(__name__)


class BaseReporter:
    """Loads verification records and renders the report's config and summary sections."""

    def __init__(
        self,
        verification_results_path: Path,
        verification_config: VerificationConfig,
        prediction_config: PredictionConfig,
        build_config: BuildConfig,
        verifier_prompt_config: VerifierPromptConfig,
        dataset_path: Optional[Path] = None,
        prediction_path: Optional[Path] = None,
        api_results_path: Optional[Path] = None,
    ):
        self.results_path = verification_results_path
        self.verification_config = verification_config
        self.prediction_config = prediction_config
        self.build_config = build_config
        self.dataset_path = dataset_path
        self.prediction_path = prediction_path
        self.api_results_path = api_results_path
        self.verifier_prompt_config = verifier_prompt_config

        ranks = self.verifier_prompt_config.category_ranks
        sorted_success_categories = sorted(
            self.verifier_prompt_config.categories,
            key=lambda cat: ranks.get(cat, 99)
        )
        
        self.ordered_report_categories = (
            sorted_success_categories + get_error_types_for_report()
        )
        
        self.records = self._load_results()
    
    def _load_results(self) -> List[VerificationRecord]:
        results = []
        if not self.results_path.exists():
            logger.error(f"Verification results file not found at {self.results_path}")
            return []
        with self.results_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = orjson.loads(line)
                    results.append(VerificationRecord(**data))
                except Exception as e:
                    logger.warning(f"Skipping malformed verification record: {e}")
        return results
    
    def generate_main_summary_table(self) -> str:
        record_dicts = [record.model_dump() for record in self.records]
        return generate_summary_table_from_records(record_dicts, self.ordered_report_categories)

    def generate_config_section(self) -> List[str]:
        content = []
        content.append("--- VLM Verification Report ---")
        content.append("## 1. Configuration Details")

        if self.dataset_path:
            content.append(f"Dataset Path: {self.dataset_path.resolve()}")
        if self.prediction_path:
            content.append(f"Prediction Path: {self.prediction_path.resolve()}")
        content.append(f"Verification Path: {self.results_path.resolve()}")
        if self.api_results_path:
            content.append(f"API Results Path: {self.api_results_path.resolve()}")
        
        content.append("### Verification Config")
        content.append("```yaml")
        content.append(yaml.dump(self.verification_config.model_dump(), indent=2, sort_keys=False))
        content.append("```")
        content.append("### Prediction Config")
        content.append("```yaml")
        content.append(yaml.dump(self.prediction_config.model_dump(), indent=2, sort_keys=False))
        content.append("```")
        content.append("### Dataset Build Config")
        content.append("```yaml")
        content.append(yaml.dump(self.build_config.model_dump(), indent=2, sort_keys=False))
        content.append("```")
        return content
    
    def generate_main_summary_section(self, score_generator) -> List[str]:
        content = []
        content.append("### 2.1. Overall Performance Summary")
        main_table = self.generate_main_summary_table()
        content.append(main_table)
        content.append("")

        content.append("### 2.2. Overall Score Summary")
        score_table = score_generator.generate_score_summary_table(self.records)
        content.append(score_table)
        
        return content