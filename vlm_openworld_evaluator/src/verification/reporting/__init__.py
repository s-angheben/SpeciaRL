import logging
from pathlib import Path

from .base import BaseReporter
from .breakdown_analysis import BreakdownAnalysisGenerator
from .score_analysis import ScoreAnalysisGenerator
from .passn_analysis import PassNAnalysisGenerator
from ..prompt_configs import VERIFIER_MANIFEST
from ..scoring import ScoreCalculator

logger = logging.getLogger(__name__)


class Reporter(BaseReporter):
    """Top-level entry point: builds and writes the full plain-text verification report."""

    def generate_summary_report(self, output_path: Path):
        if not self.records:
            logger.warning("No verification records to report on. Skipping summary generation.")
            return

        prompt_config = self.verifier_prompt_config

        score_calculator = ScoreCalculator(success_map=prompt_config.scoring_map)

        breakdown_generator = BreakdownAnalysisGenerator(self.records, self.ordered_report_categories)
        score_generator = ScoreAnalysisGenerator(self.records, score_calculator)
        passn_generator = PassNAnalysisGenerator(self.records, prompt_config.category_ranks, self.ordered_report_categories)

        report_content = []

        report_content.extend(self.generate_config_section())
        report_content.append("")

        report_content.append("## 2. Overall Verification Analysis")
        report_content.extend(self.generate_main_summary_section(score_generator))
        report_content.append("")

        report_content.append("## 3. Breakdown Analysis")
        report_content.extend(breakdown_generator.generate_breakdown_analysis_section())
        report_content.append("")

        passn_content = passn_generator.generate_passn_analysis_section(score_generator)
        if passn_content:
            report_content.append("## 4. PassN Analysis (Best per Sample)")
            report_content.extend(passn_content)
            report_content.append("")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w', encoding='utf-8') as f:
                f.write("\n".join(report_content))
            logger.info(f"Summary report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write summary report: {e}")


__all__ = ["Reporter"]