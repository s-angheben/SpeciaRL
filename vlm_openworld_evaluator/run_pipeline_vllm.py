#!/usr/bin/env python3
"""
Unified pipeline runner for VLM evaluation using vLLM.

Orchestrates dataset creation, prediction, and verification stages from a single
configuration file with hash-based artifact caching.

Usage:
    python run_pipeline_vllm.py --config experiment.yml
    python run_pipeline_vllm.py --config experiment.yml --stages dataset predict
    python run_pipeline_vllm.py --config experiment.yml --stages verify
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.unified_config import UnifiedConfig
import yaml
from src.dataset_creator.builder import DatasetBuilder
from src.dataset_creator.prompts import PromptManager
from src.predictions.predictor_vllm import VllmPredictor as Predictor
from src.verification.verifier import Verifier
from src.verification.reporting import Reporter
from src.verification import ensure_output_directory
from src.utils.hash_utils import generate_config_hash, calculate_dataset_hash, calculate_prediction_hash, calculate_verification_hash
from src.utils.seeding import set_seed
from src.utils.logging_utils import setup_logging
from src.utils.path_utils import extract_dataset_hash, extract_predictions_hash, validate_file_exists

logger = logging.getLogger(__name__)


def run_dataset_stage(build_config, force_rebuild: bool = False) -> Path:
    logger.info("=== DATASET CREATION STAGE ===")

    try:
        dataset_hash = calculate_dataset_hash(build_config)
        dataset_output_dir = Path(build_config.output_dir) / "datasets"
        dataset_path = dataset_output_dir / f"dataset_{dataset_hash}.parquet"

        if dataset_path.exists() and not force_rebuild:
            logger.info(f"Dataset already exists at {dataset_path}, skipping creation")
            return dataset_path

        prompt_manager = PromptManager(global_prompts_path=build_config.global_prompts_path)
        builder = DatasetBuilder(config=build_config, prompt_manager=prompt_manager, force=force_rebuild)
        builder.build()

        logger.info(f"Dataset created successfully at {dataset_path}")
        return dataset_path

    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise


def run_prediction_stage(prediction_config, dataset_path: Path, force_rebuild: bool = False) -> Path:
    logger.info("=== PREDICTION STAGE (vLLM) ===")

    try:
        validate_file_exists(dataset_path, "Dataset")
        dataset_hash = extract_dataset_hash(dataset_path)

        is_exploration = prediction_config.num_predictions_per_sample > 1
        output_subdir = "explorations" if is_exploration else "predictions"
        filename_prefix = "exploration" if is_exploration else "predictions"

        predictions_hash = calculate_prediction_hash(prediction_config, dataset_hash)
        predictions_output_dir = Path(prediction_config.output_dir) / output_subdir
        predictions_path = predictions_output_dir / f"{filename_prefix}_{predictions_hash}.ndjson"

        if force_rebuild and predictions_path.exists():
            logger.warning(f"Force rebuild requested. Deleting existing predictions at {predictions_path}")
            predictions_path.unlink()

        set_seed(prediction_config.seed)

        predictor = Predictor(config=prediction_config)
        predictor.run(dataset_path=dataset_path, dataset_hash=dataset_hash)

        logger.info(f"Prediction stage complete. Final results are at {predictions_path}")
        return predictions_path

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def run_verification_stage(verification_config, predictions_path: Path, build_config, prediction_config, experiment_name: str, experiment_group: str, force_rebuild: bool = False) -> Path:
    logger.info("=== VERIFICATION STAGE ===")

    try:
        validate_file_exists(predictions_path, "Predictions")
        predictions_hash = extract_predictions_hash(predictions_path)
        verification_hash = calculate_verification_hash(verification_config, predictions_hash)

        verification_output_dir = Path(verification_config.output_dir) / "verification"
        verification_path = verification_output_dir / f"verification_{verification_hash}.ndjson"

        if force_rebuild and verification_path.exists():
            logger.warning(f"Force rebuild requested. Deleting existing verification results at {verification_path}")
            verification_path.unlink()

        verifier = Verifier(verification_config, predictions_path, verification_hash, prediction_config, experiment_name, experiment_group)
        verification_results_path, api_results_path = verifier.run()

        logger.info("Generating summary report...")
        results_dir = ensure_output_directory(Path(verification_config.results_dir))

        experiment_results_dir = results_dir / experiment_group / experiment_name
        experiment_results_dir.mkdir(parents=True, exist_ok=True)

        report_path = experiment_results_dir / f"result_{verification_hash}.md"

        dataset_hash = calculate_dataset_hash(build_config)
        dataset_output_dir = Path(build_config.output_dir) / "datasets"
        dataset_path = dataset_output_dir / f"dataset_{dataset_hash}.parquet"

        reporter = Reporter(
            verification_results_path=verification_results_path,
            verification_config=verification_config,
            prediction_config=prediction_config,
            build_config=build_config,
            verifier_prompt_config=verifier.prompt_config,
            dataset_path=dataset_path,
            prediction_path=predictions_path,
            api_results_path=api_results_path
        )
        reporter.generate_summary_report(report_path)

        logger.info(f"Verification stage complete. Results are at {verification_results_path}")
        logger.info(f"Summary report generated at {report_path}")
        return verification_results_path

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run VLM evaluation pipeline with vLLM")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to unified configuration YAML file"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["dataset", "predict", "verify"],
        choices=["dataset", "predict", "verify"],
        help="Pipeline stages to run (default: all stages)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of existing artifacts"
    )

    args = parser.parse_args()

    setup_logging()

    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)

        unified_config = UnifiedConfig(**config_data)
        unified_config.validate_prompts()

        build_config = unified_config.dataset
        prediction_config = unified_config.prediction
        verification_config = unified_config.verification

        set_seed(unified_config.seed)

        logger.info(f"Running pipeline stages: {args.stages}")
        logger.info(f"Experiment: {unified_config.experiment_name}")
        if unified_config.description:
            logger.info(f"Description: {unified_config.description}")

        artifact_path = None

        if "dataset" in args.stages:
            artifact_path = run_dataset_stage(build_config, force_rebuild=args.force or unified_config.force_rebuild)

        if "predict" in args.stages:
            if artifact_path is None:
                dataset_hash = calculate_dataset_hash(build_config)
                dataset_output_dir = Path(build_config.output_dir) / "datasets"
                artifact_path = dataset_output_dir / f"dataset_{dataset_hash}.parquet"

            artifact_path = run_prediction_stage(
                prediction_config,
                dataset_path=artifact_path,
                force_rebuild=args.force or unified_config.force_rebuild
            )

        if "verify" in args.stages:
            if artifact_path is None:
                dataset_hash = calculate_dataset_hash(build_config)
                predictions_hash = calculate_prediction_hash(prediction_config, dataset_hash)

                is_exploration = prediction_config.num_predictions_per_sample > 1
                output_subdir = "explorations" if is_exploration else "predictions"
                filename_prefix = "exploration" if is_exploration else "predictions"
                predictions_output_dir = Path(prediction_config.output_dir) / output_subdir
                artifact_path = predictions_output_dir / f"{filename_prefix}_{predictions_hash}.ndjson"

            artifact_path = run_verification_stage(
                verification_config,
                predictions_path=artifact_path,
                build_config=build_config,
                prediction_config=prediction_config,
                experiment_name=unified_config.experiment_name,
                experiment_group=unified_config.experiment_group,
                force_rebuild=args.force or unified_config.force_rebuild
            )

        unified_config.cleanup_temp_files()

        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        logger.info(f"Experiment: {unified_config.experiment_name}")
        if artifact_path:
            logger.info(f"Final artifact: {artifact_path}")

        if "dataset" in args.stages or "predict" in args.stages or "verify" in args.stages:
            dataset_hash = calculate_dataset_hash(build_config)
            dataset_output_dir = Path(build_config.output_dir) / "datasets"
            dataset_path = dataset_output_dir / f"dataset_{dataset_hash}.parquet"
            logger.info(f"Dataset: {dataset_path}")

        if "predict" in args.stages or "verify" in args.stages:
            scratch = "/workspace/vllm_dir"
            os.makedirs(scratch, exist_ok=True)

            os.environ["VLLM_CACHE_DIR"]   = f"{scratch}/vllm_cache"
            os.environ["HF_HOME"]          = f"{scratch}/hf_home"
            os.environ["HF_HUB_CACHE"]     = f"{scratch}/hf_cache"
            os.environ["XDG_CACHE_HOME"]   = f"{scratch}/.cache"
            dataset_hash = calculate_dataset_hash(build_config)
            predictions_hash = calculate_prediction_hash(prediction_config, dataset_hash)

            is_exploration = prediction_config.num_predictions_per_sample > 1
            output_subdir = "explorations" if is_exploration else "predictions"
            filename_prefix = "exploration" if is_exploration else "predictions"
            predictions_output_dir = Path(prediction_config.output_dir) / output_subdir
            predictions_path = predictions_output_dir / f"{filename_prefix}_{predictions_hash}.ndjson"
            logger.info(f"Predictions: {predictions_path}")

        if "verify" in args.stages:
            logger.info(f"Verification: {artifact_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        try:
            unified_config.cleanup_temp_files()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
