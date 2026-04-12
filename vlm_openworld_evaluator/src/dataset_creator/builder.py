import os
import io
import logging
import pandas as pd
from tqdm import tqdm
from typing import Iterable, List, Dict, Any

from src.schemas.data_record import DataRecord

from .config import BuildConfig, SourceConfig
from .prompts import PromptManager
from .datasets import DATASET_REGISTRY
from .utils import generate_config_hash, parse_index_range, apply_subset_sampling
from src.schemas.base_record import RewardModel, ExtraInfo
from src.schemas.prompt_config import PromptConfig
from src.utils.seeding import set_seed
from more_itertools import chunked
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """Builds a dataset from specified sources based on a validated configuration."""

    def __init__(self, config: BuildConfig, prompt_manager: PromptManager, force: bool = False):
        self.config = config
        self.prompt_manager = prompt_manager
        self.force = force
        self.output_dir = os.path.join(self.config.output_dir, "datasets")
        os.makedirs(self.output_dir, exist_ok=True)
        set_seed(self.config.seed)

    def build(self) -> str:
        output_path = self._get_output_path()
        if not self.force and os.path.exists(output_path):
            logger.info(f"Dataset already exists at {output_path}. Skipping build.")
            logger.info("Use the --force flag to rebuild the dataset.")
            return output_path

        record_stream = (
            rec
            for source_index, source_cfg in enumerate(self.config.sources)
            for rec in self._process_source(source_cfg, source_index=source_index, start_index=0)
        )

        self._save_dataset_streaming(chunked(record_stream, 1000), output_path)

        logger.info(f"\nDataset creation complete!")
        logger.info(f"  - Saved to: {output_path}")
        return output_path

    def _save_dataset_streaming(self, records: Iterable[DataRecord], output_path: str):
        """Stream records to Parquet, casting every batch to a fixed schema (Parquet writer requires it)."""
        writer = None
        schema = None

        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Removed existing file: {output_path}")

        for batch in records:
            dict_batch = [record.model_dump() for record in batch]

            for rec in dict_batch:
                loader_args = rec['extra_info'].get('loader_args') or {}

                supers = loader_args.get('supercategories', [])
                if not isinstance(supers, list):
                    supers = []

                # supercategories must always be a non-empty list[str] so the Parquet
                # schema stays stable across batches.
                if not supers:
                    supers = ["none"]

                loader_args['supercategories'] = [
                    str(x) if x is not None else "none" for x in supers
                ]

                rec['extra_info']['loader_args'] = loader_args

            df = pd.DataFrame(dict_batch)
            table = pa.Table.from_pandas(df)

            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_path, schema)
            else:
                table = table.cast(schema)

            writer.write_table(table)

        if writer:
            writer.close()

    def _process_source(self, source_cfg: SourceConfig, source_index: int, start_index: int) -> List[DataRecord]:
        logger.info(f"Processing source: {source_cfg.name} (source index: {source_index})")

        LoaderClass = DATASET_REGISTRY.get(source_cfg.name)
        if not LoaderClass:
            raise ValueError(f"Loader class for '{source_cfg.name}' not found in registry.")

        dataset_loader = LoaderClass(name=source_cfg.name, split=source_cfg.source_split, cache_dir=source_cfg.cache_dir, **source_cfg.loader_args)

        loaded_dataset = dataset_loader.dataset
        if 'id' not in loaded_dataset.column_names:
            loaded_dataset = loaded_dataset.add_column('id', range(len(loaded_dataset)))

        # 1. Determine all possible indices
        if source_cfg.indices == "all" or source_cfg.indices is None:
            initial_indices = list(range(len(loaded_dataset)))
        else:
            initial_indices = parse_index_range(source_cfg.indices)

        # 2. Pre-filter indices based on include/exclude rules if applicable
        if source_cfg.include or source_cfg.exclude:
            logger.info("Pre-filtering indices based on include/exclude rules...")
            available_indices = []
            for i in tqdm(initial_indices, desc="Filtering indices", leave=False):
                sample = dataset_loader[i]
                if self._should_include_sample(sample, source_cfg):
                    available_indices.append(i)

            logger.info(f"  {len(available_indices)} out of {len(initial_indices)} indices remain after filtering.")
        else:
            available_indices = initial_indices

        # 3. Apply subset sampling on the *filtered* list of indices
        indices_to_process = apply_subset_sampling(available_indices, source_cfg.subset_size)

        subset_dataset = loaded_dataset.select(indices_to_process)

        processed_dataset = subset_dataset.map(
            self._process_batch,
            batched=True,
            batch_size=100,
            num_proc=4,
            fn_kwargs={
                "dataset_loader": dataset_loader,
                "source_cfg": source_cfg,
                "source_index": source_index,
                "build_config": self.config,
                "prompt_manager": self.prompt_manager,
                "start_index": start_index
            },
            remove_columns=subset_dataset.column_names
        )

        valid_records = [rec for rec in processed_dataset["data_record"] if rec is not None]
        source_records = [DataRecord(**rec) for rec in valid_records]

        return source_records

    @staticmethod
    def _process_batch(batch: Dict[str, List], dataset_loader: Any, source_cfg: SourceConfig, source_index: int, build_config: BuildConfig, prompt_manager: PromptManager, start_index: int) -> Dict[str, List[Dict]]:
        """Per-batch sample processor for HuggingFace `dataset.map(batched=True)`. Static so multiprocessing can pickle it."""
        processed_records = []

        num_samples = len(batch[next(iter(batch))])

        for i in range(num_samples):
            sample = {key: batch[key][i] for key in batch}
            original_index = sample['id']

            # Loader.__getitem__ does post-processing (label name lookup etc.), so we re-call it
            # rather than using the raw row from the batch.
            processed_sample = dataset_loader[original_index]

            vlm_prompt = prompt_manager.get_random_prompt(
                dataset_name=dataset_loader.name,
                source_prompts_path=source_cfg.source_prompts_path
            )

            record = DatasetBuilder._create_record(
                processed_sample, dataset_loader, vlm_prompt, source_cfg, build_config,
                original_index, start_index + len(processed_records), source_index
            )
            processed_records.append(record.model_dump())

        return {"data_record": processed_records}

    @staticmethod
    def _should_include_sample(sample: Dict, source_cfg: SourceConfig) -> bool:
        label_name = sample.get("label_name")
        if label_name is None:
            return True
        
        if source_cfg.include is not None and label_name not in source_cfg.include:
            return False
        
        if source_cfg.exclude is not None and label_name in source_cfg.exclude:
            return False
        
        return True

    @staticmethod
    def _create_record(
        sample: Dict, dataset: Any, vlm_prompt: PromptConfig, source_cfg: SourceConfig, build_config: BuildConfig,
        original_index: int, new_index: int, source_index: int
    ) -> DataRecord:
        with io.BytesIO() as buffer:
            sample["image"].save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        config_hash = generate_config_hash(build_config.model_dump())
        sample_id = f"{config_hash}_{source_index}_{original_index}"

        # supercategories must be a non-empty list[str] for the Parquet schema (see _save_dataset_streaming).
        loader_args = source_cfg.loader_args.copy() if source_cfg.loader_args else {}
        if 'supercategories' in loader_args:
            supers = loader_args['supercategories']
            if isinstance(supers, list):
                string_supers = [str(x) for x in supers if x is not None]
                loader_args['supercategories'] = string_supers if string_supers else ["none"]
            elif supers is None:
                loader_args['supercategories'] = ["none"]
            else:
                loader_args['supercategories'] = [str(supers)]
        else:
            loader_args['supercategories'] = ["none"]
        
        return DataRecord(
            data_source=dataset.name, 
            sample_id=sample_id,
            index=new_index,
            images=[image_bytes],
            prompt=vlm_prompt.get_vlm_prompt(),
            prompt_info=vlm_prompt,
            split=source_cfg.target_split,
            ability=source_cfg.ability or dataset.ability,
            reward_model=RewardModel(
                style=source_cfg.style or dataset.style,
                ground_truth=sample["label_name"]
            ),
            extra_info=ExtraInfo(
                data_source=dataset.name,
                index_orig=original_index,
                source_split=source_cfg.source_split,
                source_index=source_index,
                loader_args=loader_args,
            ),
        )

    def _get_output_path(self) -> str:
        config_hash = generate_config_hash(self.config.model_dump())
        output_filename = f"dataset_{config_hash}.parquet"
        return os.path.join(self.output_dir, output_filename)

    def _save_dataset(self, records: List[DataRecord], output_path: str):
        if not records:
            logger.warning("No records were generated. Skipping file save.")
            return

        dict_records = [record.model_dump() for record in records]

        # Empty loader_args dicts can't be serialized by Parquet — substitute a placeholder.
        for record in dict_records:
            if not record['extra_info']['loader_args']:
                record['extra_info']['loader_args'] = {'_empty': True}

        df = pd.DataFrame(dict_records)
        df.to_parquet(output_path, index=False)