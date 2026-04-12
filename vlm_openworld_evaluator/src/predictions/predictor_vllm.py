import logging
import io
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
import orjson
from PIL import Image
from tqdm import tqdm
import sys
from pydantic import ValidationError
from vllm import LLM, SamplingParams

from src.predictions.config import PredictionConfig, ModelConfig
from src.schemas.data_record import DataRecord
from src.schemas.prediction_record import PredictionRecord
from src.schemas.base_record import DataContext
from src.utils.json_utils import custom_json_serializer
from src.utils.hash_utils import calculate_prediction_hash
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


# Top-level so multiprocessing.Pool can pickle it.
def process_sample_for_vllm(sample_dict: Dict[str, Any], tokenizer_path: str) -> Dict | None:
    """CPU-bound per-sample preprocessing (image decode + chat template) for the worker pool."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True)
        sample = DataRecord(**sample_dict)

        MAX_PIXELS_8B = 400_000   # ~632x632, InternVL2.5-8B upper limit
        MIN_PIXELS = 3_136        # ~56x56, InternVL2.5-8B lower limit

        def resize_with_pixel_limit(img: Image.Image) -> Image.Image:
            w, h = img.size
            num_pixels = w * h
            if num_pixels > MAX_PIXELS_8B:
                scale = (MAX_PIXELS_8B / num_pixels) ** 0.5
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                img = img.resize((new_w, new_h))
            elif num_pixels < MIN_PIXELS:
                scale = (MIN_PIXELS / num_pixels) ** 0.5
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                img = img.resize((new_w, new_h))
            return img

        pil_images = [Image.open(io.BytesIO(img)).convert("RGB") for img in sample.images]

        if "internvl2_5-8b" in tokenizer_path.lower():
            resized_images = []
            for img in pil_images:
                resized = resize_with_pixel_limit(img)
                if (img.size != resized.size):
                    logger.debug(f"Resized image from {img.size} to {resized.size} for InternVL2_5-8B")
                resized_images.append(resized)
            pil_images = resized_images

        # InternVL uses plain text with <image> tokens, not the OpenAI chat format.
        is_internvl = "internvl" in tokenizer_path.lower()

        if is_internvl:
            text_parts = []
            for msg in sample.prompt:
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        text_parts.append("<image>")
                    elif item.get("type") == "text":
                        text_parts.append(item["text"])
            formatted_prompt = "\n".join(text_parts)
        else:
            messages = []
            img_idx = 0
            for msg in sample.prompt:
                role = msg.get("role", "user")
                content_list = []
                for item in msg.get("content", []):
                    if item.get("type") == "text":
                        content_list.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image" and img_idx < len(pil_images):
                        content_list.append({"type": "image"})
                        img_idx += 1
                if content_list:
                    messages.append({"role": role, "content": content_list})

            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        multi_modal_data = {"image": pil_images} if pil_images else {}

        return {
            "sample": sample,
            "prompt_dict": {
                "prompt": formatted_prompt,
                "multi_modal_data": multi_modal_data
            }
        }
    except Exception:
        return None


class VllmPredictor:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.base_output_dir = Path(config.output_dir)
        self.generation_params = config.generation_params.get_user_overrides()

        self.llm = None
        self.lora_request = None
        self.tokenizer = None
        self.sampling_params = None
        self._initialized = False

    def initialize_vllm(self, model_config):
        logger.info(f"Initializing vLLM: {model_config.model_name}")

        adapter_path = Path(model_config.path_or_id) / "lora_adapter"
        has_lora = adapter_path.exists() and adapter_path.is_dir()

        vllm_args = {
            "model": model_config.path_or_id,
            "trust_remote_code": True,
            "dtype": "auto" if model_config.dtype == "auto" else model_config.dtype,
            "enforce_eager": True,            # disables torch.compile (avoids compilation errors)
            "disable_custom_all_reduce": True,
            "enable_prefix_caching": True,
            "max_num_seqs": 256,
            "enable_chunked_prefill": True,
            **model_config.params
        }

        lora_request = None
        if has_lora:
            logger.info(f"Found LoRA adapter at {adapter_path}")

            lora_config = self._read_lora_config(adapter_path)
            lora_rank = lora_config.get("r", 64)

            logger.info(f"LoRA config: rank={lora_rank}, alpha={lora_config.get('lora_alpha', 32)}")

            vllm_args.update({
                "enable_lora": True,
                "max_lora_rank": lora_rank,
                "max_loras": 1
            })
            lora_request = LoRARequest("lora_adapter", 1, str(adapter_path))
        else:
            logger.info("No LoRA adapter found")

        llm = LLM(**vllm_args)
        return llm, lora_request

    def _read_lora_config(self, adapter_path: Path) -> dict:
        config_file = adapter_path / "adapter_config.json"
        try:
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded LoRA config: r={config.get('r')}, lora_alpha={config.get('lora_alpha')}")
                return config
            else:
                logger.warning(f"adapter_config.json not found at {config_file}")
                return {}
        except Exception as e:
            logger.error(f"Failed to read LoRA config: {e}")
            return {}

    def _ensure_initialized(self):
        if self._initialized:
            return

        logger.info("Initializing vLLM model (this may take a moment)...")

        self.llm, self.lora_request = self.initialize_vllm(self.config.model)

        logger.info(f"Loading tokenizer from '{self.config.model.path_or_id}' for chat template formatting.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.path_or_id, use_fast=True, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}. Prompt formatting may be incorrect.")
            self.tokenizer = None

        self.sampling_params = self._create_sampling_params(**self.generation_params)

        self._initialized = True
        logger.info("vLLM initialization complete")

    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        params = {}

        mappings = {
            "max_new_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repetition_penalty": "repetition_penalty",
            "seed": "seed",
        }

        for src, dst in mappings.items():
            if src in kwargs and kwargs[src] is not None:
                params[dst] = kwargs[src]

        if kwargs.get("do_sample") is False:
            params["temperature"] = 0.0

        if "stop_sequences" in kwargs or "stop_strings" in kwargs:
            stop = kwargs.get("stop_sequences") or kwargs.get("stop_strings")
            params["stop"] = [stop] if isinstance(stop, str) else stop

        return SamplingParams(**params)

    def _load_dataset(self, dataset_path: Path) -> List[DataRecord]:
        df = pd.read_parquet(dataset_path)

        if 'sample_id' not in df.columns:
            raise ValueError("Dataset must have 'sample_id' column")

        if self.config.target_splits:
            df = df[df['split'].isin(self.config.target_splits)]
            logger.info(f"Filtered to {len(df)} samples for splits: {self.config.target_splits}")

        samples = []
        for record in df.to_dict('records'):
            if 'prompt_info' in record and isinstance(record['prompt_info'], str):
                try:
                    record['prompt_info'] = orjson.loads(record['prompt_info'])
                except:
                    record['prompt_info'] = {}

            try:
                samples.append(DataRecord(**record))
            except ValidationError as e:
                logger.warning(f"Skipping invalid record: {e}")

        if self.config.prompts_override_file:
            from src.dataset_creator.prompts import PromptManager
            prompt_manager = PromptManager(self.config.prompts_override_file)
            override_prompt = prompt_manager.get_random_prompt(dataset_name=None)
            vlm_prompt = override_prompt.get_vlm_prompt()
            logger.info(f"Overriding prompts with: {self.config.prompts_override_file}")
            for sample in samples:
                sample.prompt = vlm_prompt
                sample.prompt_info = override_prompt

        return samples
    
    def _get_completed_samples(self, output_path: Path) -> Set[str]:
        """Return sample_ids that already have num_predictions_per_sample entries on disk."""
        if not output_path.exists():
            return set()

        counts = defaultdict(int)
        with output_path.open('r') as f:
            for line in f:
                try:
                    record = orjson.loads(line)
                    counts[record['sample_id']] += 1
                except:
                    continue

        return {sid for sid, count in counts.items()
                if count == self.config.num_predictions_per_sample}

    def _cleanup_partial_predictions(self, output_path: Path, completed_sample_ids: Set[str]):
        if not output_path.exists():
            return

        keep = []
        with output_path.open('r') as f:
            for line in f:
                try:
                    record = orjson.loads(line)
                    if record.get('sample_id') in completed_sample_ids:
                        keep.append(line)
                except:
                    continue

        with output_path.open('w') as f:
            f.writelines(keep)

        logger.info(f"Cleaned up partial predictions. Kept {len(keep)} complete records.")

    def run(self, dataset_path: Path, dataset_hash: str):
        is_exploration = self.config.num_predictions_per_sample > 1
        subdir = "explorations" if is_exploration else "predictions"
        prefix = "exploration" if is_exploration else "predictions"

        output_dir = self.base_output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        config_hash = calculate_prediction_hash(self.config, dataset_hash)
        output_path = output_dir / f"{prefix}_{config_hash}.ndjson"

        logger.info(f"Saving results to {output_path}")
        all_samples = self._load_dataset(dataset_path)
        completed = self._get_completed_samples(output_path)
        self._cleanup_partial_predictions(output_path, completed)

        samples = [s for s in all_samples if s.sample_id not in completed]
        if not samples:
            logger.info("All samples completed")
            return

        self._ensure_initialized()

        logger.info(f"Processing {len(samples)} samples (skipping {len(completed)} completed)")

        batch_size = self.config.inference_batch_size
        num_preds_total = self.config.num_predictions_per_sample

        num_workers = max(1, cpu_count() - 2)
        logger.info(f"Using {num_workers} worker processes for data preparation.")
        samples_as_dicts = [s.model_dump() for s in samples]

        with output_path.open('a') as f, Pool(processes=num_workers) as pool:
            worker_func = partial(process_sample_for_vllm, tokenizer_path=self.config.model.path_or_id)
            prepared_data_iter = pool.imap_unordered(worker_func, samples_as_dicts)

            progress_bar = tqdm(total=len(samples), desc="Processing samples", file=sys.stdout)
            batch_prepared = []

            def process_batch(current_batch):
                if not current_batch:
                    return

                prompts = [p['prompt_dict'] for p in current_batch]
                valid_batch_samples = [p['sample'] for p in current_batch]

                sampling_params = self.sampling_params.clone()
                sampling_params.n = num_preds_total

                try:
                    outputs = self.llm.generate(prompts, sampling_params, lora_request=self.lora_request)
                    for sample, output in zip(valid_batch_samples, outputs):
                        for idx, completion in enumerate(output.outputs):
                            record = self._create_record(sample, completion.text, idx)
                            f.write(orjson.dumps(
                                record,
                                default=lambda x: x.model_dump() if hasattr(x, 'model_dump') else custom_json_serializer(x)
                            ).decode() + '\n')
                except Exception as e:
                    logger.error(f"Batch failed: {e}", exc_info=True)

                progress_bar.update(len(current_batch))

            for prepared_item in prepared_data_iter:
                if prepared_item:
                    batch_prepared.append(prepared_item)

                if len(batch_prepared) == batch_size:
                    process_batch(batch_prepared)
                    batch_prepared = []

            process_batch(batch_prepared)

            progress_bar.close()
            f.flush()

        logger.info(f"Completed. Results in {output_path}")

    def _create_record(self, sample: DataRecord, prediction: str, idx: int) -> PredictionRecord:
        data_context = DataContext(**sample.model_dump(exclude={'images'}))
        
        return PredictionRecord(
            sample_id=sample.sample_id,
            prediction_group_id=sample.sample_id,
            model_name=self.config.model.model_name,
            vlm_prediction=prediction,
            prediction_index=idx,
            prediction_config=self.config,
            data_context=data_context
        )