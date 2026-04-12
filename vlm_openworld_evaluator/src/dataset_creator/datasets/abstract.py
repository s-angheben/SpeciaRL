import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """Common interface for dataset loaders: lazy `dataset`, indexed sample access, metadata properties."""

    def __init__(self, name: str, split: str, cache_dir: str = "data/datasets", **kwargs: Any):
        self._name = name
        self.split = split
        self.cache_dir = Path(cache_dir)
        self._dataset: Dataset | None = None

    @property
    @abstractmethod
    def cache_name(self) -> str:
        pass

    @property
    @abstractmethod
    def style(self) -> str:
        pass

    @property
    @abstractmethod
    def ability(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_available_splits(cls) -> List[str]:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample in the standardized format used by the builder."""
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self._load()
        return self._dataset

    @abstractmethod
    def _load(self) -> Dataset:
        pass

    @property
    def name(self) -> str:
        return self._name


class HuggingFaceDatasetLoader(DatasetLoader):
    """Loader for datasets fetched directly from the Hugging Face Hub."""

    @property
    @abstractmethod
    def hf_id(self) -> str:
        pass

    @property
    def cache_name(self) -> str:
        return self.hf_id.replace("/", "_")

    def _load(self) -> Dataset:
        logger.info(f"Loading '{self.name}' from Hugging Face Hub: {self.hf_id}")
        try:
            dataset_dict = load_dataset(self.hf_id, cache_dir=self.cache_dir)
            if self.split not in dataset_dict:
                raise ValueError(f"Split '{self.split}' not found. Available: {list(dataset_dict.keys())}")
            return dataset_dict[self.split]
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.hf_id}' from Hub: {e}")
            raise


class LocalDatasetBuilder(DatasetLoader):
    """Loader for datasets built from raw files: load from disk cache, build on first miss."""

    def _load(self) -> Dataset:
        processed_path = self.cache_dir / self.cache_name
        if not processed_path.exists():
            logger.info(f"Processed dataset not found at {processed_path}. Building from scratch...")
            self._build(processed_path)

        logger.debug(f"Loading dataset dictionary from {processed_path}")
        dataset_dict = load_from_disk(str(processed_path))

        if self.split not in dataset_dict:
            raise ValueError(f"Split '{self.split}' not found. Available: {list(dataset_dict.keys())}")

        return dataset_dict[self.split]

    @abstractmethod
    def _build(self, output_path: Path) -> None:
        """Download, extract, and process raw files into a `DatasetDict` saved at `output_path`."""
        pass