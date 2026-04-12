import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

import datasets

from .abstract import LocalDatasetBuilder
from ..utils import download_file, extract_tar

logger = logging.getLogger(__name__)


class Food101Loader(LocalDatasetBuilder):
    @property
    def cache_name(self) -> str:
        return "food101"

    @property
    def style(self) -> str:
        return "image_classification"

    @property
    def ability(self) -> str:
        return "accuracy"

    @classmethod
    def get_available_splits(cls) -> List[str]:
        return ["train", "test", "val"]

    def _build(self, output_path: Path):
        raw_data_dir = self.cache_dir / "raw" / self.cache_name
        data_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        tar_path = raw_data_dir / Path(data_url).name
        dataset_path = raw_data_dir / "food-101"

        if not dataset_path.exists():
            download_file(data_url, tar_path)
            extract_tar(tar_path, raw_data_dir, expected_content_name="food-101")

        asset_dir = Path("configs/dataset_assets/food101")
        if not all([
            (asset_dir / "metadata.csv").exists(),
            (asset_dir / "split_coop.csv").exists()
        ]):
            raise FileNotFoundError(
                f"Could not find `metadata.csv` or `split_coop.csv` in `{asset_dir}`. Please place them there."
            )

        metadata_df = pd.read_csv(asset_dir / "metadata.csv")
        class_names = metadata_df["class_name"].tolist()
        classes_to_idx = {str(c): i for i, c in enumerate(metadata_df["folder_name"].tolist())}
        split_df = pd.read_csv(asset_dir / "split_coop.csv")

        images_base_path = dataset_path / "images"
        data = datasets.DatasetDict()
        logger.info("Creating train/val/test splits for Food101...")
        
        for split in self.get_available_splits():
            split_filenames_df = split_df[split_df["split"] == split]
            image_paths = [str(images_base_path / fname) for fname in split_filenames_df["filename"].tolist()]
            folder_names = [Path(f).parent.name for f in image_paths]
            labels = [classes_to_idx[c] for c in folder_names]
            
            data[split] = datasets.Dataset.from_dict({
                "image": image_paths,
                "label": labels
            }).cast_column("image", datasets.Image())

        for split in data:
            features = data[split].features
            features["label"] = datasets.ClassLabel(names=class_names)
            data[split] = data[split].cast(features)

        logger.info(f"Saving processed dataset to {output_path}")
        data.save_to_disk(str(output_path))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        label_id = sample["label"]
        label_name = self.dataset.features["label"].names[label_id]

        return {
            "image": sample["image"].convert("RGB"),
            "label_id": label_id,
            "label_name": label_name.replace("_", " "),
        }