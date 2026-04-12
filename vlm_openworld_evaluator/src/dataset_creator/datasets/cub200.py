import logging
import tarfile
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

import datasets

from .abstract import LocalDatasetBuilder
from ..utils import download_file

logger = logging.getLogger(__name__)


def _extract_tar(tar_path: Path, extract_to: Path):
    expected_content = extract_to / "CUB_200_2011"
    if expected_content.exists() and any(expected_content.iterdir()):
        logger.info(f"Directory {expected_content} already contains files. Skipping extraction.")
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {tar_path.name} to {extract_to}...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_to)
    logger.info("Extraction complete.")


class CUB200Loader(LocalDatasetBuilder):
    """CUB-200-2011 loader; builds the dataset from the Caltech tarball on first use."""

    @property
    def cache_name(self) -> str:
        return "cub200_2011"

    @property
    def style(self) -> str:
        return "image_classification"

    @property
    def ability(self) -> str:
        return "accuracy"

    @classmethod
    def get_available_splits(cls) -> List[str]:
        return ["train", "test"]

    def _build(self, output_path: Path):
        # 1. Define paths and URL
        raw_data_dir = self.cache_dir / "raw" / self.cache_name
        data_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
        tar_path = raw_data_dir / Path(data_url).name
        dataset_path = raw_data_dir / "CUB_200_2011"

        # 2. Download and Extract
        if not tar_path.exists():
            download_file(data_url, tar_path)
        _extract_tar(tar_path, raw_data_dir)

        # 3. Parse the metadata files
        logger.info("Parsing CUB-200-2011 metadata files...")

        classes_df = pd.read_csv(dataset_path / 'classes.txt', sep=' ', names=['class_id', 'class_name'])
        class_id_to_name = classes_df.set_index('class_id')['class_name'].to_dict()
        class_names = classes_df['class_name'].tolist()

        images_df = pd.read_csv(dataset_path / 'images.txt', sep=' ', names=['image_id', 'filepath'])
        image_id_to_path = images_df.set_index('image_id')['filepath'].to_dict()

        image_class_labels_df = pd.read_csv(dataset_path / 'image_class_labels.txt', sep=' ', names=['image_id', 'class_id'])

        split_df = pd.read_csv(dataset_path / 'train_test_split.txt', sep=' ', names=['image_id', 'is_train'])

        # 4. Merge all metadata into a single DataFrame
        merged_df = pd.merge(image_class_labels_df, split_df, on='image_id')
        merged_df['filepath'] = merged_df['image_id'].map(image_id_to_path)
        merged_df['label_name'] = merged_df['class_id'].map(class_id_to_name)

        # 5. Create Hugging Face DatasetDict
        data = datasets.DatasetDict()
        logger.info("Creating train/test splits for CUB-200-2011...")

        for is_train_flag, split_name in [(1, "train"), (0, "test")]:
            split_df = merged_df[merged_df['is_train'] == is_train_flag]

            image_paths = (dataset_path / "images" / split_df['filepath']).astype(str).tolist()
            labels = split_df['class_id'].tolist()

            # CUB class IDs are 1-indexed; HF ClassLabel needs 0-indexed.
            labels_0_indexed = [l - 1 for l in labels]

            split_dataset = datasets.Dataset.from_dict({
                "image": image_paths,
                "label": labels_0_indexed
            }).cast_column("image", datasets.Image())

            features = split_dataset.features
            features["label"] = datasets.ClassLabel(names=class_names)
            split_dataset = split_dataset.cast(features)

            data[split_name] = split_dataset

        logger.info(f"Saving processed dataset to {output_path}")
        data.save_to_disk(str(output_path))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        label_id = sample["label"]
        label_name = self.dataset.features["label"].names[label_id]

        return {
            "image": sample["image"].convert("RGB"),
            "label_id": label_id,
            # Strip the "001." prefix and underscores from "001.Black_footed_Albatross".
            "label_name": label_name.split('.')[1].replace('_', ' '),
        }