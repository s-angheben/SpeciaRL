import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

import datasets

from .abstract import LocalDatasetBuilder
from ..utils import download_file, extract_tar

logger = logging.getLogger(__name__)


class OxfordPetsLoader(LocalDatasetBuilder):
    @property
    def cache_name(self) -> str:
        return "oxford_pets"

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
        data_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        tar_path = raw_data_dir / Path(data_url).name
        images_base_path = raw_data_dir / "images"

        if not images_base_path.exists():
            download_file(data_url, tar_path)
            extract_tar(tar_path, raw_data_dir, expected_content_name="images")

        # Idempotency check: if any image still sits at the top level, the per-class reorg hasn't happened yet.
        first_image_path = next(images_base_path.glob("*.jpg"), None)
        if first_image_path and first_image_path.parent == images_base_path:
            logger.info("Reorganizing images into class subdirectories...")
            for image_path in tqdm(list(images_base_path.glob("*.jpg")), desc="Reorganizing images"):
                if not image_path.exists():
                    continue
                class_name = "_".join(image_path.name.split("_")[:-1])
                target_dir = images_base_path / class_name
                target_dir.mkdir(exist_ok=True)
                image_path.rename(target_dir / image_path.name)

        asset_dir = Path("configs/dataset_assets/oxford_pets")
        metadata_fp = asset_dir / "metadata.csv"
        split_fp = asset_dir / "split_coop.csv"

        if not metadata_fp.exists() or not split_fp.exists():
            raise FileNotFoundError(
                f"Could not find `metadata.csv` or `split_coop.csv`. Please ensure they are placed in the "
                f"`{asset_dir}` directory."
            )

        metadata_df = pd.read_csv(metadata_fp)
        class_names = metadata_df["class_name"].tolist()
        classes_to_idx = {str(c): i for i, c in enumerate(metadata_df["folder_name"].tolist())}
        split_df = pd.read_csv(split_fp)

        data = datasets.DatasetDict()
        logger.info("Creating train/val/test splits...")
        for split in self.get_available_splits():
            split_filenames = split_df[split_df["split"] == split]["filename"].tolist()
            image_paths = [str(images_base_path / fname) for fname in split_filenames]
            folder_names = [Path(f).parent.name for f in split_filenames]
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