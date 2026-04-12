import logging
from pathlib import Path
from typing import Any, Dict, List

import datasets

from .abstract import LocalDatasetBuilder
from ..utils import download_file, extract_tar

logger = logging.getLogger(__name__)


class FGVCAircraftLoader(LocalDatasetBuilder):
    @property
    def cache_name(self) -> str:
        return "fgvc_aircraft"

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
        data_url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
        tar_path = raw_data_dir / Path(data_url).name
        dataset_path = raw_data_dir / "fgvc-aircraft-2013b"
        
        if not dataset_path.exists():
            download_file(data_url, tar_path)
            extract_tar(tar_path, raw_data_dir)
        
        data_dir = dataset_path / "data"
        metadata_fp = data_dir / "variants.txt"
        with open(metadata_fp) as f:
            class_names = [line.strip() for line in f.readlines()]
        classes_to_idx = {c: i for i, c in enumerate(class_names)}

        data = datasets.DatasetDict()
        for split in self.get_available_splits():
            split_fp = data_dir / f"images_variant_{split}.txt"
            with open(split_fp) as f:
                lines = [line.strip() for line in f.readlines()]
                filenames = [line.split(" ")[0] for line in lines]
                labels = [" ".join(line.split(" ")[1:]) for line in lines]

            image_paths = [str(data_dir / "images" / f"{x}.jpg") for x in filenames]
            labels = [classes_to_idx[c] for c in labels]
            
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