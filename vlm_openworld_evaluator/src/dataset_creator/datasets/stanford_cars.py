from typing import Any, Dict, List

from .abstract import HuggingFaceDatasetLoader


class StanfordCarsLoader(HuggingFaceDatasetLoader):
    @property
    def hf_id(self) -> str:
        return "tanganke/stanford_cars"

    @property
    def style(self) -> str:
        return "image_classification"

    @property
    def ability(self) -> str:
        return "accuracy"

    @classmethod
    def get_available_splits(cls) -> List[str]:
        return ["train", "test"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        return {
            "image": sample["image"].convert("RGB"),
            "label_id": sample["label"],
            "label_name": self.dataset.features["label"].names[sample["label"]],
        }