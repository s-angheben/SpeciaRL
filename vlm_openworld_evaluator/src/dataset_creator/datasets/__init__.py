import logging
from .cub200 import CUB200Loader
from .flowers102 import Flowers102Loader
from .oxford_pets import OxfordPetsLoader
from .fgvc_aircraft import FGVCAircraftLoader
from .stanford_cars import StanfordCarsLoader
from .caltech101 import Caltech101Loader
from .food101 import Food101Loader

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "cub200": CUB200Loader,
    "flowers102": Flowers102Loader,
    "oxford_pets": OxfordPetsLoader,
    "fgvc_aircraft": FGVCAircraftLoader,
    "stanford_cars": StanfordCarsLoader,
    "caltech101": Caltech101Loader,
    "food101": Food101Loader,
}

logger.info(f"Registered {len(DATASET_REGISTRY)} dataset handlers: {list(DATASET_REGISTRY.keys())}")

