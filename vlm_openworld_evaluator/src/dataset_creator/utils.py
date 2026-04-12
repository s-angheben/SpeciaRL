import re
import random
import zipfile
import tarfile
import requests
import logging
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Union

from src.utils.hash_utils import generate_config_hash

logger = logging.getLogger(__name__)

def parse_index_range(range_str: str) -> List[int]:
    """Parse '0-10,15,20-22' into a sorted list of indices."""
    indices = set()
    parts = range_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(list(indices))

def parse_subset_size(subset_size_str: str, total_available: int) -> int:
    """Resolve a subset size string ('100' or '10%') against the total available count."""
    if subset_size_str.endswith('%'):
        percentage = float(subset_size_str[:-1])
        if percentage <= 0 or percentage > 100:
            raise ValueError(f"Percentage must be between 0 and 100, got {percentage}%")
        subset_count = int(total_available * percentage / 100)
    else:
        subset_count = int(subset_size_str)
        if subset_count <= 0:
            raise ValueError(f"Subset size must be positive, got {subset_count}")

    if subset_count > total_available:
        raise ValueError(f"Subset size {subset_count} exceeds available samples {total_available}")

    return subset_count

def apply_subset_sampling(indices: List[int], subset_size: Optional[str]) -> List[int]:
    if subset_size is None:
        return indices
        
    subset_count = parse_subset_size(subset_size, len(indices))
    return random.sample(indices, subset_count)


def download_file(url: str, target_path: Path, chunk_size: int = 8192) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {target_path.name}"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


def extract_zip(zip_path: Path, extract_to: Path, expected_content_name: Optional[str] = None) -> None:
    """Idempotent ZIP extract: skip if `expected_content_name` already exists under `extract_to`."""
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if expected_content_name:
        if (extract_to / expected_content_name).exists():
            logger.info(f"Content '{expected_content_name}' already exists in {extract_to}. Skipping extraction.")
            return

    logger.info(f"Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info("Extraction complete.")


def extract_tar(tar_path: Path, extract_to: Path, expected_content_name: Optional[str] = None) -> None:
    """Idempotent TAR extract (including .tar.gz/.tgz); uses `pigz` if available for parallel decompression."""
    extract_to.mkdir(parents=True, exist_ok=True)

    if expected_content_name and (extract_to / expected_content_name).exists():
        logger.info(f"Content '{expected_content_name}' already exists. Skipping extraction.")
        return

    logger.info(f"Fast extracting {tar_path.name} to {extract_to}...")
    if shutil.which("pigz"):
        subprocess.run(["tar", "--use-compress-program=pigz", "-xvf", str(tar_path), "-C", str(extract_to)], check=True)
    else:
        with tarfile.open(tar_path, 'r:*') as tar:
            tar.extractall(path=extract_to)
    logger.info("Extraction complete.")
