import os
import tarfile
from typing import Dict
import click
import gdown
from enum import Enum

from utils.logs import logger

class DownloadDataType(Enum):
    TASKS = "tasks"
    HUMAN_DESIGN = "human_design"
    ZEROSHOT = "zeroshot"
    ALL = "all"

def download_file(url: str, filename: str) -> None:
    """Download a file from Google Drive using gdown."""
    gdown.download(url, filename, quiet=False)


def extract_tar_gz(filename: str, extract_path: str) -> None:
    """Extract a tar.gz file to the specified path."""
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=extract_path)


def process_dataset(url: str, filename: str, extract_path: str) -> None:
    """Download, extract, and clean up a dataset."""
    logger.info(f"Downloading {filename}...")
    download_file(url, filename)

    logger.info(f"Extracting {filename}...")
    extract_tar_gz(filename, extract_path)

    logger.info(f"{filename} download and extraction completed.")

    os.remove(filename)
    logger.info(f"Removed {filename}")
    
datasets_to_download: Dict[str, Dict[str, str]] = {
    "tasks": {
        "url": "https://drive.google.com/uc?export=download&id=16FSuvmtlYs3Mum04ERxsFN_qEPrumMAo",
        "filename": "cabench_data.tar.gz",
        "extract_path": "tasks",
    },
    "human_design": {
        "url": "https://drive.google.com/uc?export=download&id=17XZyu32IZ5VJd1bcbcReVG7ngkIv2RhA",
        "filename": "human_design.tar.gz",
        "extract_path": "results",
    },
    "zeroshot": {
        "url": "https://drive.google.com/uc?export=download&id=119-8Zjk4AOylYKD0-qCvhPx3oNG9-87n",
        "filename": "zeroshot.tar.gz",
        "extract_path": "results",
    },
}


def download(required_datasets, if_first_download: bool = True):
    """Main function to process all selected datasets"""
    if if_first_download:
        # Convert string to list if needed
        if isinstance(required_datasets, str):
            required_datasets = [required_datasets]
        
        for dataset_name in required_datasets:
            dataset = datasets_to_download[dataset_name]
            extract_path = dataset["extract_path"]
            process_dataset(dataset["url"], dataset["filename"], extract_path)
    else:
        logger.info("Skip downloading datasets")

@click.command()
@click.option("--datasets", type=str, default="tasks", help="Datasets to download")
@click.option("--if_first_download", type=bool, default=True, help="If first download")
def main(datasets, if_first_download):
    download(datasets, if_first_download)


if __name__ == "__main__":
    main()