"""
Dataset utilities for the FFMSP project.

This module provides functions to download and extract benchmark datasets 
(e.g., the FFMSP instances) into a standardized location within the repository.

Main features:
- Downloads datasets from a given URL.
- Extracts `.tgz` archives with.
- Cleans up partial downloads or old dataset directories when refreshing data.
- Ensures datasets are always placed under <repo-root>/data/datasets.

"""

import os, sys
import tarfile
import shutil
import urllib.request
from urllib.parse import urlparse
from .git_utils import get_git_root

INSTANCES_URL = "https://www.iiia.csic.es/~christian.blum/downloads/FFMSP_instances.tgz"
BAR_LEN = 40

GIT_ROOT = get_git_root()
DOWNLOAD_DIR = os.path.join(GIT_ROOT, "data")
DATASETS_DIR = os.path.join(DOWNLOAD_DIR, "datasets")

def clear_line():
    """Clear the current terminal line (used for progress bar updates)."""
    cols = shutil.get_terminal_size().columns
    sys.stdout.write("\r" + " " * cols + "\r")
    sys.stdout.flush()

def download_datasets(url, dest_path):
    """
    Download a dataset archive from `url` to `dest_path` with a progress bar.
    
    Args:
        url (str): The URL of the dataset to download.
        dest_path (str): The local file path where the dataset should be saved.
    """
    def _progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)

        filled_len = int(BAR_LEN * percent // 100)
        bar = "=" * filled_len + "-" * (BAR_LEN - filled_len)

        sys.stdout.write(f"\rDownloading dataset... [{bar}] {percent:5.1f}%")
        sys.stdout.flush()

    try:
        _, __ = urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        clear_line()
        print("Dataset download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        # Clean up if something went wrong
        if os.path.exists(dest_path):
            os.remove(dest_path)

def extract_tgz(archive_path, extract_dir):
    """
    Extract a `.tgz` archive into a directory with progress feedback.
    
    Args:
        archive_path (str): Path to the `.tgz` archive file.
        extract_dir (str): Directory where contents should be extracted.
    
    Returns:
        bool: True if extraction succeeded, False otherwise.
    """
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            total_files = len(members)

            for i, member in enumerate(members, 1):
                tar.extract(member, path=extract_dir, filter='data')

                # Progress bar
                percent = i / total_files * 100
                filled_len = int(BAR_LEN * i / total_files)
                bar = "=" * filled_len + "-" * (BAR_LEN - filled_len)

                sys.stdout.write(f"\rExtracting... [{bar}] {percent:5.1f}%")
                sys.stdout.flush()

        # Optional: verify extraction
        if not os.listdir(extract_dir):
            raise ValueError("Extraction failed: directory is empty!")

        clear_line()
        print("Extraction complete.")

        return True

    except Exception as e:
        clear_line()
        print(f"Extraction failed: {e}")

        return False

def get_datasets(url = INSTANCES_URL):
    """
    Ensure datasets are downloaded and extracted into the repository's `data/` directory.
    
    Args:
        url (str, optional): URL of the dataset archive.
    """
    def archive_filename(url):
        """Extract the archive filename from a URL."""
        return os.path.basename(urlparse(url).path)

    def delete_dir(dirpath):
        """Delete a directory tree if it exists, ignoring errors."""
        try:
            shutil.rmtree(dirpath)
        except Exception:
            pass

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    archive_location = os.path.join(DOWNLOAD_DIR, archive_filename(url))

    if os.path.exists(archive_location):
        print(f"Datasets already exist at {archive_location}, skipping download.")
    else:
        download_datasets(url, archive_location)
        # new datasets archive, delete old datasets
        delete_dir(DATASETS_DIR)

    if os.path.exists(DATASETS_DIR):
        print(f"Data already extracted at {DATASETS_DIR}, skipping extraction.")
    else:
        extract_tgz(archive_location, DATASETS_DIR)
