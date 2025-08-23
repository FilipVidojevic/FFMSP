"""
clean.py
========

Utility script for cleaning the repository from generated or downloaded files.

Typical usage:
    python scripts/clean.py

Notes:
    - Safe to run multiple times: missing files/directories are ignored.
"""

import os
import shutil
from utils.git_utils import get_git_root

GIT_ROOT = get_git_root()

def remove_dirs_by_name(root: str, target: str):
    """
    Recursively remove all directories named `target` inside `root`.
    Example: remove_dirs_by_name(".", "__pycache__")
    """
    removed = False
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        if target in dirnames:
            dir_to_remove = os.path.join(dirpath, target)
            try:
                shutil.rmtree(dir_to_remove)
                print(f"Removed directory: {dir_to_remove}")
                removed = True
            except Exception as e:
                print(f"Could not remove {dir_to_remove}: {e}")
    if not removed:
        print(f"No directories named {target} found under {root}.")

remove_dirs_by_name(GIT_ROOT, "__pycache__")
remove_dirs_by_name(GIT_ROOT, "data")