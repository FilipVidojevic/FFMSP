"""
git_utils.py
========

Utility functions which use git to obtain relevant information (e.g., repo root).

"""

import os
import subprocess

def get_git_root():
    """Return the absolute path of the git repository root."""
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return root
    except subprocess.CalledProcessError:
        return os.path.abspath(os.getcwd())  # fallback if not in a git repo