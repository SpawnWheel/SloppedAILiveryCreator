"""
File system and utility helper functions.
"""

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional


def get_app_dir() -> Path:
    """Returns the root directory of the application."""
    return Path(__file__).resolve().parent.parent.parent


def get_tools_dir() -> Path:
    """Returns the directory where external binaries (texconv.exe) reside."""
    tools_dir = get_app_dir() / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    return tools_dir


def get_cache_dir() -> Path:
    """Returns the directory used for offline template caching."""
    cache_dir = get_app_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_temp_dir() -> Path:
    """Creates and returns a temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ams2_livery_"))
    return temp_dir


def cleanup_dir(path: Path) -> None:
    """Safely removes a directory and its contents."""
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_string_hash(text: str) -> str:
    """Computes SHA-256 hash of a string (such as a URL)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    """
    Converts a string into a filesystem-safe slug.
    Example: 'Gulf Sim Racing #42' -> 'gulf_sim_racing_42'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_") or "custom_livery"
