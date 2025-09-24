"""
dataset_utils.py
Small utilities for the MVTEC dataset loader.
"""

from typing import List
from pathlib import Path
import os

def list_classes(root: str) -> List[str]:
    p = Path(root)
    classes = [d.name for d in sorted(p.iterdir()) if d.is_dir()]
    return classes

def detect_splits_for_class(class_root: str):
    p = Path(class_root)
    splits = []
    for name in ["train", "test", "test_private", "test_private_mixed"]:
        if (p / name).exists():
            splits.append(name)
    # fallback
    if not splits:
        for d in p.iterdir():
            if d.is_dir():
                splits.append(d.name)
    return splits

def find_image_files(folder: str, exts=None):
    if exts is None:
        exts = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    p = Path(folder)
    files = []
    if not p.exists():
        return files
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(str(f))
    return files
