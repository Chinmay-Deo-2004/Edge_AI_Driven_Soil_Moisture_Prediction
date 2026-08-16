from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing.labels import build_manifest


def load_features(data_dir, extractor, **kwargs):
    manifest = build_manifest(data_dir)
    features = []
    for path in manifest.path:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        features.append(extractor(image, **kwargs))
    return manifest, np.vstack(features)


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
