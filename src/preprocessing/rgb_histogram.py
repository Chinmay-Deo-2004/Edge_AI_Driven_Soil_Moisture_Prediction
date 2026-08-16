from __future__ import annotations

import cv2
import numpy as np


def extract_rgb_histogram(image_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty or unreadable")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hist, _ = np.histogramdd(rgb.reshape(-1, 3), bins=(bins, bins, bins), range=((0, 256),) * 3)
    vector = hist.ravel().astype(np.float64)
    return vector / max(vector.sum(), 1.0)
