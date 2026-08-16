from __future__ import annotations

import cv2
import numpy as np


FEATURE_NAMES = ["mean_r", "mean_g", "mean_b", "mean_hsv_v", "mean_hls_l", "mean_xyz_z", "gray_mu", "gray_sigma"]


def extract_color_texture_features(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Image is empty or unreadable")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
    xyz = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2XYZ)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return np.array([
        *rgb.mean(axis=(0, 1)), hsv[:, :, 2].mean(), hls[:, :, 1].mean(),
        xyz[:, :, 2].mean(), gray.mean(), gray.std(ddof=0),
    ], dtype=np.float64)
