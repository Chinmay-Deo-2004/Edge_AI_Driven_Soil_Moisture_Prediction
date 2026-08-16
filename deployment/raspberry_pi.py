"""Offline webcam inference with a trained SVM RGB-histogram pipeline."""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import cv2
import joblib

# Allow `python deployment/raspberry_pi.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.rgb_histogram import extract_rgb_histogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to svm_pipeline.joblib")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between predictions")
    args = parser.parse_args()
    model = joblib.load(args.model)
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Check --camera-index and permissions.")
    last_prediction, last_time = "warming up", 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam frame capture failed")
            now = time.monotonic()
            if now - last_time >= args.interval:
                feature = extract_rgb_histogram(frame, bins=args.bins).reshape(1, -1)
                last_prediction = str(model.predict(feature)[0])
                last_time = now
            cv2.putText(frame, f"Moisture: {last_prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Soil Moisture Edge - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
